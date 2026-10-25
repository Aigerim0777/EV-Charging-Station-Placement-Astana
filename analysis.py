#!/usr/bin/env python3
"""Independent reconstruction of the reduced Astana GIS-MCDM pilot.

Reads the actual OSM GeoJSON and recomputes masks, criterion scores, Entropy,
CRITIC, TOPSIS, robustness and station agreement. This is newly reconstructed
code, NOT the original program used to author the paper. The comparison report
flags differences from published values instead of silently forcing a match.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import rankdata, spearmanr
from shapely.geometry import Polygon, Point, box, shape
from shapely.ops import transform, unary_union
from shapely.prepared import prep
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CORE = (71.38, 51.10, 71.49, 51.17)  # min longitude, latitude, max longitude, latitude
CRITERIA = ['C1', 'C2', 'C3', 'C5', 'C7']
NAMES = {'C1': 'Road distance', 'C2': 'Parking distance',
         'C3': 'POI density', 'C5': 'Substation distance', 'C7': 'Fuel distance'}
BENEFIT = {'C3'}
SEED = 42
REFERENCE_WEIGHTS = {
    'Equal': [0.2]*5,
    'Entropy': [0.0646, 0.0676, 0.6199, 0.1369, 0.1110],
    'CRITIC': [0.1663, 0.1768, 0.1708, 0.2666, 0.2195],
}
REFERENCE_AUC = {'Equal': 0.7231, 'Entropy': 0.7037, 'CRITIC': 0.7248}
MAIN_ROADS = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary'}
ALL_ROADS = MAIN_ROADS | {'residential', 'service', 'unclassified', 'living_street'}


def highway_ok(p, types):
    val = str(p.get('highway', ''))
    return (val in types or (val.endswith('_link') and val[:-5] in MAIN_ROADS)) and p.get('access') not in ('no', 'private')


def classify(p):
    """Tag-defined, potentially overlapping inventory groups."""
    d = {}
    d['main_roads'] = highway_ok(p, MAIN_ROADS)
    d['drive_roads'] = highway_ok(p, ALL_ROADS)
    d['parking'] = p.get('amenity') == 'parking' and p.get('access') not in ('no', 'private', 'customers')
    d['pois'] = (p.get('shop') not in (None, 'no', 'vacant') or
                 p.get('office') not in (None, 'no', 'vacant') or
                 p.get('amenity') in ('restaurant', 'cafe', 'hospital', 'university')) and p.get('amenity') != 'charging_station'
    d['substations'] = p.get('power') == 'substation'
    d['fuel'] = p.get('amenity') == 'fuel'
    d['charging'] = p.get('amenity') == 'charging_station' and p.get('access') not in ('no', 'private')
    d['water'] = p.get('natural') in ('water', 'wetland')
    d['parks'] = p.get('leisure') == 'park' or p.get('boundary') == 'protected_area'
    d['rail'] = p.get('railway') == 'rail'
    d['airport'] = p.get('aeroway') in ('aerodrome', 'runway', 'apron')
    return d


def get_cells():
    """Generate resolution-9 cells by centre containment, including edge candidates."""
    w,s,e,n = CORE
    polygon = h3.LatLngPoly([(s,w),(s,e),(n,e),(n,w)])
    inside = set(h3.polygon_to_cells(polygon, 9))
    candidates = inside | {nb for c in inside for nb in h3.grid_disk(c, 1)}
    candidates = [c for c in candidates if s <= h3.cell_to_latlng(c)[0] <= n
                  and w <= h3.cell_to_latlng(c)[1] <= e]
    return sorted(candidates)


def auc_rank(y, scores):
    """Exact Mann-Whitney / ROC AUC, ties receive half credit."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n1, n0 = int(y.sum()), int(len(y)-y.sum())
    if not n1 or not n0: return float('nan')
    return float((rankdata(scores)[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def normalize(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(axis=0), x.max(axis=0)
    if np.any(hi <= lo): raise ValueError('Constant criterion detected')
    z = (x-lo)/(hi-lo)
    for j, name in enumerate(CRITERIA):
        if name not in BENEFIT: z[:,j] = 1-z[:,j]
    return z


def get_weights(x):
    m = len(x)
    p = x/x.sum(axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        plnp = np.where(p > 0, p*np.log(np.where(p > 0,p,1)), 0)
    entropy = -plnp.sum(axis=0)/np.log(m)
    diversity = 1-entropy
    w_entropy = diversity/diversity.sum()
    # Sample standard deviation, Pearson correlations: CRITIC.
    corr = np.corrcoef(x, rowvar=False)
    contrast = x.std(axis=0,ddof=1)*np.sum(1-corr,axis=1)
    w_critic = contrast/contrast.sum()
    return {'Equal': np.full(5,0.2),'Entropy':w_entropy,'CRITIC':w_critic}


def top_set(scores, cells, frac):
    n = math.ceil(len(cells)*frac)
    return set(sorted(cells, key=lambda c: (-scores[c], c))[:n])


def jaccard(a,b):
    return len(a & b)/len(a | b) if a | b else float('nan')


def zones_for_method(method, top, normalized, index, weights, polygon_by_cell):
    left = set(top)
    groups = []
    while left:
        current = {min(left)}; left -= current; group = set(current)
        while current:
            next_group = {z for c in current for z in h3.grid_disk(c,1) if z in left}
            group |= next_group; left -= next_group; current = next_group
        groups.append(sorted(group))
    mat = np.asarray([normalized[[index[c] for c in grp]].mean(axis=0) for grp in groups])
    den = np.linalg.norm(mat,axis=0); den[den == 0] = 1
    weighted = (mat/den)*weights
    positive,negative = weighted.max(axis=0),weighted.min(axis=0)
    dp = np.linalg.norm(weighted-positive,axis=1)
    dn = np.linalg.norm(weighted-negative,axis=1)
    cc = np.divide(dn,dp+dn,out=np.full(len(dp),0.5),where=(dp+dn)>0)
    records=[]
    for grp,v in zip(groups,cc):
        geom=unary_union([polygon_by_cell[c] for c in grp])
        centroid=geom.centroid
        lon,lat=Transformer.from_crs(32642,4326,always_xy=True).transform(centroid.x,centroid.y)
        records.append({'method':method,'cells':len(grp),'h3_cells':';'.join(grp),
                        'latitude':lat,'longitude':lon,'area_km2':geom.area/1e6,'CC':float(v)})
    return sorted(records,key=lambda r:-r['CC'])


def bootstrap_auc(y, scores, blocks, rng, reps=2000):
    b=np.asarray(blocks); unique=np.unique(b); members=[np.where(b==key)[0] for key in unique]
    samples=[]
    for _ in range(reps):
        idx=np.concatenate([members[k] for k in rng.integers(len(unique),size=len(unique))])
        if len(np.unique(y[idx]))==2: samples.append(auc_rank(y[idx],scores[idx]))
    return list(np.percentile(samples,[2.5,97.5])),len(samples)



def joint_block_bootstrap(labels, score_matrix, blocks, rng, reps=2000):
    """Paired block bootstrap: *identical* resampled rows for all methods."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(score_matrix, dtype=float)
    blocks = np.asarray(blocks)
    block_ids = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in block_ids]
    out = []
    attempts = 0
    while len(out) < reps and attempts < reps * 20:
        attempts += 1
        selected = rng.integers(0, len(members), size=len(members))
        indices = np.concatenate([members[j] for j in selected])
        if len(np.unique(labels[indices])) != 2:
            continue
        out.append([auc_rank(labels[indices], scores[indices, k]) for k in range(scores.shape[1])])
    if len(out) < reps:
        raise RuntimeError('Too few valid bootstrap draws; inspect spatial blocks')
    return np.asarray(out)


def holm_adjust(pvalues):
    """Holm step-down family-wise p-value adjustment."""
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    sorted_adj = np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order])
    out = np.empty_like(pvalues)
    out[order] = np.minimum(1.0, sorted_adj)
    return out


def block_permutation_auc(labels, baseline, alternative, blocks, rng, reps=9999):
    """Exploratory two-sided paired score-swap test on common percentile ranks.

    For each evaluation block, swap the two method-score columns with
    probability one-half; within-block labels and row positions stay fixed.
    The null depends on block-level exchangeability, not independent stations.
    """
    labels = np.asarray(labels, dtype=int)
    blocks = np.asarray(blocks)
    baseline = rankdata(baseline) / len(baseline)
    alternative = rankdata(alternative) / len(alternative)
    observed = auc_rank(labels, alternative) - auc_rank(labels, baseline)
    codes = np.unique(blocks, return_inverse=True)[1]
    nblocks = int(codes.max()) + 1
    extreme = 0
    for _ in range(reps):
        swap = rng.integers(0, 2, size=nblocks).astype(bool)[codes]
        a = np.where(swap, alternative, baseline)
        b = np.where(swap, baseline, alternative)
        diff = auc_rank(labels, b) - auc_rank(labels, a)
        extreme += abs(diff) >= abs(observed) - 1e-12
    return {'observed_difference': float(observed),
            'permutations': reps, 'extreme': extreme,
            'p_two_sided': float((extreme + 1) / (reps + 1))}


def save_suitability_map(out, eligible, polygons, score, stations):
    """Generate an original vector PDF directly from computed H3 polygons."""
    from matplotlib.collections import PolyCollection
    names = ('Equal', 'Entropy', 'CRITIC')
    fig, axs = plt.subplots(1, 3, figsize=(13, 5), sharex=True, sharey=True)
    verts = [np.asarray(polygons[c].exterior.coords) / 1000 for c in eligible]
    for ax, name in zip(axs, names):
        collection = PolyCollection(verts, array=np.asarray([score[name][c] for c in eligible]),
                                    cmap='viridis', edgecolors='none', linewidths=0,
                                    vmin=0, vmax=1)
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.set_title(name)
        ax.set_xlabel('UTM 42N easting (km)')
        if stations:
            station_xy = np.asarray([[p.x / 1000, p.y / 1000] for _, p in stations])
            ax.scatter(station_xy[:, 0], station_xy[:, 1], facecolors='none',
                       edgecolors='red', s=15, linewidths=0.6)
    axs[0].set_ylabel('UTM 42N northing (km)')
    fig.colorbar(collection, ax=axs, shrink=0.70, label='Suitability (0 to 1)')
    fig.suptitle('Astana five-criterion pilot; circles = eligible mapped stations')
    fig.savefig(out / 'suitability_maps.pdf', bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default='export.geojson')
    ap.add_argument('--output',default='results')
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    data=json.loads(Path(args.input).read_text(encoding='utf-8'))
    features=data['features']
    forward=Transformer.from_crs(4326,32642,always_xy=True).transform
    core_ll=box(*CORE);core_utm=transform(forward,core_ll)
    layers={k:[] for k in classify({})}; source_counts=Counter(); core_counts=Counter()
    for f in features:
        p=f['properties']; flags=classify(p)
        geo=shape(f['geometry']); geo_utm=transform(forward,geo)
        for k,yes in flags.items():
            if yes:
                source_counts[k]+=1
                if geo.intersects(core_ll):core_counts[k]+=1
                layers[k].append(geo_utm)
    print('Read',len(features),'features; timestamp',data.get('timestamp'))
    print('Source counts:',dict(source_counts)); print('Core counts:',dict(core_counts))
    ids=get_cells()
    polygons={c: transform(forward,Polygon([(lng,lat) for lat,lng in h3.cell_to_boundary(c)])) for c in ids}
    points={c: Point(*forward(h3.cell_to_latlng(c)[1],h3.cell_to_latlng(c)[0])) for c in ids}
    excludes={
      'boundary': [c for c in ids if not core_utm.covers(polygons[c])],
      'water': [],'parks': [],'rail': [],'airport': [],'road_access': []}
    for key in ('water','parks','rail','airport'):
        geoms=[g for g in layers[key] if isinstance(g,(Polygon,)) or g.geom_type=='MultiPolygon']
        if key=='rail': geoms=[g.buffer(20) for g in layers[key]]
        if not geoms:continue
        merged=unary_union(geoms)
        excludes[key]=[c for c in ids if polygons[c].intersects(merged) and polygons[c].intersection(merged).area>1]
    drivable=unary_union(layers['drive_roads'])
    excludes['road_access']=[c for c in ids if points[c].distance(drivable)>500]
    bad=set().union(*(set(v) for v in excludes.values()))
    eligible=sorted(set(ids)-bad)
    print('Cells:',len(ids),'excluded',len(bad),'eligible',len(eligible))
    print('Overlapping mask counts:',{k:len(v) for k,v in excludes.items()})
    if len(eligible)<2:raise RuntimeError('Too few eligible cells; inspect mask/tag rules')
    # Metric nearest-geometry distances, not distance to representative points.
    distance_features=['main_roads','parking','substations','fuel']
    merged={k:unary_union(layers[k]) for k in distance_features}
    # 500-m Gaussian KDE of one representative point per unique OSM feature.
    poi=np.array([[g.representative_point().x,g.representative_point().y] for g in layers['pois']])
    coords=np.array([[points[c].x,points[c].y] for c in eligible])
    bandwidth=500.0
    from scipy.spatial import cKDTree
    tree=cKDTree(poi)
    neighbors=tree.query_ball_point(coords,4*bandwidth)
    density=np.asarray([sum(np.exp(-np.sum((poi[near]-xy)**2,axis=1)/(2*bandwidth**2)))
                        for near,xy in zip(neighbors,coords)])/(2*np.pi*bandwidth**2)*1e6
    raw=np.column_stack([np.array([points[c].distance(merged['main_roads']) for c in eligible]),
        np.array([points[c].distance(merged['parking']) for c in eligible]),density,
        np.array([points[c].distance(merged['substations']) for c in eligible]),
        np.array([points[c].distance(merged['fuel']) for c in eligible])])
    x=normalize(raw); weights=get_weights(x); index={c:i for i,c in enumerate(eligible)}
    score={k:dict(zip(eligible,x@w)) for k,w in weights.items()}
    pd.DataFrame({'criterion':CRITERIA,**{k:v for k,v in weights.items()}}).to_csv(out/'weights.csv',index=False)
    df=pd.DataFrame({'h3':eligible,'east_m':coords[:,0],'north_m':coords[:,1],
       **{c:raw[:,j] for j,c in enumerate(CRITERIA)},
       **{k+'_score':list(s.values()) for k,s in score.items()}})
    df.to_csv(out/'cells.csv',index=False)
    top10={k:top_set(v,eligible,0.1) for k,v in score.items()}
    pairs=[]
    for a,b in [('Equal','Entropy'),('Equal','CRITIC'),('Entropy','CRITIC')]:
        pairs.append({'pair':a+'-'+b,'spearman':spearmanr(list(score[a].values()),list(score[b].values())).statistic,
                      'jaccard':jaccard(top10[a],top10[b]),'shared_cells':len(top10[a]&top10[b])})
    pd.DataFrame(pairs).to_csv(out/'agreement.csv',index=False)
    zr=[]
    for name,w in weights.items():
        zr+=zones_for_method(name,top_set(score[name],eligible,0.05),x,index,w,polygons)
    pd.DataFrame(zr).to_csv(out/'zones.csv',index=False)
    sensitivity=[]
    for name,w in weights.items():
        for j,c in enumerate(CRITERIA):
            for mag in (0.1,0.2,0.3,0.5):
                for sign in (-1,1):
                    new=w.copy();new[j]=w[j]*(1+sign*mag)
                    mask=np.arange(5)!=j;new[mask]=w[mask]*(1-new[j])/(1-w[j])
                    perturbed=dict(zip(eligible,x@new))
                    sensitivity.append({'method':name,'criterion':c,'magnitude_pct':int(100*mag),
                                        'direction':sign,'jaccard':jaccard(top10[name],top_set(perturbed,eligible,.1))})
    sens=pd.DataFrame(sensitivity);sens.to_csv(out/'sensitivity.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    for name,g in sens.groupby('method'):
        mins=g.groupby('magnitude_pct')['jaccard'].min()
        ax.plot(mins.index,mins.values,marker='o',label=name)
    ax.axhline(.8,color='gray',ls='--');ax.set(xlabel='Relative weight change (%)',ylabel='Minimum top-decile Jaccard',ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(out/'sensitivity.pdf');plt.close(fig)
    # The charging records are left out of every criterion construction stage.
    stations=[]
    for g in layers['charging']:
        point=g.representative_point()
        ll=Transformer.from_crs(32642,4326,always_xy=True).transform(point.x,point.y)
        if core_ll.covers(Point(*ll)):
            cell=h3.latlng_to_cell(ll[1],ll[0],9);stations.append((cell,point))
    retained=[(c,p) for c,p in stations if c in index]
    occupied=set(c for c,_ in retained)
    negative=sorted(set(eligible)-occupied)
    rng=np.random.default_rng(SEED)
    sampled=sorted(rng.choice(negative,size=min(110,len(negative)),replace=False).tolist())
    sample=sorted(occupied)+sampled
    labels=np.array([int(c in occupied) for c in sample])
    block_ids=np.array([f'{int(np.floor(points[c].x/1000))}:{int(np.floor(points[c].y/1000))}' for c in sample])
    # Convert strings to integer block indices so np.unique remains straightforward.
    _,block_codes=np.unique(block_ids,return_inverse=True)
    # Fixed shared evaluation sample. Compute *paired* AUC intervals and tests.
    method_names = ['Equal', 'Entropy', 'CRITIC']
    score_matrix = np.column_stack([[score[name][c] for c in sample] for name in method_names])
    observed_aucs = np.asarray([auc_rank(labels, score_matrix[:, j]) for j in range(3)])
    boot = joint_block_bootstrap(labels, score_matrix, block_codes, rng, reps=2000)
    ci = np.percentile(boot, [2.5, 97.5], axis=0)
    delta_boot = boot[:, 1:] - boot[:, [0]]
    delta_ci = np.percentile(delta_boot, [2.5, 97.5], axis=0)
    validation=[]
    for j,name in enumerate(method_names):
        high=top_set(score[name],eligible,.4)
        hits=sum(c in high for c,p in retained)
        validation.append({'method':name,'auc':float(observed_aucs[j]),
                           'auc_ci_low':float(ci[0,j]),'auc_ci_high':float(ci[1,j]),
                           'bootstrap_valid':len(boot),'hits':hits,
                           'stations_retained':len(retained),
                           'hit_rate':hits/len(retained) if retained else np.nan,
                           'auc_full_background':auc_rank(
                               [int(c in occupied) for c in eligible],
                               [score[name][c] for c in eligible])})
    pd.DataFrame(validation).to_csv(out/'validation.csv',index=False)
    pd.DataFrame({'h3_cell':sample,'station_occupied':labels,'block':block_ids,
                  **{k+'_score':score_matrix[:,j] for j,k in enumerate(method_names)}}).to_csv(
                     out/'validation_sample.csv',index=False)
    # Permutation RNG has an independent stream: adding new bootstraps will
    # not silently change the permutation test results.
    permutation_rng = np.random.default_rng(SEED + 1)
    tests = [block_permutation_auc(labels,score_matrix[:,0],score_matrix[:,j],
                                  block_codes,permutation_rng,reps=9999) for j in (1,2)]
    adjusted = holm_adjust([t['p_two_sided'] for t in tests])
    comparisons = []
    for k,(j,test) in enumerate(zip((1,2),tests)):
        comparisons.append({'comparison':method_names[j]+' vs Equal',
                            'auc_difference':float(observed_aucs[j]-observed_aucs[0]),
                            'paired_ci_low':float(delta_ci[0,k]),
                            'paired_ci_high':float(delta_ci[1,k]),
                            'permutation_p':test['p_two_sided'],
                            'holm_p':float(adjusted[k]),
                            'test_blocks':len(np.unique(block_codes)),
                            'permutations':9999})
    pd.DataFrame(comparisons).to_csv(out/'statistical_tests.csv',index=False)
    print('Statistical tests (reconstruction):',json.dumps(comparisons,indent=2))
    save_suitability_map(out,eligible,polygons,score,retained)
    # Block size robustness: independent, fixed random streams.
    spatial_checks=[]
    for width in (500,2000):
        test_blocks=np.asarray([f'{int(np.floor(points[c].x/width))}:'
                                f'{int(np.floor(points[c].y/width))}' for c in sample])
        test_codes=np.unique(test_blocks,return_inverse=True)[1]
        check_rng=np.random.default_rng(SEED + width)
        rep=joint_block_bootstrap(labels,score_matrix,test_codes,check_rng,reps=1000)
        intervals=np.percentile(rep,[2.5,97.5],axis=0)
        for j,name in enumerate(method_names):
            spatial_checks.append({'block_size_m':width,'method':name,
                                   'ci_low':float(intervals[0,j]),
                                   'ci_high':float(intervals[1,j])})
    pd.DataFrame(spatial_checks).to_csv(out/'block_size_checks.csv',index=False)
    # Comparison is never used to change computed data.
    report={'source_features':len(features),'cells_initial':len(ids),'cells_excluded':len(bad),
            'cells_eligible':len(eligible),'core_station_records':len(stations),
            'eligible_station_records':len(retained),'occupied_cells':len(occupied),
            'scenarios':len(sensitivity),
            'computed_weights':{k:[float(v) for v in ws] for k,ws in weights.items()},
            'reference_weights':REFERENCE_WEIGHTS,
            'computed_validation':validation,'computed_tests':comparisons,'reference_auc':REFERENCE_AUC,
            'note':'Independent reconstruction; discrepancies are not modified to match the paper.'}
    (out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print('All generated outputs in',out)

if __name__=='__main__':main()
