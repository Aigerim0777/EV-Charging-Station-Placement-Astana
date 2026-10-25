# Astana EV charging study: corrected analysis implementation

This is an **independently reconstructed** script, not the original code used to prepare the manuscript. It uses the OSM `export.geojson` file already in the GitHub repository.

## Run in Colab (after uploading the input and script)

```python
!pip install -r /content/requirements.txt
```

Restart the runtime after dependency installation if Colab requests it, then:

```python
!python /content/analysis.py --input /content/export.geojson --output /content/results
```

## Changes compared to the previous script

- Fixes H3 `latlng_to_cell(...,9)` call.
- Generates `suitability_maps.pdf` from computed H3 geometry and scores.
- Uses identical block-bootstrap replicates for three AUCs and paired AUC differences.
- Computes an exploratory 9,999-swap **two-sided** within-spatial-block score-swap permutation test on common-scale percentile ranks and two-test Holm adjustment.
- Produces `statistical_tests.csv` and `block_size_checks.csv` in addition to prior outputs.

## Verification boundary

- The statistical routines were tested against saved Colab `cells.csv` and `validation_sample.csv`: Equal/Entropy/CRITIC AUC = 0.7231405 / 0.7037190 / 0.7247934. With the independent permutation RNG seed 43, reconstructed p values were 0.6278 (Entropy vs Equal) and 0.8580 (CRITIC vs Equal), both Holm-adjusted to 1.0.
- These unadjusted p values are **not identical** to the previously submitted PDF (0.6324, 0.8412); different block permutation implementation/random draw conventions can affect exact values. Bootstrap interval endpoints can also differ. Do **not** claim the reconstruction exactly reproduces the manuscript's complete inference procedure.
- We could not re-run the full spatial pipeline here because H3 was unavailable. Run the script in Colab to validate fresh end-to-end outputs and compare them with the article before updating numerical claims.
- Missing work: independent verification of mapped stations, AHP expert survey, population, land-use, slope screening and full-city coverage.

If publishing, keep the generated outputs with matching source version and report departures transparently.
