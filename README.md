# GIS-Based Spatial Analytics for Optimal Electric Vehicle Charging Station Placement

## Project Overview
This repository supports Assignment 3 (Results Section) for the Research Methods and Tools course. The study uses GIS-based multi-criteria analysis to assess potential public EV charging station locations in Astana, Kazakhstan.

## Dataset
`export.geojson` contains an OpenStreetMap export with 37,917 geographic features. The preliminary study used a reduced area of Astana and 219 eligible H3 cells.

## Methods
The pilot evaluates five spatial criteria using Entropy, CRITIC, and equal weighting. It includes TOPSIS candidate-zone ranking, sensitivity analysis, and validation against mapped charging stations.

## Reproducibility
The Python analysis script and exact execution instructions must be added to this repository to enable reproduction of the results.

## Data Attribution
OpenStreetMap contributors (ODbL).

## Reproducing the Analysis

The reduced GIS-MCDM pilot can be reproduced using Python and the provided OpenStreetMap dataset.

### Installation

```bash
pip install -r requirements.txt
```

### Running the Analysis

```bash
python analysis.py --input export.geojson --output results
```

### Generated Results

The analysis generates CSV, JSON, and PDF files containing criterion weights, H3 cell suitability scores, TOPSIS candidate rankings, sensitivity scenarios, and validation metrics.

The pilot uses 219 eligible H3 cells and compares Equal, Entropy, and CRITIC weighting methods.

### Reproducibility Notes

This script is an independent reconstruction based on the methodology described in the article, rather than the original analysis script. Main AUC values and several spatial results were reproduced. Some confidence interval values differ slightly, and the complete permutation-testing procedure remains to be implemented.
