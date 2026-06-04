# Network-Aware Road Maintenance Modelling

This repository contains the modelling and analysis code for a Cambridge MPhil dissertation on graph-based pavement deterioration modelling and maintenance decision support.

## What to look at first

- `streamlit_app.py`
  Main interactive app used for presentation and dissertation walkthroughs.
- `graph_construction.py`
  Builds the section-level interdependency graph and exports graph artifacts.
- `graph_model_temporal.py`
  Prepares temporal panel data and baseline graph-learning inputs.
- `part1_extensions.py`
  Contains the relation-aware graph models and temporal R-GCN training logic.
- `evaluation.py`
  Shared evaluation metrics used across experiments.
- `ensemble.py`
  Ensemble logic combining local and graph-based predictors.

## Experiment runners

All one-off experiment and reporting entry points now live under `scripts/`.

- `scripts/run_ensemble.py`
- `scripts/run_singletask_per_distress.py`
- `scripts/run_materials_experiments.py`
- `scripts/run_traffic_sensitivity.py`
- `scripts/run_ablation.py`
- `scripts/run_multitask.py`
- `scripts/run_osm_validation_reporting.py`
- `scripts/run_distress_analysis.py`
- `scripts/run_distress_full_inventory.py`

## Final result files

Core results used by the app and dissertation are kept in:

- `reports/`
- `graph_data/`

Key current outputs include:

- `reports/treatment_feature_ablation.csv`
- `reports/part1_rgcn_temporal.csv`
- `reports/distress_model_comparison.csv`
- `reports/materials_weight_sweep.json`
- `reports/traffic_sensitivity.json`
- `graph_data/osm_validation_findings.json`
- `graph_data/ensemble_results.json`
- `graph_data/singletask_per_distress_results.json`

## Important interpretation

- The graph should be interpreted as a **section-level interdependency graph**, not as a fully routable road network.
- `spatial + route` is the most relevant physical graph view for disruption-style reasoning.
- `full_refined` is the most useful graph view for deterioration and treatment-context modelling.
- The strongest pure predictive model is the RF/graph ensemble, while the strongest interpretable graph model is the relation-aware R-GCN.

## Repository hygiene

- Large caches and intermediate archives are intentionally ignored via `.gitignore`.
- Older or intermediate experiment artifacts can be moved to local archive folders under `reports/archive/` and `graph_data/archive/` without affecting the app.
