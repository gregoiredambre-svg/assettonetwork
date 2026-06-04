# Scripts

This folder contains experiment runners and reporting utilities that are useful for reproduction, sensitivity checks, and one-off dissertation analyses.

## Main scripts

- `run_ensemble.py`
  RF + graph ensemble experiments.
- `run_singletask_per_distress.py`
  Production single-task R-GCN by distress type.
- `run_materials_experiments.py`
  Materials-enriched pavement similarity and `full_refined` weight sweep.
- `run_traffic_sensitivity.py`
  Annual traffic proxy sensitivity around maintenance events.
- `run_ablation.py`
  Similarity-factor ablation and cluster-size sweep.
- `run_multitask.py`
  Multi-task R-GCN benchmark.
- `run_osm_validation_reporting.py`
  OSM validation interpretation and reporting.
- `run_distress_analysis.py`
  Shorter distress-target profiling and comparison outputs.
- `run_distress_full_inventory.py`
  Full workbook inventory and methodology audit.

All scripts are intended to be run from the repository root, for example:

```bash
./.venv/bin/python scripts/run_ensemble.py
```
