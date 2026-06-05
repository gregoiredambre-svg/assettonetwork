# Network-Aware Road Maintenance Modelling

This is the cleaned project for the graph-based road-maintenance dissertation pipeline.
It keeps one coherent flow:

1. Build pavement material summaries.
2. Build a section-level interdependency graph.
3. Build a temporal section-year panel.
4. Train local, GCN, and relation-aware R-GCN deterioration models.
5. Estimate network-wide impacts of project/closure combinations.

The graph is a section-level interdependency graph, not a full traffic-assignment road network.

## Core Files

- `pipeline.py` - single orchestration entry point.
- `load_materials.py` - creates anti-leakage first-construction material summaries.
- `graph_construction.py` - builds nodes, graph edges, project events, and conflict edges.
- `graph_model_temporal.py` - builds the temporal panel and trains local/GCN deterioration models.
- `part1_extensions.py` - trains relation-aware R-GCN models.
- `ensemble.py` - combines local RF and graph predictions.
- `graph_model.py` - builds project-combination scenarios and trains the network-impact surrogate.
- `evaluation.py` - shared metrics and reporting helpers.

## Data Layout

Raw inputs stay under `Research Data/`.

Key generated graph/model outputs stay under `graph_data/`:

- `nodes.csv`, `edges.csv`, `projects.csv`
- `section_materials.csv`
- `temporal_rgcn_full_refined.pt`
- `ensemble_results.json`
- `network_model_metrics_full_refined.json`
- `network_scenario_predictions_full_refined.csv`

The only retained report summaries are under `reports/`:

- `gcn_temporal_metrics_full_refined.json`
- `part1_rgcn_temporal.csv`
- `part1_rgcn_temporal.json`

## Run The Pipeline

Run everything:

```bash
python3 pipeline.py
```

Run selected steps:

```bash
python3 pipeline.py materials graph
python3 pipeline.py temporal rgcn
python3 pipeline.py network
```

Preview commands without running:

```bash
python3 pipeline.py --dry-run
```

## Main Interpretation

- `full_refined` is the main graph view for deterioration modelling.
- R-GCN is the clearest interpretable graph model because it keeps separate relation channels: spatial, same-route, and same-functional-class.
- RF/R-GCN ensemble results are kept in `graph_data/ensemble_results.json`.
- The network-impact model estimates proxy impacts of project combinations: extra travel-time proxy, connectivity loss, disconnected OD share, and an overall disruption score.
