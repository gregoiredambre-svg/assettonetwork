# Part 1 Readiness Audit

Date: 2026-05-19

Project: `From Asset to Network: Graph-Based Optimisation of Interdependent Maintenance Decisions for Transportation Networks`

## Bottom line

The project has **substantively covered Part 1** of the brief:

1. a section-level graph has been built from LTPP and MERRA data;
2. historical treatment/project events are now mapped through `EXPERIMENT_SECTION`;
3. spatial, corridor, functional-similarity, diversion, and simultaneous-project conflict relationships are encoded;
4. graph-based models have been trained for both network-wide disruption proxies and section-level temporal degradation.

The main remaining issue is **not absence of work**, but **how strongly the work can be defended**. The graph should be described as a **section-level interdependency graph**, not as a complete operational road topology. The static disruption targets are **synthetic graph-based proxies**, not observed traffic diversion outcomes.

## Requirement-by-requirement assessment

### 1) Build a graph representation of the road network

Status: **Done**

Evidence:
- `graph_data/nodes.csv`: `2,426` pavement-section nodes across `49` states
- `graph_data/edges.csv`: `14,014` structural edges
  - `10,186` spatial
  - `897` same-route corridor
  - `2,931` filtered functional-similarity
- diagnostics available in:
  - `reports/graph_diagnostics.csv`
  - `reports/graph_distance_summary.csv`

Interpretation:
- This is a **section-level interdependency graph**.
- It is **not** a full national road-topology model.
- The graph is fragmented and should be interpreted as a set of **local interdependency networks**.

### 2) Identify maintenance projects and map them to graph nodes/edges

Status: **Done**

Evidence:
- `graph_construction.py` now uses `EXPERIMENT_SECTION`, not `PROJECT_HIST_AGE_EXP`
- `graph_data/projects.csv`: `7,163` graph-linked dated project/treatment records
- coverage: all `2,426` graph nodes have treatment/project linkage potential through the graph-level event table

Interpretation:
- The project-history layer is now semantically aligned with dated treatment/change events.
- This is stronger than the former `PROJECT_HIST_AGE_EXP` proxy.

### 3) Encode spatial, temporal, and functional interdependencies

Status: **Done**

Evidence:
- adjacency:
  - `spatial` edges
  - `same_route` corridor edges
- functional interdependency:
  - `same_functional_class` filtered similarity edges
- diversion logic:
  - `diversion_potential` on graph edges
  - research-network edges used in the static surrogate
- simultaneous project conflicts:
  - `graph_data/project_conflicts.csv`: `82,010` event-level conflicts
  - `graph_data/node_project_conflicts.csv`: `9,980` section-level conflict summaries
- temporal treatment context:
  - `graph_model_temporal.py` builds `EXPERIMENT_SECTION` treatment features and neighbour treatment features

Interpretation:
- The project now encodes the main interdependency types requested in Part 1.
- Conflict relationships are rule-based overlap proxies derived from dated treatment events on neighbouring sections.

### 4) Train a graph-based model variant to learn network impacts and neighbour effects

Status: **Done**

Evidence:
- static graph surrogate:
  - `graph_model.py`
  - `1,500` simulated closure scenarios
  - four disruption proxy targets
- temporal graph degradation model:
  - `graph_model_temporal.py`
  - one-year cracking prediction with neighbour and treatment context

Current best static GCN test R²:
- `delta_vht_proxy`: `0.254042` (`full_refined`)
- `connectivity_loss_pct`: `0.551947` (`full_refined`)
- `disconnected_od_pct`: `0.250141` (`spatial`)
- `disruption_score`: `0.235045` (`full_refined`)

Current temporal test R²:
- `RF local`: `0.420178` (`spatial`), `0.409446` (`spatial_route`), `0.409299` (`full_refined`)
- `GCN without project/treatment features`: `0.049204`, `0.048128`, `0.048830`
- `GCN with EXPERIMENT_SECTION treatment features`: `0.218790`, `0.218122`, `0.215056`

Interpretation:
- The graph model does learn meaningful structure, especially once treatment context is added.
- The static model estimates **network-wide proxy impacts** of project combinations.
- The temporal model does not beat RF locally, but it does show **relational signal** from treatment and neighbour context.

## What is already defensible in the dissertation

- A graph representation has been built and empirically analysed.
- Historical treatment/change events are mapped to graph nodes using `EXPERIMENT_SECTION`.
- Spatial, corridor, functional-similarity, diversion, and conflict interdependencies are encoded.
- Graph-based models have been trained for both static disruption estimation and temporal degradation forecasting.

## What still limits a “100% complete” claim

These are not blockers for saying Part 1 is completed, but they are the main places where the defence can still be strengthened:

1. **Topology realism**
   - The graph is not yet an OSM-derived operational road topology.
   - If the argument requires “the road network” in a strict topological sense, an OSM-based layer would strengthen the claim.

2. **Explicit relation-aware graph learning**
   - Multiple edge types exist, but the current GCNs do not yet use a full relational architecture such as `R-GCN`.
   - This is the clearest literature-backed methodological upgrade.

3. **Out-of-distribution validation**
   - The current splits are temporal for the degradation task and random for the static scenario surrogate.
   - Validation across unseen states/corridors would strengthen the interdependency claim.

4. **Proxy interpretation**
   - Static disruption targets are synthetic graph-based proxies, not observed diversion outcomes.
   - This is acceptable, but it must be stated explicitly.

## Recommended next moves for the rest of the week

Priority 1:
- add a short methods subsection clarifying that this is a **section-level interdependency graph**
- add a short methods subsection clarifying that static targets are **proxy disruption metrics**

Priority 2:
- run one explicit **relation-aware GNN variant** (`R-GCN`-style) as a robustness extension, especially for the temporal model

Priority 3:
- run a lightweight **OOD validation** by state or corridor holdout

Priority 4:
- if time permits, test an **OSM-derived topology layer** as a comparison graph rather than as a full replacement

## Overall judgement

Part 1 is **substantively complete** and already suitable for supervisor discussion and dissertation reporting. What remains is mainly **strengthening the methodological defence**, not filling a missing core requirement.
