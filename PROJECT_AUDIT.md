# PROJECT_AUDIT.md

## 1. Vue d'ensemble du repo
- Structure générale (max 3 niveaux, caches ignorés) :
```text
.
├── .streamlit/
│   └── config.toml
├── Research Data/
│   ├── Analysis Ready Distress.xlsx
│   ├── Annual Traffic Inputs Over Time.xlsx
│   ├── General Section Info.xlsx
│   └── MERRA - Temperature, Humidity, Precipitation, Wind, Solar/
├── graph_data/
│   ├── nodes.csv
│   ├── edges.csv
│   ├── projects.csv
│   ├── ensemble_results.json
│   └── osm_validation_findings.json
├── reports/
│   ├── graph_diagnostics.csv
│   ├── treatment_feature_ablation.csv
│   ├── part1_rgcn_temporal.csv
│   ├── distress_model_comparison.csv
│   ├── materials_weight_sweep.json
│   ├── traffic_sensitivity.json
│   └── osm_*.csv / osm_*.md / osm_*.json
├── scripts/
│   ├── run_ablation.py
│   ├── run_ensemble.py
│   ├── run_materials_experiments.py
│   ├── run_singletask_per_distress.py
│   └── ...
├── streamlit_app.py
├── graph_construction.py
├── graph_model.py
├── graph_model_temporal.py
├── part1_extensions.py
├── evaluation.py
├── ensemble.py
├── load_materials.py
└── osm_validation_full.py
```
- Stack technique détectée : Python; Streamlit; pandas; NumPy; scikit-learn; PyTorch; Plotly; NetworkX; OpenPyXL; Pillow (`requirements.txt:1-9`). Versions non épinglées dans `requirements.txt`.
- Fichiers de configuration importants : `requirements.txt`, `.streamlit/config.toml`, `.gitignore`, `README.md`.
- Taille approximative du projet : 20 fichiers `.py`, 0 notebook, ~13 194 lignes Python (comptage hors `.venv/.git/cache/archive`).
- Observation de reproductibilité : aucune version exacte de dépendances n’est figée (`requirements.txt:1-9`).

## 2. Données utilisées
- **Métadonnées sections LTPP**
  - Source : LTPP / General Section Info.
  - Format / localisation : `Research Data/General Section Info.xlsx`.
  - Utilisation : `SECTION_COORDINATES`, `PROJECT_ID_EXP`, `SECTION_GENERAL_EXP` pour `node_id`, lat/lon, `route_key`, `functional_class` (`graph_construction.py:149-205`).
  - Volume identifié : 2 426 nœuds après filtres US continentaux et bounding box (`graph_construction.py:162-177`, `reports/graph_diagnostics.csv:2-4`).
  - Préprocessing : normalisation de chaînes, filtre contiguous US, bounding box géographique, déduplication par section (`graph_construction.py:159-205`).
- **Distress LTPP**
  - Source : LTPP / Analysis Ready Distress.
  - Format / localisation : `Research Data/Analysis Ready Distress.xlsx`.
  - Sheets exploitées : `ANALYSIS_DIS_AC`, `ANALYSIS_DIS_CRCP`, `ANALYSIS_DIS_JPCC`, `TST_L05B` (`graph_construction.py:298-322`, `load_materials.py:171-220`).
  - Variables/features extraites : moyennes numériques par section pour tous les champs distress (`graph_construction.py:304-315`); panel temporel AC pour cibles `HPMS16_CRACKING_PERCENT_AC`, `MEPDG_CRACKING_PERCENT_AC`, `MEPDG_TRANS_CRACK_LENGTH_AC`, `PATCH_A`, `POTHOLES_A` (`graph_model_temporal.py:49-56`, `208-228`); matériaux/épaisseurs via `TST_L05B` (`load_materials.py:50-56`, `171-220`).
  - Volume identifiable :
    - AC : 13 097 lignes, 137 colonnes, 1 807 sections, 1988-08-05 à 2024-05-22 (`reports/analysis_ready_distress_audit.md:40-56`).
    - CRCP : 544 lignes, 123 colonnes, 108 sections, 1989-07-07 à 2022-08-10 (`reports/analysis_ready_distress_audit.md:59-75`).
    - JPCC : 5 750 observations, 794 sections, 36 années distinctes (`reports/distress_crcp_jpcc_sufficiency.md:11-17`).
  - Préprocessing : agrégation par `node_id_join`, annualisation par `YEAR` pour le modèle temporel, standardisation train-only, choix anti-leakage de la première construction pour les matériaux (`graph_model_temporal.py:208-228`, `805-872`; `load_materials.py:50-56`).
- **Trafic annuel**
  - Source : LTPP traffic trend workbook.
  - Format / localisation : `Research Data/Annual Traffic Inputs Over Time.xlsx`.
  - Sheets : `TRF_TREND`, `TRF_TREND_1`, `TRF_TREND_2` (`graph_construction.py:325-350`, `graph_model_temporal.py:237-255`).
  - Variables/features : `ANNUAL_ESAL_TREND`, `ANNUAL_GESAL_TREND`, `AADTT_ALL_TRUCKS_TREND`, `ANNUAL_TRUCK_VOLUME_TREND`, classes poids-lourds (`graph_model_temporal.py:237-255`).
  - Préprocessing : dernière année par section pour les features statiques (`graph_construction.py:338-343`), panel annuel par section pour le temporel (`graph_model_temporal.py:237-255`).
- **Climat MERRA**
  - Source : NASA MERRA dérivé local.
  - Format / localisation : dossier `Research Data/MERRA - Temperature, Humidity, Precipitation, Wind, Solar/`.
  - Variables/features : température bind, humidité relative, précipitation, évaporation, jours de pluie, vent, nébulosité, shortwave, freeze index / freeze-thaw (`graph_construction.py:208-295`).
  - Volume : 2 581 `node_id_join` avec 21 colonnes climatiques agrégées (`graph_construction.py:294`).
  - Préprocessing : jointure via `MERRA_ID`, moyenne interannuelle par grille / par section (`graph_construction.py:213-281`).
- **Événements de maintenance / projets**
  - Source : `EXPERIMENT_SECTION` LTPP.
  - Format / localisation : `Research Data/General Section Info.xlsx`, output `graph_data/projects.csv`.
  - Variables/features : dates de construction, `event_year`, `treatment_label`, `broad_treatment_group`, statut expérimental (`graph_construction.py:353-401`).
  - Préprocessing : parsing dates, fallback `ASSIGN_DATE`, classification de groupes de traitement, identifiant `project_id` (`graph_construction.py:371-399`, `1118-1125`).
- **OpenStreetMap**
  - Source : OSM via `osmnx`.
  - Format / localisation : checkpoints / synthèses dans `graph_data/osm_validation_findings.json`, `graph_data/osm_validation_partial.json`, `reports/osm_*`.
  - Utilisation : validation topologique externe des edges (`osm_validation_full.py`, `scripts/run_osm_validation_reporting.py`).
  - Volume : 16 024 arêtes évaluées au total; sous-ensemble spatial+route de 9 086 (`graph_data/osm_validation_findings.json:8-58`).

## 3. Construction des graphes
- **Spatial only**
  - Fonction principale : `graph_construction.py:build_spatial_edges` (`graph_construction.py:520-548`).
  - Paramètres détectés : `k=8`, rayon max `80 km` (`graph_construction.py:24-27`, `520-524`).
  - Logique : `NearestNeighbors` haversine, conservation des voisins à `distance_km <= 80` (`graph_construction.py:530-546`).
  - Nombre d’edges produit : 10 186 (`reports/graph_diagnostics.csv:2`).
- **Spatial + route**
  - Fonctions principales : `graph_construction.py:build_spatial_edges`, `graph_construction.py:build_route_edges` (`graph_construction.py:520-548`, `551-598`).
  - Paramètres détectés : même spatial que ci-dessus; route chain locale avec seuil `100 km` (`graph_construction.py:28`, `553-594`).
  - Logique : chaîne A-B-C sur un même `route_key`, ordonnée par `MILEPOINT` sinon ordre géographique (`graph_construction.py:555-595`).
  - Nombre d’edges produit : 11 965 au total, dont 1 779 `same_route` (`reports/graph_diagnostics.csv:3`).
- **Comprehensive / refined**
  - Fonction principale : `graph_construction.py:build_functional_edges`, assemblée par `graph_construction.py:assemble_graph` (`graph_construction.py:601-736`, `1084-1117`).
  - Paramètres détectés : `max_neighbors=5`, `max_distance_km=80` (`graph_construction.py:29-30`, `603-604`).
  - Poids par défaut détectés pour la similarité fonctionnelle : spatial `0.40`, traffic `0.15`, climate `0.20`, pavement `0.25` (`graph_construction.py:31-36`).
  - Formule de scoring : `score = w_spatial*exp(-distance/max_distance) + w_traffic*traffic_similarity + w_climate*climate_similarity + w_pavement*pavement_similarity` (`graph_construction.py:696-703`).
  - Similarité pavement actuelle : mélange continu+binaire incluant `NO_OF_LANES`, `LANE_WIDTH`, `SECTION_LENGTH`, nombre de couches, épaisseurs log-transformées, flags matériaux AC/PCC/base liée (`graph_construction.py:619-695`; `load_materials.py:171-220`).
  - Nombre d’edges produit : 17 944 au total dans le `full_refined` de référence, dont 5 979 `same_functional_class` (`reports/graph_diagnostics.csv:4`).
  - Variante matériaux/poids : un sweep ultérieur teste des poids alternatifs et reconstruit des graphes entre 17 967 et 17 981 edges (`reports/materials_weight_sweep.md:3-8`).
- **Poids de vues aval**
  - `edge_weight` générique : `0.40 spatial + 0.25 route + 0.15 traffic + 0.10 climate + 0.10 pavement` (`graph_construction.py:748-755`).
  - `weight_deterioration` : `0.35 spatial + 0.15 route + 0.20 traffic + 0.20 climate + 0.10 pavement` (`graph_construction.py:756-762`).
  - `weight_disruption` : `0.30 spatial + 0.40 route + 0.20 diversion_potential + 0.10 traffic` (`graph_construction.py:763-768`).

## 4. Modèles implémentés
- **Ridge / Random Forest temporels (tabulaires)**
  - Fichier : `graph_model_temporal.py:train_tabular_baselines` (`graph_model_temporal.py:1247-1292`).
  - Hyperparamètres : Ridge `alpha=1.0`; RF `n_estimators=300`, `max_depth=12`, `min_samples_leaf=3`, `n_jobs=-1`.
  - Target par défaut : `HPMS16_CRACKING_PERCENT_AC`; extension single-task / multi-task pour 5 distress (`graph_model_temporal.py:49-56`).
  - Features d’entrée : lagged distress, trafic annuel, climat annuel, historique de traitements/projets (via `build_temporal_panel` puis `prepare_temporal_data`; `graph_model_temporal.py:598`, `805-872`).
  - Split : train `<=2015`, val `2016-2018`, test `2019-2021` (`graph_model_temporal.py:43-48`, `805-872`).
- **Snapshot GCN**
  - Fichier : `graph_model_temporal.py:SnapshotGCN` (`graph_model_temporal.py:1010-1025`).
  - Hyperparamètres : 2 couches linéaires, `hidden_dim=64`, `dropout=0.2`; apprentissage Adam `lr=1e-3`, `weight_decay=1e-4`, `max_epochs=180`, `patience=20` (`graph_model_temporal.py:1078-1137`).
  - Target / features / split : même panel temporel que ci-dessus.
- **GCN + project history**
  - Implémentation : même base `SnapshotGCN`, variante d’entrée avec features projets activées; comparée dans `reports/treatment_feature_ablation.csv` et pilotée par le pipeline temporel (`graph_model_temporal.py`, `scripts/run_ablation.py`).
  - Différence : ajout de variables issues de `projects.csv` / `EXPERIMENT_SECTION` au panel temporel.
- **GCN-LSTM**
  - Fichier : `graph_model_temporal.py:GCNLSTM` (`graph_model_temporal.py:1028-1051`).
  - Hyperparamètres : encodeur 2 couches + LSTM `hidden_dim=64`, `dropout=0.2`; Adam `lr=1e-3`, `weight_decay=1e-4`, `max_epochs=160`, `patience=20` (`graph_model_temporal.py:1178-1233`).
  - Statut : implémenté, mais pas de métriques sauvegardées identifiées dans les rapports inspectés.
- **R-GCN temporel**
  - Fichier : `part1_extensions.py:RelationSnapshotGCN`, entraînement `part1_extensions.py:train_temporal_rgcn` (`part1_extensions.py:146-168`, `236-285`).
  - Hyperparamètres : encodeur relationnel 2 couches, `hidden_dim=64`, `dropout=0.2`; Adam `lr=1e-3`, `weight_decay=1e-4`, `max_epochs=180`, `patience=20`.
  - Relations : `spatial`, `same_route`, `same_functional_class` selon graphe choisi (`part1_extensions.py:109-143`).
  - Cibles : HPMS16 principal, puis single-task par distress et multi-task 5 distress (`scripts/run_singletask_per_distress.py`, `scripts/run_multitask.py`).
- **Multi-task R-GCN**
  - Fichier : `part1_extensions.py:MultiTaskRelationGCN`, entraînement `train_multitask_rgcn` (`part1_extensions.py:171-206`, `288-385`).
  - Hyperparamètres : encodeur partagé + têtes par cible, `hidden_dim=64`, `dropout=0.2`, mêmes réglages d’optimisation.
  - Target : 5 distress simultanés (`graph_model_temporal.py:50-56`).
- **Ensembles RF + R-GCN**
  - Fichiers : `ensemble.py`, `scripts/run_ensemble.py`, sorties `graph_data/ensemble_results.json`.
  - Variantes : ratio fixe 70/30, stacking Ridge, stacking MLP.
  - Target : HPMS16 temporal (`graph_data/ensemble_results.json:rf_local` et clés voisines).
- **Modèles statiques de disruption**
  - Fichier : `graph_model.py`.
  - Modèles : `GraphFilterNet` + Ridge (et parfois RF) sur cibles synthétiques `delta_vht_proxy`, `connectivity_loss_pct`, `disconnected_od_pct`, `disruption_score` (`reports/static_target_variable_definitions.csv:2-6`, `reports/graph_variant_model_comparison.csv:1-16`).
  - Split : `train_test_split` 20% test, puis 20% val du train (`graph_model.py:610-611`, voir aussi `reports/part1_ood_static.csv:2-4`).

## 5. Résultats numériques
- Fichiers de résultats structurés trouvés : `reports/treatment_feature_ablation.csv`, `reports/part1_rgcn_temporal.csv`, `reports/distress_model_comparison.csv`, `reports/materials_weight_sweep.json`, `graph_data/ensemble_results.json`, `reports/graph_variant_model_comparison.csv`, `reports/part1_ood_temporal.csv`, `reports/part1_ood_static.csv`.
- Incohérences / coexistence de versions :
  - `reports/distress_model_comparison.csv` garde le meilleur MEPDG single-task à `0.542448` (`reports/distress_model_comparison.csv:3`), alors que le sweep matériaux/poids publie un meilleur `0.558205` (`reports/materials_weight_sweep.md:3`, `reports/materials_weight_sweep.json:1`).
  - `reports/graph_diagnostics.csv:4` donne 17 944 edges pour `full_refined`, tandis que les variantes matériaux du sweep reconstruisent 17 967 à 17 981 edges (`reports/materials_weight_sweep.md:3-8`).
- Tableau consolidé trié par R² test décroissant :

| Modèle | Graphe | Cible / expérience | R² train | R² test | MAE | RMSE | MAPE/SMAPE | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| static Ridge | full_refined | connectivity_loss_pct | non trouvé | 0.601081 | non trouvé | 0.0030818531129526636 | non trouvé | `reports/graph_variant_model_comparison.csv:11` |
| static Ridge | spatial_route | connectivity_loss_pct | non trouvé | 0.600977 | non trouvé | 0.0030822569485386476 | non trouvé | `reports/graph_variant_model_comparison.csv:7` |
| static GraphFilterNet | full_refined | connectivity_loss_pct | non trouvé | 0.551947 | non trouvé | 0.0032661387809513488 | non trouvé | `reports/graph_variant_model_comparison.csv:11` |
| static GraphFilterNet | spatial_route | connectivity_loss_pct | non trouvé | 0.549970 | non trouvé | 0.003273334659380873 | non trouvé | `reports/graph_variant_model_comparison.csv:7` |
| R-GCN single-task | temporal/full_refined | MEPDG_CRACKING_PERCENT_AC | 0.4412527110543588 | 0.542448 | 5.690203424061046 | 9.403491667527145 | 161.26359812377493 | `reports/distress_model_comparison.csv:3` |
| R-GCN single-task | temporal/full_refined | HPMS16_CRACKING_PERCENT_AC | 0.5328963599181291 | 0.454388 | 8.59148412031286 | 11.387843029912275 | 141.97747667115448 | `reports/distress_model_comparison.csv:2` |
| Random Forest | no_graph | EXPERIMENT_SECTION treatment features | non trouvé | 0.448890 | 8.861448539787098 | 11.445081161083927 | non trouvé | `reports/treatment_feature_ablation.csv:3` |
| temporal Random Forest | spatial | HPMS16_CRACKING_PERCENT_AC_t1 | non trouvé | 0.420178 | non trouvé | 11.739423358361437 | non trouvé | `reports/graph_variant_model_comparison.csv:14` |
| temporal Random Forest | spatial_route | HPMS16_CRACKING_PERCENT_AC_t1 | non trouvé | 0.409446 | non trouvé | 11.847573236246129 | non trouvé | `reports/graph_variant_model_comparison.csv:15` |
| temporal Random Forest | full_refined | HPMS16_CRACKING_PERCENT_AC_t1 | non trouvé | 0.409299 | non trouvé | 11.849049857104644 | non trouvé | `reports/graph_variant_model_comparison.csv:16` |
| R-GCN | temporal/full_refined | HPMS16_CRACKING_PERCENT_AC | 0.3994657784402946 | 0.370796 | 9.210386670205523 | 12.229128344105881 | non trouvé | `reports/part1_rgcn_temporal.csv:4` |
| R-GCN | temporal/spatial_route | HPMS16_CRACKING_PERCENT_AC | 0.37811427455088 | 0.360650 | 9.264564347793074 | 12.3273314722414 | non trouvé | `reports/part1_rgcn_temporal.csv:3` |
| Random Forest | no_graph | No project/treatment features | non trouvé | 0.349657 | 8.88679173536631 | 12.432853371018036 | non trouvé | `reports/treatment_feature_ablation.csv:2` |
| R-GCN | temporal/spatial | HPMS16_CRACKING_PERCENT_AC | 0.30188045721608137 | 0.337484 | 9.644585717425628 | 12.54867344967584 | non trouvé | `reports/part1_rgcn_temporal.csv:2` |
| R-GCN multi-task | temporal/full_refined | HPMS16_CRACKING_PERCENT_AC | non trouvé | 0.274713 | 10.284557589362649 | 13.129691659049058 | non trouvé | `reports/distress_model_comparison.csv:2` |
| static Ridge | full_refined | delta_vht_proxy | non trouvé | 0.273844 | non trouvé | 2235496773.592371 | non trouvé | `reports/graph_variant_model_comparison.csv:10` |
| temporal Ridge | spatial | HPMS16_CRACKING_PERCENT_AC_t1 | non trouvé | 0.259403 | non trouvé | 13.267541687440847 | non trouvé | `reports/graph_variant_model_comparison.csv:14` |
| temporal Ridge | full_refined | HPMS16_CRACKING_PERCENT_AC_t1 | non trouvé | 0.259112 | non trouvé | 13.270152752009169 | non trouvé | `reports/graph_variant_model_comparison.csv:16` |
| temporal Ridge | spatial_route | HPMS16_CRACKING_PERCENT_AC_t1 | non trouvé | 0.259067 | non trouvé | 13.270555615487552 | non trouvé | `reports/graph_variant_model_comparison.csv:15` |
| static Ridge | full_refined | disruption_score | non trouvé | 0.257539 | non trouvé | 2695617459.803881 | non trouvé | `reports/graph_variant_model_comparison.csv:13` |
| static Ridge | spatial_route | delta_vht_proxy | non trouvé | 0.256132 | non trouvé | 2260859829.97063 | non trouvé | `reports/graph_variant_model_comparison.csv:6` |
| static GraphFilterNet | full_refined | delta_vht_proxy | non trouvé | 0.254042 | non trouvé | 2265772692.252739 | non trouvé | `reports/graph_variant_model_comparison.csv:10` |
| static GraphFilterNet | spatial | disconnected_od_pct | non trouvé | 0.250141 | non trouvé | 0.01785237156972153 | non trouvé | `reports/graph_variant_model_comparison.csv:4` |
| static Ridge | spatial_route | disruption_score | non trouvé | 0.239200 | non trouvé | 2726532410.4173613 | non trouvé | `reports/graph_variant_model_comparison.csv:9` |
| Ridge | no_graph | EXPERIMENT_SECTION treatment features | non trouvé | 0.237904 | non trouvé | non trouvé | non trouvé | `reports/treatment_feature_ablation.csv:3` |
| static Ridge | spatial | disconnected_od_pct | non trouvé | 0.237775 | non trouvé | 0.017998969831361494 | non trouvé | `reports/graph_variant_model_comparison.csv:4` |
| static GraphFilterNet | full_refined | disruption_score | non trouvé | 0.235045 | non trouvé | 2736147009.298278 | non trouvé | `reports/graph_variant_model_comparison.csv:13` |
| R-GCN multi-task | temporal/full_refined | MEPDG_CRACKING_PERCENT_AC | non trouvé | 0.234859 | 7.1347521206911875 | 12.160168343113257 | non trouvé | `reports/distress_model_comparison.csv:3` |
| GCN+project_history | spatial_snapshot | EXPERIMENT_SECTION treatment features | non trouvé | 0.231638 | 10.601429646155413 | 13.513956985156328 | non trouvé | `reports/treatment_feature_ablation.csv:3` |
| static GraphFilterNet | spatial_route | delta_vht_proxy | non trouvé | 0.229267 | non trouvé | 2301322983.9587975 | non trouvé | `reports/graph_variant_model_comparison.csv:6` |
| static Ridge | full_refined | disconnected_od_pct | non trouvé | 0.210483 | non trouvé | 0.026354652091768252 | non trouvé | `reports/graph_variant_model_comparison.csv:12` |
| static GraphFilterNet | spatial_route | disruption_score | non trouvé | 0.205767 | non trouvé | 2785796085.966659 | non trouvé | `reports/graph_variant_model_comparison.csv:9` |
| static GraphFilterNet | spatial | delta_vht_proxy | non trouvé | 0.202521 | non trouvé | 60932029.91768398 | non trouvé | `reports/graph_variant_model_comparison.csv:2` |
| static GraphFilterNet | spatial | disruption_score | non trouvé | 0.201278 | non trouvé | 68930901.41108349 | non trouvé | `reports/graph_variant_model_comparison.csv:5` |
| static Ridge | spatial_route | disconnected_od_pct | non trouvé | 0.200956 | non trouvé | 0.02651318115848099 | non trouvé | `reports/graph_variant_model_comparison.csv:8` |
| static GraphFilterNet | full_refined | disconnected_od_pct | non trouvé | 0.186478 | non trouvé | 0.02675229650048552 | non trouvé | `reports/graph_variant_model_comparison.csv:12` |
| static Ridge | spatial | disruption_score | non trouvé | 0.183172 | non trouvé | 69707822.42099696 | non trouvé | `reports/graph_variant_model_comparison.csv:5` |
| static Ridge | spatial | delta_vht_proxy | non trouvé | 0.179296 | non trouvé | 61812949.74292115 | non trouvé | `reports/graph_variant_model_comparison.csv:2` |
| static GraphFilterNet | spatial_route | disconnected_od_pct | non trouvé | 0.169144 | non trouvé | 0.02703581531132233 | non trouvé | `reports/graph_variant_model_comparison.csv:8` |
| static GraphFilterNet | spatial | connectivity_loss_pct | non trouvé | 0.132463 | non trouvé | 0.008588687329363624 | non trouvé | `reports/graph_variant_model_comparison.csv:3` |
| Ridge | no_graph | No project/treatment features | non trouvé | 0.046596 | non trouvé | non trouvé | non trouvé | `reports/treatment_feature_ablation.csv:2` |
| GCN | spatial_snapshot | EXPERIMENT_SECTION treatment features | non trouvé | 0.046279 | 10.991507249719957 | 15.056026009886551 | non trouvé | `reports/treatment_feature_ablation.csv:3` |
| GCN+project_history | spatial_snapshot | No project/treatment features | non trouvé | 0.042241 | 11.180516352373012 | 15.087860254963582 | non trouvé | `reports/treatment_feature_ablation.csv:2` |
| GCN | spatial_snapshot | No project/treatment features | non trouvé | 0.028294 | 11.24951607760261 | 15.197320058415086 | non trouvé | `reports/treatment_feature_ablation.csv:2` |
| R-GCN single-task | temporal/full_refined | MEPDG_TRANS_CRACK_LENGTH_AC | -0.05112825327604842 | 0.007658 | 1001.9263607499354 | 1971.7383666548367 | 145.0852592702678 | `reports/distress_model_comparison.csv:4` |
| R-GCN single-task | temporal/full_refined | POTHOLES_A | -0.004280764495331102 | -0.006125 | 0.007426764201890739 | 0.06933764647481418 | 122.34987687618808 | `reports/distress_model_comparison.csv:6` |
| static Ridge | spatial | connectivity_loss_pct | non trouvé | -0.019428 | non trouvé | 0.009310245208642542 | non trouvé | `reports/graph_variant_model_comparison.csv:3` |
| R-GCN single-task | temporal/full_refined | PATCH_A | -0.0045006960148807895 | -0.028665 | 0.46710218141841536 | 1.6464599341730262 | 180.62068380243642 | `reports/distress_model_comparison.csv:5` |
| R-GCN multi-task | temporal/full_refined | POTHOLES_A | non trouvé | -0.054741 | 0.01645144324968843 | 0.07099305190247127 | non trouvé | `reports/distress_model_comparison.csv:6` |
| R-GCN multi-task | temporal/full_refined | MEPDG_TRANS_CRACK_LENGTH_AC | non trouvé | -0.387926 | 1249.2687589392942 | 2331.8561014591405 | non trouvé | `reports/distress_model_comparison.csv:4` |
| R-GCN multi-task | temporal/full_refined | PATCH_A | non trouvé | -18.342242 | 5.511772361090955 | 7.139494995756213 | non trouvé | `reports/distress_model_comparison.csv:5` |

## 6. Tests de robustesse et validation
- **Leave-one-out / held-out geography (temporel)**
  - Fichier : `reports/part1_ood_temporal.csv`.
  - Résultats : tous les R² test sont négatifs sur état/région non vus. Ex. `full_refined`: RF `-0.2411`, Ridge `-0.1324`, GCN `-0.3938`, R-GCN `-0.3230` (`reports/part1_ood_temporal.csv:4`).
- **Leave-one-out / held-out geography (statique disruption)**
  - Fichier : `reports/part1_ood_static.csv`.
  - Résultats : sur `connectivity_loss_pct`, Ridge atteint `0.7398` pour `spatial_route` et `0.7531` pour `full_refined`; `delta_vht_proxy` reste faible (`reports/part1_ood_static.csv:3-4`).
- **Validation OpenStreetMap**
  - Scripts : `osm_validation_full.py`, `scripts/run_osm_validation_reporting.py`.
  - Résultat agrégé : 16 024 edges évalués, 6 835 validés (`42.65%` strict), 4 753 `failed_no_path`, 4 366 `failed_distance`, 70 erreurs (`graph_data/osm_validation_findings.json:8-21`).
  - Sous-ensemble spatial+route : 9 086 edges, validation `47.4%` à seuil `<=2x`, `62.3%` à `<=5x` (`graph_data/osm_validation_findings.json:39-58`).
  - Interprétation sauvegardée : conserver catégories `A/B/C/D/F`, supprimer seulement `E_no_path_long`; `58.6%` d’arêtes gardées, `1.4%` supprimées (`graph_data/osm_validation_findings.json:122-135`).
- **Analyses de corrélation par distance**
  - Fichier : `reports/cracking_correlation_spatial_bins.csv`.
  - Bins : `0-5km`, `5-20km`, `20-50km`, `50km+`.
  - Résultats : médiane de corrélation des changements = `0.509`, `0.041`, `0.249`, `0.097` respectivement (`reports/cracking_correlation_spatial_bins.csv:2-5`).
- **Corrélation par type d’arête**
  - Fichier : `reports/cracking_correlation_by_edge_type.csv`.
  - Résultats : médiane `same_functional_class=0.522`, `same_route=0.519`, `spatial=0.486` (`reports/cracking_correlation_by_edge_type.csv:2-4`).
- **Sensibilité maintenance ↔ trafic annuel**
  - Script : `scripts/run_traffic_sensitivity.py`.
  - Résultat : 4 627 événements avec pré/post valides, 55 436 contrôles, `Welch p=0.0006887`; baisse moyenne pré→année d’événement `-39888.3`, delta pré→post `+10608.9` vs `+16624.9` contrôle (`reports/traffic_sensitivity.json:2-14`).
- **Suffisance CRCP / JPCC**
  - Fichier : `reports/distress_crcp_jpcc_sufficiency.md`.
  - Verdict : CRCP marginal avec pooling AC; JPCC suffisant pour modèle séparé (`reports/distress_crcp_jpcc_sufficiency.md:3-17`).

## 7. Interface de visualisation
- Framework utilisé : Streamlit + Plotly (`requirements.txt:1-5`, `streamlit_app.py:20`).
- Fichier principal : `streamlit_app.py`.
- Configuration UI : `.streamlit/config.toml` (`.streamlit/config.toml:1-8`).
- Fonctionnalités implémentées identifiables dans le code :
  - 3 onglets majeurs `Road Section Inspector`, `From Asset to Network`, `Models and Findings` (`streamlit_app.py:1097`).
  - inspection par nœud avec normalisation `Z-score / Min-Max / Robust` (`streamlit_app.py:826-860`, `835`).
  - sélection multi-métriques temporelles (`streamlit_app.py:882-900`).
  - section `Traffic Sensitivity to Maintenance Events` (`streamlit_app.py:2175`).
  - section `Graph Topology Validation (OSM)` (`streamlit_app.py:2233`).
  - chargement direct des rapports CSV/JSON via helpers `load_report_csv/json` (`streamlit_app.py:301-317`, `991-1018`).
- Commande exacte pour lancer l’interface : non documentée explicitement dans le repo. Point d’entrée détecté : `streamlit_app.py`; commande Streamlit conventionnelle probable `./.venv/bin/streamlit run streamlit_app.py` (inférence à partir de `requirements.txt` + structure, non trouvée textuellement).

## 8. Travaux en cours / incomplets
- TODO/FIXME trouvés dans le code : **non trouvé** (`rg` sans résultat sur `TODO|FIXME|XXX|WIP`).
- Fonctions manifestement non terminées : **non trouvé** au sens de placeholder explicite.
- Incomplétudes observables par absence de brique finale :
  - aucun module d’optimisation explicite (MILP/GA/RL) malgré le cadrage dissertation; seulement des briques prédictives et des proxies de disruption (`graph_model.py`, `reports/static_target_variable_definitions.csv:2-6`).
  - `GCN-LSTM` est implémenté (`graph_model_temporal.py:1028-1051`) mais aucun fichier de résultats persistant n’a été identifié parmi les rapports inspectés.
  - les analyses OSM sont riches, mais la validation “physique” ne couvre explicitement que le sous-ensemble spatial+route dans la synthèse finale (`graph_data/osm_validation_findings.json:39-58`).
- Analyses commencées mais non finalisées dans les sorties persistées : pas de fichier trouvé pour une optimisation de scheduling budget/traffic; pas de pipeline de décision aval sauvegardé.

## 9. Ce qui manque pour la prochaine étape
- **Module d’optimisation** : non trouvé. Aucun solveur MILP/GA/RL / heuristique de scheduling dans `requirements.txt` ni dans les fichiers Python principaux.
- **Fonction objectif** : non trouvée pour la phase maintenance. Les seules fonctions cibles explicites sont des proxies de disruption synthétiques (`delta_vht_proxy`, `connectivity_loss_pct`, `disconnected_od_pct`, `disruption_score`) définies dans `reports/static_target_variable_definitions.csv:2-6`.
- **Contraintes** : non trouvées pour budget, fenêtres temporelles, non-chevauchement de travaux, ressources, ou impact trafic maximal.
- **Pipeline d’évaluation comparative optimisation** : non trouvé. Il manque un protocole du type baseline heuristique vs MILP/GA/RL avec KPI coût / trafic / état futur / résilience.
- **Couplage prédiction → décision** encore absent : pas de script qui transforme les sorties de `graph_model_temporal.py` / `part1_extensions.py` en scores d’intervention ou priorités de portefeuille.
- **Données coût / budget** : non trouvées dans les datasets chargés actuellement.

## 10. Suggestions de visuels pour la présentation
- **Comparaison des graphes (`spatial`, `spatial_route`, `full_refined`)**
  - Déjà disponible : `reports/graph_diagnostics.csv`, visualisable aussi dans `streamlit_app.py`.
  - Génération rapide : bar chart `nodes / edges / components` à partir de `reports/graph_diagnostics.csv`.
  - Export recommandé : SVG pour slides / PDF, sinon PNG 300 dpi.
- **Bar chart des performances temporelles HPMS16**
  - Source tabulaire : `reports/treatment_feature_ablation.csv`, `reports/part1_rgcn_temporal.csv`, `graph_data/ensemble_results.json`.
  - Déjà partiellement rendu dans l’app `Models and Findings` (`streamlit_app.py`).
  - Recommandation : un seul graphique comparant RF, R-GCN, ensemble 70/30, stacked ridge, stacked MLP.
- **Distress extension / single-task vs multi-task**
  - Source : `reports/distress_model_comparison.csv` + `reports/materials_weight_sweep.json`.
  - Génération rapide : grouped bar chart par cible (HPMS, MEPDG, transverse, patch, potholes).
  - Attention à l’incohérence : mentionner que le meilleur MEPDG final vient du sweep matériaux (`0.5582`) et non du tableau single-task initial (`0.5424`).
- **OSM validation**
  - Déjà prêt dans l’app (`streamlit_app.py:2233+`) et sources `graph_data/osm_validation_findings.json`, `reports/osm_bad_edge_examples.csv`, `reports/osm_good_edge_examples.csv`.
  - Visuels recommandés :
    - bar chart taux de validation selon seuil `2x/3x/4x/5x`.
    - 2 exemples concrets de “bad” et “good” edges.
- **Sensibilité maintenance ↔ trafic**
  - Déjà prêt dans l’app (`streamlit_app.py:2175+`) et `reports/traffic_sensitivity.json`.
  - Visuel recommandé : 2 barres `events vs controls` sur le delta annuel, avec p-value en annotation.
- **Robustesse géographique**
  - Source : `reports/part1_ood_temporal.csv` et `reports/part1_ood_static.csv`.
  - Génération rapide : heatmap des R² par famille de modèle et variante de graphe.
- **Candidats pour la phase 2 optimisation**
  - Pas de graphique existant final. Génération rapide possible à partir de `reports/static_target_variable_definitions.csv` : schéma pipeline “condition prediction -> graph disruption proxy -> optimization”.
  - Format recommandé : SVG vectoriel pour diagrammes de pipeline.
