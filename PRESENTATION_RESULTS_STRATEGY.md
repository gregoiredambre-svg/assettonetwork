# PRESENTATION_RESULTS_STRATEGY.md

## 1. Inventaire complet des résultats disponibles

| Fichier(s) | Nature des résultats | Date / fraîcheur | 2-3 résultats clés | Statut |
|---|---|---|---|---|
| `reports/materials_weight_sweep.json`, `reports/materials_weight_sweep.md` | Sweep des poids du graphe `full_refined` avec matériaux, cible `MEPDG_CRACKING_PERCENT_AC` | 2026-05-22, le plus récent des résultats prédictifs ciblés | meilleur `R² test = 0.5582`, baseline sans matériaux `0.5424`, gain `+0.0158` (`reports/materials_weight_sweep.json:3-6`, `41-54`) | **définitif** pour le meilleur résultat MEPDG |
| `graph_data/ensemble_results.json` | Combinaison RF + R-GCN sur HPMS16 (ratios fixes + stacking) | 2026-05-21, cohérent avec les sorties temporelles HPMS16 | RF local `R² test = 0.4489` (`13-20`), R-GCN `0.3708` (`31-38`), meilleur ensemble `Stacked MLP = 0.5356` avec gain bootstrap `+0.0867` vs RF (`139-145`, `177-188`) | **définitif** |
| `reports/distress_model_comparison.csv` | Comparaison single-task vs multi-task par cible distress | 2026-05-21, encore utile mais partiellement supersédé pour MEPDG | HPMS single-task `0.4544` vs multi-task `0.2747` (`:2`), MEPDG single-task `0.5424` vs multi-task `0.2349` (`:3`), patch multi-task `-18.3422` (`:5`) | **définitif** pour la conclusion single-task > multi-task; **obsolète** pour le meilleur MEPDG |
| `graph_data/multitask_results.json`, `reports/multitask_results.md` | Détail brut du modèle multi-task 5 distress | 2026-05-21, cohérent avec `distress_model_comparison.csv` | train HPMS `0.2578`, val MEPDG `0.2428`, macro mean train `0.0438` (`graph_data/multitask_results.json:4-18`, `44-50`, `54-68`) | **exploratoire** / fichier de support |
| `reports/part1_rgcn_temporal.csv`, `reports/part1_rgcn_temporal.json` | Comparaison R-GCN temporel par variante de graphe sur HPMS16 | 2026-05-21, cohérent avec les autres résultats HPMS16 | `spatial = 0.3375`, `spatial_route = 0.3606`, `full_refined = 0.3708` (`reports/part1_rgcn_temporal.csv:2-4`) | **définitif** |
| `reports/treatment_feature_ablation.csv`, `reports/treatment_feature_ablation.json` | Ablation avec / sans features projets / traitements, comparaison RF/Ridge/GCN | 2026-05-21, cohérent avec l’ensemble HPMS16 | RF sans traitement `0.3497` (`:2`), RF avec traitement `0.4489` (`:3`), GCN avec treatment `0.2316` vs sans `0.0463` (`:3`) | **définitif** |
| `reports/part1_ood_temporal.csv`, `reports/part1_ood_temporal.json` | Robustesse géographique temporelle (held-out states) | 2026-05-21, pas supersédé | tous les `R²` négatifs; ex. `full_refined`: RF `-0.2411`, Ridge `-0.1270`, GCN `-0.3648`, R-GCN `-0.3230` (`reports/part1_ood_temporal.csv:2-4`) | **définitif** |
| `reports/part1_ood_static.csv`, `reports/part1_ood_static.json` | Robustesse géographique des modèles statiques de disruption | 2026-05-20, secondaire mais propre | `connectivity_loss_pct` Ridge `0.7398` (`spatial_route`) et `0.7531` (`full_refined`) (`reports/part1_ood_static.csv:3-4`) | **définitif**, mais **secondaire** |
| `reports/ablation_similarity_table.json`, `graph_data/ablation_similarity.json`, `reports/ablation_similarity.md` | Ablation des facteurs de similarité du graphe | 2026-05-20 / non rafraîchi après sweep matériaux | `spatial_only = 0.3771`, `spatial+traffic = 0.3796`, `spatial+climate+pavement = 0.3849`, `all factors = 0.3845` (`reports/ablation_similarity_table.json:3-17`, `19-35`, `111-149`) | **définitif** pour la hiérarchie des facteurs, **obsolète** pour la performance absolue |
| `reports/ablation_cluster_size_table.json`, `graph_data/ablation_cluster_size.json`, `reports/ablation_cluster_size.md` | Sensibilité au filtrage par taille de cluster | 2026-05-20, cohérent | `min_size=1` meilleur `R² test = 0.3845`; `min_size=5` tombe à `0.1619`; `min_size=50` à `0.0227` (`reports/ablation_cluster_size_table.json:3-19`, `22-40`, `85-103`) | **définitif** |
| `graph_data/osm_validation_findings.json` | Synthèse finale OSM + test empirique sur utilité des edges OSM-failed | 2026-05-22, plus récent que les anciens `osm_*` | extension à `16,024` arêtes, facteur `160.2x` (`:2-7`), spatial+route validés `47.4%` à `<=2x` et `62.3%` à `<=5x` (`:39-58`), seulement `1.4%` d’arêtes recommandées à supprimer (`:122-135`) | **définitif** |
| `reports/osm_validation_by_graph_variant.csv` | Validation OSM résumée par variante de graphe | 2026-05-22, complément du JSON principal | part validée: `spatial 43.0%`, `spatial_route 41.9%`, `full_refined 39.7%` (`reports/osm_validation_by_graph_variant.csv:2-4`) | **définitif**, mais **support** |
| `reports/osm_validation_by_edge_type.csv` | Validation OSM par type d’arête | 2026-05-22, complément du JSON principal | validated share: `spatial 43.0%`, `same_route 35.7%`, `same_functional_class 35.8%` (`reports/osm_validation_by_edge_type.csv:2-4`) | **définitif**, mais **support** |
| `reports/osm_bad_edge_examples.csv`, `reports/osm_good_edge_examples.csv`, `reports/osm_validation_interpretation.md/json` | Exemples interprétables d’arêtes bonnes / mauvaises | 2026-05-22, directement utiles pour slides | fournit des cas concrets d’`opposite_carriageway`, `co-located no path`, et bons appariements OSM; utile qualitativement plus que quantitativement | **définitif**, support visuel |
| `reports/cracking_correlation_spatial_bins.csv`, `reports/cracking_correlation_by_edge_type.csv`, `reports/cracking_correlation_edge_pairs.csv` | Corrélation spatiale des changements de cracking | 2026-05-21, cohérent avec l’argument “local signal” | médianes par distance `0-5km=0.509`, `5-20km=0.041`, `20-50km=0.249` (`reports/cracking_correlation_spatial_bins.csv:2-5`); médiane `same_functional_class=0.522`, `same_route=0.519`, `spatial=0.486` (`reports/cracking_correlation_by_edge_type.csv:2-4`) | **définitif** |
| `reports/traffic_sensitivity.json`, `reports/traffic_sensitivity_per_group.csv` | Sensibilité maintenance ↔ trafic annuel (proxy `ANNUAL_GESAL_TREND`) | 2026-05-21, résultat clair mais avec caveat proxy | `n_events=4627`, `n_controls=55436`, `p=0.0006887`, chute moyenne pré→année d’événement `-39888.3` (`reports/traffic_sensitivity.json:2-13`) | **définitif** avec caveat |
| `reports/traffic_sensitivity_events.csv`, `reports/traffic_sensitivity_controls.csv` | Tables brutes par événement / contrôle pour l’analyse trafic | 2026-05-21, fichiers support | utiles pour recalculer CI et figures; pas de message autonome | **exploratoire** / support |
| `reports/graph_diagnostics.csv`, `reports/graph_diagnostics.json` | Taille/structure des 3 graphes | 2026-05-20, stable | `spatial=10,186` edges, `spatial_route=11,965`, `full_refined=17,944`; composants multiples (`reports/graph_diagnostics.csv:2-4`) | **définitif** |
| `reports/distress_crcp_jpcc_sufficiency.md`, `reports/distress_target_profile.csv` | Faisabilité des familles CRCP/JPCC et profil des targets | 2026-05-21, utile pour cadrage dataset | CRCP `108` sections = marginal; JPCC `794` sections = suffisant (`reports/distress_crcp_jpcc_sufficiency.md:3-17`) | **définitif**, mais **contexte** |
| `reports/graph_variant_model_comparison.csv/json` | Ancienne synthèse mixte static + temporal par variante | 2026-05-18, plus ancien que les fichiers ciblés | meilleurs scores sur cibles statiques proxy (`0.6011` sur `connectivity_loss_pct`), HPMS temporal RF `0.4202` (`reports/graph_variant_model_comparison.csv:11`, `14-16`) | **exploratoire** / partiellement obsolète |
| `graph_data/network_model_metrics*.json`, `graph_data/network_scenario_predictions*.csv`, `graph_data/scenario_network_impacts*.csv` | Résultats détaillés de la branche disruption réseau synthétique | 2026-05-18 à 2026-05-20, support de `graph_variant_model_comparison.csv` | utiles pour phase 2; pas nécessaires en slide tant que les proxies ne sont pas intégrés à l’optimisation | **exploratoire** |
| `reports/archive/materials/materials_impact_metrics.json` | Ancien test matériaux avant fusion dans le sweep | 2026-05-21, archivé | gain minuscule `+0.0006` sur MEPDG dans une version intermédiaire | **obsolète, version plus récente ailleurs** |
| `reports/gcn_temporal_metrics*.json`, `reports/monthly*_*.json/csv`, `reports/climate_*`, `reports/experiment_*`, `reports/project_*`, `reports/same_route_*` | Familles exploratoires supplémentaires (granularité mensuelle, climat, event-study, sémantique projets, audits same_route) | 2026-05-18 à 2026-05-22, hétérogène | beaucoup de matière, mais aucun message de slide consolidé aussi fort que les fichiers ci-dessus | **exploratoire** |

## 2. Cartographie des analyses : ce qui marche vs ce qui ne marche pas

### Catégorie A — Résultats forts à mettre en avant

- **MEPDG cracking, single-task R-GCN avec matériaux + bons poids : `R² test = 0.5582`**. Source : `reports/materials_weight_sweep.json:3-6`, `41-54`.
  - Défendable car : même pipeline temporel que le reste, comparaison explicite à une baseline sans matériaux (`0.5424`), et sweep de poids documenté.
  - Message : *refining pavement similarity with real LTPP materials and reweighting the graph produces the best cracking model in the repo*.

- **Ensemble RF + R-GCN sur HPMS16 : `R² test = 0.5356`, gain bootstrap `+0.0867` vs RF**. Source : `graph_data/ensemble_results.json:139-145`, `177-188`.
  - Défendable car : comparaison directe contre RF local (`0.4489`) et R-GCN pur (`0.3708`) sur la même cible et le même split.
  - Message : *the graph is most useful when combined with strong local tabular history, not when used alone*.

- **Single-task > multi-task sur les distress cibles**. Sources : `reports/distress_model_comparison.csv:2-6`.
  - Chiffres clés : HPMS `0.4544` vs `0.2747`; MEPDG `0.5424` vs `0.2349`; patch multi-task `-18.3422`.
  - Défendable car : comparaison cible par cible, mêmes données, mêmes horizons, même split.
  - Message : *distress processes should not be forced into one shared target; single-task modelling is materially more stable*.

- **Validation OSM à grande échelle : `16,024` edges testés, `160.2x` l’échantillon initial, seulement `1.4%` à retirer**. Source : `graph_data/osm_validation_findings.json:2-7`, `39-58`, `122-135`.
  - Défendable car : validation externe, volumineuse, et complétée par un test empirique sur l’utilité prédictive des edges OSM-failed.
  - Message : *the graph is not a routable road network, but it is a defensible section-level interdependency graph*.

- **Sensibilité trafic-maintenance significative : `p = 0.0006887`**. Source : `reports/traffic_sensitivity.json:2-13`.
  - Chiffres clés : `4627` événements, `55436` contrôles, chute moyenne pré→année d’événement `-39888.3`.
  - Défendable car : grande taille d’échantillon et test de Welch explicite.
  - Message : *maintenance years leave a measurable annual traffic-loading signature, even at coarse annual granularity*.

### Catégorie B — Résultats intermédiaires utiles contextuellement

- **RF local HPMS16 : `R² test = 0.4489`**. Source : `graph_data/ensemble_results.json:13-20`.
  - Rôle narratif : baseline solide montrant que l’historique local explique déjà beaucoup du signal.

- **R-GCN HPMS16 par variante de graphe : `spatial=0.3375`, `spatial_route=0.3606`, `full_refined=0.3708`**. Source : `reports/part1_rgcn_temporal.csv:2-4`.
  - Rôle narratif : montre qu’ajouter de la structure relationnelle aide un peu, mais pas suffisamment pour battre le RF seul.

- **Ablation des similarités : meilleur combo ancien `spatial+climate+pavement = 0.3849`; trafic marginal**. Source : `reports/ablation_similarity_table.json:111-149`, `3-35`.
  - Rôle narratif : explique pourquoi le sweep matériaux récent privilégie climat + pavement plutôt que trafic.

- **Corrélation spatiale par distance : `0-5km = 0.509` ; `5-20km = 0.041` ; `20-50km = 0.249`**. Source : `reports/cracking_correlation_spatial_bins.csv:2-5`.
  - Rôle narratif : bon argument d’ouverture pour justifier un graphe local, sans sur-vendre la propagation physique.

- **Filtrage des clusters : garder tout marche mieux (`min_size=1 -> 0.3845`)**. Source : `reports/ablation_cluster_size_table.json:3-19`, `22-40`.
  - Rôle narratif : permet de répondre à la tutrice sur le bruit des petits clusters; résultat utile mais pas “headline”.

- **Modèles statiques de disruption sur proxies : jusqu’à `R² = 0.6011` sur `connectivity_loss_pct`**. Source : `reports/graph_variant_model_comparison.csv:7`, `11`; `reports/static_target_variable_definitions.csv:2-6`.
  - Rôle narratif : bon teaser pour la phase 2 optimisation, mais il faut rappeler que la cible est synthétique, pas observée.

### Catégorie C — Résultats négatifs ou décevants

- **Leave-one-state-out : tous les modèles sont négatifs**. Source : `reports/part1_ood_temporal.csv:2-4`.
  - Chiffres : RF `-0.234` à `-0.243`, Ridge `-0.129` à `-0.127`, GCN `-0.362` à `-0.383`, R-GCN `-0.218` à `-0.323`.
  - Présenter ? **Oui, brièvement**. Cadrage : ce n’est pas “le modèle est mauvais”, c’est “le signal est régionalisé et nécessite recalibrage local”.

- **GCN purs très faibles en HPMS16**. Source : `reports/treatment_feature_ablation.csv:2-3`.
  - Chiffres : GCN sans treatment `0.0283` ou `0.0463`, avec treatment `0.0422` ou `0.2316`.
  - Présenter ? **Oui, mais seulement comme baseline faible** pour justifier pourquoi l’ensemble RF+R-GCN est plus crédible qu’un message “graph-only”.

- **Distress rares / zero-inflated** : trans crack quasi nul, patch et potholes négatifs. Source : `reports/distress_model_comparison.csv:4-6`.
  - Chiffres : `MEPDG_TRANS_CRACK_LENGTH_AC = 0.0077`; `PATCH_A = -0.0287`; `POTHOLES_A = -0.0061`; multi-task encore pire.
  - Présenter ? **Oui, en annexe ou en une phrase**, pour justifier le recentrage sur cracking comme cible principale.

- **`same_route` n’est pas plus “physique” que `spatial` dans l’OSM**. Source : `reports/osm_validation_by_edge_type.csv:2-4`.
  - Chiffres : validated share `spatial 43.0%`, `same_route 35.7%`.
  - Présenter ? **Éventuellement si on t’interroge**, mais pas en headline; sinon ça brouille le récit.

- **Cluster pruning agressif dégrade fortement la perf**. Source : `reports/ablation_cluster_size_table.json:22-40`, `85-103`.
  - Chiffres : `min_size=5 -> 0.1619`; `min_size=50 -> 0.0227`.
  - Présenter ? **Pas en slide principale**; garder pour discussion méthodologique si la question vient.

### Catégorie D — Analyses incomplètes ou non concluantes

- **GCN-LSTM** (`graph_model_temporal.py:1028-1051`) : implémenté, mais aucun résultat persistant clairement exploitable n’a été trouvé. Recommandation : **écarter** de la présentation.

- **Famille `reports/experiment_*`, `reports/project_*`, `reports/monthly*`, `reports/climate_*`** : beaucoup d’outputs, mais pas de message consolidé aussi solide que les blocs A/B. Recommandation : **écarter** pour l’interim, sauf si une question spécifique arrive.

- **Branche disruption réseau statique (`graph_data/network_*`, `scenario_*`)** : numériquement intéressante, mais encore proxy-based et non couplée à un optimiseur. Recommandation : **mentionner comme préparation de la phase 2**, pas comme résultat principal.

- **Anciennes métriques archivées (`reports/archive/...`)** : supersédées. Recommandation : **ne pas utiliser**.

## 3. Analyses MANQUANTES qui renforceraient la présentation

- **Apples-to-apples MEPDG benchmark (RF vs R-GCN vs ensemble) sur la même cible**
  - Pourquoi : aujourd’hui, le meilleur chiffre (`0.5582`) est sur MEPDG, mais la comparaison la plus propre entre familles de modèles est sur HPMS16. Il manque une comparaison équitable sur la cible la plus performante.
  - Faisabilité : élevée; le code d’ensemble existe déjà (`scripts/run_ensemble.py`) et le pipeline single-task MEPDG aussi.
  - Effort estimé : `1-2h` dev + calcul.
  - Output attendu : une table/figure “same target, same split” qui permettrait d’assumer MEPDG comme cible headline sans réserve.

- **Rerun HPMS16 avec le sweep matériaux / poids du graphe**
  - Pourquoi : on sait que les matériaux aident MEPDG; on ne sait pas si ce gain généralise à la cible comparative principale HPMS16.
  - Faisabilité : élevée; le script matériaux existe déjà (`scripts/run_materials_experiments.py`).
  - Effort estimé : `1-2h`.
  - Output attendu : un `ΔR²` clair sur HPMS16, qui dira si l’amélioration est générale ou spécifique à MEPDG.

- **Exact OOD result for the ensemble**
  - Pourquoi : le meilleur modèle global actuel est l’ensemble RF+R-GCN, mais la robustesse géographique n’est documentée que pour RF/Ridge/GCN/R-GCN de base.
  - Faisabilité : moyenne à bonne, code existant en grande partie; il faut rejouer le split leave-one-state-out pour l’ensemble.
  - Effort estimé : `1-2h`.
  - Output attendu : un seul chiffre clé disant si l’ensemble corrige — ou non — la non-transférabilité inter-états.

- **Intervalle de confiance réel pour la sensibilité trafic**
  - Pourquoi : le finding est fort, mais la présentation gagnerait en crédibilité avec un CI bootstrap sur `event_mean_delta - control_mean_delta`.
  - Faisabilité : très élevée; les tables brutes existent (`reports/traffic_sensitivity_events.csv`, `reports/traffic_sensitivity_controls.csv`).
  - Effort estimé : `<1h`.
  - Output attendu : une barre d’erreur réelle et un visuel propre, sans barres illustratives.

## 4. Recommandation narrative pour la présentation

**Q1 — Quelle est la cible principale à présenter ?**

- **Cible principale recommandée : `HPMS16_CRACKING_PERCENT_AC` pour la comparaison de modèles.**
- Justification factuelle : c’est la seule cible pour laquelle tu as, sur un même protocole, RF, Ridge, GCN, R-GCN, ablation treatment/no-treatment, ensemble RF+R-GCN, et robustness leave-one-state-out (`reports/treatment_feature_ablation.csv:2-3`; `reports/part1_rgcn_temporal.csv:2-4`; `graph_data/ensemble_results.json:13-188`; `reports/part1_ood_temporal.csv:2-4`).
- **Cible secondaire à présenter comme “best specialist result” : `MEPDG_CRACKING_PERCENT_AC`**, car c’est là que tu atteins le meilleur score absolu (`0.5582`) avec le raffinage matériaux/poids (`reports/materials_weight_sweep.json:3-6`, `41-54`).
- Donc : **HPMS16 pour la comparaison centrale, MEPDG pour la conclusion technique la plus forte**.

**Q2 — Quelle est la métrique principale à mettre en avant ?**

- **Métrique principale : `R² test`.**
- Justification : c’est la seule métrique présente de manière homogène dans presque toutes les analyses comparatives; elle permet de comparer les familles de modèles et les variantes de graphes.
- **Métrique secondaire : `MAE test`** pour les 1-2 résultats finaux seulement, afin d’éviter que le message repose sur une seule métrique.
- **Ne pas mettre `MAPE`/`SMAPE` en avant** : elles explosent à cause des valeurs proches de zéro (`graph_data/ensemble_results.json:17-18`, `35-36`; `graph_data/multitask_results.json:8-9`, `16-17`), donc elles parasitent le récit plus qu’elles ne l’éclairent.

**Q3 — Quel est le résultat HEADLINE de la présentation ?**

- **Headline recommandé : l’ensemble RF + R-GCN atteint `R² test = 0.5356` sur HPMS16, contre `0.4489` pour RF seul et `0.3708` pour R-GCN seul.** Source : `graph_data/ensemble_results.json:13-20`, `31-38`, `139-145`, `177-188`.
- Pourquoi ce chiffre-là :
  - il est **comparatif**, donc plus fort qu’un score isolé;
  - il est sur la **cible comparative principale**;
  - il raconte une vraie idée scientifique : **le graphe seul n’est pas suffisant, mais il complète utilement l’historique local**.
- Le `0.5582` MEPDG est excellent, mais il est mieux utilisé comme **résultat de raffinement / extension** que comme headline unique.

**Q4 — Faut-il présenter les R² négatifs du leave-one-state-out ?**

- **Oui, mais brièvement et cadrés comme un finding de portée, pas comme un échec.**
- Formulation recommandée : *“All models fail to transfer across held-out states, which indicates that road deterioration remains strongly region-specific and supports the need for local recalibration.”*
- Ne pas leur donner une slide entière si tu manques de temps; une demi-slide ou un panneau de limitation suffit.

**Q5 — Quel est l'ordre narratif optimal des résultats ?**

1. **Pourquoi un graphe ?**
   - Montrer la corrélation spatiale courte distance et la validation OSM.
   - Transition : *“These checks justify using a graph, but they do not yet tell us whether a graph improves prediction.”*
2. **Benchmark central sur HPMS16**
   - RF vs R-GCN vs ensemble.
   - Transition : *“The graph alone is not enough; the gain appears when relational information is combined with strong local history.”*
3. **Refinement méthodologique**
   - Single-task vs multi-task, puis MEPDG + materials/weight sweep.
   - Transition : *“This suggests the bottleneck is not only model class, but also target definition and graph semantics.”*
4. **Limite majeure : non-transférabilité géographique**
   - Leave-one-state-out négatif.
   - Transition : *“So the framework is useful, but it is not globally transferable without local adaptation.”*
5. **Pont vers la phase 2 optimisation**
   - Sensibilité trafic-maintenance significative (avec caveat proxy annuel).
   - Message final : *“Maintenance choices leave measurable network-relevant signatures, which motivates the next optimisation stage.”*

## 5. Liste finale des figures à produire pour la présentation

- **Slide 10 — “Why a graph?”**
  - Type : **dual panel** (`bar chart` + `bar chart` ou `small multiples`).
  - Données source exactes :
    - `reports/cracking_correlation_spatial_bins.csv` colonnes `distance_bin`, `median_change_corr`.
    - `graph_data/osm_validation_findings.json` blocs `spatial_route_subset.validated_at_thresholds` et `recommendation.edges_dropped_pct`.
  - Message visuel principal : la proximité spatiale porte un signal réel, et une large part des arêtes locales/route est physiquement plausible sans prétendre former un vrai réseau routable.
  - Pièges à éviter : ne pas valider `same_functional_class` comme arête “physique”; ne pas résumer l’OSM en un simple “90% correct” car ce n’est plus vrai à grande échelle.

- **Slide 11 — “Core benchmark on HPMS16”**
  - Type : **bar chart horizontal**.
  - Données source exactes :
    - `graph_data/ensemble_results.json` (`RF local`, `R-GCN`, `Stacked MLP`) ;
    - `reports/treatment_feature_ablation.csv` pour le rôle des features projets;
    - éventuellement `reports/part1_rgcn_temporal.csv` si tu veux détailler `spatial` vs `spatial_route` vs `full_refined`.
  - Message visuel principal : l’ensemble RF+R-GCN bat à la fois RF seul et R-GCN seul.
  - Pièges à éviter : **ne pas mélanger HPMS16 et MEPDG dans la même figure**; garder les modèles comparés sur la même cible.

- **Slide 12 — “Target choice and graph refinement”**
  - Type : **dual panel**.
  - Données source exactes :
    - panneau gauche : `reports/distress_model_comparison.csv` colonnes `target`, `r2_test`, `multitask_r2_test`.
    - panneau droit : `reports/materials_weight_sweep.json` liste `results` (`label`, `r2_test`, poids).
  - Message visuel principal : le multi-task n’est pas adapté ici, et le meilleur résultat absolu vient d’un single-task MEPDG avec similarité pavement/climate enrichie.
  - Pièges à éviter : faire apparaître clairement que le panneau gauche et le panneau droit ne répondent pas exactement à la même question; ne pas masquer les valeurs négatives de patch/potholes.

- **Slide 13 — “Robustness and phase-2 relevance”**
  - Type : **dual panel** (`heatmap` + `bar chart`), **à condition de le refaire proprement**.
  - Données source exactes :
    - panneau gauche : `reports/part1_ood_temporal.csv` colonnes `graph_variant`, `rf_test_r2`, `ridge_test_r2`, `gcn_test_r2`, `rgcn_test_r2`.
    - panneau droit : `reports/traffic_sensitivity.json` champs `event_mean_delta`, `control_mean_delta`, `welch_p_value_events_vs_controls`; idéalement CI calculé depuis `reports/traffic_sensitivity_events.csv` et `reports/traffic_sensitivity_controls.csv`.
  - Message visuel principal : les modèles sont région-spécifiques, mais les années de maintenance laissent déjà une signature trafic utile pour l’optimisation future.
  - Pièges à éviter : **ne pas utiliser la version actuelle du panneau trafic qui parle d’“AADTT” alors que le fichier dit `ANNUAL_GESAL_TREND` proxy** (`reports/traffic_sensitivity.json:2-3`); **ne pas utiliser une heatmap interpolée**.

- **Figure à ABANDONNER explicitement : le bar chart “model comparison” déjà généré dans `reports/presentation_figures/fig11_model_comparison.png`**.
  - Raison : il mélange des résultats HPMS16 et MEPDG dans un même classement, donc la comparaison n’est pas apples-to-apples.

- **Figure à ABANDONNER explicitement : la heatmap géographique déjà générée dans `reports/presentation_figures/fig12_geographic_robustness.png`**.
  - Raison : les valeurs ont été interpolées manuellement; elles doivent être remplacées par la matrice exacte de `reports/part1_ood_temporal.csv:2-4`.

- **Figure à ABANDONNER explicitement : le panneau trafic déjà généré dans `reports/presentation_figures/fig13_correlation_and_sensitivity.png`**.
  - Raison : il affiche des barres d’erreur illustratives et un label `AADTT` non conforme aux données réelles (`reports/traffic_sensitivity.json:2-3`).

- **Figure à AJOUTER par rapport au storyboard initial : single-task vs multi-task by distress target**.
  - Raison : c’est l’un des résultats méthodologiques les plus nets du repo (`reports/distress_model_comparison.csv:2-6`) et il justifie à la fois le choix de cible et la structure des modèles.
