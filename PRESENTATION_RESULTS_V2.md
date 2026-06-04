# PRESENTATION_RESULTS_V2.md

## Analyse 1 — Intervalle de confiance bootstrap pour la sensibilité trafic
- **Statut d'exécution** : réussi.
- **Script** : `scripts/run_traffic_sensitivity_bootstrap.py`
- **Fichiers produits** :
  - `reports/traffic_sensitivity_bootstrap.json`
- **Chiffres clés** :
  - `event_mean = 10608.91`, IC95% `[7418.05, 13883.33]` (`reports/traffic_sensitivity_bootstrap.json`)
  - `control_mean = 16624.94`, IC95% `[15332.07, 17946.60]` (`reports/traffic_sensitivity_bootstrap.json`)
  - `diff_mean = -6016.03`, IC95% `[ -9415.00, -2535.82 ]`, `welch_p_value = 0.0006887` (`reports/traffic_sensitivity_bootstrap.json`)
- **Interprétation** : les années avec maintenance présentent un delta pré→post d'`ANNUAL_GESAL_TREND` significativement plus faible que les contrôles, et le nouvel IC95% rend ce résultat beaucoup plus défendable visuellement.
- **Effet sur la stratégie narrative** : **oui, légèrement**. Le message de `PRESENTATION_RESULTS_STRATEGY.md` reste le même, mais ce résultat transforme un finding utile avec caveat en visuel solide pour la slide “phase 2 relevance”.

## Analyse 2 — Leave-one-state-out pour l'ensemble RF+R-GCN
- **Statut d'exécution** : partiel mais exploitable.
- **Script** : `scripts/run_ood_ensemble.py`
- **Fichiers produits** :
  - `reports/part1_ood_ensemble.csv`
  - `reports/part1_ood_ensemble_summary.json`
- **Portée exacte** : évaluation sur **5 états représentatifs** (`48`, `4`, `12`, `6`, `40`), pas sur tous les états (`reports/part1_ood_ensemble_summary.json`).
- **Chiffres clés** :
  - moyenne `ensemble_r2 = 0.1190` vs moyenne `rf_r2 = 0.1734` (`reports/part1_ood_ensemble_summary.json`)
  - meilleur état pour l'ensemble : `0.2796` sur l'état `6`; pire : `-0.1851` sur l'état `48` (`reports/part1_ood_ensemble.csv`)
  - l'ensemble bat le RF sur `2/5` états seulement (`reports/part1_ood_ensemble.csv`)
- **Interprétation** : l'ensemble n'élimine pas la non-transférabilité géographique; hors état d'entraînement, le RF local reste en moyenne plus robuste que l'ensemble sur ce sous-échantillon.
- **Effet sur la stratégie narrative** : **oui**. Cela renforce le cadrage déjà proposé : l'ensemble est un très bon **headline in-domain**, mais il ne doit pas être présenté comme une solution à la généralisation inter-états.

## Analyse 3 — Rerun du sweep matériaux/poids sur HPMS16
- **Statut d'exécution** : réussi.
- **Script** : `scripts/run_materials_experiments_hpms16.py`
- **Fichiers produits** :
  - `reports/materials_weight_sweep_hpms16.json`
  - `reports/materials_weight_sweep_hpms16.md`
- **Chiffres clés** :
  - baseline HPMS16 sans matériaux : `R² test = 0.4544` (`reports/materials_weight_sweep_hpms16.json`)
  - meilleure config `pavement_dominant` : `R² test = 0.5329`, `MAE = 7.6170`, `RMSE = 10.5362` (`reports/materials_weight_sweep_hpms16.json`)
  - gain absolu : `+0.0786` vs baseline (`reports/materials_weight_sweep_hpms16.json`)
- **Interprétation** : le raffinement matériaux + repondération n'est **pas spécifique à MEPDG**; il améliore fortement aussi la cible comparative principale HPMS16.
- **Effet sur la stratégie narrative** : **oui, fortement**. C'est le résultat qui modifie le plus la lecture précédente : on ne peut plus dire seulement que “les matériaux aident surtout MEPDG”. Le graph refinement devient un résultat majeur aussi sur HPMS16.

## Analyse 4 — Apples-to-apples benchmark sur MEPDG (RF vs R-GCN vs ensemble)
- **Statut d'exécution** : réussi.
- **Script** : `scripts/run_mepdg_benchmark.py`
- **Fichiers produits** :
  - `reports/mepdg_benchmark.csv`
  - `reports/mepdg_benchmark.json`
- **Chiffres clés** :
  - `RF local`: `R² test = 0.5233`, `MAE = 5.6400` (`reports/mepdg_benchmark.csv`)
  - `R-GCN baseline`: `R² test = 0.5424`, `MAE = 5.6902` (`reports/mepdg_benchmark.csv`)
  - `Stacked MLP ensemble`: `R² test = 0.5263`, `MAE = 5.1671`; meilleur résultat global de référence = `R-GCN best materials sweep 0.5582` (`reports/mepdg_benchmark.csv`)
- **Interprétation** : sur MEPDG, le graphe pur bat déjà le RF local, et l'ensemble n'apporte pas le même surcroît que sur HPMS16; le meilleur score reste un R-GCN raffiné par matériaux/poids.
- **Effet sur la stratégie narrative** : **oui, modérément**. Cela consolide l'idée de garder **HPMS16 pour le benchmark inter-modèles** et **MEPDG pour le meilleur résultat spécialisé**, plutôt que d'essayer d'utiliser un seul message pour les deux cibles.

## Synthèse finale

### Y a-t-il un résultat qui modifie significativement le message de la présentation ?
- **Oui** : `reports/materials_weight_sweep_hpms16.json` modifie le plus le message.
- Avant ce rerun, le raffinement matériaux/poids était surtout un bon résultat MEPDG.
- Maintenant, il apporte aussi un gain fort sur HPMS16, avec `R² test = 0.5329` contre `0.4544` précédemment.
- Cela veut dire que le **graph refinement** est un vrai résultat de fond, pas juste une extension annexe.

### Y a-t-il une décision narrative à reconsidérer ?
- **Oui, partiellement**.
- La recommandation “headline = ensemble HPMS16 à `0.5356`” reste valable, mais elle doit être nuancée par deux faits nouveaux :
  - sur **HPMS16**, un R-GCN raffiné par matériaux/poids monte presque au même niveau (`0.5329`) (`reports/materials_weight_sweep_hpms16.json`)
  - sur **MEPDG**, l'ensemble ne domine pas; c'est le **R-GCN raffiné** qui reste meilleur (`0.5582`) (`reports/mepdg_benchmark.csv`, `reports/materials_weight_sweep.json`)
- La décision la plus propre pour la présentation devient donc :
  - **HPMS16** = benchmark principal entre familles de modèles
  - **MEPDG** = meilleur résultat absolu après raffinement du graphe
  - **OOD** = limite claire de généralisation
  - **Traffic sensitivity** = pont vers l'optimisation

### Quels sont les 3-5 chiffres définitifs à utiliser dans les figures ?
- `R² test = 0.5356` pour le **Stacked MLP ensemble** sur HPMS16 (`graph_data/ensemble_results.json`)
- `R² test = 0.5329` pour le **HPMS16 materials/weights refined R-GCN** (`reports/materials_weight_sweep_hpms16.json`)
- `R² test = 0.5582` pour le **MEPDG refined R-GCN** (`reports/materials_weight_sweep.json`)
- `ensemble OOD mean R² = 0.1190` vs `RF OOD mean R² = 0.1734` sur 5 états représentatifs (`reports/part1_ood_ensemble_summary.json`)
- `traffic diff_mean = -6016.03`, IC95% `[ -9415.00, -2535.82 ]`, `p = 0.0006887` (`reports/traffic_sensitivity_bootstrap.json`)

### Recommandation finale pour les figures de slides
- **Slide benchmark HPMS16** : comparer `RF local 0.4489`, `R-GCN 0.3708`, `Stacked MLP ensemble 0.5356`, et **ajouter** `HPMS16 refined R-GCN 0.5329` comme nouveau point de comparaison.
- **Slide target/refinement** : garder `MEPDG refined R-GCN 0.5582` comme meilleur résultat absolu, mais l'encadrer comme “specialist target + refined graph semantics”.
- **Slide robustness** : utiliser les vrais chiffres `reports/part1_ood_ensemble_summary.json` et non une interpolation; message = l'ensemble ne résout pas l'OOD.
- **Slide traffic/phase 2** : remplacer définitivement les barres d'erreur illustratives par l'IC bootstrap réel de `reports/traffic_sensitivity_bootstrap.json`.
