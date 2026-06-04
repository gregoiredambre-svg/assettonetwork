# Presentation Results Slides

This slide pack is designed for a 5-minute dissertation presentation:

**From Asset to Network: Graph-Based Optimisation of Interdependent Maintenance Decisions for Transportation Networks**

The emphasis is on what the current results support, what they do not support, and how the prediction layer connects to network-aware maintenance decisions.

---

## Slide 1

**Slide title:** Graph construction changes prediction quality

**Main visual:** `figures/presentation/graph_construction_comparison.png`

**Suggested layout:** Left: bar chart. Right: three short bullets.

**Slide text:**
- **Exact model:** single-task temporal R-GCN, meaning a graph neural network that predicts one cracking target while passing information along graph edges.
- **Exact target:** `HPMS16_CRACKING_PERCENT_AC`, the HPMS-style asphalt cracking percentage from the LTPP AC distress panel.
- **Evaluation setup:** same target and same temporal split for all three bars: train on earlier years, validate on intermediate years, test on later unseen years (`2019-2021` test period in the main temporal benchmark family).
- **What changes across the bars:** only the graph construction:
  - spatial only
  - spatial + route
  - full refined = spatial + route + same functional class / refined similarity
- **How the graph is used here:** the graph tells the model which road sections can share information during prediction.
- **Result:** spatial only `R² = 0.337`, spatial + route `0.361`, full refined `0.371`.
- **Meaning:** better relationships between sections lead to better cracking prediction.

**Speaker notes:**
This first result asks a basic question: does the way we connect road sections matter? The answer is yes. A spatial-only graph reaches a test R² of about 0.337. Adding route continuity raises that to about 0.361. The full refined graph, which combines spatial, route, and functional similarity, reaches about 0.371. So the graph is not just decoration. The relationships we encode between sections change what the graph model can learn.

**Key conclusion:**
More meaningful graph relationships improve graph-model performance.

**Caution / what not to overclaim:**
This does not prove that the graph model is already the best overall model. It only proves that graph construction matters within the graph-model family.

---

## Slide 2

**Slide title:** For the main cracking targets, refined graph models can compete with or beat local baselines

**Main visual:** `figures/presentation/cracking_prediction_comparison.png`

**Suggested layout:** Full-width grouped bar chart with one short caption below.

**Slide text:**
- **What was tested:** four model types on the two main asphalt cracking targets, all using the same in-domain temporal evaluation logic.
- **Targets:** 
  - `HPMS16_CRACKING_PERCENT_AC` = HPMS-style cracking percentage
  - `MEPDG_CRACKING_PERCENT_AC` = mechanistic-empirical alligator cracking percentage
- **Models compared:**
  - local Random Forest = non-graph baseline using section features only
  - basic R-GCN = graph model using the baseline full refined graph
  - refined R-GCN = same graph model family, but with reweighted graph construction
  - stacked ensemble = meta-model combining Random Forest and R-GCN predictions
- **Evaluation setup:** same in-domain temporal benchmark logic, with later years held out for testing rather than random row splits.
- **How the graph is used here:** the graph lets each section learn from related sections instead of being predicted in isolation.
- **HPMS-style cracking:** local Random Forest `0.449`, basic graph model `0.371`, refined graph model `0.533`, stacked ensemble `0.536`.
- **Mechanistic-empirical cracking:** local Random Forest `0.523`, basic graph model `0.542`, refined graph model `0.558`, stacked ensemble `0.526`.
- **Meaning:** graph-aware modelling can improve the main cracking targets, but the strongest model depends on the target.

**Speaker notes:**
This is the main benchmark slide. I compare a local non-graph baseline, a basic graph model, a refined graph model, and a stacked ensemble. For HPMS-style cracking percentage, the basic graph model alone is weaker than the local Random Forest, but once the graph is refined it rises to about 0.533, which is an absolute gain of about 0.084 over the local baseline, or roughly 19 percent. The best overall HPMS result is still the stacked ensemble at about 0.536. For the mechanistic-empirical cracking target, the refined graph model is the best result at about 0.558, ahead of both the Random Forest and the ensemble by about 0.035 in absolute R². So the graph helps, but not in a universal or simplistic way.

**Key conclusion:**
Graph-aware modelling improves the main cracking targets when the graph is well designed, but the strongest model depends on the target.

**Caution / what not to overclaim:**
Do not say that graph neural networks always beat Random Forest. On HPMS-style cracking, the naive graph model does not. The gain comes from graph refinement and, in one case, from combining models.

---

## Slide 3

**Slide title:** The same full-refined graph model works well on several cracking-style targets, but not equally on every formulation

**Main visual:** `figures/presentation/distress_target_performance.png`

**Suggested layout:** Left: horizontal bar chart. Right: interpretation bullets.

**Slide text:**
- **Exact model family:** single-task temporal relation-aware R-GCN on the `full_refined` graph, run separately for each target.
- **What was tested:** a cleaner five-target set with high coverage and better suitability for regression than the earlier patching and pothole targets.
- **Targets included:** HPMS-style cracking %, mechanistic-empirical cracking %, wheel-path cracking %, longigator cracking area, and mechanistic-empirical transverse cracking length.
- **Evaluation setup:** same temporal prediction framework and same later-year test split for all five targets, so differences mainly reflect target behaviour rather than a different protocol.
- **Stronger results:** MEPDG cracking `R² = 0.542`, HPMS cracking `0.454`, wheel-path cracking `0.441`, longigator cracking `0.435`.
- **Weaker result:** transverse cracking length remains near zero at `0.008` even with high coverage.
- **Interpretation:** the graph model is consistently useful on several cracking-style targets, but not every distress formulation is equally learnable.
- **Meaning:** the issue was not only missing data; target definition and target behaviour also matter.

**Speaker notes:**
This revised slide uses a better target set than the earlier distress comparison. Instead of sparse event-like targets such as potholes and patching, I kept five better-covered asphalt distress targets and reran the same full-refined relation-aware graph model. The result is much cleaner. The graph model performs consistently on several cracking-style targets, with test R² between about 0.435 and 0.542. But one transverse cracking length target still remains near zero, even though coverage is high. So the lesson is not simply that low-data targets fail. The deeper point is that some target formulations are much better matched to this graph-based regression framework than others.

**Key conclusion:**
The graph model is credible across several well-covered cracking-style targets, but target formulation still matters.

**Caution / what not to overclaim:**
Do not claim that all distress variables are equally predictable just because coverage is high. Even with better-covered targets, some formulations remain hard for this regression setup.

---

## Slide 4

**Slide title:** Geographic transfer to unseen states remains weak

**Main visual:** `figures/presentation/ood_generalisation_limit.png`

**Suggested layout:** Centered bar chart with one red callout box underneath.

**Slide text:**
- **Exact test:** held-out-state evaluation on `HPMS16_CRACKING_PERCENT_AC`, where models are trained on some states and evaluated on entirely unseen states.
- **States evaluated in this summary:** 5 representative held-out states: `48`, `4`, `12`, `6`, `40`.
- **Models compared:** local Random Forest, baseline R-GCN, and stacked RF + R-GCN ensemble.
- **Why this matters:** this is a harder test of geographic transfer, not just normal test-set prediction.
- **Mean held-out-state performance:** local Random Forest `0.173`, graph model `-0.001`, stacked ensemble `0.119`.
- **Meaning:** the current models help most in-domain; they do not yet generalise strongly across state boundaries.
- **Interpretation:** this is a limitation, not a failure, because it defines the current boundary of the contribution.

**Speaker notes:**
This is the main limitation slide. When the models are tested on unseen states rather than later years from known states, performance drops sharply. The local Random Forest has the highest mean held-out-state R² at about 0.173. The graph model is near zero on average, and the stacked ensemble is positive but still lower than the local baseline. So the current results support in-domain temporal prediction, not universal national generalisation. That boundary matters, and it should be stated clearly.

**Key conclusion:**
The current contribution is strongest for in-domain prediction, not for out-of-state transfer.

**Caution / what not to overclaim:**
Do not claim that the graph solves geographic generalisation across the US. It does not.

---

## Slide 5

**Slide title:** The graph plays two roles in the dissertation

**Main visual:** Simple two-column diagram or flow graphic. No data chart required.

**Suggested layout:** Two side-by-side boxes:
- Left box: Prediction structure
- Right box: Decision structure

**Slide text:**
- **Prediction role:** in the R-GCN results, the graph links road sections so the model can use neighbouring or similar sections when predicting future cracking.
- **Decision role:** in the network-aware framework, the same graph defines which projects are related, potentially conflicting, corridor-linked, or disruption-sensitive.
- **Graph inputs:** spatial proximity, route continuity, functional similarity, and later proxy-based interaction structure.
- **Why this matters:** the graph is not only a modelling device; it is also a structure for maintenance coordination.
- **Meaning:** this is the bridge from asset-level prediction to network-aware maintenance planning.

**Speaker notes:**
Up to this point I have shown the prediction evidence. This slide explains why the graph matters beyond prediction. First, it is a prediction structure: it allows each section to learn from neighbouring or similar sections rather than being modelled in isolation. Second, it is a decision structure: it defines which projects are related, where simultaneous works may conflict, and which sections may be more network-sensitive. That is why the dissertation is not just about predicting pavement cracking. It is about using graph relationships to move from isolated asset decisions toward network-aware maintenance planning.

**Key conclusion:**
The graph is both a predictive device and a decision-support device.

**Caution / what not to overclaim:**
This slide explains the framework logic. It does not by itself prove that the full optimisation layer is already complete.

---

## Slide 6

**Slide title:** Network disruption is represented with proxies, not full traffic simulation

**Main visual:** `figures/presentation/proxy_ranking.png`

**Suggested layout:** Left: bar chart of most disruptive sections by proxy ranking. Right: four short proxy definitions.

**Slide text:**
- **What was tested:** synthetic closure or disruption scenarios on the graph, used to estimate which sections appear most network-sensitive.
- **What the graph is doing here:** it is no longer predicting cracking; it is representing connectivity, detours, and section interactions under disruption scenarios.
- **`delta_vht_proxy`:** approximate extra weighted travel burden after disruption.
- **`connectivity_loss_pct`:** how much the network fragments.
- **`disconnected_od_pct`:** weighted share of origin-destination pairs that become unreachable.
- **`disruption_score`:** composite proxy combining travel penalty and fragmentation.
- **Meaning:** these measures give a network-aware ranking of section importance even without full traffic simulation.

**Speaker notes:**
This slide shows how network impact is approximated without a full traffic assignment model. The figure ranks the most disruptive single-section closures using the travel-burden proxy. The four proxy outputs answer complementary questions: how much extra burden is created, how much the network fragments, how many weighted OD pairs become unreachable, and what the overall composite disruption score looks like. These are useful for maintenance decision support because they let us compare projects on network sensitivity, but they are still proxies rather than observed congestion or equilibrium traffic flow.

**Key conclusion:**
Proxy disruption metrics make network-aware decision support possible when full traffic simulation data are unavailable.

**Caution / what not to overclaim:**
These measures are not equivalent to a full traffic simulator or observed rerouting behaviour.

---

## Slide 7

**Slide title:** Decision layer: from isolated repairs to a portfolio of projects

**Main visual:** Conceptual workflow diagram. No final empirical portfolio comparison is available in the current repo.

**Suggested layout:** Horizontal pipeline with four blocks:
1. Predict deterioration
2. Estimate cost / need
3. Add disruption proxies and project interactions
4. Select a portfolio under budget and coordination constraints

**Slide text:**
- **What this step does:** it converts the cracking-prediction and proxy outputs into maintenance decisions.
- **Candidate projects:** sections with predicted deterioration and maintenance need.
- **Inputs to the decision layer:** predicted cracking / deterioration, maintenance need, cost assumptions, graph relationships, and disruption proxies.
- **Graph contribution:** project interactions, conflict penalties, corridor relationships, and disruption sensitivity.
- **Decision logic:** choose a portfolio under budget and coordination constraints, rather than ranking sections one by one.
- **Meaning:** the framework moves from isolated repair decisions toward coordinated maintenance planning.

**Speaker notes:**
I am presenting this as a decision framework slide rather than a final optimisation result. The current repo provides the prediction layer and the network proxy layer, but not yet a clean final portfolio-comparison experiment that I would defend as completed empirical evidence. The intended decision logic is: identify candidate projects from deterioration predictions, combine those with cost and network-disruption proxies, then select a portfolio under budget and coordination constraints. The key shift is from choosing isolated bad sections to choosing a portfolio that balances asset need with network sensitivity.

**Key conclusion:**
The dissertation contribution is moving from section-by-section prediction toward graph-informed portfolio decisions.

**Caution / what not to overclaim:**
Present this as the decision framework or next step, not as a fully validated optimisation result unless new empirical portfolio outputs are added.

---

## Slide 8

**Slide title:** What the results support, and where the boundary is

**Main visual:** Compact summary table with two columns: Supported / Not supported.

**Suggested layout:** Simple two-column table with four rows.

**Slide text:**
- **Supported**
  - Graph construction matters for graph-model prediction quality.
  - Refined graph-aware models improve the two main cracking targets.
  - The benefit is strongest for gradual cracking, not every distress type.
  - Proxy disruption metrics support a network-aware decision framework.
- **Not supported**
  - Graph models always beat local baselines.
  - All distress targets are predictable from this data.
  - The method already generalises well to unseen states.
  - The proxies are equivalent to full traffic simulation.

**Speaker notes:**
This final slide is the honest summary. The evidence supports that graph construction matters, that engineering-informed graph refinement improves the main cracking predictions, and that the graph provides a useful structure for network-aware decision support. But the evidence does not support universal superiority, universal predictability, strong out-of-state transfer, or equivalence between proxies and full traffic simulation. That boundary is important because it defines the actual contribution of the dissertation.

**Key conclusion:**
The contribution is a defensible step from asset-level deterioration modelling toward graph-informed, network-aware maintenance decision support.

**Caution / what not to overclaim:**
Keep the conclusion at the level of supported evidence: improved cracking prediction for key targets, plus a credible decision-support framework.

---

## 60-90 second oral narrative

The main result is that the graph is useful, but only when it is built carefully and interpreted honestly. First, different graph constructions produce different prediction quality, so the way road sections are connected is not just decorative. Second, for the two main cracking targets, refined graph-aware models become competitive with or stronger than local non-graph baselines: on HPMS-style cracking the best result is a stacked ensemble at about 0.536, while a refined graph-only model reaches about 0.533; on the mechanistic-empirical cracking target, the refined graph model is strongest at about 0.558. Third, the benefit is target-specific: it works for gradual cracking, but not for rare or noisy targets like potholes or patching. Finally, out-of-state transfer remains weak, so the contribution is not universal national generalisation. The defensible claim is that graph-informed modelling improves key cracking predictions in-domain and provides a credible bridge from asset-level deterioration modelling toward network-aware maintenance decision support using disruption proxies rather than full traffic simulation.

---

## Backup Slide

**Slide title:** OpenStreetMap checks supported the local graph interpretation

**Main visual:** `figures/presentation/osm_validation_backup.png`

**Suggested layout:** Full-width two-panel figure with one short caveat line underneath.

**Slide text:**
- **What was checked:** the resumable overnight OpenStreetMap checkpoint tested `13,267` graph edges, covering about `73.9%` of the full refined graph.
- **Main finding:** the local graph is only partly road-topological by design: about `43.0%` of tested spatial edges and `39.7%` of tested full-refined edges were strictly validated as plausible local road links.
- **Interpretation:** short spatial and many same-route links are often physically plausible, while some similarity edges behave more like contextual interdependency links than direct road-topology links.
- **Interpretation:** this supports reading the graph as a local interdependency structure rather than a purely abstract network.
- **Important caveat:** this is still a sanity check, not a complete national topological proof, and OSM failure does not automatically mean an edge is useless for deterioration modelling.

**Speaker notes:**
This is a methodological support slide, not a headline result slide. I used two OpenStreetMap checks, and this backup figure focuses on the larger overnight validation checkpoint. That checkpoint tested 13,267 graph edges, which is about 74 percent of the full refined graph. The result is deliberately mixed: many short spatial and same-route links are topologically plausible, but not every graph edge is meant to be a direct road-topology edge. In particular, same-functional-class similarity edges often encode contextual similarity rather than a guaranteed drivable local path. So OSM validation is useful here as a sanity check on local plausibility, not as a requirement that every graph edge behave like a strict navigation link.

**Key conclusion:**
The graph has meaningful external topological support, while still behaving as an interdependency graph rather than a pure road-topology graph.

**Caution / what not to overclaim:**
Do not present this as proof that every graph edge is a direct physical road link or that the graph is a complete operational road network.
