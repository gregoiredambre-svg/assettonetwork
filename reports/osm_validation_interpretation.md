# OSM Validation Interpretation

## Coverage and scope

The existing resumable OSM checkpoint already covers all three edge types and therefore all three graph variants. It tests 13,267 edges overall, including 4,657 `same_functional_class` edges. That means the current OSM evidence does reach into `full_refined`, although only 73.9% of that graph's edges are currently covered by the stored checkpoint.

## Main interpretation

- **Most physically road-like graph variant**: `spatial_route` (Spatial + Route). It keeps the local spatial layer and same-route corridor links without adding the broader same-class similarity layer.
- **Best interpreted as an interdependency graph**: `full_refined` (Full refined). Its `same_functional_class` edges are context/similarity links, so they should not be judged only as physical road adjacency.
- **Are spatial edges mostly valid road connections?** Yes, but not universally. The validated share in the tested OSM checkpoint is 43.0%, which still leaves a meaningful tail of no-path and long-detour cases.
- **Are same_route edges more strongly validated than spatial edges?** No in the current stored checkpoint. The tested validated share is 35.7% for `same_route` versus 43.0% for `spatial`, so route links should be interpreted cautiously rather than assumed superior by default.
- **Are full_refined similarity edges expected to fail OSM validation?** Often yes, and that is not necessarily a problem. `same_functional_class` edges validate at 35.8%, but they were designed as same-state similarity/interdependency links rather than guaranteed routable adjacency.

## Concrete bad-edge examples

- `51_5009` ↔ `51_A340` (spatial, failed_no_path): The graph links these sections, but OSM found no local drivable connection between them, so the edge is weak as a physical road link. Straight-line distance 55.97 km.
- `51_5009` ↔ `51_A310` (spatial, failed_no_path): The graph links these sections, but OSM found no local drivable connection between them, so the edge is weak as a physical road link. Straight-line distance 55.76 km.
- `4_0122` ↔ `4_A310` (same_route, failed_no_path): The graph links these sections, but OSM found no local drivable connection between them, so the edge is weak as a physical road link. Straight-line distance 39.17 km.
- `19_0759` ↔ `19_5042` (same_route, failed_no_path): The graph links these sections, but OSM found no local drivable connection between them, so the edge is weak as a physical road link. Straight-line distance 26.02 km.
- `51_5010` ↔ `51_A310` (same_functional_class, failed_no_path): The sections look similar in context, but OSM found no practical local drivable connection, so this behaves as a similarity edge rather than a physical link. Straight-line distance 51.67 km.
- `51_5010` ↔ `51_A350` (same_functional_class, failed_no_path): The sections look similar in context, but OSM found no practical local drivable connection, so this behaves as a similarity edge rather than a physical link. Straight-line distance 51.46 km.

## Concrete good-edge examples

- `4_0265` ↔ `4_0266` (spatial, validated): Both LTPP sections snap to the same OSM road edge, which is the strongest possible local topological support. Straight-line distance 0.21 km, detour ratio 0.00.
- `55_0214` ↔ `55_0222` (spatial, validated): Both LTPP sections snap to the same OSM road edge, which is the strongest possible local topological support. Straight-line distance 0.21 km, detour ratio 0.00.
- `4_B902` ↔ `4_B903` (same_route, validated): Both LTPP sections snap to the same OSM road edge, which is the strongest possible local topological support. Straight-line distance 1.61 km, detour ratio 0.00.
- `39_0203` ↔ `39_0209` (same_route, validated): These sections are on a clearly connected route corridor in OSM, so the same-route interpretation is well supported. Straight-line distance 1.04 km, detour ratio 0.02.
- `39_0106` ↔ `39_0265` (same_functional_class, validated): Even though this edge was added for similarity, OSM still finds a short drivable connection, so it is both contextually similar and locally plausible. Straight-line distance 0.86 km, detour ratio 0.02.
- `55_0114` ↔ `55_C903` (same_functional_class, validated): Even though this edge was added for similarity, OSM still finds a short drivable connection, so it is both contextually similar and locally plausible. Straight-line distance 1.16 km, detour ratio 0.02.

## Recommended use in the dissertation

- **Should the graph be presented as a true road network?** No. It should not be presented as a fully routable national road network.
- **Should it be presented as a section-level interdependency graph?** Yes. That framing is both more accurate and more defensible.
- **Which graph should be used for shortest-path disruption calculations?** Prefer the spatial + same-route view, because those edges are the most interpretable as physical road-like connectivity.
- **Which graph should be used for broader deterioration / treatment-context analysis?** The full refined graph remains appropriate, because similarity and contextual interdependence matter even when the links are not locally routable in OSM.
- **Should OSM validation be treated as full validation?** No. It is best used as a local sanity check that helps distinguish physically road-like edges from broader similarity/interdependency edges.

The tested checkpoint contains 865 failed same_route edges, so corridor links should still be inspected rather than assumed correct by default.
