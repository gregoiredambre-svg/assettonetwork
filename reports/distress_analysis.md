# Distress Analysis

## Target profiles

| Target | Transform | Coverage % | Zero % | Median | P99 | Max | Modelling note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HPMS cracking (%) | identity | 99.8 | 45.6 | 1.000 | 54.000 | 65.000 | Stable headline cracking target with broad coverage; suitable for standard regression. |
| MEPDG cracking (%) | identity | 99.8 | 64.5 | 0.000 | 63.000 | 100.000 | Broader alligator-cracking measure; also strong enough for direct regression. |
| Transverse cracking length | log1p_winsor99 | 98.7 | 40.4 | 156.000 | 7831.540 | 18522.000 | Right-skewed length metric; needs log-style transform and careful error interpretation. |
| Patched area | log1p_winsor99 | 100.0 | 88.4 | 0.000 | 153.554 | 530.400 | Highly zero-inflated and heavy-tailed; regression alone is difficult and event-style framing may help. |
| Pothole area | log1p_winsor99 | 100.0 | 97.6 | 0.000 | 0.200 | 7.150 | Rare-event distress with extreme sparsity; strongest candidate for two-stage or hurdle modelling. |

## Single-task versus multi-task

| Target | Single-task test R² | Multi-task test R² | ΔR² | Transform | n_test |
| --- | --- | --- | --- | --- | --- |
| HPMS cracking (%) | 0.454 | 0.275 | +0.180 | identity | 170 |
| MEPDG cracking (%) | 0.542 | 0.235 | +0.308 | identity | 170 |
| Transverse cracking length | 0.008 | -0.388 | +0.396 | log1p_winsor99 | 170 |
| Patched area | -0.029 | -18.342 | +18.314 | log1p_winsor99 | 170 |
| Pothole area | -0.006 | -0.055 | +0.049 | log1p_winsor99 | 170 |
