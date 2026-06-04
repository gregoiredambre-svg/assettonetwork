| target | transform | winsor_cap | coverage_pct | zero_pct_observed | r2_train | r2_val | r2_test | mae_test | rmse_test | smape_test | n_train | n_val | n_test | modelling_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MEPDG_CRACKING_PERCENT_AC | identity |  | 99.839658 | 64.530437 | 0.441253 | 0.305054 | 0.542448 | 5.690203 | 9.403492 | 161.263598 | 10804 | 236 | 170 | Broader alligator-cracking measure; also strong enough for direct regression. |
| HPMS16_CRACKING_PERCENT_AC | identity |  | 99.755669 | 45.641026 | 0.532896 | 0.476277 | 0.454388 | 8.591484 | 11.387843 | 141.977477 | 10800 | 236 | 170 | Stable headline cracking target with broad coverage; suitable for standard regression. |
| ME_PERCENT_WHEEL_PATH_CRACK | identity |  |  |  | 0.532208 | 0.475499 | 0.441083 | 5.356449 | 7.029957 | 146.031998 | 10800 | 236 | 170 |  |
| LONGIGATOR_CRACKING | identity |  |  |  | 0.496054 | 0.297966 | 0.435395 | 34.554306 | 54.055574 | 157.622359 | 10808 | 236 | 170 |  |
| MEPDG_TRANS_CRACK_LENGTH_AC | log1p_winsor99 | 7678.260000 | 98.701993 | 40.380599 | -0.051128 | -0.056336 | 0.007658 | 1001.926361 | 1971.738367 | 145.085259 | 10692 | 236 | 170 | Right-skewed length metric; needs log-style transform and careful error interpretation. |
