# Edge Weight OOD Ablation

- Target: `HPMS16_CRACKING_PERCENT_AC`
- Graph variant: `full_refined`
- States evaluated: `12, 4, 40, 48, 6`

## Mean OOD R² by variant

| variant | rgcn_mean_r2 | ensemble_mean_r2 | rf_mean_r2 |
| --- | --- | --- | --- |
| baseline_mixed_current | -0.0526 | 0.1388 | 0.1734 |
| distance_only | -0.1539 | 0.1235 | 0.1734 |
| unified_full_formula | -0.2328 | 0.1286 | 0.1734 |

## Per-state results

| variant | state_held_out | n_test_nodes | n_test_transitions | rf_test_r2 | rgcn_test_r2 | ensemble_test_r2 | rf_test_mae | rgcn_test_mae | ensemble_test_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_mixed_current | 48 | 172 | 1254 | -0.040407 | -0.097782 | -0.142431 | 7.876225 | 8.166105 | 8.546791 |
| unified_full_formula | 48 | 172 | 1254 | -0.040407 | -0.410076 | -0.179645 | 7.876225 | 9.517246 | 8.872944 |
| distance_only | 48 | 172 | 1254 | -0.040407 | -0.496697 | -0.159241 | 7.876225 | 10.670890 | 8.622451 |
| baseline_mixed_current | 4 | 104 | 681 | 0.360333 | -0.090242 | 0.207160 | 8.868337 | 11.662143 | 10.172087 |
| unified_full_formula | 4 | 104 | 681 | 0.360333 | -0.135557 | 0.203054 | 8.868337 | 12.440858 | 10.446126 |
| distance_only | 4 | 104 | 681 | 0.360333 | -0.079819 | 0.265124 | 8.868337 | 12.487969 | 10.015044 |
| baseline_mixed_current | 12 | 75 | 572 | 0.140420 | -0.067308 | 0.132350 | 10.909778 | 12.585464 | 9.669467 |
| unified_full_formula | 12 | 75 | 572 | 0.140420 | -0.811568 | 0.102307 | 10.909778 | 18.731356 | 9.850307 |
| distance_only | 12 | 75 | 572 | 0.140420 | -0.232444 | 0.038632 | 10.909778 | 15.027364 | 10.099090 |
| baseline_mixed_current | 6 | 69 | 564 | 0.232626 | -0.051807 | 0.266411 | 11.190995 | 13.350644 | 10.750944 |
| unified_full_formula | 6 | 69 | 564 | 0.232626 | -0.031041 | 0.265660 | 11.190995 | 13.647262 | 10.800200 |
| distance_only | 6 | 69 | 564 | 0.232626 | -0.208171 | 0.243564 | 11.190995 | 14.365431 | 10.972046 |
| baseline_mixed_current | 40 | 59 | 495 | 0.174225 | 0.044029 | 0.230686 | 6.369122 | 8.136719 | 6.335138 |
| unified_full_formula | 40 | 59 | 495 | 0.174225 | 0.224162 | 0.251734 | 6.369122 | 7.266384 | 5.908625 |
| distance_only | 40 | 59 | 495 | 0.174225 | 0.247498 | 0.229613 | 6.369122 | 6.980271 | 6.181100 |
