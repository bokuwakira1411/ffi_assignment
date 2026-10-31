## (i) Phi / return by primary label

| task_id | primary | n | phi_T | phi_max | G_base | len_mean |
|---|---|---|---|---|---|---|
| 0 | A | 11 | 0.045 | 0.045 | -0.497 | 520.000 |
| 0 | C | 1 | 0.000 | 0.000 | -0.520 | 520.000 |
| 0 | O | 1 | 0.000 | 0.000 | -0.520 | 520.000 |
| 0 | S | 7 | 1.000 | 1.000 | 1.213 | 287.429 |
| 8 | A | 2 | 0.500 | 0.500 | -0.437 | 520.000 |
| 8 | B | 1 | 0.667 | 0.667 | -0.353 | 520.000 |
| 8 | D | 3 | 0.667 | 0.667 | -0.353 | 520.000 |
| 8 | E | 6 | 0.611 | 0.611 | -0.381 | 520.000 |
| 8 | S | 8 | 1.000 | 1.000 | 0.903 | 430.250 |

## (ii) Does G rank episodes sensibly?

| task_id | all_success_G_above_all_failure_G | min_G_success | max_G_failure | spearman(G, success) | spearman(G, phi_T) among failures | distinct G among failures | n_fail |
|---|---|---|---|---|---|---|---|
| 0 | True | 1.124 | -0.270 | 0.934 | 1.000 | 2 | 13 |
| 8 | True | 0.819 | -0.353 | 0.908 | 1.000 | 2 | 12 |

## (iii) Init states with mixed success over baseline seeds

| task_id | n_init_states | n_seeds | all_fail | mixed | all_success | mixed_init_ids |
|---|---|---|---|---|---|---|
| 0 | 50 | 3 | 22 | 16 | 12 | 1 4 6 7 9 16 19 21 22 28 31 34 37 38 41 48 |
| 8 | 50 | 3 | 14 | 23 | 13 | 6 7 9 10 12 13 14 16 17 22 26 27 30 33 38 39 40 41 43 44 46 47 48 |
