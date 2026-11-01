# DSRL nhỏ (task 8): kết quả

Sinh bởi `scripts/dsrl_train.py`. Eval: task 8 × 50 init × seed 3000/4000/5000, noise ~ N(mean của actor, I). official và v2 θ4 lấy từ `outputs/failure_labels/labels.csv` (cùng seed env, cùng init state; noise của policy là các lần rút khác).

## Thu thập theo iteration

| iter | thành công | rejected groups | PPO dừng | approx KL cuối | clip frac cuối | |mu| cuối |
|---|---|---|---|---|---|---|
| 0 | 52/128 | 2/16 | target_kl at epoch 6 | 0.0337 | 0.463 | 0.0114 |
| 1 | 49/128 | 2/16 | target_kl at epoch 5 | 0.0351 | 0.446 | 0.0164 |
| 2 | 48/128 | 1/16 | target_kl at epoch 5 | 0.0322 | 0.452 | 0.0187 |
| 3 | 50/128 | 1/16 | target_kl at epoch 4 | 0.0320 | 0.444 | 0.0217 |

## Eval

| arm | s3000 | s4000 | s5000 | gộp |
|---|---|---|---|---|
| dsrl | 25/50 | 21/50 | 27/50 | 73/150 = 48.7% |
| official | 18/50 | 22/50 | 20/50 | 60/150 = 40.0% |
| fpo_v2_sel | 26/50 | 29/50 | 21/50 | 76/150 = 50.7% |

McNemar chính xác, cặp theo (seed, init):

- dsrl vs official: chỉ dsrl 26, chỉ official 13, p = 0.053
- dsrl vs fpo_v2_sel: chỉ dsrl 28, chỉ fpo_v2_sel 31, p = 0.795

Nhãn chính v2 của các thất bại dsrl (77): A 26, E_contact 25, B 15, F 5, G 3, D 2, E_free 1
Số ấm đã lên bếp trong thất bại (0/1/2): 19/57/1

Actor: 820,638 tham số. Episode env: thu thập 512, eval 150.
Thời gian: {'eval': 322.7016966342926}
