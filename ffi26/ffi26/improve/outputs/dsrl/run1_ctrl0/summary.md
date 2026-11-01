# Đối chứng DSRL: actor chưa train, cùng noise

Sinh bởi `scripts/dsrl_eval_ctrl.py`. ctrl0 = `run1/actor_iter0.pt` (mean = 0, tức SmolVLA official với noise N(0, I)), cùng seed env, cùng init, cùng seed noise với eval của dsrl. Hai arm chỉ khác nhau ở mean của actor.

| arm | s3000 | s4000 | s5000 | gộp |
|---|---|---|---|---|
| dsrl (actor_iter4) | 25/50 | 21/50 | 27/50 | 73/150 = 48.7% |
| ctrl0 (actor_iter0, cùng noise) | 26/50 | 24/50 | 22/50 | 72/150 = 48.0% |
| official (eval cũ, noise khác) | 18/50 | 22/50 | 20/50 | 60/150 = 40.0% |

McNemar chính xác, cặp theo (seed, init):

- dsrl vs ctrl0: chỉ dsrl 18, chỉ ctrl0 17, p = 1.000
- ctrl0 vs official: chỉ ctrl0 31, chỉ official 19, p = 0.119

Nhãn chính v2 của các thất bại ctrl0 (78): E_contact 29, A 26, B 18, F 2, D 2, G 1
