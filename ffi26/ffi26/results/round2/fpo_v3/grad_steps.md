# FPO v2 vs v3: số bước gradient, số mẫu, và vì sao v3 chạy nhiều bước hơn

Script `improve/scripts/fpo_v3_grad_steps.py` (CPU, không train). Nguồn: `outputs/fpo_v{2,3}/iter{k}/train/train_info.json`, `train_log.csv`, advantage trong `iter{k}/decisions/g*.npz` (lọc như `load_decisions`). Mẫu mỗi bước theo code: PPO = 128 decision; BC rollout = 32; v3 thêm 32 mẫu demo. Iter 0 dùng CÙNG decision, cùng θ0, cùng c = 24,61 cho v2 và v3, nên là so sánh sạch nhất.

## Tổng

| run | steps | ppo_samples | bc_samples | demo_samples | train_s |
|---|---|---|---|---|---|
| v2 | 84 | 10752 | 2688 | 0 | 3066 |
| v3 | 402 | 51456 | 12864 | 12864 | 5611 |

## Theo vòng

| run | iter | decisions | steps | steps_per_epoch | epochs | ppo_samples | bc_samples | demo_samples | clip_frac_mean | clip_frac_last10 | kl_end | train_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 0 | 11245 | 23 | 87 | 0.264 | 2944 | 736 | 0 | 0.1287 | 0.2109 | 0.0325 | 811 |
| v2 | 1 | 10880 | 27 | 84 | 0.321 | 3456 | 864 | 0 | 0.1201 | 0.2125 | 0.0456 | 809 |
| v2 | 2 | 10757 | 16 | 83 | 0.193 | 2048 | 512 | 0 | 0.1357 | 0.2125 | 0.1825 | 731 |
| v2 | 3 | 10256 | 18 | 79 | 0.228 | 2304 | 576 | 0 | 0.1536 | 0.2258 | 0.1986 | 715 |
| v3 | 0 | 11245 | 131 | 87 | 1.506 | 16768 | 4192 | 4192 | 0.1422 | 0.2039 | 0.0329 | 1637 |
| v3 | 1 | 11007 | 138 | 85 | 1.624 | 17664 | 4416 | 4416 | 0.1486 | 0.2055 | 0.0275 | 1668 |
| v3 | 2 | 10389 | 44 | 80 | 0.55 | 5632 | 1408 | 1408 | 0.1355 | 0.2023 | 0.1077 | 980 |
| v3 | 3 | 10995 | 89 | 85 | 1.047 | 11392 | 2848 | 2848 | 0.1487 | 0.2008 | 0.1322 | 1326 |

## Advantage dùng trong loss (Â), theo vòng × task

v2: Â = clip(A / std(A cả hai task), ±5), KHÔNG trừ trung bình; v3: Â = clip((A − mean_t) / std_t, ±5).

| run | iter | task | n | A_mean | A_std | Ahat_mean | Ahat_absmean | Ahat_std | frac_Ahat_neg |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 0 | 0 | 5054 | -0.0948 | 0.3547 | -0.306 | 0.776 | 1.144 | 0.633 |
| v2 | 0 | 8 | 6191 | -0.0486 | 0.2664 | -0.157 | 0.606 | 0.859 | 0.613 |
| v2 | 1 | 0 | 4843 | -0.096 | 0.3163 | -0.337 | 0.785 | 1.111 | 0.644 |
| v2 | 1 | 8 | 6037 | -0.0433 | 0.2542 | -0.152 | 0.622 | 0.893 | 0.548 |
| v2 | 2 | 0 | 5107 | -0.0565 | 0.2347 | -0.221 | 0.529 | 0.916 | 0.689 |
| v2 | 2 | 8 | 5650 | -0.0583 | 0.2744 | -0.227 | 0.815 | 1.071 | 0.584 |
| v2 | 3 | 0 | 4662 | -0.0957 | 0.3316 | -0.346 | 0.781 | 1.2 | 0.559 |
| v2 | 3 | 8 | 5594 | -0.0275 | 0.2152 | -0.099 | 0.459 | 0.779 | 0.51 |
| v3 | 0 | 0 | 5054 | -0.0948 | 0.3547 | -0.0 | 0.72 | 1.0 | 0.321 |
| v3 | 0 | 8 | 6191 | -0.0486 | 0.2664 | 0.0 | 0.709 | 1.0 | 0.459 |
| v3 | 1 | 0 | 5038 | -0.0906 | 0.3405 | -0.0 | 0.713 | 1.0 | 0.325 |
| v3 | 1 | 8 | 5969 | -0.0498 | 0.2976 | 0.0 | 0.732 | 0.999 | 0.388 |
| v3 | 2 | 0 | 4468 | -0.1066 | 0.3535 | 0.0 | 0.757 | 1.0 | 0.328 |
| v3 | 2 | 8 | 5921 | -0.0449 | 0.2928 | 0.0 | 0.708 | 1.0 | 0.415 |
| v3 | 3 | 0 | 5434 | -0.0884 | 0.3274 | 0.0 | 0.712 | 1.0 | 0.356 |
| v3 | 3 | 8 | 5561 | -0.0593 | 0.3027 | -0.0 | 0.735 | 1.0 | 0.416 |

## Iter 0 (cùng dữ liệu): theo bước

| run | step | clip_frac | clip_frac_roll10 | ratio_mean | std_logr | dl_absmean | grad_norm |
|---|---|---|---|---|---|---|---|
| v2 | 5 | 0.023 | 0.012 | 1.0022 | 0.0612 | 9.53e-04 | 9.78 |
| v2 | 10 | 0.055 | 0.045 | 0.96 | 0.1083 | 2.24e-03 | 8.44 |
| v2 | 15 | 0.203 | 0.126 | 1.4116 | 0.4539 | 7.52e-03 | 1256.68 |
| v2 | 20 | 0.211 | 0.19 | 0.915 | 0.2838 | 7.06e-03 | 28.83 |
| v2 | 23 | 0.195 | 0.211 | 0.953 | 0.3224 | 7.60e-03 | 13.39 |
| v3 | 5 | 0.039 | 0.011 | 1.0015 | 0.102 | 1.43e-03 | 17.76 |
| v3 | 10 | 0.023 | 0.019 | 0.9895 | 0.0635 | 1.57e-03 | 13.94 |
| v3 | 15 | 0.031 | 0.041 | 0.9906 | 0.0765 | 1.93e-03 | 16.04 |
| v3 | 20 | 0.086 | 0.066 | 0.9594 | 0.1191 | 3.04e-03 | 12.66 |
| v3 | 23 | 0.094 | 0.081 | 0.9985 | 0.1767 | 3.59e-03 | 9.95 |
| v3 | 30 | 0.102 | 0.105 | 0.9649 | 0.1563 | 3.35e-03 | 14.28 |
| v3 | 50 | 0.172 | 0.148 | 0.9257 | 0.1668 | 4.71e-03 | 12.63 |
| v3 | 80 | 0.141 | 0.149 | 0.9368 | 0.2339 | 4.94e-03 | 14.97 |
| v3 | 110 | 0.195 | 0.19 | 0.9611 | 0.1995 | 4.83e-03 | 15.68 |
| v3 | 131 | 0.273 | 0.204 | 0.8985 | 0.3499 | 8.04e-03 | 22.88 |

## clip_frac theo task và độ lệch ratio, theo vòng

v2: mỗi bước một task (cột `task` của log); v3: cột `clip_frac_t0/t8` (nửa batch mỗi task).

| run | iter | clip_t0_mean | ratio_mean_t0 | clip_t8_mean | ratio_mean_t8 | ratio_mean_all | ratio_mean_last10 | grad_norm_mean | std_logr_last10 |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 0 | 0.165 | 1.0185 | 0.082 | 0.973 | 0.9987 | 0.997 | 89.38 | 0.3139 |
| v2 | 1 | 0.131 | 0.9547 | 0.109 | 0.9512 | 0.953 | 0.9105 | 14.57 | 0.2087 |
| v2 | 2 | 0.162 | 0.9234 | 0.115 | 0.9453 | 0.9357 | 0.9016 | 15.03 | 0.2586 |
| v2 | 3 | 0.213 | 0.9097 | 0.06 | 0.9729 | 0.9343 | 0.8987 | 17.31 | 0.1818 |
| v3 | 0 | 0.149 | — (không log theo task) | 0.135 | — (không log theo task) | 0.9919 | 1.0787 | 29.95 | 0.2859 |
| v3 | 1 | 0.168 | — (không log theo task) | 0.129 | — (không log theo task) | 0.9551 | 0.9439 | 17.43 | 0.2606 |
| v3 | 2 | 0.152 | — (không log theo task) | 0.119 | — (không log theo task) | 0.9522 | 0.9395 | 17.07 | 0.2305 |
| v3 | 3 | 0.184 | — (không log theo task) | 0.113 | — (không log theo task) | 0.9535 | 0.9335 | 18.53 | 0.211 |

## Iter 0, cùng khoảng bước: clip_frac theo task

v2: trung bình trên các bước của task đó; v3: cột `clip_frac_t0/t8` (nửa batch của task).

| bước | v2 bước task 0 | v2 bước task 8 | v3 nửa task 0 | v3 nửa task 8 | v3 cả batch |
|---|---|---|---|---|---|
| 1–10 | 0.061 (n = 4) | 0.035 (n = 6) | 0.030 | 0.008 | 0.019 |
| 11–23 | 0.211 (n = 9) | 0.152 (n = 4) | 0.095 | 0.059 | 0.077 |
| 24–50 | — (đã dừng) | — (đã dừng) | 0.155 | 0.097 | 0.126 |
| 51–100 | — (đã dừng) | — (đã dừng) | 0.159 | 0.161 | 0.160 |
| 101–131 | — (đã dừng) | — (đã dừng) | 0.190 | 0.201 | 0.195 |

## v2: cửa sổ 10 bước cuối (quyết định early stop), theo task

| vòng | số bước task 0 / 10 | clip bước task 0 | clip bước task 8 |
|---|---|---|---|
| 0 | 7 | 0.225 | 0.177 |
| 1 | 5 | 0.234 | 0.191 |
| 2 | 4 | 0.277 | 0.169 |
| 3 | 5 | 0.38 | 0.072 |

Bước có grad_norm > 100 (trước khi clip về 10): v2 iter0: bước 14 (349), bước 15 (1257); v3 iter0: bước 84 (1960); v3 iter3: bước 24 (100), bước 42 (123), bước 46 (112).

## Kết luận (chỉ từ log, không có ablation)

- Tổng: v2 84 bước, v3 402 bước; mẫu PPO / BC rollout / demo ở bảng Tổng. Cả 8 vòng dừng do early stop (clip_frac 10 bước cuối > 0,2), không vòng nào chạm trần 3 epoch.
- Iter 0 (cùng dữ liệu, cùng θ0, cùng c): ở bước 11–23, clip_frac của TỪNG task trong v3 chỉ khoảng một nửa v2. Vậy v3 không chỉ "trung bình hoá" hai task: ratio của mỗi task trôi chậm hơn mỗi bước. Riêng việc trộn task (batch cân bằng) không giải thích được điều này.
- Thành phần cửa sổ của v2 có góp phần: bước task 0 clip nhiều hơn bước task 8 (bảng trên), và cửa sổ cuối của iter 0 có 7/10 bước task 0.
- Hai cơ chế khớp với log nhưng KHÔNG tách được nếu không train lại: (a) v2 không trừ trung bình advantage (Â trung bình −0,31 ở task 0, 63 % mẫu Â < 0), nên mỗi bước đẩy l_new của đa số mẫu theo cùng một hướng; v3 trừ trung bình theo task (Â trung bình 0). (b) v3 thêm λ_demo × L_CFM(demo) (32 mẫu, λ 5): AdamW + clip grad norm 10 (v2: 80 % số bước có grad_norm > 10, trung vị 15.2; v3: 91 % số bước có grad_norm > 10, trung vị 13.7) nên một phần bước cập nhật dành cho demo thay vì PPO.
- Gai grad_norm xuất hiện ở cả hai (v2 iter0 bước 14–15 ngay trước early stop; v3 iter0 bước 84), nên gai không phải khác biệt hệ thống. Xác nhận nguyên nhân cần ablation (v3 bỏ demo; v3 không trừ trung bình), chưa chạy (yêu cầu: không train thêm).
