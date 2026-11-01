# Kiểm tra advantage thô của FPO (kiểm tra 1–3)

Script `improve/scripts/fpo_adv_check.py` (CPU, chỉ đọc). Nguồn: `outputs/fpo_v{2,3}/iter{0..3}/{decisions/g*.npz, rollouts.csv, segments.csv, groups.csv}`. v3 iter 0 = v2 iter 0 (dùng lại). Advantage tính trong `src/reward_r2.py:173–217`, gán cho decision theo bước đầu của nó (`scripts/fpo_collect.py:89–92`). "Lọc" = như khi train (`fpo_train.load_decisions`: rejected = 0 và adv ≠ 0). TC = rollout thành công, TB = rollout thất bại.

## Kiểm tra 3: khớp decision ↔ rollout ↔ segment

| run | iter | task | rollouts | n_dec == ceil(len/10) | bước = 0,10,20,… | n_exec đúng | đúng 1 segment chứa bước | max |adv − adv segment| |
|---|---|---|---|---|---|---|---|---|
| v2 | 0 | 0 | 128 | 128/128 | 128/128 | 5470/5470 | 5470/5470 | 5.5e-08 |
| v2 | 0 | 8 | 128 | 128/128 | 128/128 | 6191/6191 | 6191/6191 | 2.8e-08 |
| v2 | 1 | 0 | 128 | 128/128 | 128/128 | 5696/5696 | 5696/5696 | 2.9e-08 |
| v2 | 1 | 8 | 128 | 128/128 | 128/128 | 6050/6050 | 6050/6050 | 2.5e-08 |
| v2 | 2 | 0 | 128 | 128/128 | 128/128 | 5589/5589 | 5589/5589 | 2.9e-08 |
| v2 | 2 | 8 | 128 | 128/128 | 128/128 | 6068/6068 | 6068/6068 | 2.9e-08 |
| v2 | 3 | 0 | 128 | 128/128 | 128/128 | 5494/5494 | 5494/5494 | 2.7e-08 |
| v2 | 3 | 8 | 128 | 128/128 | 128/128 | 6063/6063 | 6063/6063 | 2.8e-08 |
| v3 | 0 | 0 | 128 | 128/128 | 128/128 | 5470/5470 | 5470/5470 | 5.5e-08 |
| v3 | 0 | 8 | 128 | 128/128 | 128/128 | 6191/6191 | 6191/6191 | 2.8e-08 |
| v3 | 1 | 0 | 128 | 128/128 | 128/128 | 5454/5454 | 5454/5454 | 4.8e-08 |
| v3 | 1 | 8 | 128 | 128/128 | 128/128 | 5978/5978 | 5978/5978 | 2.9e-08 |
| v3 | 2 | 0 | 128 | 128/128 | 128/128 | 5335/5335 | 5335/5335 | 5.8e-08 |
| v3 | 2 | 8 | 128 | 128/128 | 128/128 | 5969/5969 | 5969/5969 | 2.9e-08 |
| v3 | 3 | 0 | 128 | 128/128 | 128/128 | 5434/5434 | 5434/5434 | 3.6e-08 |
| v3 | 3 | 8 | 128 | 128/128 | 128/128 | 6004/6004 | 6004/6004 | 2.9e-08 |

## Kiểm tra 1: trung bình advantage trong từng (nhóm, stage), mỗi rollout một phiếu, nhóm không bị loại

Nếu V[k] đúng thì mean ≈ 0. Không bằng 0 chính xác vì γ^(độ dài segment) khác nhau giữa các rollout và rollout không tới stage k+1 thì không có bootstrap.

| run | iter | task | (nhóm, stage) n ≥ 2 | trung vị |mean| | max |mean| | trung vị mean |adv| | tỉ lệ |mean| / mean|adv| (trung vị) | (nhóm, stage) n = 1 |
|---|---|---|---|---|---|---|---|---|
| v2 | 0 | 0 | 28 | 0.0000 | 0.0190 | 0.1207 | 0.000 | 0 |
| v2 | 0 | 8 | 32 | 0.0000 | 0.0113 | 0.1776 | 0.000 | 0 |
| v2 | 1 | 0 | 25 | 0.0000 | 0.0133 | 0.1929 | 0.000 | 2 |
| v2 | 1 | 8 | 32 | 0.0000 | 0.0221 | 0.1642 | 0.000 | 2 |
| v2 | 2 | 0 | 24 | 0.0000 | 0.0005 | 0.0429 | 0.000 | 3 |
| v2 | 2 | 8 | 30 | 0.0000 | 0.0368 | 0.1705 | 0.000 | 1 |
| v2 | 3 | 0 | 26 | 0.0000 | 0.0053 | 0.0671 | 0.000 | 0 |
| v2 | 3 | 8 | 28 | 0.0000 | 0.0084 | 0.0435 | 0.000 | 3 |
| v3 | 0 | 0 | 28 | 0.0000 | 0.0190 | 0.1207 | 0.000 | 0 |
| v3 | 0 | 8 | 32 | 0.0000 | 0.0113 | 0.1776 | 0.000 | 0 |
| v3 | 1 | 0 | 28 | 0.0000 | 0.0268 | 0.0595 | 0.000 | 0 |
| v3 | 1 | 8 | 32 | 0.0000 | 0.0227 | 0.1838 | 0.001 | 1 |
| v3 | 2 | 0 | 26 | 0.0000 | 0.0376 | 0.0927 | 0.000 | 1 |
| v3 | 2 | 8 | 31 | 0.0000 | 0.0207 | 0.1822 | 0.000 | 4 |
| v3 | 3 | 0 | 28 | 0.0000 | 0.0063 | 0.0521 | 0.000 | 0 |
| v3 | 3 | 8 | 29 | 0.0000 | 0.0241 | 0.1583 | 0.000 | 2 |

## Kiểm tra 2: trung bình theo decision vs theo segment / rollout (decision đã lọc)

| run | iter | task | mean theo decision | mean theo segment | mean theo rollout | decision / rollout TC | decision / rollout TB | corr(số decision của segment, adv) | % decision của rollout TB |
|---|---|---|---|---|---|---|---|---|---|
| v2 | 0 | 0 | -0.0948 | -0.0008 | -0.0448 | 30.4 | 52.0 | -0.474 | 66.9% |
| v2 | 0 | 8 | -0.0486 | -0.0006 | -0.0292 | 40.9 | 52.0 | -0.406 | 72.2% |
| v2 | 1 | 0 | -0.096 | 0.0006 | -0.0454 | 31.1 | 52.0 | -0.507 | 70.9% |
| v2 | 1 | 8 | -0.0433 | -0.0012 | -0.0254 | 41.9 | 52.0 | -0.416 | 58.4% |
| v2 | 2 | 0 | -0.0565 | -0.0 | -0.0258 | 29.3 | 52.0 | -0.393 | 73.6% |
| v2 | 2 | 8 | -0.0583 | -0.002 | -0.0362 | 41.1 | 52.0 | -0.476 | 60.7% |
| v2 | 3 | 0 | -0.0957 | 0.0002 | -0.0488 | 28.8 | 52.0 | -0.505 | 69.2% |
| v2 | 3 | 8 | -0.0275 | -0.0002 | -0.017 | 41.9 | 52.0 | -0.281 | 55.8% |
| v3 | 0 | 0 | -0.0948 | -0.0008 | -0.0448 | 30.4 | 52.0 | -0.474 | 66.9% |
| v3 | 0 | 8 | -0.0486 | -0.0006 | -0.0292 | 40.9 | 52.0 | -0.406 | 72.2% |
| v3 | 1 | 0 | -0.0906 | -0.0011 | -0.0411 | 31.3 | 52.0 | -0.456 | 64.0% |
| v3 | 1 | 8 | -0.0498 | -0.0013 | -0.0281 | 42.2 | 52.0 | -0.447 | 51.2% |
| v3 | 2 | 0 | -0.1066 | -0.002 | -0.0461 | 30.3 | 52.0 | -0.513 | 58.6% |
| v3 | 2 | 8 | -0.0449 | -0.0023 | -0.0206 | 41.4 | 52.0 | -0.381 | 54.9% |
| v3 | 3 | 0 | -0.0884 | -0.0001 | -0.0432 | 29.4 | 52.0 | -0.463 | 70.8% |
| v3 | 3 | 8 | -0.0593 | -0.0021 | -0.0309 | 40.9 | 52.0 | -0.457 | 57.0% |

## Kiểm tra 3 (thêm): dấu advantage của decision thuộc rollout thành công / thất bại

Chuẩn hoá của v3: Â = clip((A − mean_t) / std_t, ±5), mean_t và std_t tính **theo decision** trên decision đã lọc của task t (`fpo_train_v3.py:101–102`). Cột "mean theo rollout" chỉ để so sánh, không dùng khi train. Với v2, cột "sau chuẩn hoá v3" là giả định (v2 thực tế dùng A / std, không đổi dấu).

| run | iter | task | decision (lọc) | của rollout thành công | mean_t (theo decision) | mean theo rollout | std_t | TC: A < 0 (thô) | TC: Â < 0 (sau chuẩn hoá v3) | TB: A > 0 (thô) | TB: Â > 0 (sau chuẩn hoá v3) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 | 0 | 0 | 5054 | 1674 | -0.0948 | -0.0448 | 0.3547 | 29.5% | 1.1% | 19.9% | 52.6% |
| v2 | 0 | 8 | 6191 | 1719 | -0.0486 | -0.0292 | 0.2664 | 16.6% | 1.6% | 21.5% | 37.0% |
| v2 | 1 | 0 | 4843 | 1411 | -0.096 | -0.0454 | 0.3163 | 28.3% | 6.9% | 20.8% | 32.3% |
| v2 | 1 | 8 | 6037 | 2514 | -0.0433 | -0.0254 | 0.2542 | 16.9% | 6.5% | 18.3% | 26.0% |
| v2 | 2 | 0 | 5107 | 1347 | -0.0565 | -0.0258 | 0.2347 | 45.4% | 1.3% | 22.7% | 56.9% |
| v2 | 2 | 8 | 5650 | 2220 | -0.0583 | -0.0362 | 0.2744 | 22.2% | 1.2% | 18.2% | 22.4% |
| v2 | 3 | 0 | 4662 | 1438 | -0.0957 | -0.0488 | 0.3316 | 33.5% | 1.9% | 34.1% | 55.9% |
| v2 | 3 | 8 | 5594 | 2475 | -0.0275 | -0.017 | 0.2152 | 31.4% | 9.9% | 33.4% | 38.2% |
| v3 | 0 | 0 | 5054 | 1674 | -0.0948 | -0.0448 | 0.3547 | 29.5% | 1.1% | 19.9% | 52.6% |
| v3 | 0 | 8 | 6191 | 1719 | -0.0486 | -0.0292 | 0.2664 | 16.6% | 1.6% | 21.5% | 37.0% |
| v3 | 1 | 0 | 5038 | 1814 | -0.0906 | -0.0411 | 0.3405 | 26.8% | 1.6% | 23.4% | 50.1% |
| v3 | 1 | 8 | 5969 | 2910 | -0.0498 | -0.0281 | 0.2976 | 19.5% | 3.6% | 15.7% | 27.7% |
| v3 | 2 | 0 | 4468 | 1851 | -0.1066 | -0.0461 | 0.3535 | 26.3% | 0.9% | 17.1% | 44.7% |
| v3 | 2 | 8 | 5921 | 2670 | -0.0449 | -0.0206 | 0.2928 | 21.0% | 2.1% | 18.1% | 26.1% |
| v3 | 3 | 0 | 5434 | 1586 | -0.0884 | -0.0432 | 0.3274 | 24.9% | 0.0% | 27.9% | 49.8% |
| v3 | 3 | 8 | 5561 | 2394 | -0.0593 | -0.0309 | 0.3027 | 13.6% | 2.2% | 15.3% | 28.5% |

## Nhận xét (chỉ từ số liệu trên)

- Kiểm tra 3: mọi decision khớp với đúng một rollout và đúng một segment; số decision = ⌈độ dài/10⌉, bước 0, 10, 20, …, n_exec đúng ở mọi (run, vòng, task); adv lưu trùng với `segments.csv` (sai lệch tối đa 6e-08, do làm tròn float32). Không thấy lỗi gán advantage.
- Kiểm tra 1: trong từng (nhóm, stage), trung bình advantage theo rollout có trung vị 0,0000 ở mọi (run, vòng, task); lớn nhất 0.0376. V[k] khớp với thiết kế.
- Kiểm tra 2: trung bình theo segment ≈ 0 (|mean| ≤ 0,0023); trung bình theo decision âm (−0,028 đến −0,107). Rollout thất bại luôn có 52,0 decision (mọi thất bại chạy hết 520 bước), rollout thành công 29–42; tương quan (số decision của segment, adv) từ −0,28 đến −0,51. Vậy trung bình âm là do **trọng số theo decision**, không phải do tính sai advantage.
- Dấu sau chuẩn hoá của v3: mean_t tính theo decision (âm), nên trừ mean_t đẩy mọi Â lên khoảng |mean_t|/std_t (0,15–0,30 std). Sau chuẩn hoá: decision của rollout thành công có Â < 0 chỉ còn 0,0–3,6 % (thô: 13,6–29,5 %); decision của rollout **thất bại** có Â > 0 tăng lên 26–53 % (thô: 15,3–27,9 %), cao nhất ở task 0 (44,7–52,6 %).
- Kiểm tra 4–7 không chạy: kiểm tra 1–3 không cho thấy lỗi trong cách tính advantage thô. Điểm đáng chú ý (trọng số theo decision, và dấu sau chuẩn hoá) là hệ quả của thiết kế, không phải lỗi tính toán.
