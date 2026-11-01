# FFI26 – Analyse: điểm mạnh và giới hạn của SmolVLA (`lerobot/smolvla_libero`)

> `failure_analysis.md` đã xoá khỏi repo ngày 31/10/2026; bản cuối: `git show e852bb7:failure_analysis.md`.

Cấu hình chính: libero_10, NAS = 10, relative (`docs/reproduce.md` §4). Nguồn dữ liệu:

| Viết tắt | Dữ liệu | Cách gán nhãn |
|---|---|---|
| **D10** | D_nas10: 10 task × 50 init × 3 seed (`docs/failure_overview.md` §1) | tỉ lệ thành công; nhãn F0–F6 bằng mắt cho 28 thất bại có video (seed 1000, ep 0–9) |
| **D1 / D50 / B** | D_nas1, D_nas50, B_main (NAS = 1, 50, 1) (`docs/failure_overview.md`, `e852bb7:failure_analysis.md`) | nhãn F0–F6 bằng mắt (44 / 48 / 142 thất bại) |
| **LOG** | checkpoint chính thức, NAS = 10, có log predicate BDDL và vị trí vật mỗi bước: task 0, 8 × 50 ep; 8 task khác × 20 ep (`ffi26/ffi26/results/round2/episodes.csv`) | nhãn tự động A–E, O (`improve/src/failure_labels.py`); sự kiện slip/collision (`improve/src/reward_r2.py`) |
| **AUTO** | arm `official` của eval FPO v2 (NAS = 10, expert fp32): task 0, 8 × 50 init × seed 3000/4000/5000 (150 ep/task); 8 task khác × 20 init, seed 1000. Tổng 460 ep, 212 thất bại, 248 thành công (`improve/outputs/failure_labels/labels.csv`) | nhãn tự động v2 A, C, F, G, D, E, B (`improve/src/failure_labels.py`, `label_episode_v2`); sự kiện slip (`improve/src/reward_r2.py`, `detect_events`). Số đếm: `docs/failures/tools/l2_counts.py` |

Nhãn bằng mắt do Claude gán, một người, phần lớn chỉ dựa trên khung điển hình (`docs/failure_overview.md` §5).
Nhãn tự động dùng ngưỡng tự chọn. Detector collision đã được kiểm bằng ảnh (`results/round2/detector_check/`); detector slip chưa.

## 1. Điểm mạnh

| # | Điểm mạnh | Bằng chứng |
|---|---|---|
| S1 | **Task một bước hoặc có vật khớp nối làm rất tốt** | D10 (/150): task 3 "bát vào ngăn kéo rồi đóng" 146, task 2 "bật bếp + đặt ấm" 138, task 1 136, task 5 135. Spatial: 9/10 task đạt 237/270 = 87,8% khi bỏ task 5 (`docs/failures/spatial_results.md`) |
| S2 | **Hiểu đúng câu lệnh: hiếm khi nhầm vật hay nhầm đích** | Nhầm vật/đích (F5) chỉ 1–2 thất bại mỗi run: B 1/142, D10 1/28. LOG: nhãn C (nhầm vật) là nhãn chính ở 4/90 thất bại |
| S3 | **Action chunk + quan sát lại mỗi 10 bước: vừa mượt vừa sửa được sai** | NAS=10 đạt 73,6%, hơn NAS=1 (54,6%) ở cả 10 task và hơn NAS=50 (47,8%) (`docs/failure_overview.md` §1). Cùng cảnh: NAS=10 cứu 10/16 lần gắp hỏng (F1) của NAS=50 và 7/12 lần ra khỏi vùng làm việc (F6) của NAS=1. LOG: 14/170 episode thành công có ≥ 1 lần trượt vật (slip) rồi gắp lại thành công, ví dụ task 0 init 31 (2 lần) |
| S4 | **Nhỏ và nhanh** | 450M tham số, 96,5 ms / chunk trên RTX 5070 Ti (`reproduce.md` §2.3). Một chunk 50 action dùng được cho nhiều bước: 10 job D_nas1 mất 6.309 s, D_nas10 1.597 s, D_nas50 1.336 s (`differences.md` §3b, `docs/failures/D_nas10.md`). Train/eval/RL đều chạy được trên 1 GPU 16 GB |

## 2. Giới hạn

| # | Giới hạn | Bằng chứng |
|---|---|---|
| L1 | **Task dài nhiều bước: xong bước con thứ nhất rồi không làm tiếp** | Hai task yếu nhất là task hai vật: task 0 60/150, task 8 71/150 (D10). LOG task 8: 19/27 thất bại xong đúng một nửa goal (một ấm trên bếp). LOG task 0: 9/29 thất bại xong một lon. B: 47 thất bại "xong bước 1, hỏng bước 2"; 91/142 thất bại có chuyển động 100 bước cuối < 0,3, tức robot dừng hẳn từ trước khi hết giờ (`e852bb7:failure_analysis.md` §2.3, §4) |
| L2 | **Thao tác tinh kém: gắp hỏng, trượt khỏi kẹp, xô đổ vật bên cạnh, làm rơi, kẹp sai tư thế, đặt lệch đích** | AUTO: 142/212 thất bại có ít nhất một lỗi thao tác (nhãn A/C/F/G/D), so với 5/248 thành công. Lỗi lớn nhất là gắp hỏng (A): 111/212 thất bại, 0/248 thành công. Chi tiết từng loại lỗi ở §2.1 |
| L3 | **Rất nhạy với cách thực thi, mà giá trị mặc định lại tệ nhất** | Cùng checkpoint, NAS 1 / 10 / 50: 54,6% / 73,6% / 47,8%; NAS = 50 là giá trị trong config. absolute thay relative: 2/100, và 2 lần "thành công" là do va chạm (`e852bb7:failure_analysis.md` §6.4). Hai kiểu lỗi ngược nhau: NAS=1 hay dừng hoặc đi lạc (F3 + F6 = 26/44 ở D1), NAS=50 gắp trượt rồi vẫn làm tiếp (F1 16/48) |
| L4 | **Kết quả do cảnh quyết định, ít đa dạng khi lấy mẫu** | 500 cảnh × 3 seed: 289 cảnh thành công cả 3, 69 thất bại cả 3, chỉ 142 cho kết quả lẫn lộn (D10). B: 111/142 thất bại thuộc 37 cảnh thất bại cả 3 seed (`e852bb7:failure_analysis.md` §2.2). Đổi nhiễu ε chỉ đổi chunk 0,044 (trung bình, đơn vị chuẩn hoá; độ lệch chuẩn của chunk 0,77) (`improve/outputs/fpo/smoke/verify.json`). Hệ quả cho RL: thử lại ít khi ra kết quả khác; khi thu 8 rollout / cảnh, 8/32 cảnh thất bại cả 8 lần (`improve/outputs/r2/round1/groups.csv`) |
| L5 | **Khó fine-tune thêm bằng recipe mặc định** | Action expert được nạp ở bf16. Khi fine-tune bằng `lerobot-train` với lr 1e-5, bước cập nhật nhỏ hơn độ phân giải bf16: sau 6.000 step chỉ 5,5% phần tử của expert (96,6M) từng thay đổi, phần fp32 thay đổi 97–100% (`results/round2/fpo_plan.md` §9). Không biến thể fine-tune nào ở round 1 tốt hơn checkpoint gốc (`results/round2/summary.md`). Cập nhật 01/11: FPO v2 trở đi giữ trọng số chính ở fp32 và nạp expert fp32 khi eval, nên vấn đề làm tròn không còn (đã kiểm: 155/155 tensor đã train nạp đúng từng bit, `results/round2/fpo_v3/REVIEW_CHECKLIST.md` mục 8) |

### 2.1 Chi tiết L2: các loại lỗi thao tác

Hai nguồn bổ sung cho nhau. Nhãn tự động (AUTO) tính từ log mô phỏng, đếm được trên nhiều episode, nhưng không đo
hướng của vật nên không nhận ra "kẹp sai tư thế". Nhãn bằng mắt (D10, 28 thất bại có video) thấy được tư thế nhưng mẫu
nhỏ. Một episode có thể mang nhiều nhãn, nên các dòng không cộng lại thành tổng. Cột "thất bại / thành công" đếm số
episode có nhãn đó. Nếu nhãn hầu như chỉ xuất hiện ở thất bại thì nó gắn với thất bại; con số này không chứng minh
nhân quả.

| # | Loại lỗi | Cách nhận biết | AUTO: thất bại / thành công | Xảy ra ở đâu | Nhãn bằng mắt (D10) |
|---|---|---|---|---|---|
| L2a | **Gắp hỏng**: tay tới vật nhưng không nhấc được | Nhãn A: vật goal chưa từng đạt predicate, chưa từng được nhấc ≥ 3 cm, nhưng có ≥ 2 lần đóng kẹp khi đầu tay cách tâm vật < 10 cm, hoặc giữ lệnh đóng ≥ 20 bước gần vật | **111/212** / 0/248 | task 0: 73/89 thất bại (lon alphabet soup 57, tomato sauce 17); 63/89 thất bại task 0 không nhấc được vật goal nào. Task 8: 25/90 (moka_pot_1 17, moka_pot_2 8). 8 task khác: 13/33 (pudding 5, cream cheese 3) | F1 1/28: task 7 kẹp lon tại chỗ nhưng không nhấc |
| L2b | **Trượt khỏi kẹp** | Sự kiện slip: vật đang được giữ với kẹp đóng, mất tiếp xúc ≥ 5 bước trong khi kẹp vẫn đóng, độ mở hai ngón < 1,5 cm, vật chưa đạt goal | 90/212 / 22/248 | task 0: 58/89 thất bại so với 7/61 thành công; task 8: 17/90 so với 6/60; 8 task khác: 15/33 so với 9/127. 22 episode thành công có slip đều gắp lại được (xem S3) | không có nhãn riêng (gộp vào F1) |
| L2c | **Chạm, xô lệch hoặc cầm nhầm vật không thuộc goal** | Nhãn C: gripper cầm một vật không có trong goal, hoặc chạm vào nó rồi vật dịch > 5 cm | 35/212 / 1/248 | gần như chỉ ở task 0: 33/89 thất bại, chủ yếu hộp nước cam (30). Trong 33 episode đó, 11 có **cầm** vật ngoài goal (nước cam là chủ yếu), 22 chỉ chạm rồi làm dịch vật. 8 task khác: 2/33 | F5 1/28 (task 0: cho hộp nước cam vào giỏ); F2 task 0: hộp nước cam bị làm rơi cạnh giỏ |
| L2d | **Làm rơi vật đang mang** | Sự kiện drop: vật đang được giữ và đã nhấc, tách khỏi kẹp rồi rơi > 5 cm trong 30 bước, chưa đạt goal | 13/212 / 3/248 | task 0: 6/89 (alphabet soup 4); task 8: 3/90 (moka_pot_1 3); 8 task khác: 4/33 | gộp trong F2 |
| L2e | **Làm đổ, kẹp sai tư thế** (vật bị kẹp nằm ngang/nghiêng) | Chỉ nhãn bằng mắt F2; nhãn tự động không đo hướng vật | không đo được | task 8: 15/90 thất bại nhấc được cả hai ấm nhưng ấm thứ hai không được tính là trên bếp (60/60 thành công nhấc cả hai). Có thể đây là các ca kẹp ấm nằm ngang, nhưng **chưa kiểm bằng video** | **F2 9/28**, nhóm lớn nhất: task 8 ấm 2 bị kẹp nằm ngang/nghiêng (3), task 0 lon bị kẹp nghiêng hoặc đổ (2) và hộp nước cam rơi (1), task 2 ấm moka đổ cạnh bếp, task 4 cốc nghiêng ở mép đĩa, task 7 lon đổ nằm ngang |
| L2f | **Đặt gần đích nhưng env không tính** | Nhãn G: vật được thả, đứng yên (10 bước, < 1 mm/bước) cách tâm đích < 10 cm, nhưng predicate goal vẫn sai | 13/212 / 0/248 | task 4: 4/7 thất bại, task 6: 4/11 (cốc sứ 6 lần); task 8: 4/90 (moka_pot_2). Khoảng cách tới tâm đích 1,2–8,6 cm, trung vị 4,3 cm. Predicate `On` cần tâm hai vật lệch < 3 cm (`docs/failure_overview.md` §4.1) | **F4 5/28**: task 4 hai cốc có vẻ ở hai đĩa (2), task 5 sách nằm nghiêng ở ngăn, task 9 cốc ở mép cửa lò / cửa lò chưa đóng (2) |
| L2g | **Làm mất tiến độ đã đạt** | Nhãn D: một predicate goal đã đúng (≥ 5 bước) rồi sai lại (≥ 5 bước) | 8/212 / 1/248 | task 8: 5/90 (ấm moka_pot_2 đã lên bếp rồi bị kéo ra); task 6: 3/11 | không có nhãn riêng |

Nhận xét:
1. **Gắp hỏng (L2a) là lỗi thao tác chính**: có ở 111/212 thất bại và không có ở episode thành công nào. Task 0 bị nặng
   nhất. Ở task này lỗi đến sớm: 63/89 thất bại không nhấc được lon nào, vì vậy bước con đầu tiên không hoàn thành.
   Kết quả cũ trên LOG (bf16, 50 ep) cho cùng hình ảnh: task 0 có 25/29 thất bại là A, 20/29 có slip so với 5/21
   thành công.
2. **Gắp hỏng và trượt khỏi kẹp đi cùng nhau ở task 0**: 58/89 thất bại có slip, so với 7/61 thành công. Robot có
   kẹp trúng nhưng không giữ chắc.
3. **Ở task 0, lỗi thao tác hay ảnh hưởng vật bên cạnh** (L2c). Nhãn bằng mắt ở B_main ghi rằng robot với vào cụm hộp
   bên phải, nơi lon alphabet soup nằm sau hộp sữa (`docs/failures/notes/B_main.md` mục 2). Điều này khớp với AUTO:
   alphabet soup là vật bị gắp hỏng nhiều nhất (57/73), và hộp nước cam bị chạm hoặc cầm trong 30/89 thất bại.
   11 episode có cầm vật ngoài goal. Vì vậy S2 ("hiếm khi nhầm vật") đúng với nhãn bằng mắt F5, nhưng cần đọc cùng
   con số này. AUTO không phân biệt được cầm nhầm do hiểu sai câu lệnh với cầm nhầm do va vào vật khác khi với tới
   vật goal.
4. **Task 8 khác task 0**: chỉ 33/90 thất bại có nhãn A/C/F/G/D. 57 thất bại còn lại chỉ có nhãn B (xong đúng một ấm)
   và/hoặc E (đứng yên). Như vậy phần lớn thất bại task 8 thuộc L1 (không chuyển sang bước con thứ hai), hoặc là kẹp
   sai tư thế (L2e) mà AUTO không thấy. Muốn tách hai khả năng này phải xem video.
5. **Đặt chưa đủ chính xác (L2f)** ít gặp hơn (13/212) nhưng không xảy ra ở episode thành công nào. Lỗi này tập trung ở
   các task đặt cốc lên đĩa (task 4, 6), nơi ngưỡng 3 cm của `On` chặt.
6. Lỗi theo task giữ ổn định giữa các cấu hình (`docs/failure_overview.md` §3): task 8 kẹp ấm nằm ngang và task 0 làm đổ
   đồ ở cụm hộp bên phải đều xuất hiện ở nhiều run. Ở Spatial task 5 (bát chồng trên ramekin), 30/30 lần thất bại, trong
   đó 25 lần robot gạt bát rơi rồi dừng (`e852bb7:failure_analysis.md` §5.2).

Giới hạn của bảng này: ngưỡng của nhãn tự động do người viết tự chọn. Theo đầu file này, detector collision đã được
kiểm bằng ảnh, còn slip thì chưa. Với drop, A, G, D, repo chưa có ghi nhận kiểm tra bằng ảnh. `manual_check.csv` (20 episode) chưa được gán nhãn tay. Arm `official` ở đây nạp expert fp32,
còn D10 dùng `lerobot-eval` (bf16), nên tỉ lệ thành công khác nhau một chút: task 0 61/150 ở đây, 60/150 ở D10.

## 3. Quỹ đạo thành công và thất bại

| | Thành công | Thất bại |
|---|---|---|
| Độ dài | B: 151 / 245 / 456 bước (min / trung vị / max). LOG task 0: 249 / 266 / 432; task 8: 350 / 391 / 478 | luôn đúng giới hạn 520 bước (hết giờ), không có thất bại sớm (`e852bb7:failure_analysis.md` §2.1) |
| Hành vi | các bước con nối tiếp không dừng. Ví dụ D1 task 1 ep 1: nhấc hộp cream cheese ở bước ~50–74, vào giỏ ~150–170. D50 task 8 ep 0: ấm 1 lên bếp ~159, ấm 2 ~372 | dừng hoặc lơ lửng sau bước con đầu. Ví dụ B task 0 ep 1 seed 1000: lon xanh vào giỏ ~222, sau đó cầm lon đỏ lơ lửng từ ~297 tới 519 (`docs/failures/frames/B_main/t0_e1_s1000.jpg`) |
| Phục hồi | gắp lại sau khi trượt (LOG: 14 episode) | có thử lại nhưng không xong: Spatial task 9 ep 5 seed 1001 làm rơi bát cạnh đĩa rồi gắp lại, bát vẫn nghiêng ở mép đĩa (`e852bb7:failure_analysis.md` §5.1) |
| Lỗi thao tác | – | kẹp ấm thứ hai nằm ngang (`frames/B_main/t8_e3_s1000.jpg`); ôm cốc trên đĩa tới hết giờ (`frames/B_main/t6_e3_s1000.jpg`); gắp trượt rồi mang tay không tới giỏ (D50 task 1 ep 1, `e852bb7:failure_analysis.md` §6.5) |

Kết luận: điểm yếu chính nằm ở chỗ **chuyển sang bước con tiếp theo** và **độ chính xác khi gắp/đặt**, không nằm ở
hiểu câu lệnh. Đổi NAS sửa được phần lỗi do cách thực thi (S3), còn L1 và L2 thì không. Vì vậy phần Improve nhắm vào
task 0 và task 8.
