# FFI26: so sánh các cấu hình và nguyên nhân thất bại chung (libero_10)

Viết ngày 31/10/2026. Bài toán chính là **libero_10** (LIBERO-Long). LIBERO-Spatial chỉ dùng để kiểm tra checkpoint,
nên chỉ có kết quả cuối ở `docs/failures/spatial_results.md`.

File chi tiết theo từng run (bảng theo task, danh sách từng episode thất bại kèm ảnh):

| Run | File | Cấu hình |
|---|---|---|
| B_main | `docs/failures/B_main.md` | NAS=1, relative, 3 seed (1000–1002) x 10 episode/task |
| D_nas1 | `docs/failures/D_nas1.md` | NAS=1, relative, seed 1000 x 50 episode/task |
| D_nas10 | `docs/failures/D_nas10.md` | NAS=10, relative, seed 1000 x 50 episode/task (chạy 31/10/2026) |
| D_nas50 | `docs/failures/D_nas50.md` | NAS=50, relative, seed 1000 x 50 episode/task |

NAS = `--policy.n_action_steps`: số action trong một chunk được thực thi trước khi policy quan sát lại. Checkpoint
`lerobot/smolvla_libero` sinh chunk 50 action (`differences.md` mục 2). Mọi run dùng cùng checkpoint, giới hạn 520
bước/episode, `--eval.batch_size=5`.

## 1. Kết quả

Nguồn: `per_task[0].n_success` trong `eval_info.json` của từng job. Khoảng tin cậy: Wilson 95%, cùng công thức với
`lerobot/src/lerobot/utils/eval_stats.py:45`.

| Task | B_main (/30) | D_nas1 (/50) | **D_nas10 (/50)** | D_nas50 (/50) |
|---|---|---|---|---|
| 0 alphabet soup + tomato sauce → giỏ | 1 | 12 | **16** | 12 |
| 1 cream cheese + butter → giỏ | 16 | 29 | **44** | 21 |
| 2 bật bếp + đặt ấm moka | 28 | 45 | **48** | 37 |
| 3 bát vào ngăn kéo + đóng | 30 | 46 | **49** | 40 |
| 4 cốc trắng → đĩa trái, cốc vàng-trắng → đĩa phải | 7 | 20 | **36** | 17 |
| 5 sách → ngăn sau | 27 | 43 | **45** | 37 |
| 6 cốc → đĩa, pudding → bên phải đĩa | 9 | 18 | **27** | 11 |
| 7 alphabet soup + cream cheese → giỏ | 16 | 23 | **36** | 18 |
| 8 hai ấm moka → bếp | 1 | 7 | **26** | 19 |
| 9 cốc → lò vi sóng + đóng | 23 | 30 | **41** | 27 |
| **Tổng** | **158/300 = 52,7%** (47,0–58,2) | **273/500 = 54,6%** (50,2–58,9) | **368/500 = 73,6%** (69,6–77,3) | **239/500 = 47,8%** (43,5–52,2) |

Nhận xét:
1. **NAS=10 tốt nhất**, hơn NAS=1 ở cả 10 task, và khoảng tin cậy không chồng với NAS=1. So từng cảnh (cùng task, cùng
   init state, 500 cặp): NAS=1 thất bại còn NAS=10 thành công ở 118 cặp, ngược lại 23 cặp. Số của paper cho LIBERO-Long
   là 71% [CHƯA KIỂM CHỨNG, `differences.md` mục 3]; D_nas10 đạt 73,6%.
2. B_main và D_nas1 cùng cấu hình (NAS=1), cho 52,7% và 54,6%: hai lần đo nhất quán.
3. NAS=50 (giá trị trong config của checkpoint) thấp nhất.
4. **Đề xuất cần bạn quyết định:** dùng NAS=10 làm cấu hình chính cho phần Analyse/Improve thay cho NAS=1. Lưu ý: giá trị
   10 lấy từ tài liệu LeRobot cho Pi0.5 (`lerobot/docs/source/libero.mdx:205`), không phải khuyến nghị cho SmolVLA, và
   mới đo với 1 seed.

## 2. Bộ nguyên nhân thất bại chung

Mỗi episode thất bại có video được gán **đúng một nguyên nhân chính**, kèm giai đoạn và mức bằng chứng.
Mọi thất bại đều kéo dài đúng 520 bước (hết giờ; `failure_analysis.md` mục 2.1), nên nguyên nhân chỉ đọc được từ video.

| Mã | Tên | Định nghĩa (nhận ra trong video như thế nào) | Ví dụ |
|---|---|---|---|
| **F1** | Gắp hỏng | Tay tới vật nhưng vật không được nhấc lên: gắp trượt rồi vẫn mang tay không tới đích, kẹp tại chỗ mà không nhấc, hoặc lượn quanh vật bị che mà không gắp được | D_nas50 `t1 e1`: hộp cream cheese vẫn ở chỗ cũ, tay không tới giỏ |
| **F2** | Làm đổ / kẹp sai tư thế | Vật cần thao tác hoặc vật bên cạnh bị đổ, bị xô lệch, hoặc bị kẹp nằm ngang / nghiêng | B_main task 8: ấm thứ hai bị kẹp nằm ngang |
| **F3** | Đứng yên / lơ lửng | Robot dừng, lơ lửng hoặc giữ vật mà không làm thao tác tiếp theo, cho tới hết giờ | B_main task 6: cốc đã trên đĩa, tay vẫn ôm cốc |
| **F4** | Đặt không đúng / gần đích | Vật đã tới vùng đích nhưng env không tính: đặt lệch, sai ngăn, cửa/ngăn kéo chưa đóng hẳn, để vật ngay trước đích | D_nas50 task 4: hai cốc có vẻ ở hai đĩa; task 3: ngăn kéo chưa đóng hẳn |
| **F5** | Nhầm vật / nhầm đích | Thao tác với vật không thuộc mục tiêu hoặc đặt vào đích sai | Nhấc cả giỏ (task 0, 7); cho hộp nước cam vào giỏ (D_nas10 `t0 e0`) |
| **F6** | Ra khỏi vùng làm việc | Cánh tay đi ra mép hoặc ra khỏi khung hình và ở đó tới hết giờ | B_main task 1: cầm hộp đi về góc trên trái |
| **F0** | Không xác định | Video không đủ để quyết định | |

Quy tắc khi một episode có nhiều dấu hiệu: chọn sự kiện **xảy ra đầu tiên và không được sửa lại** (ví dụ D_nas50
`t7 e0`: gắp trượt rồi nhấc cả giỏ, gán F1).

Giai đoạn: **0** = chưa xong bước con thứ nhất; **1** = xong bước con thứ nhất, hỏng ở bước sau; **?** = camera ngoài không
cho biết. Mức bằng chứng: **timeline** (đã xem chuỗi khung của chính episode), **khung điển hình** (dải 4 khung: bước 10,
1/3 episode, khung lúc robot vừa dừng, khung cuối), **suy luận**.

Công cụ (trong repo, chạy lại được): `docs/failures/tools/extract_frames.py` (ảnh + chỉ số chuyển động),
`make_run_report.py` (bảng của từng run), `sheet.py` (trang xem). Nhãn: `docs/failures/labels/RUN.csv`.

## 3. Phân bố nguyên nhân theo cấu hình

Chỉ tính các episode thất bại có video. B_main có video cho mọi episode; D_* chỉ có cho episode 0–9 mỗi task.

| Nguyên nhân | B_main (NAS=1) | D_nas1 (NAS=1) | D_nas10 (NAS=10) | D_nas50 (NAS=50) |
|---|---|---|---|---|
| F1 gắp hỏng | 8 (6%) | 0 (0%) | 1 (4%) | **16 (33%)** |
| F2 làm đổ / kẹp sai | 39 (27%) | 13 (30%) | 9 (32%) | 3 (6%) |
| F3 đứng yên / lơ lửng | **53 (37%)** | **14 (32%)** | 4 (14%) | 15 (31%) |
| F4 đặt không đúng / gần đích | 12 (8%) | 2 (5%) | 5 (18%) | 11 (23%) |
| F5 nhầm vật / đích | 1 (1%) | 2 (5%) | 1 (4%) | 2 (4%) |
| F6 ra khỏi vùng làm việc | 27 (19%) | 12 (27%) | 7 (25%) | **0 (0%)** |
| F0 không xác định | 2 (1%) | 1 (2%) | 1 (4%) | 1 (2%) |
| **Thất bại có video** | **142** | **44** | **28** | **48** |
| Giai đoạn 0 / 1 / ? | 59 / 51 / 32 | 18 / 16 / 10 | 11 / 15 / 2 | 21 / 27 / 0 |
| Chuyển động 100 bước cuối < 0,3 | 91 | 26 | 11 | 10 |

Cùng cảnh, cấu hình khác (episode 0–9, cùng init state): NAS=10 thành công ở bao nhiêu cảnh mà cấu hình khác thất bại.

| Thất bại của | F1 | F2 | F3 | F4 | F5 | F6 | F0 | Tổng |
|---|---|---|---|---|---|---|---|---|
| D_nas1, NAS=10 thành công | – | 4/13 | 5/14 | 2/2 | 1/2 | 7/12 | 1/1 | 20/44 |
| D_nas50, NAS=10 thành công | 10/16 | 1/3 | 7/15 | 8/11 | 1/2 | – | 1/1 | 28/48 |

Nhận xét **[suy luận từ các bảng trên, chưa kiểm chứng cơ chế]**:
1. **Hai kiểu lỗi đối nghịch theo NAS.** NAS=1 (quan sát lại mỗi bước) hay dừng (F3) và đi lạc khỏi vùng làm việc (F6), ít khi
   gắp hỏng (F1: 8/142 ở B_main, 0/44 ở D_nas1). NAS=50 (chạy 50 action không quan sát) không đi lạc (F6 = 0) nhưng gắp
   trượt rồi vẫn làm tiếp (F1) và đặt thiếu chính xác (F4). NAS=10 giảm được cả hai: cứu 10/16 F1 của NAS=50 và 7/12 F6 của NAS=1.
2. **Lỗi còn lại sau khi sửa NAS là lỗi thao tác**: với NAS=10, F2 (làm đổ, kẹp ấm nằm ngang, task 0 và 8) chiếm 9/28,
   F4 (đặt chưa đủ chính xác, task 4, 5, 9) 5/28. Đây là phần mà thay đổi cách thực thi không sửa được, cần cải thiện
   bản thân policy.
3. **Giai đoạn 1 tăng tỉ trọng khi policy tốt hơn**: ở D_nas10, 15/28 thất bại đã xong bước con thứ nhất. Task nhiều bước
   vẫn là điểm yếu (task 6, 8 chiếm 9/28).
4. **Nguyên nhân theo task khá ổn định giữa các cấu hình**: task 1 đi về góc trên trái (F6, B_main và D_nas1); task 6 cầm
   pudding / ôm cốc cạnh đĩa (F3, cả 4 run); task 8 kẹp ấm thứ hai nằm ngang (F2, cả 4 run); task 0 với vào cụm hộp
   bên phải làm đổ đồ (F2, B_main, D_nas1, D_nas10).

## 4. Can thiệp reward (bản thiết kế, chưa cài đặt)

Theo yêu cầu, phần này chỉ là thiết kế bằng văn bản; chưa có code, chưa huấn luyện.

### 4.1 Tín hiệu có sẵn trong mô phỏng

- Reward hiện tại của eval là thưa: env chỉ báo thành công khi **mọi** điều kiện của goal đúng
  (`libero/envs/problems/libero_kitchen_tabletop_manipulation.py:136-144`, lerobot gọi `check_success()` ở
  `lerobot/src/lerobot/envs/libero.py:376`).
- Goal của mỗi task là một phép AND của các predicate trong file BDDL
  (`site-packages/libero/libero/bddl_files/libero_10/*.bddl`). Ví dụ task 8:
  `(On moka_pot_1 flat_stove_1_cook_region) (On moka_pot_2 flat_stove_1_cook_region) (Turnon flat_stove_1)`.
  Mỗi predicate có thể đánh giá riêng (`_eval_predicate`, cùng file, dòng 146), nên **đếm được số bước con đã xong** ở mọi bước.
- `On(a, b)` yêu cầu tâm hai vật cách nhau **< 3 cm theo phương ngang**, có tiếp xúc và a không thấp hơn b
  (`libero/envs/object_states/base_object_states.py:87-93`). Đây là giải thích khả dĩ cho nhóm F4 "hai cốc ở đĩa nhưng
  không tính" ở task 4 **[suy luận: chưa đo khoảng cách thật trong các episode đó]**.
- Từ `env.sim` đọc được vị trí vật (`body_xpos`), trạng thái tiếp xúc giữa ngón kẹp và vật, độ mở tay kẹp, và trạng
  thái khớp của cửa / ngăn kéo (`is_close`, `articulated_objects.py`).

### 4.2 Các thành phần reward, mỗi thành phần nhắm vào một nguyên nhân

| Thành phần | Công thức (đề xuất) | Nhắm vào | Bằng chứng từ phân tích |
|---|---|---|---|
| R_sub: reward theo bước con | +1 mỗi khi một predicate của goal chuyển từ sai sang đúng; −1 nếu predicate đã đúng chuyển lại thành sai | F3 giai đoạn 1, các task nhiều bước | Giai đoạn 1 chiếm 51/142 (B_main), 15/28 (D_nas10); task 6 và 8 là hai task yếu nhất |
| R_grasp: chỉ thưởng mang vật khi vật đang được giữ | Thưởng tiến lại gần đích chỉ khi có tiếp xúc hai ngón kẹp với vật **và** vật cao hơn mặt bàn ≥ 2 cm; phạt nhỏ khi tay đóng mà không giữ vật | F1 (gắp trượt rồi mang tay không) | F1 = 16/48 ở D_nas50; cũng có ở B_main (`t0 e0 s1001`, `s1002`) |
| R_place: shaping cho thao tác đặt | Thưởng theo hiệu thế −‖xy(vật) − xy(đích)‖, kích hoạt khi vật cách đích < 10 cm; cộng thêm khi đạt < 3 cm (ngưỡng của `On`); với cửa / ngăn kéo: thưởng theo độ đóng của khớp | F4 | F4 = 11/48 (D_nas50), 5/28 (D_nas10): ngăn kéo chưa đóng, cốc lệch đĩa, sách sai ngăn, cửa lò chưa đóng |
| P_disturb: phạt làm đổ / xô lệch | Phạt khi trục đứng của một vật (kể cả vật nhiễu) nghiêng > 45°, hoặc vật không cần thao tác bị dịch > 5 cm | F2 | F2 = 39/142 (B_main), 9/28 (D_nas10); task 0 (cụm hộp bên phải), task 8 (ấm nằm ngang) |
| P_stall: phạt không tiến triển | Phạt nhỏ mỗi bước nếu trong N = 50 bước liền không có predicate nào đổi trạng thái và không vật nào dịch > 1 cm | F3 | F3 = 53/142 (B_main); 91/142 thất bại có chuyển động cuối < 0,3 |
| P_workspace: phạt ra khỏi vùng làm việc | Phạt khi đầu tay kẹp ra ngoài hộp bao vùng bàn (xác định từ vị trí các vật lúc khởi tạo cộng lề) | F6 | F6 = 27/142 (B_main), 7/28 (D_nas10); task 1 và 4 |
| P_wrong: phạt nhầm vật | Phạt khi nhấc một vật không có tên trong goal (giỏ, hộp nước cam, ramekin) | F5 | Ít (1–2 episode mỗi run) nhưng là lỗi rõ ràng |

Reward tổng (đề xuất): `r = 10·success + R_sub + 0,1·(R_grasp + R_place) − (P_disturb + P_wrong) − 0,01·(P_stall + P_workspace)`.
Các hệ số chỉ là điểm khởi đầu, cần tinh chỉnh. Hai thành phần shaping khoảng cách (R_grasp, R_place) nên viết dưới
dạng hiệu thế (potential-based) để không thay đổi chính sách tối ưu của bài toán gốc.

### 4.3 Thứ tự ưu tiên và rủi ro

1. **Làm trước R_sub và R_place**: lấy trực tiếp từ predicate của LIBERO, không cần ngưỡng tự chọn ngoài ngưỡng 3 cm
   có sẵn, và nhắm vào nhóm lỗi còn lại lớn nhất khi chạy với NAS=10 (giai đoạn 1, F4).
2. **R_grasp và P_disturb** cần đọc tiếp xúc và góc nghiêng của vật: phải kiểm tra tên body của từng vật trong mô phỏng.
3. **P_stall và P_workspace** dùng ngưỡng tự chọn, dễ gây hành vi phụ (ví dụ robot "rung" để tránh phạt đứng yên);
   chỉ nên dùng hệ số nhỏ.
4. Lỗi do cách thực thi chunk (F1 với NAS=50, F6 với NAS=1) đã giảm mạnh khi đổi NAS sang 10 mà không cần reward.
   Nếu huấn luyện thêm, nên đo lại trên NAS=10 để không gán nhầm cải thiện của NAS cho reward.
5. Rủi ro chung: reward theo predicate chỉ đúng nếu predicate trùng với ý định của task; với task 4, ngưỡng 3 cm làm
   cả những lần đặt "trông đúng" bị tính sai, nên R_place sẽ đẩy policy đặt chính giữa đĩa, đúng với tiêu chí của env.

## 5. Giới hạn

- D_* chỉ có video cho episode 0–9 mỗi task (`lerobot_eval.py:823`): bảng nguyên nhân của D_nas1, D_nas10, D_nas50 chỉ
  phủ 44/227, 28/132 và 48/261 thất bại.
- D_nas10 mới đo 1 seed (1000). B_main cho thấy với NAS=1, độ lệch giữa các seed nhỏ (`differences.md` mục 3), nhưng điều
  này chưa được kiểm tra với NAS=10.
- Phân loại bằng mắt do Claude làm, không có người thứ hai kiểm tra. Phần lớn nhãn chỉ dựa trên khung điển hình;
  số episode có timeline: B_main 6, D_nas50 22, D_nas1 và D_nas10 0.
- Chỉ có camera ngoài: không biết vật có nằm trong giỏ / lò vi sóng hay không khi bị che (giai đoạn "?").
- Không đọc trạng thái mô phỏng của các episode thất bại (không có trong output của eval), nên "vì sao env không tính"
  ở nhóm F4 là suy luận.
