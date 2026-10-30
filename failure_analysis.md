# FFI26 - Phân tích các ca thất bại từ video (bản nháp để kiểm tra luận lý)

Trạng thái: bản viết ngày 30/10/2026. Bản đầu (chỉ có libero_10) được viết để bạn kiểm tra cách lập luận; bản này
bổ sung LIBERO-Spatial và ablation. Phạm vi:

| Phần | Trạng thái |
|---|---|
| B_main (libero_10, NAS=1, relative, seed 1000/1001/1002): 142 episode thất bại | **đã phân tích**, 10/10 task (mục 3, 4) |
| F_spatial (libero_spatial, NAS=1, relative, 3 seed): 63 episode thất bại | **đã phân tích**, 9 task có thất bại (mục 5) |
| Ablation D_nas1/D_nas50, F_spatial_nas50, F_spatial_abs | số đo chuyển động cho toàn bộ video, và xem bằng mắt một số cặp episode cùng cảnh ban đầu (mục 6.1–6.4) |
| D_nas50 (libero_10, NAS=50): 48 thất bại có video | **đã phân tích** (bổ sung 30/10/2026): 22 cặp [timeline], 26 episode [khung cuối] (mục 6.5) |
| D_nas1 (libero_10, NAS=1): 44 thất bại có video | **đã phân loại** ngày 31/10/2026 (khung điển hình): `docs/failures/D_nas1.md` |
| D_nas10 (libero_10, NAS=10, chạy 31/10/2026): 28 thất bại có video | **đã phân loại** (khung điển hình): `docs/failures/D_nas10.md` |

Từ 31/10/2026, so sánh giữa các cấu hình, bộ nguyên nhân chung F0–F6 và thiết kế can thiệp reward nằm ở
`docs/failure_overview.md`. File này giữ phân tích chi tiết ban đầu.

Các đường dẫn `results/...` là tương đối với `~/ffi26/`. Video của một episode nằm ở:
`results/runs/B_main/lerobot_smolvla_libero/task{T}_seed{S}/videos/libero_10_{T}/eval_episode_{E}.mp4`.
Trong file này, episode được viết tắt là `eE sS` (ví dụ `e4 s1000` = episode 4, seed 1000).

## 1. Phương pháp (để bạn kiểm tra từng bước)

### 1.1 Dữ liệu đầu vào

1. Danh sách video và cờ thành công lấy từ `eval_info.json` của từng job: `per_task[0].metrics.successes[i]` đi cặp với
   `per_task[0].metrics.video_paths[i]`.
2. Mỗi video là camera bên ngoài (agentview), 360×360, **một khung hình cho mỗi bước env**. Số khung hình đo bằng
   `ffprobe -count_packets`.
3. Video **không có camera cổ tay**. Vì vậy những gì camera ngoài không nhìn thấy (bên trong giỏ, ngăn sau của hộp,
   lò vi sóng, hoặc khi tay robot ra khỏi khung hình) được ghi là **không xác định**.

### 1.2 Các bước phân tích

| Bước | Làm gì | Kết quả dùng để làm gì |
|---|---|---|
| 1 | Đếm khung hình của toàn bộ 1.000 video, so với cờ thành công | Xác định thất bại có phải do hết giờ hay không (mục 2.1) |
| 2 | Tính **chỉ số chuyển động**: trung bình trị tuyệt đối hiệu hai khung liên tiếp (ảnh xám, thu nhỏ 90×90), lấy trên 100 bước cuối | Phát hiện robot gần như đứng yên khi hết giờ (mục 2.3) |
| 3 | Với mỗi task: xem **khung cuối** của mọi episode thất bại, đặt cạnh nhau, có cắt vùng bàn | Phân loại trạng thái cuối |
| 4 | Với 2–3 episode đại diện của task: xem **dòng thời gian** 8 khung chia đều từ bước 0 đến bước cuối | Hiểu hành vi dẫn tới trạng thái cuối, kiểm tra cách đọc khung cuối |
| 5 | Xem khung cuối của một vài episode **thành công** của cùng task | Biết cấu hình đích trông như thế nào trong ảnh |
| 6 | Gom các episode cùng trạng thái cuối thành một nhóm, đếm | Bảng theo task (mục 3) |

### 1.3 Mức độ tin cậy của từng nhận định

| Nhãn | Ý nghĩa |
|---|---|
| **[timeline]** | Đã xem dòng thời gian của chính episode đó |
| **[khung cuối]** | Chỉ dựa trên khung hình cuối |
| **[suy luận]** | Kết luận rút ra từ quan sát khác, có nêu lý do |
| **[không xác định]** | Video không đủ thông tin |

Phân loại bằng mắt là đánh giá của Claude, không phải phép đo. Bạn nên mở lại video được trích để kiểm tra.

## 2. Các phát hiện có thể đo được (không phụ thuộc vào việc xem bằng mắt)

### 2.1 Mọi thất bại đều là hết giờ

| Run | Thất bại | Số khung hình của thất bại | Thành công: số khung hình (min / trung vị / max) |
|---|---|---|---|
| B_main (libero_10) | 142 | luôn đúng **520** | 151 / 245 / 456 |
| F_spatial (libero_spatial) | 63 | luôn đúng **280** | 71 / 105 / 239 |

520 và 280 là giới hạn bước của hai suite (`libero.py:98-104`). Env chỉ kết thúc sớm khi thành công
(`libero.py:376-377`). Vì vậy không có loại "thất bại sớm": mọi episode thất bại đều chạy đến hết giới hạn bước,
và chỉ video mới phân biệt được các kiểu thất bại.

### 2.2 Thất bại do cảnh ban đầu quyết định, không do seed

B_main có 10 task × 10 init state = 100 cặp (task, init state). Mỗi cặp được chạy với 3 seed (lý do: episode i luôn
dùng init state i, xem `differences.md` mục 3).

| Số seed thành công trên 3 | Số cặp (task, init state) |
|---|---|
| 0 (thất bại cả 3 seed) | 37 |
| 1 | 10 |
| 2 | 11 |
| 3 (thành công cả 3 seed) | 42 |

**111/142 thất bại (78%) đến từ 37 cảnh thất bại ở cả 3 seed.** Khi xem video, các seed của cùng một cảnh cũng cho
trạng thái cuối giống nhau ở phần lớn trường hợp (ví dụ task 1 e3, task 6 e3/e5/e8, task 8 e3/e5).
Hệ quả cho báo cáo: 30 episode/task không phải 30 lần thử độc lập. Để cải thiện, cần nhắm vào các cảnh cụ thể.

### 2.3 Phần lớn thất bại kết thúc với robot gần như đứng yên

Chỉ số chuyển động ở 100 bước **đầu** của các episode thất bại có trung vị 1,70. Ngưỡng "gần như đứng yên" được đặt
là **< 0,3**. Ngưỡng này do Claude chọn sau khi đối chiếu bằng mắt: các episode dưới 0,3 đã xem đều là robot lơ lửng
một chỗ (ví dụ task 0 `e4 s1001` = 0,029, task 1 `e0 s1002` = 0,101). **Ngưỡng này không phải hằng số chuẩn**:
với ngưỡng 0,5 con số là 104/142.

| Task libero_10 | Thất bại | Chuyển động 100 bước cuối < 0,3 |
|---|---|---|
| 0 | 29 | 17 |
| 1 | 14 | 13 |
| 2 | 2 | 0 |
| 3 | 0 | – |
| 4 | 23 | 16 |
| 5 | 3 | 3 |
| 6 | 21 | 18 |
| 7 | 14 | 5 |
| 8 | 29 | 14 |
| 9 | 7 | 5 |
| **Tổng** | **142** | **91 (64%)** |

Diễn giải: ở 91/142 thất bại, robot đã ngừng tiến triển từ trước khi hết giờ, chứ không phải đang làm dở thì hết giờ.
Vì vậy tăng giới hạn bước nhiều khả năng không cứu được những episode này **[suy luận]**.

## 3. Phân tích theo task (B_main, libero_10)

Quy ước về hướng: camera nhìn từ phía trước robot nên **trái/phải trong ảnh ngược với trái/phải của robot**.
Điều này được kiểm chứng ở task 4 (mục 3.4) bằng episode thành công.

### 3.0 Task 0: put both the alphabet soup and the tomato sauce in the basket (thất bại 29/30)

Cách nhận diện vật: lon **nhãn xanh** = alphabet soup, lon **nhãn đỏ** = tomato sauce **[suy luận]**: trong episode
thành công của task 7 (`t7 e0 s1000`), giỏ chứa hộp cream cheese và một lon, còn lon nhãn đỏ ở lại trên bàn.
Task 7 yêu cầu alphabet soup, nên lon nhãn xanh là alphabet soup.

| Nhóm | Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|---|
| A1 | Lon xanh đã vào giỏ; robot cầm lon đỏ và lơ lửng phía trên bàn, không mang tới giỏ | e1 (3 seed), e9 s1001, e9 s1002 | 5 | `e1 s1000` [timeline]: lon xanh vào giỏ ở khoảng bước 222, sau đó cầm lon đỏ lơ lửng từ bước ~297 đến 519. Các episode còn lại [khung cuối] |
| A2 | Lon xanh đã vào giỏ; lon đỏ vẫn trên bàn, robot ở phía trên giỏ | e9 s1000 | 1 | [khung cuối] |
| A3 | Lon xanh đã vào giỏ; robot lơ lửng đứng yên phía trên vùng lon đỏ | e4 s1001, e4 s1002 | 2 | `e4 s1001` [timeline]; `e4 s1002` [khung cuối], giống s1001 |
| A4 | Lon xanh đã vào giỏ; khi đi lấy vật thứ hai, robot **kẹp vào thành giỏ và nhấc cả giỏ lên** | e4 s1000 | 1 | [timeline], thấy rõ từ bước ~371 |
| B | Giỏ trống ở khung cuối; không lon nào vào giỏ | e0, e2, e3, e5, e6, e8 (mỗi cảnh 3 seed), e7 s1001, e7 s1002 | 20 | `e2 s1000` [timeline]: lon xanh nằm sau hộp sữa/nước cam, tay robot len vào giữa các hộp, làm đổ hộp nước cam và lon xanh. `e0 s1000` [timeline]: không lấy được lon xanh, sau đó lơ lửng ở giữa bàn. Các episode khác [khung cuối]: giỏ trống, robot ở vùng các hộp bên phải ảnh (trừ e0) |

Tóm tắt: 9/29 hoàn thành vật thứ nhất rồi hỏng ở vật thứ hai; 20/29 không hoàn thành vật nào.
Trong nhóm B, lon xanh ở sau các hộp trên bàn. Nguyên nhân "lon bị che, robot va chạm khi với tới" mới được
kiểm chứng ở `e2 s1000`, chưa kiểm chứng cho cả nhóm.

### 3.1 Task 1: put both the cream cheese box and the butter in the basket (thất bại 14/30)

| Nhóm | Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|---|
| A | Robot gắp hộp xanh, đi về góc trên bên trái (phía sau giỏ) và ở đó đến hết giờ, một phần hoặc toàn bộ tay ra khỏi khung hình. Hộp nhỏ màu cam (butter) vẫn nằm trên bàn ở khung cuối | e2, e3, e8 (mỗi cảnh 3 seed), e5 s1000, e5 s1001 | 11 | `e3 s1000`, `e5 s1000` [timeline]; các episode khác [khung cuối]. **Hộp xanh có nằm trong giỏ hay không: [không xác định]** (không nhìn thấy trong khung cuối) |
| B | Robot kẹt ở chỗ hộp xanh phía sau hộp sữa khoảng 450 bước, không nhấc hộp lên được | e0 s1001, e0 s1002, e1 s1002 | 3 | `e0 s1002` [timeline]; còn lại [khung cuối] |

Nhóm A là kiểu thất bại riêng của task này: hành vi sau khi gắp vật thứ nhất là đi ra khỏi vùng làm việc.
Với camera hiện có, không phân biệt được "đã thả hộp vào giỏ rồi đứng yên" và "chưa thả".

### 3.2 Task 2: turn on the stove and put the moka pot on it (thất bại 2/30)

| Episode | Mô tả | Tin cậy |
|---|---|---|
| e1 s1000 | Bếp đã đỏ (đã bật; ở khung đầu cuộn bếp màu tối), ấm moka đứng trên bàn cạnh bếp | [khung cuối] |
| e5 s1002 | Bếp đã đỏ; robot cầm ấm ở mép bếp, ấm chưa nằm trên bếp | [khung cuối] |

Cả hai: bước "bật bếp" xong, bước "đặt ấm lên bếp" chưa xong.

### 3.3 Task 3: put the black bowl in the bottom drawer of the cabinet and close it

Không có thất bại (30/30).

### 3.4 Task 4: put the white mug on the left plate and put the yellow and white mug on the right plate (thất bại 23/30)

Kiểm chứng hướng: trong episode thành công `e1 s1000` [khung cuối], cốc **trắng** nằm trên đĩa **bên phải ảnh**,
cốc **vàng-trắng** nằm trên đĩa **bên trái ảnh**. Vậy đích trong ảnh: cốc trắng → đĩa phải ảnh, cốc vàng-trắng → đĩa trái ảnh.

| Nhóm | Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|---|
| A | Cốc vàng-trắng **không bị di chuyển**, vẫn ở giữa bàn ở khung cuối | e2 (3 seed), e3 s1000, e3 s1002, e6 (3 seed), e7 s1001, e7 s1002, e8 s1002, e9 (3 seed) | 14 | [khung cuối]. Hai timeline: `e2 s1000`: robot mang cốc trắng sang phải, ra khỏi khung hình và ở đó đến hết giờ (cốc trắng ở đâu: [không xác định]). `e9 s1000`: đặt cốc trắng lên đĩa phải ở khoảng bước 148–222, rồi lơ lửng tại đĩa đến hết giờ. Với 12 episode còn lại, cốc trắng đã lên đĩa hay chưa: [không xác định] từ khung cuối |
| B | Một cốc bị đổ hoặc bị cầm nghiêng | e0 s1000 (cốc trắng nằm đổ), e0 s1002, e3 s1001 (cốc vàng-trắng bị cầm nghiêng), e4 s1001 (cốc vàng-trắng nằm đổ ở mép đĩa trái), e7 s1000 (cốc trắng nghiêng) | 5 | [khung cuối] |
| C | **Gần đúng cấu hình đích nhưng env không tính thành công**: cốc trắng ở đĩa phải, cốc vàng-trắng ở mép đĩa trái | e0 s1001, e4 s1000, e4 s1002 | 3 | `e4 s1000` [timeline]: cốc trắng lên đĩa phải ở bước ~148, cốc vàng-trắng được đặt ở đĩa trái từ bước ~297. Predicate nào chưa thỏa (có thể cốc chỉ nằm một phần trên đĩa): **[không xác định]** |
| – | Không phân loại được: cốc trắng ở đĩa phải, cốc vàng-trắng không thấy trong khung | e5 s1002 | 1 | [không xác định] |

### 3.5 Task 5: pick up the book and place it in the back compartment of the caddy (thất bại 3/30)

| Episode | Mô tả | Tin cậy |
|---|---|---|
| e1 s1001, e1 s1002 | Sách dựng nghiêng, nhô lên ở ngăn bên phải của hộp. Ở episode thành công `e0 s1000`, sách không nhìn thấy (đã nằm trong ngăn sau) | [khung cuối] |
| e7 s1002 | Robot vẫn cầm sách ở ngăn bên phải | [khung cuối] |

Sách nằm sai ngăn hoặc chưa vào hẳn ngăn sau **[suy luận]**, dựa trên việc sách còn nhìn thấy được.

### 3.6 Task 6: put the white mug on the plate and put the chocolate pudding to the right of the plate (thất bại 21/30)

| Nhóm | Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|---|
| A | Cốc trắng đã trên đĩa; robot ở cạnh hộp pudding (bên trái đĩa trong ảnh), cầm hoặc lơ lửng, chưa đặt xong | e0 (3 seed), e1 s1001, e2 s1001, e4 s1001, e4 s1002, e9 s1001, e9 s1002 | 9 | [khung cuối]. Vì sao điều kiện "bên phải đĩa" chưa thỏa: [không xác định] |
| B | Cốc trắng đã trên đĩa nhưng **tay kẹp vẫn ôm cốc** đến hết giờ; pudding chưa được đụng tới | e3 (3 seed), e5 (3 seed), e4 s1000 | 7 | [khung cuối] |
| C | Cốc trắng không lên đĩa; **cốc đỏ (vật gây nhiễu) bị làm đổ** | e8 (3 seed) | 3 | [khung cuối] |
| D | Cốc trắng bị cầm nghiêng cạnh đĩa (`e2 s1000`); cốc trắng đặt trước đĩa, không nằm trên đĩa (`e1 s1000`) | e2 s1000, e1 s1000 | 2 | [khung cuối] |

Tóm tắt: 16/21 (A + B) hoàn thành bước 1 (cốc lên đĩa) và hỏng ở bước 2.

### 3.7 Task 7: put both the alphabet soup and the cream cheese box in the basket (thất bại 14/30)

| Nhóm | Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|---|
| A1 | Lon xanh (alphabet soup) bị đổ, nằm trên bàn hoặc bị cầm nghiêng | e0 s1002, e3 s1000, e5 s1001, e6 s1002 | 4 | [khung cuối] |
| A2 | Lon xanh vẫn đứng trên bàn, robot lơ lửng gần hoặc đang kẹp | e3 s1002, e5 s1000, e5 s1002 | 3 | [khung cuối] |
| A3 | Lon xanh bị cầm nằm ngang ở mép giỏ | e3 s1001 | 1 | [khung cuối] |
| B | Có một lon trong giỏ; hộp cream cheese đang được cầm phía trên hoặc ở mép giỏ, chưa tính thành công | e8 (3 seed), e9 s1000 | 4 | [khung cuối] |
| – | Không xác định: tay robot hoặc vật ra khỏi khung hình | e1 s1000, e7 s1001 | 2 | [không xác định] |

Tóm tắt: 8/14 hỏng ngay ở vật thứ nhất (lon xanh), 4/14 hỏng khi đặt vật thứ hai.

### 3.8 Task 8: put both moka pots on the stove (thất bại 29/30)

Đếm trên khung cuối đã cắt vùng bếp và bàn ở độ phân giải đầy đủ **[khung cuối]**.

| Nhóm | Mô tả | Episode | Số |
|---|---|---|---|
| A1 | 1 ấm trên bếp; ấm thứ hai bị **kẹp nằm ngang** | e3 (3 seed), e4 s1000, e5 s1001, e8 s1002 | 6 |
| A2 | 1 ấm trên bếp; ấm thứ hai được cầm thẳng đứng nhưng không đặt lên | e1 s1001, e4 s1002, e5 s1000, e5 s1002, e6 s1001, e8 s1000 | 6 |
| A3 | 1 ấm trên bếp; ấm thứ hai vẫn trên bàn | e6 s1000, e8 s1001, e9 s1001 | 3 |
| A4 | Tay vẫn kẹp ấm ở trên bếp; ấm thứ hai nằm ngoài vùng cắt | e0 s1000 | 1 |
| B1 | Không ấm nào trên bếp; một ấm bị **đổ hoặc bị cầm nghiêng** cạnh bếp | e0 s1001, e0 s1002, e2 s1000, e2 s1002, e6 s1002, e7 s1000, e7 s1002 | 7 |
| B2 | Không ấm nào trên bếp; robot cầm ấm thẳng đứng ở giữa bàn | e1 s1000, e1 s1002, e2 s1001, e4 s1001, e7 s1001, e9 s1002 | 6 |

Tóm tắt: 16/29 đặt được ấm thứ nhất, 13/29 không đặt được ấm nào. Kiểu "kẹp ấm nằm ngang hoặc làm đổ ấm"
xuất hiện ở 13/29 (A1 + B1). Liên hệ: ở ablation D, task 8 là task duy nhất mà NAS=50 tốt hơn NAS=1 rõ rệt
(19/50 so với 7/50, `differences.md` mục 3b). Chưa xem video của D để giải thích điều này.

### 3.9 Task 9: put the yellow and white mug in the microwave and close it (thất bại 7/30)

| Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|
| Robot cầm cốc vàng-trắng ở trước cửa lò vi sóng, chưa đưa vào trong; chưa tới bước đóng cửa | e7 (3 seed), e8 s1000, e8 s1002, e9 s1000, e9 s1001 | 7 | [khung cuối] |

Mọi thất bại của task 9 nằm ở init state 7–9.

## 4. Tổng hợp các kiểu thất bại trên libero_10 (B_main)

Các kiểu dưới đây **không loại trừ nhau** (một episode có thể vừa "đứng yên" vừa "hỏng ở bước 2"), nên không cộng dồn.

| Kiểu thất bại | Bằng chứng | Task liên quan |
|---|---|---|
| **Hỏng ở bước con thứ hai** (bước 1 thấy rõ đã xong, bước 2 không xong) | task 0: 9/29; task 2: 2/2; task 6: 16/21; task 7: 4/14; task 8: 16/29. Cộng các nhóm được xác định rõ: 47 episode | 0, 2, 6, 7, 8 |
| **Đứng yên / lơ lửng đến hết giờ** | 91/142 có chuyển động 100 bước cuối < 0,3 (mục 2.3) | tất cả, rõ nhất ở 1, 4, 6 |
| **Kẹp sai tư thế, làm đổ hoặc nghiêng vật** | task 8: 13/29; task 4: 5/23; task 7: 5/14 (A1 + A3); task 6: 3/21 (đổ cốc đỏ) | 4, 6, 7, 8 |
| **Va chạm khi vật bị che** | task 0 `e2 s1000` [timeline] | 0 (mới kiểm chứng 1 episode) |
| **Cầm nhầm vật** | task 0 `e4 s1000`: nhấc cả giỏ | 0 |
| **Tay đi ra khỏi vùng làm việc / khung hình** | task 1: 11/14; task 4: `e2 s1000` [timeline] | 1, 4 |
| **Gần đích nhưng không được tính thành công** | task 4: 3/23; task 7: 4/14; task 5: 3/3 | 4, 5, 7 |

Hai nhận xét rút ra **[suy luận]**:
1. Nhóm lớn nhất không phải là "thao tác vụng" mà là **không chuyển sang bước con tiếp theo**: robot dừng lại hoặc
   lơ lửng sau khi xong bước 1 (task 0, 1, 6, 8). Điều này khớp với giả thuyết ở phần Improve của `readme.md`
   (reward theo từng bước con).
2. Thất bại tập trung vào một số cảnh ban đầu cố định (mục 2.2). Khi chọn task và cảnh cho RL, nên ưu tiên 37 cảnh
   thất bại cả 3 seed.

## 5. Phân tích LIBERO-Spatial (F_spatial: NAS=1, relative, 3 seed, 63 thất bại)

Video: `results/runs/F_spatial/lerobot_smolvla_libero/task{T}_seed{S}/videos/libero_spatial_{T}/eval_episode_{E}.mp4`.
Ở suite này mọi task đều là "nhặt một bát đen và đặt lên đĩa", chỉ khác **vị trí** của bát mục tiêu. Vì vậy, ngoài khung
cuối, mỗi episode được xem thêm **khung đầu đặt cạnh khung cuối** để biết bát mục tiêu ban đầu ở đâu và cuối cùng ở đâu.

### 5.1 Bảng theo task

| Task (thất bại) | Mô tả task | Nhóm và episode | Tin cậy |
|---|---|---|---|
| 0 (1/30) | bowl between the plate and the ramekin | S1: `e9 s1001` | [khung đầu/cuối] |
| 1 (6/30) | bowl next to the ramekin | S1: `e1 s1000`, `e1 s1001`, `e2 s1002`, `e4 s1001`. S6: `e7 s1000`, `e7 s1002` (mang vật sang phía trên bên phải, không tới đĩa; vật gì: [không xác định]) | S6 `e7 s1000` [timeline]; còn lại [khung đầu/cuối] |
| 3 (2/30) | bowl on the cookie box | S1: `e5 s1002`. S3: `e7 s1000` (bát vẫn trên hộp bánh, robot ở gần ramekin) | [khung đầu/cuối] |
| 4 (10/30) | bowl in the top drawer of the wooden cabinet | S3: `e1 s1000`, `e1 s1002`, `e3` (3 seed), `e8 s1000`, `e9 s1001` (bát vẫn trong ngăn kéo ở khung cuối). S1: `e6 s1002`, `e9 s1000`. Không xác định: `e4 s1002` | `e3 s1000` [timeline]: tay vào vùng ngăn kéo ở bước ~40, không lấy được bát, **vẫn di chuyển tới đĩa với tay không** (bước ~120), rồi lơ lửng bên trái. Còn lại [khung đầu/cuối] |
| 5 (30/30) | bowl on the ramekin | xem mục 5.2 | |
| 6 (1/30) | bowl next to the cookie box | S1: `e3 s1000` | [khung đầu/cuối] |
| 7 (4/30) | bowl on the stove | S2: `e8 s1000` (bát vẫn trên bếp đến cuối; robot mang một vật khác tới đĩa, có thể là ramekin, chưa chắc). S4: `e8 s1002`, `e9 s1001` (kẹp ở bát trên bếp, đứng yên: chuyển động 0,013 và 0,124). S3: `e5 s1001` (bát vẫn trên bếp) | `e8 s1000` [timeline]; còn lại [khung đầu/cuối] |
| 8 (7/30) | bowl next to the plate | S1: `e1 s1001`, `e1 s1002`. S2: `e2 s1002`, `e4 s1000` (lấy một vật ở cụm phía trên; **bát cạnh đĩa vẫn ở chỗ cũ đến cuối**). S3: `e4 s1001`, `e4 s1002` (mọi bát ở chỗ cũ, robot lơ lửng). Không xác định: `e2 s1000` | `e2 s1002`, `e4 s1000` [timeline]; còn lại [khung đầu/cuối] |
| 9 (2/30) | bowl on the wooden cabinet | S1: `e5 s1000`, `e5 s1001` | `e5 s1001` [timeline]: đưa bát tới đĩa ở bước ~120, bát rơi xuống **cạnh** đĩa (~159), robot gắp lại, cuối cùng bát nghiêng ở mép đĩa. `e5 s1000` [khung cuối] |

Định nghĩa nhóm:

| Nhóm | Nghĩa |
|---|---|
| S1 | Một bát đã được mang tới đĩa; ở khung cuối tay kẹp vẫn giữ bát, hoặc bát nghiêng ở mép đĩa. Env không tính thành công; vì sao predicate "bát trên đĩa" chưa thỏa: [không xác định] |
| S2 | **Lấy nhầm vật**: bát mục tiêu vẫn ở chỗ cũ đến hết giờ, robot mang một vật khác tới đĩa |
| S3 | Bát mục tiêu không được nhấc lên; robot lơ lửng hoặc đi nơi khác |
| S4 | Kẹp ở bát mục tiêu nhưng đứng yên, không nhấc lên |
| S5 | Chồng bát–ramekin bị xáo trộn tại chỗ (chỉ task 5, mục 5.2) |
| S6 | Mang vật đi sai hướng, không tới đĩa |

### 5.2 Task 5: pick up the black bowl on the ramekin and place it on the plate (thất bại 30/30)

Cảnh ban đầu: **bát đen nằm nghiêng trên một ramekin** (đĩa nhỏ màu trắng có gờ) ở góc trên bên trái
**[khung đầu, độ phân giải đầy đủ]**. Bản nháp trước nêu hai giả thuyết (gắp nhầm ramekin, nhấc cả đĩa).
Sau khi xem dòng thời gian dày hơn (16 khung), kết quả như sau:

| Nhóm | Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|---|
| S5 | Chồng bị xáo trộn tại chỗ: tay hạ xuống chồng, bát bị gạt khỏi ramekin và nằm nghiêng cạnh đó, rồi robot lơ lửng ngay bên trái. Đĩa trống | e0, e1, e2, e5, e7, e8, e9 (mỗi cảnh 3 seed), e3 s1000, e3 s1002, e6 s1000, e6 s1002 | 25 | `e0 s1000` [timeline 16 khung]; còn lại [khung cuối]. Robot có đang cầm vật nào hay không: **[không xác định]** |
| S2 | **Gắp nhầm ramekin** (vật nằm dưới bát) thay vì bát | e4 s1001 (đặt ramekin lên đĩa ở bước ~211; bát rơi lại xuống bàn gần chỗ cũ), e4 s1000 (tay đang giữ ramekin trên không, bát ở chỗ cũ) | 2 | `e4 s1001` **[timeline 16 khung, đã xác nhận]**; `e4 s1000` [khung cuối] |
| S1 | Gắp **đúng bát**, mang tới đĩa nhưng **không thả**: tay tiếp tục ấn xuống làm đĩa bị nghiêng lên (bước ~200–279) | e3 s1001 | 1 | **[timeline 16 khung, đã xác nhận]**. Giả thuyết "nhấc cả đĩa" ở bản nháp được giải thích bằng hiện tượng này: đĩa nghiêng do bị ấn, không phải do bị gắp |
| – | Không xác định: tay ở vùng đĩa, không nhìn rõ vật | e4 s1002, e6 s1001 | 2 | [không xác định] |

Kết luận cho task 5: giả thuyết "gắp nhầm ramekin" **đúng ở 2/30** (1 xác nhận bằng timeline). Kiểu phổ biến nhất
(25/30) là **không tách được bát khỏi ramekin một cách có kiểm soát**: bát bị gạt rơi, sau đó robot dừng lại.
Bát nằm chồng nghiêng trên ramekin là cấu hình chỉ có ở task này **[suy luận]**, nên thất bại 30/30 nhiều khả năng do
kỹ năng tách vật chồng, không do hiểu sai câu lệnh. Từ video không chứng minh được điều này.

### 5.3 Tổng hợp LIBERO-Spatial

| Nhóm | Số episode | Task |
|---|---|---|
| S5: chồng bị xáo trộn, robot dừng | 25 | 5 |
| S1: mang bát tới đĩa nhưng không đặt được (giữ mãi, bát nghiêng ở mép, ấn đĩa nghiêng) | 14 | 0, 1, 3, 4, 5, 6, 8, 9 |
| S3: không nhấc được bát mục tiêu | 11 | 3, 4, 7, 8 |
| S2: lấy nhầm vật | 5 | 5, 7, 8 |
| S4: kẹp ở bát nhưng đứng yên | 2 | 7 |
| S6: mang vật sai hướng | 2 | 1 |
| Không xác định | 4 | 4, 5, 8 |
| **Tổng** | **63** | |

So với libero_10, khác biệt rõ nhất là chỉ số chuyển động. Ở Spatial, trung vị chuyển động 100 bước cuối của thất bại là
0,50 và chỉ 11/63 dưới 0,3. Ở libero_10 (B_main) trung vị là 0,21 và 91/142 dưới 0,3. Như vậy ở Spatial, robot ít khi
"đứng yên" hơn và thường **vẫn đang thao tác ở đĩa** khi hết giờ (nhóm S1).

Hai hành vi đáng chú ý cho phần Analyse **[timeline]**:
- `t4 e3 s1000`: không lấy được bát nhưng vẫn thực hiện chuyển động mang tới đĩa. Policy thực hiện tiếp chuỗi chuyển
  động mà không kiểm tra vật đã nằm trong tay hay chưa **[suy luận]**.
- `t9 e5 s1001`: làm rơi bát cạnh đĩa rồi gắp lại. Đây là một trường hợp **có phục hồi sau lỗi**, dù cuối cùng vẫn thất bại.

## 6. Ablation

### 6.1 Số đo chuyển động trên toàn bộ video thất bại có video

Mỗi job chỉ lưu video cho 10 episode đầu (`lerobot_eval.py:823`). Vì vậy D_nas1 và D_nas50 chỉ có video cho
episode 0–9 của mỗi task.

| Run | NAS | control_mode | Thất bại có video | Trung vị chuyển động 100 bước cuối | Số thất bại có chuyển động < 0,3 |
|---|---|---|---|---|---|
| D_nas1 (libero_10) | 1 | relative | 44 | 0,23 | 26 |
| D_nas50 (libero_10) | 50 | relative | 48 | **1,27** | 10 |
| F_spatial | 1 | relative | 63 | 0,50 | 11 |
| F_spatial_nas50 | 50 | relative | 42 | **1,36** | 2 |
| F_spatial_abs | 1 | absolute | 98 | 0,88 | 28 |

Với NAS=50, thất bại **gần như không bao giờ kết thúc bằng trạng thái đứng yên**: robot vẫn chuyển động lớn khi hết giờ.
Lưu ý: NAS=50 thực thi 50 action liên tiếp mà không quan sát lại, nên chuyển động liên tục là hệ quả trực tiếp của cấu
hình này. Không nên diễn giải chỉ số này là "NAS=50 cố gắng hơn".

### 6.2 libero_10 task 8: vì sao NAS=50 tốt hơn (19/50 so với 7/50)

Trong episode 0–9 (có video), NAS=50 thành công ở e0, e6, e8, e9; NAS=1 thành công ở e8, e9. Đã xem hai cặp mà
NAS=50 thành công còn NAS=1 thất bại **[timeline, 2 cặp]**:

| Cặp | NAS=1 (thất bại, 520 bước) | NAS=50 (thành công) |
|---|---|---|
| e0 | Đặt ấm thứ nhất lên bếp ở khoảng bước 148; trong ~370 bước còn lại không đặt được ấm thứ hai, cuối cùng tay ở bếp, ấm thứ hai vẫn bên trái | Ấm thứ nhất trên bếp ở ~bước 159, ấm thứ hai ở bước 372 |
| e6 | Tay lơ lửng quanh ấm ở giữa bàn từ khoảng bước 148 đến 445; ấm thứ nhất chỉ lên bếp gần bước 519 | Ấm thứ nhất trên bếp ở ~bước 236, ấm thứ hai ở bước 413 |

Quan sát: ở hai cặp này, NAS=1 **chậm và chần chừ** ở thao tác gắp, còn NAS=50 hoàn thành cả hai ấm trong giới hạn
520 bước. Từ 2 cặp, chưa đủ để kết luận đây là cơ chế chung của task 8.

### 6.3 LIBERO-Spatial: NAS=50 kém hơn NAS=1 (58/100 so với 78/100, cùng seed 1000)

Đã xem hai cặp mà NAS=1 thành công còn NAS=50 thất bại **[timeline, 2 cặp]**:

| Cặp | NAS=1 (thành công) | NAS=50 (thất bại, 280 bước) |
|---|---|---|
| t6 e0 (bát cạnh hộp bánh) | Gắp bát ở ~bước 50, đặt lên đĩa ở bước 116 | Tay hạ xuống gần bát (bước ~40) nhưng không lấy được; bát vẫn ở chỗ cũ đến cuối; tay tới đĩa, đĩa có vẻ bị nghiêng |
| t7 e2 (bát trên bếp) | Gắp bát ở ~bước 66, đặt ở bước 115 | Gắp được bát, đưa tới đĩa ở ~bước 120, nhưng không thả; từ bước ~159 tay tiếp tục ấn ở mép đĩa và đĩa bị nghiêng |

Quan sát: hai kiểu thất bại của NAS=50 ở đây (trượt khi gắp; tới đĩa mà không thả) giống nhóm S3 và S1 của NAS=1, nhưng
xảy ra ở những cảnh mà NAS=1 làm được. Giả thuyết: khi thực thi 50 action không quan sát lại, policy không sửa được
sai lệch nhỏ lúc gắp hoặc lúc thả **[suy luận, chưa kiểm chứng]**.

### 6.4 control_mode absolute (F_spatial_abs: 2/100)

Đã xem 2 thất bại và cả 2 "thành công" **[timeline]**:

| Episode | Quan sát |
|---|---|
| t2 e0, t6 e3 (thất bại) | Cánh tay **quét những chuyển động lớn, tư thế xoắn bất thường** ngay từ bước đầu, va vào và làm xê dịch hộp bánh và đĩa; không có thao tác gắp–đặt nào |
| t0 e1 (thành công sau **25 bước**) | Cánh tay quét qua cụm bát, một bát bị **va rơi lên đĩa** (bước ~21–24) |
| t1 e2 (thành công sau **60 bước**) | Chuyển động quét lớn tương tự; không thấy thao tác gắp–đặt bình thường |

Để so sánh: thành công **ngắn nhất** của F_spatial (relative) cần 71 bước. Hai "thành công" của absolute
(25 và 60 bước) **là do va chạm ngẫu nhiên, không phải kỹ năng** **[suy luận từ timeline và độ dài episode]**.
Kết quả này củng cố kết luận ở `differences.md` mục 3b: checkpoint phải chạy với `relative`.

### 6.5 libero_10: các thất bại của D_nas50 (bổ sung ngày 30/10/2026)

Mục 6.1 chỉ đo chuyển động, mục 6.2 chỉ xem task 8 (task duy nhất NAS=50 tốt hơn). Mục này xem trực tiếp các thất
bại của D_nas50, vì NAS=50 **kém hơn NAS=1 ở 8/10 task** của libero_10.

**Số liệu** (đọc từ `per_task[0].n_success` trong `eval_info.json` của từng job, seed 1000, 50 episode mỗi task):

| Task | D_nas1 | D_nas50 | Chênh lệch |
|---|---|---|---|
| 0 | 12 | 12 | 0 |
| 1 | 29 | 21 | −8 |
| 2 | 45 | 37 | −8 |
| 3 | 46 | 40 | −6 |
| 4 | 20 | 17 | −3 |
| 5 | 43 | 37 | −6 |
| 6 | 18 | 11 | −7 |
| 7 | 23 | 18 | −5 |
| 8 | 7 | 19 | +12 |
| 9 | 30 | 27 | −3 |
| **Tổng** | **273/500** | **239/500** | **−34** |

**Phạm vi xem video.** Chỉ có video cho episode 0–9 của mỗi task (mục 6.1). Trong 100 episode này, D_nas50 có
**48 thất bại**, tất cả đều dài đúng 520 bước (hết giờ). Hai nhóm:
- **22 cặp** mà NAS=1 thành công còn NAS=50 thất bại, cùng task và cùng episode (nên cùng cảnh ban đầu). Đã xem
  **[timeline]** 12 khung chia đều của cả hai bên.
- **26 thất bại còn lại**, nơi NAS=1 cũng thất bại. Chỉ xem khung ở bước 150 và khung cuối (**[khung cuối]**).

Đường dẫn video: `results/runs/D_nas50/lerobot_smolvla_libero/task{T}_seed1000/videos/libero_10_{T}/eval_episode_{E}.mp4`.
Trong mục này, `tT eE` = task T, episode E. Chỉ số chuyển động lấy từ `motion.csv` của mục 6.1 (cùng script).

#### 6.5.1 Phân loại 48 thất bại

Mỗi episode được xếp vào đúng một nhóm, nên tổng là 48.

| Nhóm | Mô tả | Episode | Số | Tin cậy |
|---|---|---|---|---|
| N1 | **Gắp trượt vật thứ nhất nhưng vẫn tiếp tục chuỗi chuyển động**: ở khung bước 48 tay đang ở vật, ở khung bước 95 vật vẫn ở chỗ cũ và ở đó đến cuối episode, sau đó tay đi tới đích (giỏ) **với tay không**, hoặc chuyển sang vật khác | t1 e1, t1 e7, t7 e4, t7 e6 (tay không tới giỏ); t7 e0 (tay không tới giỏ, rồi nhấc cả giỏ lên từ bước ~330); t6 e6 (bỏ qua cốc trắng, chuyển sang hộp pudding); t0 e7 (trượt lon alphabet soup, sau đó cho một lon khác vào giỏ ở bước ~330, rồi lơ lửng trên giỏ) | 7 | [timeline] |
| | Như trên, nhưng chỉ dựa trên 2 khung: vật thứ nhất vẫn ở chỗ cũ, tay ở giỏ (t1 e8, t7 e5) hoặc ở hộp pudding (t6 e8) | t1 e8, t7 e5, t6 e8 | 3 | [khung cuối] + khung bước 150 |
| N2 | **Thả vật trước đích rồi làm bước sau**: cầm cốc tới trước lò vi sóng nhưng để cốc lại trên bàn bên ngoài (t9 e2: cốc bị nghiêng), sau đó thao tác với cửa lò đến hết giờ; cửa không đóng | t9 e0, t9 e2 | 2 | [timeline] |
| N3 | **Xong bước con thứ nhất, hỏng bước thứ hai** (giống kiểu chính của B_main, mục 4) | xem bảng dưới | 21 | xem bảng dưới |
| N5 | **Nhấc cả giỏ** (cầm nhầm vật) | t7 e3, t7 e8 | 2 | [khung cuối]: giỏ đang bị cầm lơ lửng ở khung cuối. (t7 e0 cũng nhấc giỏ nhưng đã tính ở N1) |
| N6 | **Hai cốc có vẻ đã nằm ở hai đĩa, robot đứng yên, nhưng env không tính thành công** | t4 e1, t4 e5 [timeline]; t4 e3, t4 e6, t4 e7 [khung cuối] | 5 | Xem ghi chú dưới bảng |
| N7 | **Sách vào sai ngăn**: sách được thả ở ngăn bên phải của hộp (t5 e8, khoảng bước 236, dựng đứng), hoặc bị đẩy nằm nghiêng ở ngăn phải (t5 e5) | t5 e5, t5 e8 | 2 | [timeline] |
| U | **Chưa xong bước con thứ nhất, không rõ cơ chế**: giỏ trống, tay ở vùng các lon (t0 e0, e3, e8, e9); hộp cream cheese vẫn trên bàn, tay ở vùng các lon (t1 e5); cầm cốc trắng lơ lửng cạnh đĩa phải đến hết giờ, cốc vàng-trắng chưa đụng tới (t4 e8, [timeline]) | t0 e0, t0 e3, t0 e8, t0 e9, t1 e5, t4 e8 | 6 | [khung cuối], trừ t4 e8 |

Chi tiết nhóm N3 (21 episode):

| Task | Bước 1 đã xong | Bước 2 hỏng như thế nào | Episode | Số | Tin cậy |
|---|---|---|---|---|---|
| 0 | Một lon trong giỏ (thấy nắp lon trong giỏ; lon nào: [không xác định]) | lon thứ hai chưa vào giỏ; ở t0 e2 một lon nằm đổ cạnh giỏ | t0 e2, t0 e4, t0 e6 | 3 | [khung cuối] |
| 1 | Hộp cream cheese đã được mang đi khỏi bàn (có trong giỏ hay không: [không xác định]) | tay lượn qua lại giữa lon soup và butter từ bước ~150 đến 519, không gắp được butter | t1 e0 | 1 | [timeline] |
| 2 | Bếp đã bật (đỏ) ở bước ~95 | t2 e2: không nhấc được ấm, tay tì lên ấm từ bước ~330 đến hết giờ; t2 e7: ấm bị **làm đổ** ở khoảng bước 236, sau đó tay đẩy quanh ấm đổ; t2 e1: tay lơ lửng trên quai ấm | t2 e1, t2 e2, t2 e7 | 3 | t2 e2, t2 e7 [timeline]; t2 e1 [khung cuối] |
| 3 | Bát đã ở trong ngăn kéo ở bước ~189 | **ngăn kéo chưa được đóng hẳn** (vẫn thấy bát); tay đứng yên ở trước/dưới mặt ngăn kéo (chuyển động 0,13 và 0,11) | t3 e0, t3 e8 | 2 | [timeline] + cắt vùng ngăn kéo ở khung cuối |
| 6 | Cốc trắng trên đĩa ở bước ~142 | t6 e7, t6 e9: cầm hộp pudding cạnh đĩa đến hết giờ; t6 e1, t6 e4: pudding chưa được đặt; t6 e5: tay vẫn ôm cốc trắng | t6 e1, t6 e4, t6 e5, t6 e7, t6 e9 | 5 | t6 e7, t6 e9 [timeline]; còn lại [khung cuối]. Vì sao điều kiện "bên phải đĩa" chưa thỏa: [không xác định] |
| 7 | Lon alphabet soup trong giỏ ở bước ~142 | tay lơ lửng phía trên hộp cream cheese từ bước ~189 đến hết giờ, không gắp | t7 e2 | 1 | [timeline] |
| 8 | Một ấm trên bếp | ấm thứ hai chưa lên bếp; ở t8 e4 ấm thứ hai bị kẹp nằm ngang | t8 e1, t8 e2, t8 e3, t8 e4, t8 e5, t8 e7 | 6 | [khung cuối] |

Ghi chú nhóm N6 (task 4): ở khung cuối, cốc vàng-trắng nằm ở đĩa trái ảnh và cốc trắng ở đĩa phải ảnh, giống khung
cuối của episode thành công `t4 e1` của D_nas1. Đĩa trái nằm sát mép ảnh nên không thấy được cốc nằm trọn trên đĩa
hay không. Vì sao env không tính thành công: **[không xác định]**. Cả 5 episode có chuyển động 100 bước cuối
≤ 0,04, tức robot đứng yên. Ở `t4 e1` và `t4 e5` [timeline], cả hai cốc đã được đặt xong trước bước ~236, sau đó
robot rút tay ra và đứng yên gần 300 bước. Nhóm này tương tự nhóm C của B_main ở task 4 (mục 3.4).

#### 6.5.2 22 cặp NAS=1 thành công, NAS=50 thất bại

| Nhóm của NAS=50 | Cặp | Số |
|---|---|---|
| N1 gắp trượt rồi vẫn tiếp tục | t0 e7, t1 e1, t1 e7, t6 e6, t7 e0, t7 e4, t7 e6 | 7 |
| N3 xong bước 1, hỏng bước 2 | t1 e0, t2 e2, t2 e7, t3 e0, t3 e8, t6 e7, t6 e9, t7 e2 | 8 |
| N2 thả vật trước đích | t9 e0, t9 e2 | 2 |
| N6 gần đích, không tính | t4 e1, t4 e5 | 2 |
| N7 sách sai ngăn | t5 e5, t5 e8 | 2 |
| U | t4 e8 | 1 |
| **Tổng** | | **22** |

Trong 7 cặp N1, ở cùng cảnh ban đầu, NAS=1 gắp được vật thứ nhất ngay lần đầu (ví dụ `t1 e1`: hộp cream cheese
được nhấc lên ở khoảng bước 50–74 và vào giỏ ở khoảng bước 150–170).

#### 6.5.3 Nhận xét

1. **Kiểu mới so với B_main: gắp trượt rồi vẫn làm tiếp (N1, 10/48).** Ở cả 7 episode N1 có timeline, khung bước 48
   cho thấy tay đang ở vật và khung bước 95 cho thấy vật vẫn ở chỗ cũ. Lần gắp trượt vì vậy nằm trước bước 95, nhiều
   khả năng **bên trong chunk 50 action đầu tiên** (bước 0–49) **[suy luận]**: timeline chỉ lấy mẫu mỗi ~47 bước. Với NAS=50, policy chỉ quan sát lại ở bước 50, 100, ... nên trong
   chunk đầu tay vẫn thực hiện tiếp đoạn "nhấc lên" dù vật chưa được kẹp **[suy luận]**. Sau đó policy không quay lại
   gắp lại mà tiếp tục mang tay không tới giỏ hoặc chuyển sang vật khác: đây là quan sát **[timeline]**, còn nguyên
   nhân là **[suy luận, chưa kiểm chứng]**. Kiểu này giống `t4 e3 s1000` của Spatial (mục 5) và các cặp ở mục 6.3.
   Cập nhật 31/10/2026: khi xem lại khung điển hình, kiểu này cũng thấy ở B_main (NAS=1) nhưng hiếm hơn: task 0
   `e0 s1001` và `e0 s1002`, tức 2/142 so với 10/48 ở NAS=50 (`docs/failures/B_main.md`).
2. **Kiểu chung với NAS=1 vẫn là lớn nhất: hỏng ở bước con thứ hai (N3, 21/48).** NAS=50 không làm kiểu này biến mất.
3. **Đứng yên khi thất bại không biến mất hoàn toàn với NAS=50.** 10/48 thất bại có chuyển động < 0,3 (mục 6.1).
   Gồm 5 episode N6 (task 4), 2 episode task 3, 2 episode task 5 và t2 e2 (0,25). Ở 9 episode của task 3, 4, 5,
   **vật đã ở gần đích và robot dừng lại**, khác với kiểu lơ lửng giữa chừng hay gặp ở B_main.
4. **Task 4, 3, 5 thất bại "gần đích" (N6, N3-task 3, N7: 9/48).** Ở các task này, thao tác đặt cuối cùng (đặt cốc
   trọn trên đĩa, đẩy ngăn kéo đóng hẳn, đưa sách vào đúng ngăn) không hoàn thành. Giả thuyết: thao tác chính xác ở
   cuối cần sửa nhỏ dựa trên quan sát, mà NAS=50 không làm được **[suy luận, chưa kiểm chứng]**, cùng hướng với giả
   thuyết ở mục 6.3.
5. Các nhận xét về cơ chế dựa trên 22 cặp ở episode 0–9 của seed 1000. Không có video cho episode 10–49, nên không
   kiểm chứng được tỉ lệ các nhóm trên toàn bộ 261 thất bại của D_nas50.

## 7. Kết luận chung (để dùng cho phần Analyse)

1. **Mọi thất bại đều là hết giờ** (mục 2.1). Cần video để phân loại.
2. **Thất bại do cảnh ban đầu quyết định** ở libero_10 (111/142 thuộc 37 cảnh thất bại cả 3 seed, mục 2.2).
3. **libero_10, task nhiều bước:** kiểu phổ biến nhất là **xong bước con thứ nhất rồi không chuyển sang bước thứ hai**,
   thường kèm trạng thái đứng yên (91/142 có chuyển động cuối < 0,3).
4. **LIBERO-Spatial, task một bước:** kiểu phổ biến nhất ngoài task 5 là **mang bát tới đĩa nhưng không đặt được**
   (S1, 14/63). Task 5 thất bại hoàn toàn vì **bát chồng trên ramekin** (25/30 làm xáo trộn chồng, 2/30 gắp nhầm ramekin).
5. **Lấy nhầm vật** có xảy ra nhưng ít: task 0 `e4 s1000` nhấc giỏ; Spatial 5 episode (S2).
6. **NAS=1 so với 50:** NAS=1 hay đứng yên khi thất bại, NAS=50 thì không. NAS=50 tốt hơn ở libero_10 task 8 (2 cặp
   cho thấy NAS=1 chậm và chần chừ khi gắp) nhưng kém hơn ở Spatial (2 cặp: trượt khi gắp, không thả). Các nhận xét
   cơ chế dựa trên 2 cặp mỗi bên, chỉ nên trình bày như ví dụ minh họa.
   Bổ sung (mục 6.5): trên libero_10, NAS=50 kém hơn ở 8/10 task (tổng 239/500 so với 273/500). Trong 22 cặp cùng
   cảnh mà NAS=1 thành công còn NAS=50 thất bại: 7 cặp **gắp trượt trong chunk đầu rồi vẫn làm tiếp với tay không**,
   8 cặp hỏng ở bước con thứ hai, 6 cặp hỏng ở thao tác đặt cuối cùng (task 4, 5, 9), 1 cặp không rõ.
7. **absolute:** chuyển động bất thường; 2 "thành công" là do va chạm.

## 8. Giới hạn của phân tích

- Chỉ có camera ngoài; không có camera cổ tay, không có trạng thái của predicate. Mọi nhận định "vì sao env không tính
  thành công" đều là [không xác định].
- Phân loại bằng mắt do Claude thực hiện, một người, không có người thứ hai kiểm tra chéo. Nhãn [khung cuối] có độ tin
  cậy thấp hơn [timeline].
- Ablation D: chỉ có video cho episode 0–9 trên tổng 50 episode của mỗi task. Thất bại của D_nas1 và D_nas10 chỉ được phân loại bằng khung điển hình (`docs/failures/`).
- Ngưỡng chuyển động 0,3 do Claude chọn (mục 2.3).

## 9. Công cụ đã dùng (để tái lập)

Các script nằm trong thư mục nháp của phiên làm việc
(`/tmp/claude-1003/-home-student-Lan-K68AI-temp/92c565a4-1d53-4fbd-9f85-01bf079683ec/scratchpad/`), **ngoài repo**,
và có thể bị xóa khi hết phiên. Nếu bạn muốn giữ, Claude có thể chép chúng vào repo khi được bạn cho phép.

| Script | Chức năng |
|---|---|
| `vidindex.py` | Lập bảng video, cờ thành công, số khung hình → `vids.csv` |
| `motion.py` | Chỉ số chuyển động của các episode thất bại → `motion.csv` |
| `grid.py` | Lưới khung hình (khung cuối hoặc khung bất kỳ), tùy chọn cắt vùng (`CROP`) |
| `sheet.py` | Dòng thời gian N khung chia đều của từng episode, tùy chọn cắt vùng (`CROP`) và khoảng bước (`video|bắt đầu|kết thúc`) |
| `pairs.py` | Khung đầu và khung cuối đặt cạnh nhau cho từng episode |
| `mk.py` | Liệt kê các video thất bại của một task, kèm nhãn |

Mục 6.5 dùng các script khác, nằm trong thư mục nháp của một phiên sau
(`/tmp/claude-1003/-home-student-Lan-K68AI-temp/02a22dbf-5493-45c5-a81f-8cbc51c58a97/scratchpad/`), cũng **ngoài repo**:

| Script | Chức năng |
|---|---|
| `idx.py` / `idx_lib.py` | Lập bảng episode 0–9 của D_nas50 với cờ thành công của D_nas1 và D_nas50 → `out/idx.txt` |
| `pair.py` | Timeline 12 khung của NAS=50 (trên) và NAS=1 (dưới) cho cùng một task/episode |
| `grid.py` | Khung bước 150 và khung cuối của các episode D_nas50 |
| `crop.py` | Cắt một vùng của khung bất kỳ, đặt cạnh nhau |

`idx.py` cũng tính chỉ số chuyển động, nhưng thu nhỏ ảnh bằng OpenCV thay vì ffmpeg nên ra trung vị 1,51 thay vì
1,27 (số thất bại < 0,3 vẫn là 10). Mục 6.5 dùng số của `motion.csv` (mục 6.1), đã chạy lại và khớp 1,27.
