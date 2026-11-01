# Video Improve

Mỗi video replay chính xác một episode của eval gốc: cùng init state và seed. Bên trái là camera agent, bên phải là camera cổ tay.
Tên file: `{phương pháp}_t{task}_init{k}_s{seed}_{success|fail}.mp4`.

| Thư mục | Nội dung | Mở trước |
|---|---|---|
| `3_v4_paired/` | official, FPO v3, FPO v4 trên cùng 8 episode (task 0, 8) | `sidebyside/` (3 video cạnh nhau), `keyframes/` |
| `1_v1_v2/` | official, FPO v1, FPO v2 trên 9 episode held-out (task 0, 8) | `index.md` |
| `2_v2_forget/` | FPO v2 thất bại ở task 1, 7: gắp nhầm vật của task 0 | `index.md` |

Mỗi thư mục có một thư mục con cho từng phương pháp, chứa video của phương pháp đó.
`_meta/` chứa danh sách episode, kết quả kiểm tra replay, log và config; không cần mở khi xem video.
Video được sinh bằng `improve/scripts/phase3_eval_video.py`.
