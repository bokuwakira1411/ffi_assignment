#!/bin/bash
# Chỉ chạy khi GPU rảnh. Mỗi lệnh replay các episode trong manual_check.csv và ghi video vào
# /home/student/Lan_K68AI/temp/ffi26/ffi26/improve/outputs/failure_labels/videos/<config>/. Kiểm tra tất định: replay_check_<arm>.csv (so với file eval gốc).
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && cd ~/ffi26/improve
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python scripts/phase3_eval_video.py --config /home/student/Lan_K68AI/temp/ffi26/ffi26/improve/outputs/failure_labels/render/fpo_v1_it3_s3k5k.yaml --blind  # 7 episode
python scripts/phase3_eval_video.py --config /home/student/Lan_K68AI/temp/ffi26/ffi26/improve/outputs/failure_labels/render/fpo_v2_sel_s1000.yaml --blind  # 2 episode
python scripts/phase3_eval_video.py --config /home/student/Lan_K68AI/temp/ffi26/ffi26/improve/outputs/failure_labels/render/fpo_v2_sel_s3k5k.yaml --blind  # 4 episode
python scripts/phase3_eval_video.py --config /home/student/Lan_K68AI/temp/ffi26/ffi26/improve/outputs/failure_labels/render/official_s1000.yaml --blind  # 1 episode
python scripts/phase3_eval_video.py --config /home/student/Lan_K68AI/temp/ffi26/ffi26/improve/outputs/failure_labels/render/official_s3k5k.yaml --blind  # 6 episode
