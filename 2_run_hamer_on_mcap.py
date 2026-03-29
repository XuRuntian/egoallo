import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
import cv2  # 在文件顶部添加
# ==========================================
# 核心修复：把 HaMeR 的源码目录加入 Python 搜索路径
# ==========================================
# 推测你的 hamer 源码克隆在了 /home/user/hamer 目录下
hamer_src_path = "/home/user/hamer" 
if hamer_src_path not in sys.path:
    sys.path.insert(0, hamer_src_path)
# ==========================================

from hamer_helper import HamerHelper 

# 导入你的 EgoAdapter
project_src = "/home/user/robo-auto-annation/src"
if project_src not in sys.path:
    sys.path.insert(0, project_src)
from adapters.ego_adapter import EgoAdapter


def process_mcap_for_hands(mcap_path, output_pkl_path, focal_length=470.0):
    adapter = EgoAdapter()
    if not adapter.load(mcap_path): 
        print("❌ 无法加载 MCAP 文件！")
        return

    hamer_helper = HamerHelper() 

    detections_left = {}
    detections_right = {}

    # [qx,qy,qz,qw, x,y,z]
    T_device_cam = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32) 
    T_cpf_cam = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)

    print(f"🚀 开始运行 HaMeR 检测手部 (共 {adapter.get_length()} 帧)...")
    for i in tqdm(range(adapter.get_length())):
        frame = adapter.get_frame(i)
        if frame is None: continue

        img = frame.images['camera0']
        timestamp_ns = int(frame.timestamp * 1e9) 
        out_left, out_right = hamer_helper.look_for_hands(img, focal_length=focal_length)

        # 提取左手数据
        if out_left is not None:
            detections_left[timestamp_ns] = {
                "verts": out_left["verts"], 
                "keypoints_3d": out_left["keypoints_3d"], 
                "mano_hand_pose": out_left["mano_hand_pose"],
                "mano_hand_betas": out_left["mano_hand_betas"],
                "mano_hand_global_orient": out_left["mano_hand_global_orient"],
            }
        else:
            detections_left[timestamp_ns] = None

        # 提取右手数据
        if out_right is not None:
            detections_right[timestamp_ns] = {
                "verts": out_right["verts"], 
                "keypoints_3d": out_right["keypoints_3d"], 
                "mano_hand_pose": out_right["mano_hand_pose"],
                "mano_hand_betas": out_right["mano_hand_betas"],
                "mano_hand_global_orient": out_right["mano_hand_global_orient"],
            }
        else:
            detections_right[timestamp_ns] = None

    # 保存结果
    outputs_dict = {
        "mano_faces_right": hamer_helper.get_mano_faces("right"),
        "mano_faces_left": hamer_helper.get_mano_faces("left"),
        "detections_right_wrt_cam": detections_right,
        "detections_left_wrt_cam": detections_left,
        "T_device_cam": T_device_cam,
        "T_cpf_cam": T_cpf_cam,
    }

    with open(output_pkl_path, "wb") as f:
        pickle.dump(outputs_dict, f)
    print(f"✅ 手部检测完成，已保存至 {output_pkl_path}")

# ==========================================
# 程序的执行入口
# ==========================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="运行 HaMeR 提取 MCAP 中的手部动作")
    parser.add_argument("--mcap_path", type=str, default="ego1.mcap", help="你的 mcap 视频路径")
    parser.add_argument("--output", type=str, default="hamer_outputs.pkl", help="输出的 pkl 路径")
    parser.add_argument("--focal_length", type=float, default=512.17, help="相机的像素焦距")

    args = parser.parse_args()

    process_mcap_for_hands(
        mcap_path=args.mcap_path, 
        output_pkl_path=args.output, 
        focal_length=args.focal_length
    )