import dataclasses
import time
import math  
from pathlib import Path
import numpy as np
import torch
import viser
import yaml
import pickle
from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.inference_utils import load_denoiser
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3  
from egoallo.vis_helpers import visualize_traj_and_hand_detections
from egoallo.hand_detection_structs import (
    CorrespondedHamerDetections, 
    SavedHamerOutputs,
)

# 导入你迁移过来的 EgoAdapter
import sys
import os
project_src = "/home/user/robo-auto-annation/src"
if project_src not in sys.path:
    sys.path.insert(0, project_src)

try:
    from adapters.ego_adapter import EgoAdapter
    print("✅ 成功通过 adapters.ego_adapter 导入")
except ImportError as e:
    print(f"❌ 导入失败，错误信息: {e}")

@dataclasses.dataclass
class Args:
    mcap_path: str = "ego1.mcap"
    checkpoint_dir: Path = Path("./egoallo_checkpoint_april13/checkpoints_3000000/")
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")

    # === 新增：引入手部视觉检测结果 ===
    hamer_pkl_path: str | None = "hamer_outputs.pkl" 

    # === 新增：保存动作数据的相关参数 ===
    save_traj: bool = True
    out_dir: str = "egoallo_outputs"

    start_index: int = 0
    traj_length: int = 1280     
    num_samples: int = 1

    guidance_mode: str = "aria_hamer" # 使用视觉引导
    guidance_inner: bool = True       # 必须为 True 才能让视觉生效
    guidance_post: bool = True
    visualize_traj: bool = True 

    # 用于矫正相机朝向的欧拉角 (度数)
    rot_x_deg: float = 0.0  
    rot_y_deg: float = 90.0
    rot_z_deg: float = 180.0

def main(args: Args) -> None:
    device = torch.device("cuda")

    # ==========================================
    # 1. 使用 EgoAdapter 加载数据并提取原始轨迹
    # ==========================================
    adapter = EgoAdapter()
    if not adapter.load(args.mcap_path):
        print("❌ MCAP 加载失败")
        return

    print(f"✅ 成功加载 MCAP，总帧数: {adapter.get_length()}")

    raw_poses = []
    pose_timestamps_sec = []
    end_index = min(args.start_index + args.traj_length + 1, adapter.get_length())

    for i in range(args.start_index, end_index):
        frame = adapter.get_frame(i)
        if frame is None or 'qpos' not in frame.state:
            continue

        qpos = frame.state['qpos'] # [x, y, z, qx, qy, qz, qw]
        x, y, z, qx, qy, qz, qw = qpos
        # 转换为 EgoAllo 需要的 [qw, qx, qy, qz, x, y, z] 排布
        raw_poses.append([qw, qx, qy, qz, x, y, z])
        pose_timestamps_sec.append(frame.timestamp)

    # ==========================================
    # 2. 坐标系旋转对齐
    # ==========================================
    Ts_raw = SE3(torch.tensor(raw_poses, dtype=torch.float32, device=device))

    rx = math.radians(args.rot_x_deg)
    ry = math.radians(args.rot_y_deg)
    rz = math.radians(args.rot_z_deg)

    R_align = (
        SO3.from_x_radians(torch.tensor(rx, device=device)) @
        SO3.from_y_radians(torch.tensor(ry, device=device)) @
        SO3.from_z_radians(torch.tensor(rz, device=device))
    )

    T_align = SE3.from_rotation(R_align)
    Ts_world_cpf = (Ts_raw @ T_align).parameters()

    print(f"📦 提取并对齐后的轨迹形状: {Ts_world_cpf.shape}")
    print(f"📐 矫正参数: X={args.rot_x_deg}°, Y={args.rot_y_deg}°, Z={args.rot_z_deg}°")

    # ==========================================
    # 🌟 新增：加载手部视觉先验 (HaMeR)
    # ==========================================
    if args.hamer_pkl_path is not None and Path(args.hamer_pkl_path).exists():
        print(f"👀 正在加载手部视觉先验: {args.hamer_pkl_path}")
        hamer_detections = CorrespondedHamerDetections.load(
            Path(args.hamer_pkl_path),
            pose_timestamps_sec, 
        ).to(device)
        print("✅ HaMeR 数据加载并对齐成功")
    else:
        print("⚠️ 未找到 HaMeR 手部检测结果，手部将不使用视觉引导进行盲猜。")
        hamer_detections = None

    # ==========================================
    # 3. 地平面与模型加载
    # ==========================================
    floor_z = -0.75
    points_data = np.zeros((10, 3)) 

    print("⏳ 正在加载模型...")
    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    # ==========================================
    # 4. 推理采样 (扩散模型 + 引导优化)
    # ==========================================
    print("🚀 开始推理采样...")
    server = viser.ViserServer() if args.visualize_traj else None

    traj = run_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode=args.guidance_mode,
        guidance_inner=args.guidance_inner,
        guidance_post=args.guidance_post,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=hamer_detections, # 👈 传入真正的检测结果参与优化
        aria_detections=None,
        num_samples=args.num_samples,
        device=device,
        floor_z=floor_z,
    )

    # ==========================================
    # 🌟 新增：保存动作结果到文件 (方便下游使用)
    # ==========================================
    if args.save_traj:
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        save_name = time.strftime("%Y%m%d-%H%M%S") + f"_mcap_{args.start_index}-{end_index}"
        out_path = out_root / (save_name + ".npz")

        # 将局部预测映射到全身计算真实的世界坐标根节点
        posed = traj.apply_to_body(body_model)
        Ts_world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            posed, Ts_world_cpf[..., 1:, :]
        )

        print(f"💾 正在保存动作结果至 {out_path} ...")
        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].numpy(force=True),
            Ts_world_root=Ts_world_root.numpy(force=True),
            body_quats=posed.local_quats[..., :21, :].numpy(force=True),
            left_hand_quats=posed.local_quats[..., 21:36, :].numpy(force=True),
            right_hand_quats=posed.local_quats[..., 36:51, :].numpy(force=True),
            contacts=traj.contacts.numpy(force=True),
            betas=traj.betas.numpy(force=True), # 身高体型参数
            timestamps_ns=(np.array(pose_timestamps_sec) * 1e9).astype(np.int64),
        )

        # 顺便把这此运行的参数也保存下来，方便回溯
        (out_root / (save_name + "_args.yaml")).write_text(yaml.dump(dataclasses.asdict(args)))
        print("✅ 保存完成！")

    # ==========================================
    # 5. 可视化
    # ==========================================
    if args.visualize_traj and server is not None:
        server.gui.configure_theme(dark_mode=True)
        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf[1:],
            traj,
            body_model,
            hamer_detections, # 👈 可视化中也会显示检测出来的蓝色小手作为对比
            None,
            points_data=points_data,
            splat_path=None,
            floor_z=floor_z,
        )
        print("🌐 可视化运行中...")
        while True:
            loop_cb()

if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))