# -*- coding: utf-8 -*-
import os
import os.path
import sys
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path

# Resolve GVHMR root from ./GVHMR (relative to script or cwd)
_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

GVHMR_ROOT = (_SCRIPT_DIR / "GVHMR").resolve()
if not GVHMR_ROOT.exists():
    GVHMR_ROOT = (_CWD / "GVHMR").resolve()

assert GVHMR_ROOT.exists(), (
    f"GVHMR not found.\n"
    f"Expected at: {_SCRIPT_DIR / 'GVHMR'} or {_CWD / 'GVHMR'}\n"
    f"Current script dir: {_SCRIPT_DIR}\n"
    f"Current working dir: {_CWD}"
)

# ultralytics yolov8x.pt is a pickle checkpoint, blocked by weights_only=True default.
# We force weights_only=False by default BEFORE Tracker/YOLO is imported.
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat  # type: ignore

# Make GVHMR importable
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))

from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_dir, compose
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default=None, help="By default uses GVHMR config output_root")
    parser.add_argument("-s", "--static_cam", action="store_true", help="If true, skip VO (slam)")
    parser.add_argument("--use_dpvo", action="store_true", help="If true, use DPVO. By default not using DPVO.")
    parser.add_argument("--f_mm", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Folder containing video.mp4",
    )
    args = parser.parse_args()

    # Always use video.mp4
    video_path = Path(args.video_dir) / "video.mp4"
    assert video_path.exists(), f"Video not found at {video_path}"

    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input] {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    # Use filesystem config dir (avoid pkg:// missing config issues)
    cfg_dir = (GVHMR_ROOT / "hmr4d" / "configs").resolve()
    assert cfg_dir.exists(), f"GVHMR config dir not found: {cfg_dir}"

    with initialize_config_dir(version_base="1.3", config_dir=str(cfg_dir)):
        overrides = [
            f"video_name={Path(args.video_dir).resolve().name}",
            f"static_cam={args.static_cam}",
            f"verbose={args.verbose}",
            f"use_dpvo={args.use_dpvo}",
            f"+video_dir={args.video_dir}",  # add new key into structured config
            f"video_path={str(video_path)}",
        ]
        if args.f_mm is not None:
            overrides.append(f"f_mm={args.f_mm}")
        if args.output_root is not None:
            overrides.append(f"output_root={args.output_root}")

        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info("[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3)
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")

    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")

    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get visual odometry results
    if not static_cam:
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()
                torch.save(vo_results, paths.slam)
            else:
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
        T_w2c = torch.zeros(length, 3)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
            T_w2c = torch.from_numpy(traj[:, :3])
        else:
            R_w2c = torch.from_numpy(traj[:, :3, :3])
            T_w2c = torch.from_numpy(traj[:, :3, 3])

    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
    }
    return data


def render_global(cfg, global_path, result_path):
    global_video_path = Path(global_path)

    pred = torch.load(result_path)
    smplx = make_smplx("supermotion").cuda()

    smplx2smpl_path = (GVHMR_ROOT / "hmr4d" / "utils" / "body_model" / "smplx2smpl_sparse.pt").resolve()
    J_reg_path = (GVHMR_ROOT / "hmr4d" / "utils" / "body_model" / "smpl_neutral_J_regressor.pt").resolve()

    smplx2smpl = torch.load(str(smplx2smpl_path)).cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(str(J_reg_path)).cuda()

    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        verts = verts.clone()
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")

    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=0,
        target_center_height=1.0,
    )
    pred["global_R"] = global_R.cpu()
    pred["global_T"] = global_T.cpu()
    torch.save(pred, result_path)

    length, width, height = get_video_lwh(cfg.video_path)
    _, _, K = create_camera_sensor(width, height, 24)

    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(length), desc="Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()


if __name__ == "__main__":
    # Ensure PROJ_ROOT-relative paths resolve correctly inside GVHMR
    os.chdir(str(GVHMR_ROOT))

    cfg = parse_args_to_cfg()
    Log.info(f"[GPU]: {torch.cuda.get_device_name()}")

    motion_output_dir = str(Path(cfg.video_dir) / "motion")
    if not os.path.exists(motion_output_dir):
        os.makedirs(motion_output_dir)

    run_preprocess(cfg)
    data = load_data_dict(cfg)

    result_path = os.path.join(motion_output_dir, "result.pt")
    global_path = os.path.join(motion_output_dir, "global.mp4")

    Log.info("[HMR4D] Predicting")
    model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(cfg.ckpt_path)
    model = model.eval().cuda()

    tic = Log.sync_time()
    pred = model.predict(data, static_cam=cfg.static_cam)
    pred = detach_to_cpu(pred)
    Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s")
    torch.save(pred, result_path)
    render_global(cfg, global_path, result_path)
