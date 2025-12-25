# -*- coding: utf-8 -*-
import os
import json
from copy import deepcopy
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
import scipy
import smplx
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from hoi_utils import align, get_scene_pcd, get_obj_pcd, get_front_points


def _result_pt_path(video_dir: str) -> str:
    p = os.path.join(video_dir, "motion", "result.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"result.pt not found (only allowed path): {p}")
    return p


def _load_depths(video_dir: str) -> np.ndarray:
    depth_path = os.path.join(video_dir, "depth.npy")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"depth.npy not found: {depth_path}")
    return np.load(depth_path)


def _load_mask(video_dir: str, t: int) -> np.ndarray:
    # prefer per-frame; fallback to 00000.png
    cand = os.path.join(video_dir, "mask_dir", f"{t:05d}.png")
    if not os.path.exists(cand):
        cand0 = os.path.join(video_dir, "mask_dir", "00000.png")
        if not os.path.exists(cand0):
            raise FileNotFoundError(f"Mask not found: {cand} and fallback {cand0} also missing.")
        cand = cand0
    m = Image.open(cand).convert("L")
    m = np.asarray(m)
    return (m == 255).astype(np.uint8)


def apply_transform_to_model(verts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    verts: (N, 3)
    T: (4, 4)
    Apply: [v,1] @ T^T
    """
    ones = np.ones((verts.shape[0], 1), dtype=verts.dtype)
    hom = np.concatenate([verts, ones], axis=1)      # (N,4)
    out = hom @ T.T                                  # (N,4)
    return out[:, :3]


def _mean_pairwise_distance(x: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    return float(np.mean(scipy.spatial.distance.cdist(x, x)))


# SMPL-X model
def build_smplx_model() -> Tuple[smplx.body_models.SMPLXLayer, np.ndarray]:
    model_type = "smplx"
    model_path = os.path.join(
        "GVHMR", "inputs", "checkpoints", "body_models", "smplx", "SMPLX_NEUTRAL.npz"
    )
    layer_arg = {
        "create_global_orient": False,
        "create_body_pose": False,
        "create_left_hand_pose": False,
        "create_right_hand_pose": False,
        "create_jaw_pose": False,
        "create_leye_pose": False,
        "create_reye_pose": False,
        "create_betas": False,
        "create_expression": False,
        "create_transl": False,
    }
    model = smplx.create(
        model_path,
        model_type=model_type,
        gender="neutral",
        num_betas=10,
        num_expression_coeffs=10,
        use_pca=False,
        use_face_contour=True,
        flat_hand_mean=True,
        **layer_arg,
    )
    return model, model.faces


# Mandatory visualization
def render_global_video(
    verts: torch.Tensor,          # (T, V, 3) on CPU
    faces: np.ndarray,            # (F, 3) int
    output_path: str,
    video_path: str,
    j_regressor_path: str,
):
    """
    Force render global.mp4 every run.
    This requires your global_utils + Renderer environment.
    """
    try:
        from einops import einsum
        import torch.nn.functional as F
        from global_utils.utils import Renderer, get_global_cameras_static, get_ground_params_from_points
        from global_utils.video_io_utils import get_video_lwh, get_writer
        from global_utils.hmr_cam import create_camera_sensor
    except Exception as e:
        raise ImportError(
            "Visualization is mandatory, but required modules are missing.\n"
            "Need: einops + global_utils (Renderer / video_io_utils / hmr_cam).\n"
            f"Original error: {repr(e)}"
        )

    def apply_T_on_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ki,...ji->...jk", T[..., :3, :3], points) + T[..., None, :3, 3]

    def transform_mat(Rm: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if len(Rm.shape) > len(t.shape):
            t = t[..., None]
        return torch.cat([F.pad(Rm, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=-1)

    def compute_T_ayfz2ay(joints: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        t_ayfz2ay = joints[:, 0, :].detach().clone()
        t_ayfz2ay[:, 1] = 0

        RL_xz_h = joints[:, 1, [0, 2]] - joints[:, 2, [0, 2]]
        RL_xz_s = joints[:, 16, [0, 2]] - joints[:, 17, [0, 2]]
        RL_xz = RL_xz_h + RL_xz_s
        I_mask = RL_xz.pow(2).sum(-1) < 1e-4

        x_dir = torch.zeros_like(t_ayfz2ay)
        x_dir[:, [0, 2]] = F.normalize(RL_xz, 2, -1)
        y_dir = torch.zeros_like(x_dir)
        y_dir[..., 1] = 1
        z_dir = torch.cross(x_dir, y_dir, dim=-1)
        R_ayfz2ay = torch.stack([x_dir, y_dir, z_dir], dim=-1)
        R_ayfz2ay[I_mask] = torch.eye(3).to(R_ayfz2ay)

        if inverse:
            R_ay2ayfz = R_ayfz2ay.transpose(1, 2)
            t_ay2ayfz = -einsum(R_ayfz2ay, t_ayfz2ay, "b i j , b i -> b j")
            return transform_mat(R_ay2ayfz, t_ay2ayfz)
        else:
            return transform_mat(R_ayfz2ay, t_ayfz2ay)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"video.mp4 not found: {video_path}")
    if not os.path.exists(j_regressor_path):
        raise FileNotFoundError(f"J_regressor not found: {j_regressor_path}")

    J_regressor = torch.load(j_regressor_path).double().cuda()

    faces_t = torch.from_numpy(faces.astype(np.int64)).cuda()
    verts = verts.double().cuda()  # (T, V, 3)

    human_vertex_count = 10475

    def move_to_start_point_face_z(verts_in: torch.Tensor) -> torch.Tensor:
        verts_in = verts_in.clone()
        human_part = verts_in[:, :human_vertex_count, :]
        offset = einsum(J_regressor, human_part[0], "j v, v i -> j i")[0]
        offset[1] = human_part[:, :, [1]].min()
        verts_in = verts_in - offset
        T_ay2ayfz = compute_T_ayfz2ay(
            einsum(J_regressor, human_part[[0]], "j v, l v i -> l j i"),
            inverse=True
        )
        verts_in = apply_T_on_points(verts_in, T_ay2ayfz)
        return verts_in

    verts_glob = move_to_start_point_face_z(verts)

    human_part = verts_glob[:, :human_vertex_count, :]
    joints_glob = einsum(J_regressor, human_part, "j v, l v i -> l j i")

    global_R, global_T, global_lights = get_global_cameras_static(
        human_part.float().cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)

    renderer = Renderer(width, height, device="cuda", faces=faces_t, K=K)
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], human_part)
    renderer.set_ground(scale * 1.5, cx, cz)

    color = torch.ones(3).float().cuda() * 0.8

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = get_writer(output_path, fps=30, crf=23)

    Tm = int(verts_glob.shape[0])
    render_len = min(length, Tm)

    for i in tqdm(range(render_len), desc="Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(
            verts_glob[[i]].float(),
            color[None],
            cameras,
            global_lights
        )
        writer.write_frame(img)

    writer.close()


def main(args):
    video_dir = args.video_dir
    select_t = int(args.select_index)
    j_regressor_path = args.j_regressor_path

    # 1) Load result.pt strictly
    result_pt = _result_pt_path(video_dir)
    output = torch.load(result_pt, map_location="cpu")

    global_param = output["smpl_params_incam"]
    G_global_param = output["smpl_params_global"]

    # 2) Load depths
    depths = _load_depths(video_dir)

    # 3) SMPL-X
    model, faces = build_smplx_model()

    # 4) Load object mesh + rotate z180
    obj_path = os.path.join(video_dir, "obj_org.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"obj_org.obj not found: {obj_path}")

    obj_pcd = o3d.io.read_triangle_mesh(obj_path)
    R_z180 = obj_pcd.get_rotation_matrix_from_xyz((0.0, 0.0, np.pi))
    obj_pcd.rotate(R_z180, center=(0, 0, 0))
    obj_pcd.compute_vertex_normals()

    overts_base = np.asarray(obj_pcd.vertices).astype(np.float32)
    ofaces = np.asarray(obj_pcd.triangles).astype(np.int32)

    # 5) Sequence length
    t_len = int(global_param["global_orient"].shape[0])
    if not (0 <= select_t < t_len):
        raise ValueError(f"select_index out of range: {select_t}, valid [0, {t_len-1}]")

    # 6) Build h_list (incam) and o_list (base overts)
    h_list: List[np.ndarray] = []
    o_list: List[np.ndarray] = []

    zero_pose = torch.zeros((1, 3)).float()
    for t in tqdm(range(t_len), desc="SMPL-X verts (incam)"):
        left_hand_pose = torch.zeros((1, 45)).float()
        right_hand_pose = torch.zeros((1, 45)).float()

        params = {
            "global_orient": global_param["global_orient"][t].reshape(1, -1),
            "body_pose": global_param["body_pose"][t].reshape(1, -1),
            "betas": global_param["betas"][t].reshape(1, -1),
            "expression": torch.zeros((1, 10)).float(),
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "jaw_pose": zero_pose,
            "leye_pose": zero_pose,
            "reye_pose": zero_pose,
            "transl": global_param["transl"][t].reshape(1, -1),
        }
        hverts = model(**params).vertices.detach().cpu().numpy()[0].astype(np.float32)
        h_list.append(hverts)
        o_list.append(overts_base.copy())

    # 7) Compute object_pose (scale + center[t]) in incam space
    # front points (for stable scale compare)
    try:
        obj_pcd_n = deepcopy(obj_pcd).simplify_quadric_decimation(target_number_of_triangles=512)
        overts_n = np.asarray(obj_pcd_n.vertices).astype(np.float32)
        overts_c = np.asarray(get_front_points(overts_n, obj_pcd)).astype(np.float32)
        if overts_c.ndim != 2 or overts_c.shape[1] != 3:
            raise ValueError("front points shape unexpected")
    except Exception:
        idx = np.random.choice(len(overts_base), min(500, len(overts_base)), replace=False)
        overts_c = overts_base[idx]

    object_pose: Dict = {"scale": 0.0, "center": []}
    scale_value: Optional[float] = None

    for t in tqdm(range(t_len), desc="Object center/scale (incam)"):
        depth = depths[t]
        obj_mask = _load_mask(video_dir, t)
        K = output["K_fullimg"][t]

        scene, _ = get_scene_pcd(depth)
        obj_pts, _ = get_obj_pcd(obj_mask, depth)

        s, b, *_ = align(scene, h_list[t], obj_mask.shape[0], obj_mask.shape[1], K)

        obj_pts = (obj_pts - b) / s
        if obj_pts.shape[0] == 0:
            if scale_value is None:
                scale_value = 1.0
                object_pose["scale"] = float(scale_value)
            center = np.mean(overts_base * scale_value, axis=0)
            object_pose["center"].append(center.tolist())
            continue

        m = min(500, obj_pts.shape[0])
        pick = np.random.choice(obj_pts.shape[0], m, replace=False)
        obj_samp = obj_pts[pick].astype(np.float32)

        if t == select_t:
            dis_b = _mean_pairwise_distance(obj_samp)
            dis_s = _mean_pairwise_distance(overts_c)
            if dis_s < 1e-8:
                raise RuntimeError("degenerate front points, cannot estimate scale")
            scale_value = dis_b / dis_s
            object_pose["scale"] = float(scale_value)

        if scale_value is None:
            scale_value = 1.0
            object_pose["scale"] = float(scale_value)

        displace = np.mean(obj_samp, axis=0) - np.mean(overts_c * scale_value, axis=0)
        center = np.mean(overts_base * scale_value, axis=0) + displace
        object_pose["center"].append(center.tolist())

    # 8) Save obj_poses.json
    output_dir = os.path.join(video_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "obj_poses.json"), "w", encoding="utf-8") as f:
        json.dump(object_pose, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved: {os.path.join(output_dir, 'obj_poses.json')}")

    # incam to global
    all_frames: List[torch.Tensor] = []
    combined_faces: Optional[np.ndarray] = None

    for t, (hverts, overts) in tqdm(enumerate(zip(h_list, o_list)), total=t_len, desc="Incam->Global + combine"):
        center = np.asarray(object_pose["center"][t], dtype=np.float32)
        scale = float(object_pose["scale"])

        overts = overts * scale
        overts = overts - np.mean(overts, axis=0)
        overts = overts + center

        axis_angle = global_param["global_orient"][t].cpu().numpy()
        R_2ori = R.from_rotvec(axis_angle).as_matrix()

        T_2ori = global_param["transl"][t].cpu().numpy().squeeze()
        T_2ori[1] -= 0.7
        T_2ori[0] += 0.13

        transformation_matrix = np.eye(4, dtype=np.float32)
        transformation_matrix[:3, :3] = R_2ori.T.astype(np.float32)
        transformation_matrix[:3, 3] = (-R_2ori.T @ T_2ori).astype(np.float32)

        hverts = apply_transform_to_model(hverts, transformation_matrix)
        overts = apply_transform_to_model(overts, transformation_matrix)

        axis_angle = G_global_param["global_orient"][t].cpu().numpy()
        global_R = R.from_rotvec(axis_angle).as_matrix()
        global_T = G_global_param["transl"][t].cpu().numpy().squeeze()
        global_T[0] -= 0.12
        global_T[1] -= 0.01
        global_T[2] += 0.03

        transformation_matrix = np.eye(4, dtype=np.float32)
        transformation_matrix[:3, :3] = global_R.astype(np.float32)
        transformation_matrix[:3, 3] = global_T.astype(np.float32)

        hverts = apply_transform_to_model(hverts, transformation_matrix)
        overts = apply_transform_to_model(overts, transformation_matrix)

        if combined_faces is None:
            num_human_verts = hverts.shape[0]
            obj_faces_offset = ofaces.astype(np.int32) + num_human_verts
            combined_faces = np.concatenate([faces.astype(np.int32), obj_faces_offset], axis=0).astype(np.int32)

        combined_verts = np.concatenate([hverts, overts], axis=0).astype(np.float32)
        all_frames.append(torch.from_numpy(combined_verts))

    assert combined_faces is not None

    all_frames_t = torch.stack(all_frames, dim=0)  # (T, V, 3)
    torch.save(all_frames_t, os.path.join(output_dir, "all_frames.pt"))
    np.save(os.path.join(output_dir, "combined_faces.npy"), combined_faces)

    print(f"[OK] Saved: {os.path.join(output_dir, 'all_frames.pt')} shape={tuple(all_frames_t.shape)}")
    print(f"[OK] Saved: {os.path.join(output_dir, 'combined_faces.npy')} shape={tuple(combined_faces.shape)}")

    # 9) MANDATORY visualization: output/global.mp4
    video_path = os.path.join(video_dir, "video.mp4")
    out_mp4 = os.path.join(output_dir, "global.mp4")
    print(f"[INFO] Rendering mandatory global video:\n  video={video_path}\n  out={out_mp4}")

    render_global_video(
        verts=all_frames_t,
        faces=combined_faces,
        output_path=out_mp4,
        video_path=video_path,
        j_regressor_path=j_regressor_path,
    )

    print(f"[OK] Saved: {out_mp4}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("HOI pose export + incam->global (hard-coded biases) + mandatory render")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--select_index", type=int, default=0)
    parser.add_argument("--j_regressor_path", type=str, required=True, help="Path to J_regressor.pt")
    args = parser.parse_args()

    main(args)
