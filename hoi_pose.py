import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU device to use
import torch
import numpy as np
import smplx
import json
import scipy
import open3d as o3d
import cv2
import shutil
import trimesh
# from hoi_utils import align, get_scene_pcd, get_obj_pcd,load_transformation_matrix,replace_depth,get_front_points,process_depth2square,process_frame2square
from hoi_utils import align, get_scene_pcd, get_obj_pcd, load_transformation_matrix, replace_depth, get_front_points
from PIL import Image
import argparse
from tqdm import tqdm
# from get_obj_pose import get_transformation_matrix, try_rotate

from scipy.spatial.transform import Rotation as R
from copy import deepcopy


# -------------------------
# SMPL-X model (relative path)
# -------------------------
model_type = 'smplx'
model_folder = os.path.join(
    "GVHMR", "inputs", "checkpoints", "body_models", "smplx", "SMPLX_NEUTRAL.npz"
)
layer_arg = {
    'create_global_orient': False,
    'create_body_pose': False,
    'create_left_hand_pose': False,
    'create_right_hand_pose': False,
    'create_jaw_pose': False,
    'create_leye_pose': False,
    'create_reye_pose': False,
    'create_betas': False,
    'create_expression': False,
    'create_transl': False
}

model = smplx.create(
    model_folder,
    model_type=model_type,
    gender='neutral',
    num_betas=10,
    num_expression_coeffs=10,
    use_pca=False,
    use_face_contour=True,
    flat_hand_mean=True,
    **layer_arg
)
faces = model.faces


def _resolve_result_pt(video_dir: str) -> str:
    """Prefer motion/result.pt; fallback to video_dir/result.pt."""
    cand1 = os.path.join(video_dir, "motion", "result.pt")
    cand2 = os.path.join(video_dir, "result.pt")
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    raise FileNotFoundError(
        "result.pt not found. Tried:\n"
        f"  1) {cand1}\n"
        f"  2) {cand2}\n"
    )


def main(args):
    result_pt_path = _resolve_result_pt(args.video_dir)
    output = torch.load(result_pt_path, map_location='cpu')

    global_param = output['smpl_params_incam']

    depths = np.load(os.path.join(args.video_dir, "depth.npy"))  # 深度图数据

    Rx, Ry, Rz = np.array(np.eye(3)), np.array(np.eye(3)), np.array(np.eye(3))
    obj_org = o3d.io.read_triangle_mesh(os.path.join(args.video_dir, 'obj_org.obj'))

    # Rotate 180 degrees around Z-axis
    R_z180 = obj_org.get_rotation_matrix_from_xyz((0, 0, np.pi))
    obj_org.rotate(R_z180, center=(0, 0, 0))

    select_t = args.select_index
    object_pose = {}
    object_pose['scale'] = 0
    object_pose['center'] = []
    output_dir = os.path.join(args.video_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    h_list = []
    o_list = []

    t = select_t
    if True:
        obj_pcd = deepcopy(obj_org)

        obj_mask = Image.open(os.path.join(args.video_dir, "mask_dir", "00000.png")).convert("L")
        obj_mask = np.asarray(obj_mask)
        obj_mask = np.asarray(obj_mask == 255).astype(np.uint8)

        overts = np.asarray(obj_pcd.vertices).copy()
        overts_save = np.asarray(obj_pcd.vertices).copy()
        o_list.append(overts_save)

        obj_pcd.compute_vertex_normals()

        depth = depths[t]
        left_hand_pose = torch.zeros((1, 15 * 3)).float()
        right_hand_pose = torch.zeros((1, 15 * 3)).float()
        body_pose = global_param['body_pose']
        global_orient = global_param['global_orient']
        betas = global_param['betas']
        transl = global_param['transl']
        zero_pose = torch.zeros((1, 3)).float().repeat(1, 1)

        params = {
            "global_orient": global_orient[t].reshape(1, -1),
            "body_pose": body_pose[t].reshape(1, -1),
            "betas": betas[t].reshape(1, -1),
            "expression": torch.zeros((1, 10)).float(),
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "jaw_pose": zero_pose,
            "leye_pose": zero_pose,
            "reye_pose": zero_pose,
            "transl": transl[t].reshape(1, -1)
        }

        hverts = model(**params).vertices.detach().cpu().numpy()[0]
        h_list.append(hverts)

        K = output["K_fullimg"][t]

        scene, index_s = get_scene_pcd(depth)
        obj, index_o = get_obj_pcd(obj_mask, depth)
        s, b, front_s, front_h, pid = align(scene, hverts, obj_mask.shape[0], obj_mask.shape[1], K)

        if t == select_t:
            obj_pcd_n = obj_pcd.simplify_quadric_decimation(target_number_of_triangles=512)
            overts_n = np.asarray(obj_pcd_n.vertices)
            overts_c = get_front_points(overts_n, obj_pcd)
        else:
            indices = np.random.choice(len(overts), 500)
            overts_c = overts[indices]

        obj -= b
        obj /= s

        oindices = np.random.choice(len(obj), 500)
        obj = obj[oindices]

        dis_b = np.mean(scipy.spatial.distance.cdist(obj, obj))
        dis_s = np.mean(scipy.spatial.distance.cdist(overts_c, overts_c))
        scale = dis_b / dis_s

        overts *= scale
        overts_c *= scale

        displace = np.mean(obj, axis=0) - np.mean(overts_c, axis=0)
        overts += displace
        overts_c += displace

        center = np.mean(overts, axis=0)

        object_pose['scale'] = float(scale)
        object_pose['center'].append(center)

        o_list.append(overts)
        object_pose['x'] = 0.0
        object_pose['y'] = 0.0
        object_pose['z'] = 0.0

        obj_pcd.vertices = o3d.utility.Vector3dVector(overts)
        obj_output_path = os.path.join(args.video_dir, f'output/{str(t).zfill(5)}_o.obj')
        o3d.io.write_triangle_mesh(obj_output_path, obj_pcd)
        print(f"[OK] Object mesh saved: {obj_output_path}")

        h_save = o3d.geometry.TriangleMesh()
        h_save.vertices = o3d.utility.Vector3dVector(hverts)
        h_save.triangles = o3d.utility.Vector3iVector(faces)
        h_save.compute_vertex_normals()
        h_output_path = os.path.join(args.video_dir, f'output/{str(t).zfill(5)}_h.obj')
        o3d.io.write_triangle_mesh(h_output_path, h_save)
        print(f"[OK] Human mesh saved: {h_output_path}")

    object_pose['center'] = np.array(object_pose['center']).tolist()

    with open(os.path.join(output_dir, "obj_poses.json"), 'w') as f:
        json.dump(object_pose, f)
    print(f"[OK] Object pose json saved: {os.path.join(output_dir, 'obj_poses.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process 3D objects.')
    parser.add_argument('--video_dir', type=str, required=True, help='Base directory containing the objects')
    parser.add_argument('--select_index', type=int, default=0, help='Index of the selected frame')
    args = parser.parse_args()
    main(args)
