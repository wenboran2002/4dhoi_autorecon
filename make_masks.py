# -*- coding: utf-8 -*-
import sys
import json
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch

# Resolve SAM2 root from ./sam2 (script dir or cwd)
_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

SAM2_ROOT = (_SCRIPT_DIR / "sam2").resolve()
if not SAM2_ROOT.exists():
    SAM2_ROOT = (_CWD / "sam2").resolve()

assert SAM2_ROOT.exists(), (
    f"sam2 not found.\n"
    f"Expected at: {_SCRIPT_DIR / 'sam2'} or {_CWD / 'sam2'}\n"
    f"Current script dir: {_SCRIPT_DIR}\n"
    f"Current working dir: {_CWD}"
)

SAM2_PKG_DIR = (SAM2_ROOT / "sam2").resolve()
assert (SAM2_PKG_DIR / "build_sam.py").exists(), (
    f"SAM2 python package not found or invalid.\n"
    f"Expected build_sam.py at: {SAM2_PKG_DIR / 'build_sam.py'}\n"
    f"SAM2_ROOT: {SAM2_ROOT}"
)

sys.path.insert(0, str(SAM2_ROOT))


def _save_mask(mask_any: np.ndarray, out_path: Path):
    """
    Save mask as 8-bit PNG.

    SAM2 may output masks with an extra singleton dimension, e.g. (1, H, W).
    We squeeze it to (H, W) before saving.
    """
    m = np.asarray(mask_any)

    # Common SAM2 shapes: (H,W) or (1,H,W) or (H,W,1)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        raise RuntimeError(f"Unexpected mask shape {m.shape} for {out_path}")

    img = (m.astype(np.uint8) * 255)
    Image.fromarray(img, mode="L").save(str(out_path), compress_level=0)


def _resolve_cfg_and_ckpt():
    """
    Hydra search path is pkg://sam2, so config_name must be relative to SAM2_PKG_DIR.
    That means config_name must start with "configs/..."
    """
    cfg_abs = SAM2_PKG_DIR / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    if not cfg_abs.exists():
        raise RuntimeError(f"SAM2 config file not found: {cfg_abs}")

    cfg_name = "configs/sam2.1/sam2.1_hiera_l.yaml"

    ckpt_abs = SAM2_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
    if not ckpt_abs.exists():
        raise RuntimeError(f"SAM2 checkpoint not found: {ckpt_abs}")

    return cfg_name, ckpt_abs


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_dir",
        required=True,
        type=str,
        help="Demo directory containing frames/ and points.json.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    demo_dir = Path(args.video_dir).expanduser().resolve()
    if not demo_dir.exists():
        raise RuntimeError(f"video_dir not found: {demo_dir}")

    frames_dir = demo_dir / "frames"
    if not frames_dir.exists():
        raise RuntimeError(f"frames directory not found: {frames_dir}")

    num_frames = (
        len(list(frames_dir.glob("*.jpg")))
        + len(list(frames_dir.glob("*.jpeg")))
        + len(list(frames_dir.glob("*.png")))
    )
    if num_frames == 0:
        raise RuntimeError(f"No frame images found under: {frames_dir} (expected .jpg/.jpeg/.png)")

    points_path = demo_dir / "points.json"
    if not points_path.exists():
        raise RuntimeError(f"points.json not found: {points_path}")

    # outputs
    out_obj_dir = demo_dir / "mask_dir"
    out_human_dir = demo_dir / "human_mask_dir"
    out_obj_dir.mkdir(parents=True, exist_ok=True)
    out_human_dir.mkdir(parents=True, exist_ok=True)

    # load points
    data = json.load(open(points_path, "r", encoding="utf-8"))
    human_pts = np.asarray(data["human_points"], dtype=np.float32)
    object_pts = np.asarray(data["object_points"], dtype=np.float32)

    human_lbl = np.ones((len(human_pts),), dtype=np.int32)
    object_lbl = np.ones((len(object_pts),), dtype=np.int32)

    prompt_frame = 0

    cfg_name, ckpt_path = _resolve_cfg_and_ckpt()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] SAM2_ROOT    : {SAM2_ROOT}")
    print(f"[INFO] cfg_name     : {cfg_name}")
    print(f"[INFO] ckpt_path    : {ckpt_path}")
    print(f"[INFO] device       : {device}")
    print(f"[INFO] video_dir    : {demo_dir}")
    print(f"[INFO] frames_dir   : {frames_dir}")
    print(f"[INFO] prompt_frame : {prompt_frame}")

    from sam2.build_sam import build_sam2_video_predictor

    predictor = build_sam2_video_predictor(cfg_name, str(ckpt_path))

    obj_id_object = 1
    obj_id_human = 2

    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    with torch.inference_mode(), autocast_ctx:
        state = predictor.init_state(video_path=str(frames_dir))

        predictor.add_new_points_or_box(
            state,
            frame_idx=prompt_frame,
            obj_id=obj_id_object,
            points=object_pts,
            labels=object_lbl,
        )
        predictor.add_new_points_or_box(
            state,
            frame_idx=prompt_frame,
            obj_id=obj_id_human,
            points=human_pts,
            labels=human_lbl,
        )

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            obj_ids = list(obj_ids)
            for i, oid in enumerate(obj_ids):
                m = (mask_logits[i] > 0).detach().cpu().numpy()
                if oid == obj_id_object:
                    _save_mask(m, out_obj_dir / f"{frame_idx:05d}.png")
                elif oid == obj_id_human:
                    _save_mask(m, out_human_dir / f"{frame_idx:05d}.png")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
