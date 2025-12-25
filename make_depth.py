# -*- coding: utf-8 -*-

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_frames(frames_dir: Path):
    exts = (".png", ".jpg", ".jpeg")
    paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    def _key(p: Path):
        try:
            return int(p.stem)
        except Exception:
            return p.stem

    return sorted(paths, key=_key)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Demo sequence directory",
    )
    return p.parse_args()

# Resolve Depth-Anything-V2 root from ./Depth-Anything-V2 (script dir or cwd)
_SCRIPT_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

DA_ROOT = (_SCRIPT_DIR / "Depth-Anything-V2").resolve()
if not DA_ROOT.exists():
    DA_ROOT = (_CWD / "Depth-Anything-V2").resolve()

assert DA_ROOT.exists(), (
    f"Depth-Anything-V2 not found.\n"
    f"Expected at: {_SCRIPT_DIR / 'Depth-Anything-V2'} or {_CWD / 'Depth-Anything-V2'}\n"
    f"Current script dir: {_SCRIPT_DIR}\n"
    f"Current working dir: {_CWD}"
)

# Add repo to sys.path so `import depth_anything_v2...` works
sys.path.insert(0, str(DA_ROOT))

# Verify package import target exists (best-effort sanity check)
DA_PKG_DIR = (DA_ROOT / "depth_anything_v2").resolve()
assert DA_PKG_DIR.exists(), (
    f"Depth-Anything-V2 python package not found.\n"
    f"Expected package dir at: {DA_PKG_DIR}\n"
    f"DA_ROOT: {DA_ROOT}"
)

def main():
    args = parse_args()
    demo_dir = Path(args.video_dir).resolve()
    frames_dir = demo_dir / "frames"
    select_json = demo_dir / "select_id.json"

    if not demo_dir.exists():
        raise RuntimeError(f"video_dir not found: {demo_dir}")
    if not frames_dir.exists():
        raise RuntimeError(f"frames folder not found: {frames_dir}")
    if not select_json.exists():
        raise RuntimeError(f"select_id.json not found: {select_json}")

    # ---- Depth-Anything-V2 import ----
    try:
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import DepthAnythingV2 from {DA_ROOT}: {repr(e)}")

    # ---- Choose encoder + checkpoint ----
    encoder = "vitl"
    ckpt_path = DA_ROOT / "checkpoints" / "depth_anything_v2_vitl.pth"
    if not ckpt_path.exists():
        raise RuntimeError(f"Depth-Anything-V2 checkpoint not found: {ckpt_path}")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    if encoder not in model_configs:
        raise RuntimeError(f"Unknown encoder: {encoder}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DepthAnythingV2(**model_configs[encoder])
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    frame_paths = _list_frames(frames_dir)
    T = len(frame_paths)

    sel = _load_json(select_json)
    select_id = int(sel.get("select_id", 0))
    if not (0 <= select_id < T):
        raise RuntimeError(f"select_id out of range: {select_id}, frames={T}")

    depths = []
    input_size = 518  # official common setting; infer_image will resize internally

    for i, fp in enumerate(frame_paths):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame: {fp}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            d = model.infer_image(rgb, input_size=input_size)  # HxW np
            d = np.asarray(d, dtype=np.float32)

        depths.append(d)

        if (i + 1) % 25 == 0 or (i + 1) == T:
            print(f"[Depth] {i+1}/{T} done")

    depth_arr = np.stack(depths, axis=0).astype(np.float32)  # (T,H,W)

    # save full sequence depth
    depth_npy = demo_dir / "depth.npy"
    np.save(str(depth_npy), depth_arr)

    # save select frame depth for hoi_pose's replace_depth() branch
    out_dir = demo_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_depth_npy = out_dir / "raw_depth.npy"
    np.save(str(raw_depth_npy), depth_arr[select_id])

    print(f"[OK] Saved: {depth_npy}  shape={depth_arr.shape} dtype={depth_arr.dtype}")
    print(f"[OK] Saved: {raw_depth_npy}  frame={select_id}")


if __name__ == "__main__":
    main()
