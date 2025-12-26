## 🧩 4DHOI Reconstruction

This repository is a component of the implementation for the paper "Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction".

🔗 Main Repository: https://github.com/wenboran2002/open4dhoi_code

This pipeline handles 4D Human-Object Interaction (HOI) Automatic Reconstruction, including object mesh generation and full 4D HOI alignment.


## 🚀 To Do

[x] Release reconstruction pipeline

[x] Release visualization code

[ ] Release sam3d-body hand pose

### 🧰 Data Preparation

1. Download the demo file from [Google Drive](https://drive.google.com/uc?export=download&id=18-FiQL4Ew7G7_iyc_yMmcAajYKKBvQch) and place it in ./demo.

The data structure should be like this:
   ```text
   ./demo/
   ├── points.json       # annotation points used by the pipeline
   └── video.mp4         # original video
   ```

2. Go to the demo directory:
   ```bash
   cd demo
   ```

3. Extract video frames:
    ```bash
    mkdir -p frames
    ffmpeg -i video.mp4 -q:v 2 -start_number 0 frames/%05d.jpg
    ```

### 🧱 Build Object Mesh

1. Clone dependencies and set up the environment (follow each repo’s installation instructions):

   - To generate the object mesh, we use [**SAM2**](https://github.com/facebookresearch/sam2) and [**sam-3d-objects**](https://github.com/facebookresearch/sam-3d-objects) 
   for mask generation and 3D object reconstruction.

   ```bash
   cd ..
   git clone https://github.com/facebookresearch/sam2.git              # SAM 2
   git clone https://github.com/facebookresearch/sam-3d-objects.git    # SAM 3D Objects
   ```

   - Environment setup:

   ```bash
   # create env
   conda create -n reconobj python=3.11 -y
   conda activate reconobj

   # install Pytorch
   pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
      --index-url https://download.pytorch.org/whl/cu128
   ```

   Then install all required dependencies by following each repo’s installation guide.

2. Generate masks (human + object):

   ```bash
   python make_masks.py --video_dir ./demo
   ```

This will create:

* `human_mask_dir/` : human masks (per-frame)
* `mask_dir/`       : object masks (per-frame)

3. Export the object mesh:

   ```bash
   python make_obj_org.py --video_dir ./demo --config "your config path"
   ```

   Output:

   * `obj_org.obj` : the exported object mesh in the demo directory

### 🤖 Reconstruct 4D HOI

1. Prerequisites: [**GVHMR**](https://github.com/zju3dv/GVHMR), [**Orient-Anything**](https://github.com/SpatialVision/Orient-Anything), [**Depth-Anything-V2**](https://github.com/DepthAnything/Depth-Anything-V2)

   ```bash
   git clone https://github.com/zju3dv/GVHMR.git                       # GVHMR
   git clone https://github.com/SpatialVision/Orient-Anything.git      # Orient-Anything
   git clone https://github.com/DepthAnything/Depth-Anything-V2.git    # Depth Anything V2
   ```

   ⚠️ Important:

   * Download required GVHMR weights and place them in GVHMR/inputs/checkpoints/.

   * Download required Depth-Anything-V2 checkpoints and place them in Depth-Anything-V2/checkpoints/.

2. Environment setup：
   ```bash
   conda create -n reconhoi python=3.10 -y
   conda activate reconhoi

   conda install -c conda-forge pymomentum-cpu -y

   # install pytorch3d
   pip install "git+https://github.com/facebookresearch/pytorch3d.git"
   # install other required dependencies
   pip install -r requirements.txt
   ```

3. Generate human motion: 
   ```bash
   python human_motion.py --video_dir ./demo
   ```

4. Depth estimation:
   ```bash
   python make_depth.py --video_dir ./demo
   ```

5. HOI pose alignment:
   ```bash
   python hoi_pose.py --video_dir ./demo --j_regressor_path ./J_regressor.pt
   ```

After reconstruction, the demo folder should look like this:

```text
./demo/
├── frames/                     # extracted RGB frames from video.mp4
│   ├── 00000.jpg
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── 00004.jpg
│   └── ...
├── human_mask_dir/             # per-frame human segmentation masks
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   ├── 00003.png
│   ├── 00004.png
│   └── ...
├── mask_dir/                   # per-frame object segmentation masks
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   ├── 00003.png
│   ├── 00004.png
│   └── ...
├── motion/                     # GVHMR human motion reconstruction outputs
│   ├── global.mp4              # visualization of reconstructed human motion
│   └── result.pt               # SMPL-X params (per-frame), camera intrinsics, etc.
├── output/                     # final alignment / reconstruction outputs
│   ├── all_frames.pt           # per-frame combined vertices in global space (T, V_total, 3), used for rendering
│   ├── combined_faces.npy      # face indices for the combined human+object mesh (F_total, 3), aligned with all_frames.pt
│   ├── global.mp4              # global-space rendering video of the combined human + object sequence
│   ├── obj_poses.json          # estimated object pose info (e.g., scale/center or per-frame pose)
│   ├── raw_depth.npy           # raw depth cache
│   ├── 00000_h.obj             # exported human mesh (SMPL-X)
│   └── 00000_o.obj             # aligned object mesh
├── points.json                 # user-provided annotation points for the pipeline
├── video.mp4                   # original input video (raw)
├── depth.npy                   # per-frame depth maps generated by Depth-Anything-V2
└── obj_org.obj                 # reconstructed object mesh in its original coordinate/
```


### 📖 Citation
If you find this code useful for your research, please consider citing our paper:

```
@misc{wen2025efficientscalablemonocularhumanobject,
      title={Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction}, 
      author={Boran Wen and Ye Lu and Keyan Wan and Sirui Wang and Jiahong Zhou and Junxuan Liang and Xinpeng Liu and Bang Xiao and Dingbang Huang and Ruiyang Liu and Yong-Lu Li},
      year={2025},
      eprint={2512.00960},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.00960}, 
}
```