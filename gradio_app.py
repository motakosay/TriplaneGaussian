import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
import tempfile
from functools import partial

CACHE_EXAMPLES = os.environ.get("CACHE_EXAMPLES", "0") == "1"
DEFAULT_CAM_DIST = 1.9

import gradio as gr
from image_preprocess.utils import image_preprocess, resize_image, sam_out_nosave, pred_bbox, sam_init
from gradio_splatting.backend.gradio_model3dgs import Model3DGS
from tgs.data import CustomImageOrbitDataset
from tgs.utils.misc import todevice
from tgs.utils.config import ExperimentConfig, load_config
from infer import TGS

# NOTE: removed automatic hf_hub_download to avoid Hugging Face hub dependency.
# Place your checkpoint at 'checkpoints/model_lvis_rel.ckpt' or set env var MODEL_CKPT_PATH.
MODEL_CKPT_PATH = os.environ.get("MODEL_CKPT_PATH", "checkpoints/model_lvis_rel.ckpt")
# MODEL_CKPT_PATH = "checkpoints/model_lvis_rel.ckpt"
SAM_CKPT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
CONFIG = "config.yaml"
EXP_ROOT_DIR = "./outputs-gradio"

os.makedirs(EXP_ROOT_DIR, exist_ok=True)

gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
device = "cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu"

print("device: ", device)

# init model
base_cfg: ExperimentConfig
base_cfg = load_config(CONFIG, cli_args=[], n_gpus=1)
base_cfg.system.weights = MODEL_CKPT_PATH
model = TGS(cfg=base_cfg.system).to(device)
print("load model ckpt done.")

HEADER = """
# Triplane Meets Gaussian Splatting: Fast and Generalizable Single-View 3D Reconstruction with Transformers

<div>
</div>

TGS enables fast reconstruction from single-view image in a few seconds based on a hybrid Triplane-Gaussian 3D representation.

This model is trained on Objaverse-LVIS (**~45K** synthetic objects) only. And note that we normalize the input camera pose to a pre-set viewpoint during training stage following LRM, rather than directly using camera pose of input camera as implemented in our original paper.

**Tips:**
1. If you find the result is unsatisfied, please try to change the camera distance. It perhaps improves the results.

**Notes:**
1. Please wait until the completion of the reconstruction of the previous model before proceeding with the next one, otherwise, it may cause bug. We will fix it soon.
2. We currently conduct image segmentation (SAM) by invoking subprocess, which consumes more time as it requires loading SAM checkpoint each time. We have observed that directly running SAM in app.py often leads to queue blocking, but we haven't identified the cause yet. We plan to fix this issue for faster segmentation running time later. 
"""

def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")

def preprocess(input_raw, sam_predictor=None):
    save_path = model.get_save_path("seg_rgba.png")
    input_raw = resize_image(input_raw, 512)
    image_sam = sam_out_nosave(
        sam_predictor, input_raw.convert("RGB"), pred_bbox(input_raw)
    )
