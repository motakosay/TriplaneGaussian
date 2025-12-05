#!/usr/bin/env python3
"""
infer.py

Updated inference entrypoint for TriplaneGaussian.

This version attempts to be runnable from a notebook or when the current working
directory is not the repository root by automatically adding the repository root
to sys.path before importing the project's `tgs` package.

Usage example (notebook/cell):
!python infer.py --config config.yaml data.image_list=[/content/thread_00.jpg,] --image_preprocess

Notes:
- Place your model checkpoint locally (default: checkpoints/model_lvis_rel.ckpt) or set the
  MODEL_CKPT_PATH environment variable.
- This script expects the tgs package to be available under the repository root.
"""

import os
import re
import sys
import argparse
import subprocess
from typing import Any

# Ensure the repository root (the directory containing this file) is on sys.path
# This helps running infer.py from a notebook or arbitrary working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# If the code is inside a subdirectory (rare), also add parent as a fallback.
REPO_ROOT = SCRIPT_DIR
PARENT = os.path.dirname(SCRIPT_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

# Now import project modules (fail with a clear message if still not importable).
try:
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.utils.misc import todevice, get_device
except Exception as e:
    print("Failed to import project modules from the repository. Details:")
    print(f"  Exception: {e}")
    print("")
    print("Make sure you run this script from the repository root or that the repository root")
    print("is on PYTHONPATH. If you are running in a notebook, ensure the notebook's working")
    print("directory is the repository root, or set PYTHONPATH to include the repo root, e.g.:")
    print("")
    print("  import sys")
    print(f"  sys.path.insert(0, '{REPO_ROOT}')")
    print("")
    print("Alternatively, install the package in editable mode (from the repo root):")
    print("  pip install -e .")
    sys.exit(1)

import torch
from torch.utils.data import DataLoader
import imageio
import numpy as np

class TGS:
    """
    Minimal inference wrapper around a trained model checkpoint.

    The wrapper expects cfg (namespace-like) with attribute `.weights` pointing to
    a checkpoint path. The wrapper will:
      - load a pickled torch.nn.Module checkpoint if present
      - error with guidance if checkpoint contains only a state_dict
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.device = "cpu"
        self.output_dir = os.getcwd()

        ckpt_path = getattr(cfg, "weights", None)
        ckpt_path = os.environ.get("MODEL_CKPT_PATH", ckpt_path or "checkpoints/model_lvis_rel.ckpt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. Place your checkpoint at that path or set MODEL_CKPT_PATH."
            )

        self.checkpoint_path = ckpt_path
        self.model = None
        self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, torch.nn.Module):
            self.model = ckpt
        elif isinstance(ckpt, dict):
            if "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
                self.model = ckpt["model"]
            elif "state_dict" in ckpt or any(k.endswith("state_dict") for k in ckpt.keys()):
                raise RuntimeError(
                    "Checkpoint appears to contain only a state_dict. Please instantiate the model class "
                    "from the repository and load the state dict manually. Example:\n\n"
                    "from tgs.models.your_model import YourModel\n"
                    "model = YourModel(cfg=...)\n"
                    "model.load_state_dict(torch.load('path')['state_dict'])\n"
                    "model.to(device)\n"
                )
            else:
                raise RuntimeError("Unknown checkpoint dict format; expected a saved model or a 'state_dict' key.")
        else:
            raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

        try:
            self.model.eval()
        except Exception:
            pass

    def to(self, device: torch.device):
        self.device = device
        if self.model is not None:
            try:
                self.model.to(device)
            except Exception:
                pass
        return self

    def set_save_dir(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.output_dir = out_dir

    def get_save_path(self, relpath: str) -> str:
        full = os.path.join(self.output_dir, relpath)
        d = os.path.dirname(full)
        if d:
            os.makedirs(d, exist_ok=True)
        return full

    def save_image_grid(self, path: str, images: list):
        if not images:
            return
        entry = images[0]
        img = entry.get("img", None)
        if img is None:
            return
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)
        save_path = self.get_save_path(path)
        imageio.imwrite(save_path, img)

    def save_img_sequences(self, directory: str, pattern: str, save_format="mp4", fps=30, delete=False):
        dir_full = os.path.join(self.output_dir, directory)
        if not os.path.isdir(dir_full):
            print(f"No directory {dir_full} to save sequences from.")
            return
        regex = re.compile(pattern)
        frames = []
        for fname in sorted(os.listdir(dir_full)):
            if regex.match(fname):
                frames.append(os.path.join(dir_full, fname))
        if not frames:
            print("No frames matched the provided pattern.")
            return
        imgs = [imageio.imread(f) for f in frames]
        out_path = os.path.join(self.output_dir, f"{directory}.{save_format}")
        print(f"Saving video to {out_path} ({len(imgs)} frames, {fps} fps)")
        imageio.mimwrite(out_path, imgs, fps=fps, macro_block_size=None)
        if delete:
            for f in frames:
                try:
                    os.remove(f)
                except Exception:
                    pass

def main():
    parser = argparse.ArgumentParser("Triplane Gaussian Splatting - Inference")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="camera distance")
    parser.add_argument("--image_preprocess", action="store_true", help="run image preprocess (SAM)")
    args, extras = parser.parse_known_args()

    device = get_device()

    cfg = load_config(args.config, cli_args=extras)
    model_path = os.environ.get("MODEL_CKPT_PATH", "checkpoints/model_lvis_rel.ckpt")
    cfg.system.weights = model_path

    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)
    print("Loaded checkpoint:", model.checkpoint_path)
    print("Saving outputs to:", args.out)

    if args.image_preprocess:
        segmented_image_list = []
        for image_path in cfg.data.image_list:
            filepath, ext = os.path.splitext(image_path)
            save_path = os.path.join(filepath + "_rgba.png")
            segmented_image_list.append(save_path)
            cmd = f"python image_preprocess/run_sam.py --image_path {image_path} --save_path {save_path}"
            print("Running:", cmd)
            subprocess.run([cmd], shell=True)
        cfg.data.image_list = segmented_image_list

    cfg.data.cond_camera_distance = args.cam_dist
    cfg.data.eval_camera_distance = args.cam_dist

    dataset = CustomImageOrbitDataset(cfg.data)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.eval_batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=False,
        collate_fn=dataset.collate
    )

    for batch in dataloader:
        batch = todevice(batch)
        try:
            # If the saved object is the full model, calling the wrapper triggers the model behavior.
            # We attempt calling the wrapper first; if it fails, try to call the underlying model.
            model(batch)
        except Exception as e:
            if hasattr(model, "model") and callable(getattr(model.model, "__call__", None)):
                try:
                    model.model(batch)
                except Exception as e2:
                    print("Model invocation failed:", e2)
                    raise
            else:
                print("Model invocation failed:", e)
                raise

    model.save_img_sequences("video", r"(\d+)\.png", save_format="mp4", fps=30, delete=True)

if __name__ == "__main__":
    main()
