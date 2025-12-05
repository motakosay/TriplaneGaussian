#!/usr/bin/env python3
"""
infer.py

Light-weight inference wrapper for the TriplaneGaussian project.

This file provides:
- A TGS wrapper class that loads a checkpoint (best-effort) and exposes a minimal
  interface used by the rest of the repository (to(device), set_save_dir, get_save_path,
  forward, save_image_grid, save_img_sequences).
- A CLI entrypoint that loads a config, instantiates dataset/dataloader, runs inference,
  and saves outputs.

Notes:
- This implementation intentionally avoids downloading checkpoints from external hubs.
  Place your checkpoint locally (default: checkpoints/model_lvis_rel.ckpt) or set the
  MODEL_CKPT_PATH environment variable.
- The wrapper will accept a checkpoint that already contains a serialized model object
  (i.e., saved via torch.save(model, ...)). If your checkpoint only contains a state_dict,
  you must instantiate the model class (from the project code) and load the state_dict
  before using this script. This generic wrapper will raise an informative error in that case.
"""

import os
import re
import sys
import argparse
import subprocess
import glob
from typing import Optional, Any, Dict

import torch
from torch.utils.data import DataLoader

import imageio
import numpy as np

# Project imports (these must exist in your repository)
# The wrapper is intentionally lightweight: it does not assume deep integration details.
try:
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.utils.misc import todevice, get_device
except Exception:
    # If these imports fail, we'll still provide helpful error messages later when used.
    ExperimentConfig = None
    load_config = None
    CustomImageOrbitDataset = None
    todevice = None
    get_device = None


class TGS:
    """
    Minimal inference wrapper around a trained model checkpoint.

    Behavior:
    - If the checkpoint file (cfg.weights) contains a pickled model object (torch.nn.Module),
      we use it directly.
    - If the checkpoint contains only state_dict, we raise an error explaining how to
      instantiate the model class and load the state dict.
    """

    def __init__(self, cfg: Any):
        """
        cfg is expected to be a namespace-like object that has at least a `.weights` attribute
        pointing to the checkpoint path. This matches how ExperimentConfig.system is used in the repo.
        """
        self.cfg = cfg
        self.device = "cpu"
        self.output_dir = os.getcwd()

        ckpt_path = None
        if cfg is not None:
            # Common attribute path used in project: cfg.weights
            ckpt_path = getattr(cfg, "weights", None)

        # Allow override via environment variable
        ckpt_path = os.environ.get("MODEL_CKPT_PATH", ckpt_path or "checkpoints/model_lvis_rel.ckpt")

        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {ckpt_path}. Place your checkpoint at this path or set MODEL_CKPT_PATH."
            )

        self.checkpoint_path = ckpt_path
        self.model = None
        self._load_checkpoint(ckpt_path)

    def _load_checkpoint(self, path: str) -> None:
        """
        Load a checkpoint. Best-effort approach:
        - If the checkpoint file is a pickled torch.nn.Module (saved via torch.save(model, ...)),
          torch.load will return a Module instance and we keep it.
        - If the checkpoint is a dict and contains a 'model' key that is a Module, use that.
        - If the checkpoint is a dict and contains only state_dict, we raise an explicit error
          telling the user to instantiate their model class and load the state dict.
        """
        map_location = "cpu"
        ckpt = torch.load(path, map_location=map_location)

        if isinstance(ckpt, torch.nn.Module):
            self.model = ckpt
        elif isinstance(ckpt, dict):
            # Common conventions to check:
            if "model" in ckpt and isinstance(ckpt["model"], torch.nn.Module):
                self.model = ckpt["model"]
            elif "state_dict" in ckpt:
                # We cannot know how to build the model class here. Provide guidance.
                raise RuntimeError(
                    "Checkpoint contains only state_dict. Please instantiate the model class from the repository "
                    "and load the state_dict manually. Example:\n\n"
                    "from tgs.models.your_model import YourModel\n"
                    "model = YourModel(cfg=...)\n"
                    "model.load_state_dict(torch.load('path/to/checkpoint')['state_dict'])\n"
                    "model = model.to(device)\n\n"
                    "Alternatively, save the full model object (torch.save(model, 'model.ckpt')) and re-run."
                )
            else:
                # Some checkpoints store keys differently (e.g., 'model_state_dict', 'net', etc.)
                # Try a few common keys:
                for key in ["model_state_dict", "net", "network", "state"]:
                    if key in ckpt and isinstance(ckpt[key], dict):
                        raise RuntimeError(
                            f"Checkpoint contains '{key}' dict (likely a state_dict). Please instantiate the model "
                            "class and load it as described in the README."
                        )

                # Unknown dict contents: try to use as-is if there's a callable 'forward' inside (unlikely)
                # Fall back to raising an error.
                raise RuntimeError(
                    "Unknown checkpoint format. Expect either a saved torch.nn.Module or a dict with a 'state_dict'. "
                    "Please check your checkpoint file."
                )
        else:
            raise RuntimeError("Unsupported checkpoint type: {}.".format(type(ckpt)))

        # Put model in eval mode by default
        self.model.eval()

    def to(self, device: torch.device):
        """
        Move the wrapped model to device and store device.
        """
        self.device = device
        if self.model is not None:
            try:
                self.model.to(device)
            except Exception:
                # Some saved models may not be proper Modules; ignore and keep device setting.
                pass
        return self

    def set_save_dir(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.output_dir = out_dir

    def get_save_path(self, relpath: str) -> str:
        # Ensure directories are created for nested paths
        full = os.path.join(self.output_dir, relpath)
        d = os.path.dirname(full)
        if d:
            os.makedirs(d, exist_ok=True)
        return full

    def save_image_grid(self, path: str, images: list):
        """
        Save a single image (or image grid) to disk.
        Expected format for images param (compatible with usage in repository):
            images: list of dicts, each dict is {'type': 'rgb', 'img': np.ndarray or torch tensor, 'kwargs': {...}}
        We support saving the first provided RGB image. This is intentionally simple.
        """
        if not images:
            return

        entry = images[0]
        img = entry.get("img", None)
        if img is None:
            return

        # Convert torch tensor to numpy HWC uint8 if necessary
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)

        # Support channel-first or channel-last
        if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW
            img = np.transpose(img, (1, 2, 0))
        # Clip and convert
        if img.dtype != np.uint8:
            # Assume floating point [0,1] or [0,255]
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        save_path = self.get_save_path(path)
        imageio.imwrite(save_path, img)

    def save_img_sequences(self, directory: str, pattern: str, save_format="mp4", fps=30, delete=False):
        """
        Collect frames matching pattern under output_dir/directory and save as a single video file.
        pattern is a regex used to match filenames within that directory (not the full path).
        Example from repo: save_img_sequences("video", "(\d+)\.png", save_format="mp4")
        """
        dir_full = os.path.join(self.output_dir, directory)
        if not os.path.isdir(dir_full):
            print(f"No directory {dir_full} to save sequences from.")
            return

        # Find files that match the regex pattern
        regex = re.compile(pattern)
        frames = []
        for fname in sorted(os.listdir(dir_full)):
            if regex.match(fname):
                frames.append(os.path.join(dir_full, fname))

        if not frames:
            print("No frames matched the provided pattern.")
            return

        # Read images
        imgs = [imageio.imread(f) for f in frames]

        # Save video
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
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    args, extras = parser.parse_known_args()

    # Validate project helpers are importable
    if load_config is None:
        print("Project configuration helpers could not be imported. Ensure this script runs from the project root where 'tgs' package is available.", file=sys.stderr)
        sys.exit(1)
    if CustomImageOrbitDataset is None:
        print("Dataset class could not be imported. Ensure 'tgs.data.CustomImageOrbitDataset' exists.", file=sys.stderr)
        sys.exit(1)
    if todevice is None or get_device is None:
        print("Utility functions 'tgs.utils.misc.todevice' or 'tgs.utils.misc.get_device' could not be imported.", file=sys.stderr)
        sys.exit(1)

    device = get_device()

    cfg = load_config(args.config, cli_args=extras)
    # Use local checkpoint path (no external downloads)
    model_path = os.environ.get("MODEL_CKPT_PATH", "checkpoints/model_lvis_rel.ckpt")
    cfg.system.weights = model_path

    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)
    print("Loaded model checkpoint and set save directory:", args.out)

    # run image segmentation for images if requested (delegates to external script)
    if args.image_preprocess:
        segmented_image_list = []
        for image_path in cfg.data.image_list:
            filepath, ext = os.path.splitext(image_path)
            save_path = os.path.join(filepath + "_rgba.png")
            segmented_image_list.append(save_path)
            # Using subprocess to call the project's SAM runner script (if available)
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

    # Run inference loop
    for batch in dataloader:
        batch = todevice(batch)
        try:
            # Expect the model wrapper to accept a batch dict and perform processing
            model(batch)
        except Exception as e:
            # Try to call model.model if the wrapper forwards to an internal model object
            if hasattr(model, "model") and callable(getattr(model.model, "__call__", None)):
                try:
                    model.model(batch)
                except Exception as e2:
                    print("Model call failed:", e2, file=sys.stderr)
                    raise
            else:
                print("Model invocation failed:", e, file=sys.stderr)
                raise

    # Optionally pack frames into video(s)
    model.save_img_sequences(
        "video",
        r"(\d+)\.png",
        save_format="mp4",
        fps=30,
        delete=True,
    )


if __name__ == "__main__":
    main()
