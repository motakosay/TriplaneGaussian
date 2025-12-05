        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        rend_out = self.renderer(scene_codes,
                                query_points=pointclouds,
                                additional_features=proj_feats,
                                **batch)

        return {**out, **rend_out}
    
    def forward(self, batch):
        out = self._forward(batch)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                out["3dgs"][b].save_ply(self.get_save_path(f"3dgs/{batch['instance_id'][b]}.ply"))

            for index, render_image in enumerate(out["comp_rgb"][b]):
                view_index = batch["view_index"][b, index]
                self.save_image_grid(
                    f"video/{batch['instance_id'][b]}/{view_index}.png",
                    [
                        {
                            "type": "rgb",
                            "img": render_image,
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                )
        

if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from tgs.data import CustomImageOrbitDataset
    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    args, extras = parser.parse_known_args()

    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)
    # NOTE: removed automatic hf_hub_download to avoid Hugging Face hub dependency.
    # Place your checkpoint at 'checkpoints/model_lvis_rel.ckpt' or set env var MODEL_CKPT_PATH.
    model_path = os.environ.get("MODEL_CKPT_PATH", "checkpoints/model_lvis_rel.ckpt")
    # model_path = "checkpoints/model_lvis_rel.ckpt"
    cfg.system.weights=model_path
    model = TGS(cfg=cfg.system).to(device)
    model.set_save_dir(args.out)
    print("load model ckpt done.")

    # run image segmentation for images
    if args.image_preprocess:
        segmented_image_list = []
        for image_path in cfg.data.image_list:
            filepath, ext = os.path.splitext(image_path)
            save_path = os.path.join(filepath + "_rgba.png")
            segmented_image_list.append(save_path)
            subprocess.run([f"python image_preprocess/run_sam.py --image_path {image_path} --save_path {save_path}"], shell=True)
        cfg.data.image_list = segmented_image_list

    cfg.data.cond_camera_distance = args.cam_dist
    cfg.data.eval_camera_distance = args.cam_dist
    dataset = CustomImageOrbitDataset(cfg.data)
    dataloader = DataLoader(dataset,
                        batch_size=cfg.data.eval_batch_size, 
                        num_workers=cfg.data.num_workers,
                        shuffle=False,
                        collate_fn=dataset.collate
                    )

    for batch in dataloader:
        batch = todevice(batch)
        model(batch)
    
    model.save_img_sequences(
        "video",
        "(\d+)\.png",
        save_format="mp4",
        fps=30,
        delete=True,
    )
