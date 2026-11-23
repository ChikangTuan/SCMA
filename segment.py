import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--device", choices=("cuda", "cpu"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".npy" or args.output.exists():
        parser.error("--output must be a new .npy file")
    if args.fold < 0 or Path(args.checkpoint).name != args.checkpoint:
        parser.error("expected a nonnegative fold and a checkpoint filename")
    dataset = json.loads((args.model_dir / "dataset.json").read_text(encoding="utf-8"))
    plans = json.loads((args.model_dir / "plans.json").read_text(encoding="utf-8"))
    config = plans.get("configurations", {}).get("2d", {})
    if (dataset.get("labels") != {"background": 0, "metal": 1}
            or list(dataset.get("channel_names", {})) != ["0"]
            or plans.get("image_reader_writer") != "NaturalImage2DIO"
            or config.get("normalization_schemes") != ["ZScoreNormalization"]
            or config.get("spacing") != [1, 1]):
        parser.error("expected a Dataset501-compatible, binary, single-channel 2D NaturalImage2DIO model")

    with h5py.File(args.case, "r") as h5:
        image = np.asarray(h5["ma_CT"], dtype=np.float32)
    if image.shape != (416, 416) or not np.isfinite(image).all():
        parser.error("ma_CT must be a finite 416 by 416 native attenuation slice (cm^-1)")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=(args.device == "cuda"), device=torch.device(args.device),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(str(args.model_dir), (args.fold,), args.checkpoint)
    if len(predictor.configuration_manager.patch_size) != 2:
        parser.error("the selected checkpoint must use the 2D configuration")
    labels = np.asarray(predictor.predict_single_npy_array(
        image[None, None], {"spacing": (999, 1, 1)}
    ))
    if labels.shape != (1, *image.shape) or not np.isin(labels, (0, 1)).all():
        raise ValueError("nnU-Net did not return a co-registered binary segmentation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, labels[0].astype(np.uint8), allow_pickle=False)


if __name__ == "__main__":
    main()
