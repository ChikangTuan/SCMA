from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .flow import distance_to_metal


IMAGE_SHAPE = (416, 416)
SINOGRAM_SHAPE = (640, 641)


def _array(h5, name, shape):
    data = np.asarray(h5[name], dtype=np.float32)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError(f"{h5.filename}:{name} must be finite with shape {shape}")
    return data


def _as_binary(mask, shape, label):
    mask = np.asarray(mask)
    if mask.shape != shape or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{label} must contain binary labels (0 and 1) at shape {shape}")
    return mask.astype(np.float32)


def load_condition(case_file):
    with h5py.File(case_file, "r") as h5:
        return torch.from_numpy(_array(h5, "LI_CT", IMAGE_SHAPE)[None])


def read_measurement(case_file):
    with h5py.File(case_file, "r") as h5:
        y_ma = _array(h5, "ma_sinogram", SINOGRAM_SHAPE)
        trace = _as_binary(h5["metal_trace"], SINOGRAM_SHAPE, "metal_trace")
    return y_ma, trace


def prepare_case(case_file, segmentation_file, output_file, projector):
    segmentation_file = Path(segmentation_file)
    if segmentation_file.suffix.lower() == ".npy":
        mask = np.load(segmentation_file, allow_pickle=False)
    else:
        with h5py.File(segmentation_file, "r") as h5:
            mask = h5["metal_mask"][()]
    mask = _as_binary(mask, IMAGE_SHAPE, "metal_mask")
    with h5py.File(case_file, "r") as h5:
        y_ma = _array(h5, "ma_sinogram", SINOGRAM_SHAPE)
    condition, trace = projector.prepare_li(y_ma, mask)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stream = output_file.open("xb")
    try:
        with stream, h5py.File(stream, "w") as h5:
            for name, data in (("ma_sinogram", y_ma), ("metal_mask", mask.astype(np.uint8)),
                               ("metal_trace", trace.astype(np.uint8)), ("LI_CT", condition)):
                h5.create_dataset(name, data=data, compression="gzip", shuffle=True)
    except BaseException:
        output_file.unlink(missing_ok=True)
        raise


def _case_index(root, split, case_limit):
    listing = root / f"{split}_640geo_dir.txt"
    paths = [Path(s.strip()) for s in listing.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    if not paths or any(p.is_absolute() or ".." in p.parts or p.name != "gt.h5" for p in paths):
        raise ValueError(f"Expected relative paths to gt.h5 in {listing}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Duplicate cases in {listing}")
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be positive")
        paths = paths[:case_limit]
    bases = [root / f"{split}_640", root / f"{split}_640geo"]
    for base in bases:
        if (base / paths[0]).is_file():
            return base, paths
    first_matches = [p for base in bases if base.is_dir() for p in base.rglob(paths[0].as_posix())]
    if len(first_matches) != 1:
        raise FileNotFoundError(f"Expected a unique {paths[0]} under {bases}")
    return first_matches[0].parents[len(paths[0].parts) - 1], paths


class SynDeepLesion(Dataset):
    def __init__(self, root, split="train", case_limit=None, versions=None, mask_dir=None):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.root = Path(root)
        self.case_root, self.cases = _case_index(self.root, split, case_limit)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        n_versions = 90 if split == "train" else 10
        self.versions = tuple(range(n_versions) if versions is None else versions)
        if (not self.versions or len(set(self.versions)) != len(self.versions)
                or any(not isinstance(v, (int, np.integer)) or not 0 <= v < n_versions
                       for v in self.versions)):
            raise ValueError(f"Expected distinct integer versions between 0 and {n_versions-1}")
        self.templates = {}
        bank_path = self.root / f"{split}mask.npy"
        if self.mask_dir is None and bank_path.is_file():
            bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
            if bank.ndim != 3 or max(self.versions) >= bank.shape[2]:
                raise ValueError("The mask bank must have shape [H, W, number of versions]")
            for version in self.versions:
                raw = _as_binary(bank[:, :, version], bank.shape[:2], f"template {version}")
                resized = Image.fromarray(raw).resize(IMAGE_SHAPE[::-1], Image.Resampling.BILINEAR)
                mask = (np.asarray(resized) > 0.5).astype(np.float32)
                self.templates[version] = (mask, distance_to_metal(mask))

    def __len__(self):
        return len(self.cases) * len(self.versions)

    def __getitem__(self, index):
        case_i, version_i = divmod(index, len(self.versions))
        relative_gt = self.cases[case_i]
        version = self.versions[version_i]
        relative_case = relative_gt.with_name(f"{version}.h5")
        with h5py.File(self.case_root / relative_gt, "r") as h5:
            clean = _array(h5, "image", IMAGE_SHAPE)
        with h5py.File((self.mask_dir or self.case_root) / relative_case, "r") as h5:
            condition = _array(h5, "LI_CT", IMAGE_SHAPE)
            if "metal_mask" in h5:
                mask = _as_binary(h5["metal_mask"], IMAGE_SHAPE, "metal_mask")
                distance = distance_to_metal(mask)
            elif self.mask_dir is None and version in self.templates:
                mask, distance = self.templates[version]
            else:
                raise ValueError(f"Missing metal_mask for {relative_case}")
        return {
            key: torch.from_numpy(value[None])
            for key, value in (("clean", clean), ("condition", condition), ("mask", mask),
                               ("distance", distance))
        }
