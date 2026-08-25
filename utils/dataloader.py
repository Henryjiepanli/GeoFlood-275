# -*- coding: utf-8 -*-
"""
dataloader.py (TIF-based, Flood AOI dataset)  Python 3.8 compatible

Returns:
  __getitem__ -> (sample_dict, stem)

sample_dict keys:
  - "S1":      FloatTensor (2,H,W)   normalized to [-1,1], fill->-1
  - "S2":      FloatTensor (4,H,W)   normalized to [-1,1], fill->-1
  - "DEM":     FloatTensor (H,W)     normalized (same as your NPZ fast: /5000 then (x-0.5)/0.5)
  - "Slope":   FloatTensor (H,W)     already 0..1 (clip), fill->0
  - "ESA":     LongTensor  (H,W)     int64, nodata=0 (kept)
  - "DW":      LongTensor  (H,W)     int64, nodata=0 (optional; if not exist -> zeros)
  - "seg_mask":LongTensor  (H,W)     flood mask; if values {0,255} will be remapped to {0,1}

Notes:
- Random flips in train split.
- Uses rasterio to read GeoTIFF. Geoinfo not needed for training; reading array only.
- Efficient: reads each patch file directly (already tiled to 512x512).

S1 normalization:
- unified range: [-45, 25]
- clipped then mapped to [-1,1]
- invalid/fill -> -1
"""

import os
import random
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.utils.data as data
import rasterio


# -------------------------
# Utils
# -------------------------
def _as_contig(x: np.ndarray) -> np.ndarray:
    return x if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x)


def _list_tifs(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ]
    files.sort()
    return files


def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _read_tif(path: str) -> np.ndarray:
    """
    Returns:
      - if raster has multiple bands: (C,H,W)
      - if single band: (H,W)
    """
    with rasterio.open(path) as src:
        arr = src.read()  # (C,H,W)
    if arr.shape[0] == 1:
        return arr[0]
    return arr


# -------------------------
# Normalization
# -------------------------
class NormalizeFlood(object):
    """
    Match your NPZ FastPreprocessor behavior, with updated S1 normalization.
    """

    def __init__(
        self,
        sar_fill: float = -50.0,
        s2_fill: float = -100.0,
        s2_clip_max: float = 6000.0,
        slope_fill: float = 0.0,
        dem_mean: float = 0.5,
        dem_std: float = 0.5,
        reorder_s2_to_rgbn: bool = False,   # if stored as [NIR,R,G,B], reorder-> [R,G,B,NIR]
        s1_min: float = -45.0,
        s1_max: float = 25.0,
    ):
        self.sar_fill = float(sar_fill)
        self.s2_fill = float(s2_fill)
        self.s2_clip_max = float(s2_clip_max)
        self.slope_fill = float(slope_fill)
        self.dem_mean = float(dem_mean)
        self.dem_std = float(dem_std)
        self.reorder_s2_to_rgbn = bool(reorder_s2_to_rgbn)

        # fixed unified SAR range
        self.s1_min = float(s1_min)
        self.s1_max = float(s1_max)
        self.esa_raw_values = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], dtype=np.int64)
        self.esa_to_idx = {v: i for i, v in enumerate(self.esa_raw_values.tolist())}

    @staticmethod
    def _to_minus1_1(x01: np.ndarray) -> np.ndarray:
        return x01 * 2.0 - 1.0

    def norm_s1(self, s1: np.ndarray) -> np.ndarray:
        """
        s1: (2,H,W)
        Normalize using unified fixed range [-45, 25] -> [-1, 1]
        invalid/fill -> -1
        """
        s1 = s1.astype(np.float32, copy=False)

        valid = np.isfinite(s1) & (np.abs(s1 - self.sar_fill) > 1e-3)

        s1 = np.clip(s1, self.s1_min, self.s1_max)
        s1 = (s1 - self.s1_min) / (self.s1_max - self.s1_min + 1e-6)
        s1 = self._to_minus1_1(s1).astype(np.float32, copy=False)

        s1[~valid] = -1.0
        return _as_contig(s1)

    def norm_s2(self, s2: np.ndarray) -> np.ndarray:
        """
        s2: (4,H,W)
        """
        s2 = s2.astype(np.float32, copy=False)

        if self.reorder_s2_to_rgbn:
            # stored [NIR,R,G,B] -> [R,G,B,NIR]
            s2 = s2[[1, 2, 3, 0], ...]

        valid = np.isfinite(s2) & (s2 != self.s2_fill)
        np.clip(s2, 0.0, self.s2_clip_max, out=s2)
        s2 = s2 / (self.s2_clip_max + 1e-6)
        s2 = self._to_minus1_1(s2).astype(np.float32, copy=False)
        s2[~valid] = -1.0
        return _as_contig(s2)

    def norm_dem(self, dem: np.ndarray) -> np.ndarray:
        """
        dem: (H,W)
        """
        dem = dem.astype(np.float32, copy=False)
        dem = dem / 5000.0
        dem = (dem - self.dem_mean) / self.dem_std
        return _as_contig(dem.astype(np.float32, copy=False))

    def norm_slope(self, slope: np.ndarray) -> np.ndarray:
        """
        slope already 0..1, but keep safe
        """
        slope = slope.astype(np.float32, copy=False)
        slope = np.where(np.isfinite(slope), slope, self.slope_fill)
        slope = np.clip(slope, 0.0, 1.0).astype(np.float32, copy=False)
        return _as_contig(slope)

    def norm_esa(self, esa: np.ndarray) -> np.ndarray:
        """
        Convert ESA raw codes:
        [0,10,20,30,40,50,60,70,80,90,95,100]
        ->
        contiguous ids [0..11]
        """
        esa = esa.astype(np.int64, copy=False)
        out = np.zeros_like(esa, dtype=np.int64)

        # default unknown -> 0
        for raw_val, idx in self.esa_to_idx.items():
            out[esa == raw_val] = idx

        return _as_contig(out)

    @staticmethod
    def norm_mask(mask: np.ndarray) -> np.ndarray:
        """
        Flood mask: may be {0,255} -> convert to {0,1}.
        """
        m = mask.astype(np.uint8, copy=False)
        if m.max() > 1:
            m = (m >= 128).astype(np.uint8)
        return _as_contig(m.astype(np.int64, copy=False))

    @staticmethod
    def to_long(x: np.ndarray) -> np.ndarray:
        return _as_contig(x.astype(np.int64, copy=False))


# -------------------------
# Dataset
# -------------------------
class FloodAOITifDataset(data.Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "Train",          # Train / Val / Test
        mode: str = "both",            # "both" | "s1" | "s2"
        use_dynamic_world: bool = True,
        augment: bool = True,
        reorder_s2_to_rgbn: bool = False,
        sar_fill: float = -50.0,
        s2_fill: float = -100.0,
        s2_clip_max: float = 6000.0,
        s1_min: float = -45.0,
        s1_max: float = 25.0,
    ):
        super().__init__()
        assert split in ("Train", "Val", "Test")
        assert mode in ("both", "s1", "s2")

        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.use_dynamic_world = bool(use_dynamic_world)
        self.do_aug = bool(augment) and (split == "Train")

        self.norm = NormalizeFlood(
            sar_fill=sar_fill,
            s2_fill=s2_fill,
            s2_clip_max=s2_clip_max,
            reorder_s2_to_rgbn=reorder_s2_to_rgbn,
            s1_min=s1_min,
            s1_max=s1_max,
        )

        split_dir = os.path.join(root_dir, split)

        self.dir_s1 = os.path.join(split_dir, "Sentinel-1")
        self.dir_s2 = os.path.join(split_dir, "Sentinel-2")
        self.dir_dem = os.path.join(split_dir, "DEM")
        self.dir_esa = os.path.join(split_dir, "ESAV200")
        self.dir_slope = os.path.join(split_dir, "Slope_Norm")
        self.dir_mask = os.path.join(split_dir, "CD_Flood")
        self.dir_dw = os.path.join(split_dir, "Dynamic_Landcover")

        # index by stems from S1
        s1_files = _list_tifs(self.dir_s1)
        if len(s1_files) == 0:
            raise RuntimeError("No tif found in: {}".format(self.dir_s1))

        self.items = []
        for p_s1 in s1_files:
            stem = _stem(p_s1)
            p_s2 = os.path.join(self.dir_s2, stem + ".tif")
            p_dem = os.path.join(self.dir_dem, stem + ".tif")
            p_esa = os.path.join(self.dir_esa, stem + ".tif")
            p_slope = os.path.join(self.dir_slope, stem + ".tif")
            p_mask = os.path.join(self.dir_mask, stem + ".tif")
            p_dw = os.path.join(self.dir_dw, stem + ".tif")

            if not (os.path.exists(p_s2) and os.path.exists(p_dem) and os.path.exists(p_esa)
                    and os.path.exists(p_slope) and os.path.exists(p_mask)):
                continue

            self.items.append({
                "stem": stem,
                "S1": p_s1,
                "S2": p_s2,
                "DEM": p_dem,
                "ESA": p_esa,
                "Slope": p_slope,
                "Mask": p_mask,
                "DW": p_dw,
            })

        if len(self.items) == 0:
            raise RuntimeError("No complete samples found under split: {}".format(split_dir))

    def __len__(self):
        return len(self.items)

    def _maybe_flip(self, s1, s2, dem, slope, esa, dw, mask):
        if not self.do_aug:
            return s1, s2, dem, slope, esa, dw, mask

        if random.random() < 0.5:
            # vertical flip
            if s1 is not None:
                s1 = s1[:, ::-1, :]
            if s2 is not None:
                s2 = s2[:, ::-1, :]
            dem = dem[::-1, :]
            slope = slope[::-1, :]
            esa = esa[::-1, :]
            if dw is not None:
                dw = dw[::-1, :]
            mask = mask[::-1, :]

        if random.random() < 0.5:
            # horizontal flip
            if s1 is not None:
                s1 = s1[:, :, ::-1]
            if s2 is not None:
                s2 = s2[:, :, ::-1]
            dem = dem[:, ::-1]
            slope = slope[:, ::-1]
            esa = esa[:, ::-1]
            if dw is not None:
                dw = dw[:, ::-1]
            mask = mask[:, ::-1]

        return s1, s2, dem, slope, esa, dw, mask

    def __getitem__(self, idx):
        it = self.items[idx]
        stem = it["stem"]

        # read arrays
        s1 = _read_tif(it["S1"]) if self.mode in ("both", "s1") else None
        s2 = _read_tif(it["S2"]) if self.mode in ("both", "s2") else None
        dem = _read_tif(it["DEM"])      # (H,W)
        slope = _read_tif(it["Slope"])  # (H,W)
        esa = _read_tif(it["ESA"])      # (H,W)
        mask = _read_tif(it["Mask"])    # (H,W)
        dw = _read_tif(it["DW"])

        # safety shapes
        if s1 is not None and s1.ndim != 3:
            raise RuntimeError("S1 should be (2,H,W), got {} for {}".format(s1.shape, it["S1"]))
        if s2 is not None and s2.ndim != 3:
            raise RuntimeError("S2 should be (4,H,W), got {} for {}".format(s2.shape, it["S2"]))

        # augmentation
        s1, s2, dem, slope, esa, dw, mask = self._maybe_flip(s1, s2, dem, slope, esa, dw, mask)

        # normalize
        out = {}  # type: Dict[str, torch.Tensor]

        if s1 is not None:
            out["S1"] = torch.from_numpy(self.norm.norm_s1(s1)).float()
        if s2 is not None:
            out["S2"] = torch.from_numpy(self.norm.norm_s2(s2)).float()

        out["DEM"] = torch.from_numpy(self.norm.norm_dem(dem)).float()
        out["Slope"] = torch.from_numpy(self.norm.norm_slope(slope)).float()
        out["ESA"] = torch.from_numpy(self.norm.norm_esa(esa)).long()
        out["DW"] = torch.from_numpy(self.norm.to_long(dw)).long()
        out["seg_mask"] = torch.from_numpy(self.norm.norm_mask(mask)).long()

        return out, stem


# -------------------------
# Worker init
# -------------------------
def worker_init_fn(worker_id):
    seed = 2333 + worker_id
    random.seed(seed)
    np.random.seed(seed)


# -------------------------
# DataLoader builder
# -------------------------
def get_loader(
    S1_root, S2_root, DEM_root, ESA_root, gt_root,   # kept for compatibility, only S1_root used
    trainsize, mode, batchsize,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    drop_last=None,
    DW_root=None,
    Slope_root=None,
    split_dir_root=None,
    split_name=None,
    tif_mode="both",                # "both"|"s1"|"s2"
    use_dynamic_world=True,
    reorder_s2_to_rgbn=False,
    s2_clip_max=6000.0,
    sar_fill=-50.0,
    s2_fill=-100.0,
    s1_min=-45.0,
    s1_max=25.0,
):
    """
    Compatibility wrapper:
      - If you pass split_dir_root + split_name:
            root_dir = split_dir_root, split = split_name
      - Else:
            root_dir = S1_root, and we infer split from 'mode' ("train"/"val"/"test")
            expecting: S1_root/{Train|Val|Test}/...
    """
    if split_dir_root is not None:
        root_dir = split_dir_root
    else:
        root_dir = S1_root

    if split_name is not None:
        split = split_name
    else:
        m = str(mode).lower()
        split = "Train" if m == "train" else ("Val" if m == "val" else "Test")

    dataset = FloodAOITifDataset(
        root_dir=root_dir,
        split=split,
        mode=tif_mode,
        use_dynamic_world=use_dynamic_world,
        augment=True,
        reorder_s2_to_rgbn=reorder_s2_to_rgbn,
        sar_fill=sar_fill,
        s2_fill=s2_fill,
        s2_clip_max=s2_clip_max,
        s1_min=s1_min,
        s1_max=s1_max,
    )

    if drop_last is None:
        drop_last = (split == "Train")

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle if split == "Train" else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    if num_workers > 0:
        loader_kwargs.update(
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
            worker_init_fn=worker_init_fn,
        )

    return data.DataLoader(**loader_kwargs)


# -------------------------
# Quick test
# -------------------------
if __name__ == "__main__":
    root = "/path/to/GeoFlood-275/Benchmark_all"
    dl = get_loader(
        root, None, None, None, None,
        512, "train", 2,
        num_workers=2,
        tif_mode="both",
        s1_min=-45.0,
        s1_max=25.0,
    )
    x, stem = next(iter(dl))
    print(stem[:2])
    print(
        x["S1"].shape,
        x["S2"].shape,
        x["DEM"].shape,
        x["Slope"].shape,
        x["ESA"].shape,
        x["DW"].shape,
        x["seg_mask"].shape
    )
    print("S1 min/max:", x["S1"].min().item(), x["S1"].max().item())