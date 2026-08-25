# -*- coding: utf-8 -*-
"""
dataload_v2.py (TIF-based GeoFlood Benchmark dataloader)

This is a v2 dataloader for the Benchmark/{Train|Val|Test} layout. It keeps the
old dataloader keys and adds the newly tiled extension products.

Returned sample_dict keys:
  - "S1":                   FloatTensor (2,H,W), post-event SAR, normalized to [-1,1]
  - "S1_pre":               FloatTensor (2,H,W), pre-event SAR, normalized to [-1,1]
  - "S2":                   FloatTensor (4,H,W), normalized to [-1,1]
  - "DEM":                  FloatTensor (H,W)
  - "Slope":                FloatTensor (H,W), clipped to [0,1]
  - "ESA":                  LongTensor  (H,W), ESA raw codes mapped to contiguous ids
  - "DW":                   LongTensor  (H,W), DynamicWorld; 0 is nodata/outside
  - "GPM_Rainfall":         FloatTensor (4,H,W), log-normalized rainfall in [0,1]
  - "ERA5_Soil_Moisture":   FloatTensor (16,H,W), clipped to [0,1]
  - "seg_mask":             LongTensor  (H,W), flood mask in {0,1}

Optional convenience keys:
  - "S1_pair":              FloatTensor (4,H,W), concat[S1_pre, S1]
  - "Climate":              FloatTensor (20,H,W), concat[GPM_Rainfall, ERA5_Soil_Moisture]

The existing training code can still read "S1", "S2", "ESA", "DW", "Slope" and
"seg_mask". To actually use the new products in the model, update train.py and
the model forward to consume "S1_pre", "GPM_Rainfall" and "ERA5_Soil_Moisture".
"""

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
import torch.utils.data as data


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


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _read_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    if arr.shape[0] == 1:
        return arr[0]
    return arr


def _ensure_chw(arr: np.ndarray, path: str, key: str) -> np.ndarray:
    if arr.ndim != 3:
        raise RuntimeError("{} should be (C,H,W), got {} for {}".format(key, arr.shape, path))
    return arr


def _ensure_hw(arr: np.ndarray, path: str, key: str) -> np.ndarray:
    if arr.ndim != 2:
        raise RuntimeError("{} should be (H,W), got {} for {}".format(key, arr.shape, path))
    return arr


def _flip_array(arr: Optional[np.ndarray], vertical: bool, horizontal: bool) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim == 3:
        if vertical:
            arr = arr[:, ::-1, :]
        if horizontal:
            arr = arr[:, :, ::-1]
    elif arr.ndim == 2:
        if vertical:
            arr = arr[::-1, :]
        if horizontal:
            arr = arr[:, ::-1]
    else:
        raise RuntimeError("Unsupported array ndim for flip: {}".format(arr.ndim))
    return arr


# -------------------------
# Normalization
# -------------------------
class NormalizeFloodV2(object):
    def __init__(
        self,
        sar_fill: float = -50.0,
        s2_fill: float = -100.0,
        dem_fill: float = -9999.0,
        slope_fill: float = 0.0,
        climate_fill: float = -9999.0,
        s2_clip_max: float = 6000.0,
        dem_mean: float = 0.5,
        dem_std: float = 0.5,
        reorder_s2_to_rgbn: bool = False,
        s1_min: float = -45.0,
        s1_max: float = 25.0,
        rain_clip_max: float = 300.0,
        rain_norm: str = "log",       # "log" | "linear" | "none"
        soil_clip_min: float = 0.0,
        soil_clip_max: float = 1.0,
        soil_auto_scale: bool = False,
    ):
        self.sar_fill = float(sar_fill)
        self.s2_fill = float(s2_fill)
        self.dem_fill = float(dem_fill)
        self.slope_fill = float(slope_fill)
        self.climate_fill = float(climate_fill)
        self.s2_clip_max = float(s2_clip_max)
        self.dem_mean = float(dem_mean)
        self.dem_std = float(dem_std)
        self.reorder_s2_to_rgbn = bool(reorder_s2_to_rgbn)
        self.s1_min = float(s1_min)
        self.s1_max = float(s1_max)
        self.rain_clip_max = float(rain_clip_max)
        self.rain_norm = str(rain_norm).lower()
        self.soil_clip_min = float(soil_clip_min)
        self.soil_clip_max = float(soil_clip_max)
        self.soil_auto_scale = bool(soil_auto_scale)

        if self.rain_norm not in ("log", "linear", "none"):
            raise ValueError("rain_norm must be log, linear or none")

        self.esa_raw_values = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], dtype=np.int64)
        self.esa_to_idx = {v: i for i, v in enumerate(self.esa_raw_values.tolist())}

    @staticmethod
    def _to_minus1_1(x01: np.ndarray) -> np.ndarray:
        return x01 * 2.0 - 1.0

    def norm_s1(self, s1: np.ndarray) -> np.ndarray:
        s1 = s1.astype(np.float32, copy=False)
        valid = np.isfinite(s1) & (np.abs(s1 - self.sar_fill) > 1e-3)
        s1 = np.clip(s1, self.s1_min, self.s1_max)
        s1 = (s1 - self.s1_min) / (self.s1_max - self.s1_min + 1e-6)
        s1 = self._to_minus1_1(s1).astype(np.float32, copy=False)
        s1[~valid] = -1.0
        return _as_contig(s1)

    def norm_s2(self, s2: np.ndarray) -> np.ndarray:
        s2 = s2.astype(np.float32, copy=False)
        if self.reorder_s2_to_rgbn:
            s2 = s2[[1, 2, 3, 0], ...]
        valid = np.isfinite(s2) & (s2 != self.s2_fill)
        np.clip(s2, 0.0, self.s2_clip_max, out=s2)
        s2 = s2 / (self.s2_clip_max + 1e-6)
        s2 = self._to_minus1_1(s2).astype(np.float32, copy=False)
        s2[~valid] = -1.0
        return _as_contig(s2)

    def norm_dem(self, dem: np.ndarray) -> np.ndarray:
        dem = dem.astype(np.float32, copy=False)
        valid = np.isfinite(dem) & (dem != self.dem_fill)
        dem = dem / 5000.0
        dem = (dem - self.dem_mean) / (self.dem_std + 1e-6)
        dem[~valid] = 0.0
        return _as_contig(dem.astype(np.float32, copy=False))

    def norm_slope(self, slope: np.ndarray) -> np.ndarray:
        slope = slope.astype(np.float32, copy=False)
        slope = np.where(np.isfinite(slope), slope, self.slope_fill)
        slope = np.clip(slope, 0.0, 1.0).astype(np.float32, copy=False)
        return _as_contig(slope)

    def norm_esa(self, esa: np.ndarray) -> np.ndarray:
        esa = esa.astype(np.int64, copy=False)
        out = np.zeros_like(esa, dtype=np.int64)
        for raw_val, idx in self.esa_to_idx.items():
            out[esa == raw_val] = idx
        return _as_contig(out)

    def norm_dw(self, dw: np.ndarray) -> np.ndarray:
        dw = dw.astype(np.float32, copy=False)
        valid = np.isfinite(dw)
        out = np.zeros(dw.shape, dtype=np.int64)
        out[valid] = dw[valid].astype(np.int64)
        return _as_contig(out)

    def norm_rainfall(self, rain: np.ndarray) -> np.ndarray:
        rain = rain.astype(np.float32, copy=False)
        valid = np.isfinite(rain) & (rain != self.climate_fill)
        rain = np.where(valid, rain, 0.0)
        rain = np.clip(rain, 0.0, self.rain_clip_max)

        if self.rain_norm == "log":
            rain = np.log1p(rain) / (np.log1p(self.rain_clip_max) + 1e-6)
        elif self.rain_norm == "linear":
            rain = rain / (self.rain_clip_max + 1e-6)

        rain[~valid] = 0.0
        return _as_contig(rain.astype(np.float32, copy=False))

    def norm_soil_moisture(self, soil: np.ndarray) -> np.ndarray:
        soil = soil.astype(np.float32, copy=False)
        valid = np.isfinite(soil) & (soil != self.climate_fill)
        soil = np.where(valid, soil, 0.0)

        if self.soil_auto_scale:
            finite_valid = soil[valid]
            if finite_valid.size > 0 and np.nanpercentile(finite_valid, 99) > 1.5:
                soil = soil / 100.0

        soil = np.clip(soil, self.soil_clip_min, self.soil_clip_max)
        soil[~valid] = 0.0
        return _as_contig(soil.astype(np.float32, copy=False))

    @staticmethod
    def norm_mask(mask: np.ndarray) -> np.ndarray:
        m = mask.astype(np.uint8, copy=False)
        if m.max() > 1:
            m = (m >= 128).astype(np.uint8)
        return _as_contig(m.astype(np.int64, copy=False))


# -------------------------
# Dataset
# -------------------------
class FloodAOITifDatasetV2(data.Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "Train",
        mode: str = "both",             # "both" | "s1" | "s2"
        use_dynamic_world: bool = True,
        use_pre_s1: bool = True,
        use_rainfall: bool = True,
        use_soil_moisture: bool = True,
        require_new_products: bool = True,
        return_s1_pair: bool = False,
        return_climate: bool = False,
        augment: bool = True,
        reorder_s2_to_rgbn: bool = False,
        sar_fill: float = -50.0,
        s2_fill: float = -100.0,
        dem_fill: float = -9999.0,
        climate_fill: float = -9999.0,
        s2_clip_max: float = 6000.0,
        s1_min: float = -45.0,
        s1_max: float = 25.0,
        rain_clip_max: float = 300.0,
        rain_norm: str = "log",
        soil_auto_scale: bool = False,
        strict_channels: bool = True,
    ):
        super().__init__()
        assert split in ("Train", "Val", "Test")
        assert mode in ("both", "s1", "s2")

        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.use_dynamic_world = bool(use_dynamic_world)
        self.use_pre_s1 = bool(use_pre_s1)
        self.use_rainfall = bool(use_rainfall)
        self.use_soil_moisture = bool(use_soil_moisture)
        self.require_new_products = bool(require_new_products)
        self.return_s1_pair = bool(return_s1_pair)
        self.return_climate = bool(return_climate)
        self.do_aug = bool(augment) and (split == "Train")
        self.strict_channels = bool(strict_channels)
        self.climate_fill = float(climate_fill)

        self.norm = NormalizeFloodV2(
            sar_fill=sar_fill,
            s2_fill=s2_fill,
            dem_fill=dem_fill,
            climate_fill=climate_fill,
            s2_clip_max=s2_clip_max,
            reorder_s2_to_rgbn=reorder_s2_to_rgbn,
            s1_min=s1_min,
            s1_max=s1_max,
            rain_clip_max=rain_clip_max,
            rain_norm=rain_norm,
            soil_auto_scale=soil_auto_scale,
        )

        split_dir = os.path.join(root_dir, split)
        self.dir_s1 = os.path.join(split_dir, "Sentinel-1")
        self.dir_s1_pre = os.path.join(split_dir, "Sentinel-1-pre")
        self.dir_s2 = os.path.join(split_dir, "Sentinel-2")
        self.dir_dem = os.path.join(split_dir, "DEM")
        self.dir_esa = os.path.join(split_dir, "ESAV200")
        self.dir_slope = os.path.join(split_dir, "Slope_Norm")
        self.dir_mask = os.path.join(split_dir, "CD_Flood")
        self.dir_dw = os.path.join(split_dir, "Dynamic_Landcover")
        self.dir_rain = os.path.join(split_dir, "GPM_Rainfall")
        self.dir_soil = os.path.join(split_dir, "ERA5_Soil_Moisture")

        s1_files = _list_tifs(self.dir_s1)
        if len(s1_files) == 0:
            raise RuntimeError("No tif found in: {}".format(self.dir_s1))

        self.items = []
        self.skipped = 0
        for p_s1 in s1_files:
            stem = _stem(p_s1)
            item = {
                "stem": stem,
                "S1": p_s1,
                "S1_pre": os.path.join(self.dir_s1_pre, stem + ".tif"),
                "S2": os.path.join(self.dir_s2, stem + ".tif"),
                "DEM": os.path.join(self.dir_dem, stem + ".tif"),
                "ESA": os.path.join(self.dir_esa, stem + ".tif"),
                "Slope": os.path.join(self.dir_slope, stem + ".tif"),
                "Mask": os.path.join(self.dir_mask, stem + ".tif"),
                "DW": os.path.join(self.dir_dw, stem + ".tif"),
                "GPM_Rainfall": os.path.join(self.dir_rain, stem + ".tif"),
                "ERA5_Soil_Moisture": os.path.join(self.dir_soil, stem + ".tif"),
            }

            required = [item["S2"], item["DEM"], item["ESA"], item["Slope"], item["Mask"]]
            if self.mode in ("both", "s1"):
                required.append(item["S1"])
            if self.mode in ("both", "s2"):
                required.append(item["S2"])
            if self.use_pre_s1 and self.require_new_products:
                required.append(item["S1_pre"])
            if self.use_rainfall and self.require_new_products:
                required.append(item["GPM_Rainfall"])
            if self.use_soil_moisture and self.require_new_products:
                required.append(item["ERA5_Soil_Moisture"])

            if not all(os.path.exists(p) for p in required):
                self.skipped += 1
                continue
            self.items.append(item)

        if len(self.items) == 0:
            raise RuntimeError("No complete samples found under split: {}".format(split_dir))

    def __len__(self):
        return len(self.items)

    def _maybe_flip(self, arrays: Dict[str, Optional[np.ndarray]]) -> Dict[str, Optional[np.ndarray]]:
        if not self.do_aug:
            return arrays

        vertical = random.random() < 0.5
        horizontal = random.random() < 0.5
        if not vertical and not horizontal:
            return arrays

        for key in list(arrays.keys()):
            arrays[key] = _flip_array(arrays[key], vertical=vertical, horizontal=horizontal)
        return arrays

    def _zeros_chw(self, channels: int, hw: Tuple[int, int], fill: float = 0.0) -> np.ndarray:
        h, w = hw
        return np.full((channels, h, w), fill, dtype=np.float32)

    def _zeros_hw(self, hw: Tuple[int, int], fill: int = 0) -> np.ndarray:
        h, w = hw
        return np.full((h, w), fill, dtype=np.int64)

    def _read_optional_chw(self, path: str, channels: int, hw: Tuple[int, int], fill: float) -> np.ndarray:
        if os.path.exists(path):
            return _read_tif(path)
        return self._zeros_chw(channels, hw, fill=fill)

    def __getitem__(self, idx):
        item = self.items[idx]
        stem = item["stem"]

        s1 = _read_tif(item["S1"]) if self.mode in ("both", "s1") else None
        s2 = _read_tif(item["S2"]) if self.mode in ("both", "s2") else None

        dem = _read_tif(item["DEM"])
        slope = _read_tif(item["Slope"])
        esa = _read_tif(item["ESA"])
        mask = _read_tif(item["Mask"])

        if s1 is not None:
            _ensure_chw(s1, item["S1"], "S1")
            hw = (s1.shape[-2], s1.shape[-1])
        elif s2 is not None:
            _ensure_chw(s2, item["S2"], "S2")
            hw = (s2.shape[-2], s2.shape[-1])
        else:
            hw = dem.shape[-2], dem.shape[-1]

        s1_pre = None
        if self.use_pre_s1:
            s1_pre = self._read_optional_chw(item["S1_pre"], 2, hw, fill=-50.0)

        rain = None
        if self.use_rainfall:
            rain = self._read_optional_chw(item["GPM_Rainfall"], 4, hw, fill=self.climate_fill)

        soil = None
        if self.use_soil_moisture:
            soil = self._read_optional_chw(item["ERA5_Soil_Moisture"], 16, hw, fill=self.climate_fill)

        if self.use_dynamic_world and os.path.exists(item["DW"]):
            dw = _read_tif(item["DW"])
        else:
            dw = self._zeros_hw(hw, fill=0)

        _ensure_hw(dem, item["DEM"], "DEM")
        _ensure_hw(slope, item["Slope"], "Slope")
        _ensure_hw(esa, item["ESA"], "ESA")
        _ensure_hw(mask, item["Mask"], "Mask")
        _ensure_hw(dw, item["DW"], "DW")

        if s2 is not None:
            _ensure_chw(s2, item["S2"], "S2")
        if s1_pre is not None:
            _ensure_chw(s1_pre, item["S1_pre"], "S1_pre")
        if rain is not None:
            _ensure_chw(rain, item["GPM_Rainfall"], "GPM_Rainfall")
        if soil is not None:
            _ensure_chw(soil, item["ERA5_Soil_Moisture"], "ERA5_Soil_Moisture")

        if self.strict_channels:
            if s1 is not None and s1.shape[0] != 2:
                raise RuntimeError("S1 should have 2 bands, got {} for {}".format(s1.shape[0], item["S1"]))
            if s1_pre is not None and s1_pre.shape[0] != 2:
                raise RuntimeError("S1_pre should have 2 bands, got {} for {}".format(s1_pre.shape[0], item["S1_pre"]))
            if s2 is not None and s2.shape[0] != 4:
                raise RuntimeError("S2 should have 4 bands, got {} for {}".format(s2.shape[0], item["S2"]))
            if rain is not None and rain.shape[0] != 4:
                raise RuntimeError("GPM_Rainfall should have 4 bands, got {} for {}".format(rain.shape[0], item["GPM_Rainfall"]))
            if soil is not None and soil.shape[0] != 16:
                raise RuntimeError("ERA5_Soil_Moisture should have 16 bands, got {} for {}".format(soil.shape[0], item["ERA5_Soil_Moisture"]))

        arrays = {
            "S1": s1,
            "S1_pre": s1_pre,
            "S2": s2,
            "DEM": dem,
            "Slope": slope,
            "ESA": esa,
            "DW": dw,
            "GPM_Rainfall": rain,
            "ERA5_Soil_Moisture": soil,
            "Mask": mask,
        }
        arrays = self._maybe_flip(arrays)

        out = {}  # type: Dict[str, torch.Tensor]
        if arrays["S1"] is not None:
            out["S1"] = torch.from_numpy(self.norm.norm_s1(arrays["S1"])).float()
        if arrays["S1_pre"] is not None:
            out["S1_pre"] = torch.from_numpy(self.norm.norm_s1(arrays["S1_pre"])).float()
        if arrays["S2"] is not None:
            out["S2"] = torch.from_numpy(self.norm.norm_s2(arrays["S2"])).float()

        out["DEM"] = torch.from_numpy(self.norm.norm_dem(arrays["DEM"])).float()
        out["Slope"] = torch.from_numpy(self.norm.norm_slope(arrays["Slope"])).float()
        out["ESA"] = torch.from_numpy(self.norm.norm_esa(arrays["ESA"])).long()
        out["DW"] = torch.from_numpy(self.norm.norm_dw(arrays["DW"])).long()

        if arrays["GPM_Rainfall"] is not None:
            out["GPM_Rainfall"] = torch.from_numpy(self.norm.norm_rainfall(arrays["GPM_Rainfall"])).float()
        if arrays["ERA5_Soil_Moisture"] is not None:
            out["ERA5_Soil_Moisture"] = torch.from_numpy(
                self.norm.norm_soil_moisture(arrays["ERA5_Soil_Moisture"])
            ).float()

        if self.return_s1_pair and ("S1_pre" in out) and ("S1" in out):
            out["S1_pair"] = torch.cat([out["S1_pre"], out["S1"]], dim=0)
        if self.return_climate and ("GPM_Rainfall" in out) and ("ERA5_Soil_Moisture" in out):
            out["Climate"] = torch.cat([out["GPM_Rainfall"], out["ERA5_Soil_Moisture"]], dim=0)

        out["seg_mask"] = torch.from_numpy(self.norm.norm_mask(arrays["Mask"])).long()
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
    S1_root, S2_root, DEM_root, ESA_root, gt_root,
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
    tif_mode="both",
    use_dynamic_world=True,
    use_pre_s1=True,
    use_rainfall=True,
    use_soil_moisture=True,
    require_new_products=True,
    return_s1_pair=False,
    return_climate=False,
    reorder_s2_to_rgbn=False,
    s2_clip_max=6000.0,
    sar_fill=-50.0,
    s2_fill=-100.0,
    dem_fill=-9999.0,
    climate_fill=-9999.0,
    s1_min=-45.0,
    s1_max=25.0,
    rain_clip_max=300.0,
    rain_norm="log",
    soil_auto_scale=False,
    strict_channels=True,
):
    """
    Compatibility wrapper. Existing train.py can switch from:
        from utils.dataloader import get_loader
    to:
        from utils.dataload_v2 import get_loader

    S1_root is treated as the Benchmark root unless split_dir_root is provided.
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

    dataset = FloodAOITifDatasetV2(
        root_dir=root_dir,
        split=split,
        mode=tif_mode,
        use_dynamic_world=use_dynamic_world,
        use_pre_s1=use_pre_s1,
        use_rainfall=use_rainfall,
        use_soil_moisture=use_soil_moisture,
        require_new_products=require_new_products,
        return_s1_pair=return_s1_pair,
        return_climate=return_climate,
        augment=True,
        reorder_s2_to_rgbn=reorder_s2_to_rgbn,
        sar_fill=sar_fill,
        s2_fill=s2_fill,
        dem_fill=dem_fill,
        climate_fill=climate_fill,
        s2_clip_max=s2_clip_max,
        s1_min=s1_min,
        s1_max=s1_max,
        rain_clip_max=rain_clip_max,
        rain_norm=rain_norm,
        soil_auto_scale=soil_auto_scale,
        strict_channels=strict_channels,
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
        num_workers=0,
        tif_mode="both",
        use_pre_s1=True,
        use_rainfall=True,
        use_soil_moisture=True,
        return_s1_pair=True,
        return_climate=True,
        drop_last=False,
    )
    x, stems = next(iter(dl))
    print(stems[:2])
    for key in sorted(x.keys()):
        print(key, tuple(x[key].shape), x[key].dtype, float(x[key].min()), float(x[key].max()))
