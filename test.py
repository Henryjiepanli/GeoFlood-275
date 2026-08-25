# -*- coding: utf-8 -*-
"""Evaluate GeoFloodNet on Val/Test and save visual results.

Example:
    python test.py \
        --model floodnet \
        --dataset_root /path/to/GeoFlood-275/Benchmark_all \
        --load ./Experiments/GeoFloodNet/Val_best.pth \
        --batchsize 4 \
        --amp
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from utils.dataloader import get_loader
from network.GeoFloodNet import FloodNet


def parse_ws(s):
    xs = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not xs:
        raise ValueError("Empty window set, e.g. '4,8'")
    return xs


def parse_ints(s):
    xs = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not xs:
        raise ValueError("Expected comma separated integer list")
    return xs


def build_model(cfg):
    model_name = cfg.model.lower()
    if model_name == "floodnet":
        return FloodNet(
            num_classes=cfg.num_classes,
            base=cfg.base,
            depths=(2, 2, 2, 2),
            heads=(2, 4, 8, 8),
            drop=cfg.drop,
            win8=tuple(cfg.win8),
            win16=tuple(cfg.win16),
            win32=tuple(cfg.win32),
            psdp_mix=bool(cfg.psdp_mix),
        )
    raise ValueError("Unsupported model: {}".format(cfg.model))


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def normalize_state_dict_keys(state_dict, model):
    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())
    if not state_keys:
        return state_dict

    model_has_module = model_keys[0].startswith("module.")
    state_has_module = state_keys[0].startswith("module.")

    if state_has_module and not model_has_module:
        return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    if model_has_module and not state_has_module:
        return {"module." + k: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, ckpt_path, strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = normalize_state_dict_keys(extract_state_dict(ckpt), model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Checkpoint does not strictly match model. missing={}, unexpected={}".format(
                len(missing), len(unexpected)
            )
        )

    print("[INFO] Loaded checkpoint: {}".format(ckpt_path))
    if missing:
        print("[WARN] Missing keys: {}{}".format(missing[:10], " ..." if len(missing) > 10 else ""))
    if unexpected:
        print("[WARN] Unexpected keys: {}{}".format(unexpected[:10], " ..." if len(unexpected) > 10 else ""))


def forward_model(model, sample, device, use_amp=False):
    s1 = sample["S1"].to(device, non_blocking=True)
    s2 = sample["S2"].to(device, non_blocking=True)
    esa = sample["ESA"].to(device, non_blocking=True).long().unsqueeze(1)
    dw = sample["DW"].to(device, non_blocking=True).long().unsqueeze(1)
    slope = sample["Slope"].to(device, non_blocking=True).float().unsqueeze(1)
    gts = sample["seg_mask"].to(device, non_blocking=True).long()

    with torch.cuda.amp.autocast(enabled=bool(use_amp) and device.type == "cuda"):
        logits = model(s2, s1, slope, esa, dw)
        if isinstance(logits, (tuple, list)):
            logits = logits[-1]
        if logits.shape[-2:] != gts.shape[-2:]:
            logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
    return logits, gts


@torch.no_grad()
def update_confusion_matrix(confmat, gt, pred, num_classes, ignore_index=None):
    if ignore_index is not None:
        mask = gt != ignore_index
        gt = gt[mask]
        pred = pred[mask]

    gt = gt.reshape(-1).long()
    pred = pred.reshape(-1).long()
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt = gt[valid]
    pred = pred[valid]

    idx = gt * num_classes + pred
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    confmat += bins.reshape(num_classes, num_classes)


def compute_metrics(confmat, positive_class=1):
    conf = confmat.detach().cpu().numpy().astype(np.float64)
    total = conf.sum()
    correct = np.trace(conf)
    oa = correct / max(total, 1.0)

    if positive_class >= conf.shape[0]:
        raise ValueError("positive_class={} out of range for {} classes".format(positive_class, conf.shape[0]))

    tp = conf[positive_class, positive_class]
    gt_sum = conf[positive_class, :].sum()
    pred_sum = conf[:, positive_class].sum()
    fp = pred_sum - tp
    fn = gt_sum - tp
    tn = total - tp - fp - fn
    eps = 1e-12

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "oa": float(oa),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "total": int(total),
    }


def to_uint8_rgb(s2_tensor, channels, percentile=(2.0, 98.0)):
    s2 = s2_tensor.detach().float().cpu().numpy()
    c, h, w = s2.shape
    chosen = []
    for ch in channels[:3]:
        chosen.append(s2[ch] if ch < c else s2[min(c - 1, 0)])
    while len(chosen) < 3:
        chosen.append(chosen[-1])

    rgb = np.stack(chosen, axis=-1)
    rgb = np.clip((rgb + 1.0) * 0.5, 0.0, 1.0)

    lo_p, hi_p = percentile
    if hi_p > lo_p:
        out = np.empty_like(rgb)
        for ch in range(3):
            lo, hi = np.percentile(rgb[..., ch], [lo_p, hi_p])
            if hi > lo:
                out[..., ch] = np.clip((rgb[..., ch] - lo) / (hi - lo), 0.0, 1.0)
            else:
                out[..., ch] = rgb[..., ch]
        rgb = out

    return (rgb * 255.0).round().astype(np.uint8)


def mask_to_gray(mask):
    return (mask.astype(np.uint8) * 255)


def error_to_color(gt, pred, ignore_index=None):
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = (35, 35, 35)       # TN background
    out[(gt == 1) & (pred == 1)] = (40, 190, 90)    # TP
    out[(gt != 1) & (pred == 1)] = (230, 70, 60)    # FP
    out[(gt == 1) & (pred != 1)] = (70, 130, 255)   # FN
    if ignore_index is not None:
        out[gt == ignore_index] = (150, 150, 150)
    return out


def add_title(img, title):
    title_h = 24
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + title_h), (20, 20, 20))
    canvas.paste(Image.fromarray(img), (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(255, 255, 255))
    return canvas


def save_visualization(s2, gt, pred, stem, out_dir, rgb_channels, percentile, ignore_index=None):
    out_dir = Path(out_dir)
    pred_dir = out_dir / "pred_masks"
    vis_dir = out_dir / "visualizations"
    pred_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    gt_np = gt.detach().cpu().numpy().astype(np.int64)
    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    rgb = to_uint8_rgb(s2, rgb_channels, percentile=percentile)
    gt_img = mask_to_gray(gt_np)
    pred_img = mask_to_gray(pred_np)
    err_img = error_to_color(gt_np, pred_np, ignore_index=ignore_index)

    Image.fromarray(pred_img).save(pred_dir / (stem + ".png"))

    panels = [
        add_title(rgb, "S2 RGB"),
        add_title(gt_img, "GT"),
        add_title(pred_img, "Pred"),
        add_title(err_img, "Error TP/FP/FN"),
    ]
    width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(vis_dir / (stem + ".png"))


def build_loader(cfg, split):
    return get_loader(
        cfg.dataset_root,
        None,
        None,
        None,
        None,
        cfg.trainsize,
        split.lower(),
        cfg.batchsize,
        num_workers=cfg.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=bool(cfg.num_workers > 0),
        drop_last=False,
        DW_root=None,
        Slope_root=None,
    )


@torch.no_grad()
def evaluate_split(model, cfg, split, device, output_root):
    loader = build_loader(cfg, split)
    confmat = torch.zeros((cfg.num_classes, cfg.num_classes), dtype=torch.int64, device=device)
    split_out = Path(output_root) / split
    split_out.mkdir(parents=True, exist_ok=True)

    saved_vis = 0
    pbar = tqdm(loader, desc="Eval-{}".format(split), ncols=120)
    for batch_idx, (sample, stems) in enumerate(pbar, start=1):
        if cfg.max_batches > 0 and batch_idx > cfg.max_batches:
            break
        logits, gts = forward_model(model, sample, device, use_amp=cfg.amp)
        pred = torch.argmax(logits, dim=1)
        update_confusion_matrix(confmat, gts, pred, cfg.num_classes, ignore_index=cfg.ignore_index)

        if not cfg.no_vis and (cfg.vis_limit < 0 or saved_vis < cfg.vis_limit):
            s2_cpu = sample["S2"]
            gts_cpu = gts.detach().cpu()
            pred_cpu = pred.detach().cpu()
            for b, stem in enumerate(stems):
                if cfg.vis_limit >= 0 and saved_vis >= cfg.vis_limit:
                    break
                save_visualization(
                    s2_cpu[b],
                    gts_cpu[b],
                    pred_cpu[b],
                    str(stem),
                    split_out,
                    cfg.vis_rgb_channels,
                    cfg.vis_percentile,
                    ignore_index=cfg.ignore_index,
                )
                saved_vis += 1

    metrics = compute_metrics(confmat, positive_class=cfg.positive_class)
    metrics["split"] = split
    metrics["num_samples"] = len(loader.dataset)
    metrics["visualizations"] = saved_vis

    with open(split_out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(split_out / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(confmat.detach().cpu().tolist(), f, indent=2)

    print(
        "{} | IoU={:.4f} F1={:.4f} Precision={:.4f} Recall={:.4f} OA={:.4f} | samples={} vis={}".format(
            split,
            metrics["iou"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["oa"],
            metrics["num_samples"],
            metrics["visualizations"],
        )
    )
    return metrics


def write_summary(metrics_list, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "split", "iou", "f1", "precision", "recall", "oa",
        "tp", "fp", "fn", "tn", "total", "num_samples", "visualizations",
    ]
    with open(output_root / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_list:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def default_output_dir(cfg):
    ckpt_path = Path(cfg.load).resolve()
    return str(ckpt_path.parent / "eval_results")


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoFloodNet on Val/Test splits and save visualizations.")
    parser.add_argument("--model", type=str, default="floodnet", choices=["floodnet"])
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--load", type=str, required=True, help="Checkpoint path, e.g. Val_best.pth or current_state.pth")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--splits", type=str, default="Val,Test", help="Comma separated split names: Val,Test")
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1, help="Debug only. -1 evaluates the full split.")
    parser.add_argument("--trainsize", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--ignore_index", type=int, default=None)
    parser.add_argument("--positive_class", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--strict_load", action="store_true")

    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--vis_limit", type=int, default=-1, help="Max visualizations per split. -1 saves all.")
    parser.add_argument("--vis_rgb_channels", type=str, default="0,1,2",
                        help="S2 channels used as RGB, e.g. 0,1,2 or 1,2,3")
    parser.add_argument("--vis_percentile", type=str, default="2,98",
                        help="Percentile stretch for RGB visualization. Use 0,0 to disable.")

    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--psdp_mix", action="store_true")
    parser.add_argument("--win8", type=str, default="2,4,8")
    parser.add_argument("--win16", type=str, default="2,4,8")
    parser.add_argument("--win32", type=str, default="2,4,8")

    cfg = parser.parse_args()
    cfg.win8 = parse_ws(cfg.win8)
    cfg.win16 = parse_ws(cfg.win16)
    cfg.win32 = parse_ws(cfg.win32)
    cfg.vis_rgb_channels = parse_ints(cfg.vis_rgb_channels)
    vp = [float(x) for x in str(cfg.vis_percentile).split(",") if x.strip()]
    if len(vp) != 2:
        raise ValueError("--vis_percentile expects two numbers, e.g. 2,98")
    cfg.vis_percentile = tuple(vp)
    cfg.splits = [x.strip() for x in cfg.splits.split(",") if x.strip()]
    cfg.output_dir = cfg.output_dir or default_output_dir(cfg)
    return cfg


def main():
    cfg = get_args()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("[INFO] Building model: {}".format(cfg.model))
    model = build_model(cfg)
    load_checkpoint(model, cfg.load, strict=cfg.strict_load)
    model.to(device)
    model.eval()

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for split in cfg.splits:
        all_metrics.append(evaluate_split(model, cfg, split, device, cfg.output_dir))
    write_summary(all_metrics, cfg.output_dir)
    print("[INFO] Results saved to: {}".format(Path(cfg.output_dir).resolve()))


if __name__ == "__main__":
    main()
