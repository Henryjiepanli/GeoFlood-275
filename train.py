# -*- coding: utf-8 -*-

import os
import sys
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.dataloader import get_loader
from utils.utils import clip_gradient
from network.GeoFloodNet import FloodNet


# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed: int = 2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# -------------------------
# Pure poly LR (NO warmup)
# -------------------------
class IterPolyLR:
    """
    Poly decay: base_lr -> min_lr over total_iters
      lr = (base_lr - min_lr) * (1 - t)^power + min_lr
      t = cur_iter / (total_iters - 1)
    """
    def __init__(self, optimizer, base_lr: float, total_iters: int, min_lr: float = 1e-6, power: float = 0.9):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.total_iters = int(max(total_iters, 1))
        self.min_lr = float(min_lr)
        self.power = float(power)

    def get_lr(self, cur_iter: int) -> float:
        cur_iter = int(max(cur_iter, 0))
        if self.total_iters <= 1:
            return self.base_lr
        t = float(cur_iter) / float(self.total_iters - 1)
        t = min(max(t, 0.0), 1.0)
        lr = (self.base_lr - self.min_lr) * ((1.0 - t) ** self.power) + self.min_lr
        return lr

    def step(self, cur_iter: int) -> float:
        lr = self.get_lr(cur_iter)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# -------------------------
# GPU confusion-matrix metrics
# -------------------------
@torch.no_grad()
def update_confusion_matrix(confmat: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor,
                            num_classes: int, ignore_index=None):
    if ignore_index is not None:
        mask = (gt != ignore_index)
        gt = gt[mask]
        pred = pred[mask]

    gt = gt.view(-1).long()
    pred = pred.view(-1).long()

    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt = gt[valid]
    pred = pred[valid]

    idx = gt * num_classes + pred
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    confmat += bins.view(num_classes, num_classes)


@torch.no_grad()
def confmat_to_binary_stats(confmat: torch.Tensor, positive_class: int = 1):
    tp = confmat[positive_class, positive_class]
    gt_sum = confmat[positive_class, :].sum()
    pred_sum = confmat[:, positive_class].sum()
    fp = pred_sum - tp
    fn = gt_sum - tp

    eps = 1e-12
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return iou, f1, precision, recall


# -------------------------
# Trainer
# -------------------------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- build GeoFloodNet model ----
        self.model = FloodNet(
            num_classes=self.cfg.num_classes,
            base=self.cfg.base,
            depths=(2, 2, 2, 2),
            heads=(2, 4, 8, 8),
            drop=self.cfg.drop,
            win8=tuple(self.cfg.win8),
            win16=tuple(self.cfg.win16),
            win32=tuple(self.cfg.win32),
            psdp_mix=bool(self.cfg.psdp_mix),
            # geo_ch = self.cfg.geo_ch,
        ).to(self.device)

        # ---- DataParallel ----
        self.is_dp = False
        if torch.cuda.device_count() > 1 and (not getattr(self.cfg, "force_single_gpu", False)):
            print("[INFO] Using {} GPUs (DataParallel)".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
            self.is_dp = True

        # ---- optimizer / loss / amp ----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        ignore_index = self.cfg.ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=int(ignore_index)).to(self.device) \
            if ignore_index is not None else nn.CrossEntropyLoss().to(self.device)

        self.use_amp = bool(self.cfg.amp) and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ---- logging ----
        os.makedirs(self.cfg.save_path, exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.cfg.save_path, "summary"))
        self._setup_logging()

        self.best_iou = float("-inf")
        self.best_epoch = 0

        self.global_iter = 0
        self.start_epoch = 1

        # ---- optionally load pretrained encoders ----
        self._maybe_load_pretrained_encoders()

        # ---- data (TIF-based) ----
        dataset_root = self.cfg.dataset_root

        self.train_loader = get_loader(
            dataset_root, None, None, None, None,
            self.cfg.trainsize, "train", self.cfg.batchsize,
            num_workers=self.cfg.num_workers, shuffle=True, pin_memory=True,
            DW_root=None, Slope_root=None,
        )
        self.val_loader = get_loader(
            dataset_root, None, None, None, None,
            self.cfg.trainsize, "val", self.cfg.batchsize,
            num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True,
            DW_root=None, Slope_root=None,
        )

        self.test_loader = get_loader(
            dataset_root, None, None, None, None,
            self.cfg.trainsize, "test", self.cfg.batchsize,
            num_workers=self.cfg.num_workers, shuffle=False, pin_memory=True,
            DW_root=None, Slope_root=None,
        )

        # ---- LR scheduler (NO warmup) ----
        iters_per_epoch = len(self.train_loader)
        self.total_iters = int(self.cfg.epoch * iters_per_epoch)
        self.lr_sched = IterPolyLR(
            optimizer=self.optimizer,
            base_lr=self.cfg.lr,
            total_iters=self.total_iters,
            min_lr=self.cfg.min_lr,
            power=self.cfg.poly_power,
        )

        # ---- resume training (full ckpt) ----
        if getattr(self.cfg, "load", None):
            self.start_epoch = self._load_any(self.cfg.load)

        logging.info(
            "IterLR(no-warmup) total_iters={} iters/epoch={} base_lr={} min_lr={} power={}".format(
                self.total_iters, iters_per_epoch, self.cfg.lr, self.cfg.min_lr, self.cfg.poly_power
            )
        )

    # -------------------------
    # init helpers
    # -------------------------
    def _setup_logging(self):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=os.path.join(self.cfg.save_path, "log.log"),
            format="[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]",
            level=logging.INFO,
            filemode="a",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("GeoFloodNet train (polyLR-no-warmup fast-metrics, TIF)")
        logging.info("Config: {}".format(self.cfg))
        sys.stdout.flush()

    def _get_state_dict(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()

    def _maybe_load_pretrained_encoders(self):
        pre_s1 = getattr(self.cfg, "pre_s1", None)
        pre_s2 = getattr(self.cfg, "pre_s2", None)
        if (not pre_s1) and (not pre_s2):
            return

        target = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if not hasattr(target, "load_pretrained_encoders"):
            print("[WARN] Model has no load_pretrained_encoders(); skip pretrained load.")
            return

        print("[INFO] Loading pretrained encoders:")
        print("  pre_s2 =", pre_s2)
        print("  pre_s1 =", pre_s1)
        target.load_pretrained_encoders(s2_path=pre_s2, s1_path=pre_s1, strict=False, verbose=True)

    def _load_any(self, path: str):
        ckpt = torch.load(path, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and self.use_amp and ckpt["scaler_state_dict"] is not None:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])

            self.best_iou = ckpt.get("best_iou", float("-inf"))
            self.best_epoch = ckpt.get("best_epoch", 0)

            if "global_iter" in ckpt:
                self.global_iter = int(ckpt["global_iter"])
            else:
                prev_epoch = int(ckpt.get("epoch", 0))
                self.global_iter = prev_epoch * len(self.train_loader)

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            logging.info("Resumed from {}, start_epoch={}, global_iter={}".format(path, start_epoch, self.global_iter))
            print("[INFO] Resuming training from epoch {}, global_iter={}".format(start_epoch, self.global_iter))
            return start_epoch

        # state_dict only
        self.model.load_state_dict(ckpt, strict=False)
        logging.info("Loaded state_dict: {}".format(path))
        print("[INFO] Loaded weights from {} (state_dict)".format(path))
        return 1

    # -------------------------
    # forward / loops
    # -------------------------
    def _forward_model(self, sample):
        s1 = sample["S1"].to(self.device, non_blocking=True)
        s2 = sample["S2"].to(self.device, non_blocking=True)
        esa = sample["ESA"].to(self.device, non_blocking=True).long().unsqueeze(1)
        dw = sample["DW"].to(self.device, non_blocking=True).long().unsqueeze(1)
        slope = sample["Slope"].to(self.device, non_blocking=True).float().unsqueeze(1)
        gts = sample["seg_mask"].to(self.device, non_blocking=True).long()

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits = self.model(s2, s1, slope, esa, dw)
            if logits.shape[-2:] != gts.shape[-2:]:
                logits = F.interpolate(logits, size=gts.shape[-2:], mode="bilinear", align_corners=False)
            loss_ce = self.ce_loss(logits, gts)

        return loss_ce, logits, gts

    def train_one_epoch(self, epoch: int):
        self.model.train()
        confmat = torch.zeros((self.cfg.num_classes, self.cfg.num_classes), dtype=torch.int64, device=self.device)

        loss_all = 0.0
        n_steps = 0

        pbar = tqdm(self.train_loader, desc="Train-{:03d}".format(epoch), ncols=140)
        for i, (sample, stems) in enumerate(pbar, start=1):
            cur_lr = self.lr_sched.step(self.global_iter)
            self.optimizer.zero_grad(set_to_none=True)

            loss, logits, gts = self._forward_model(sample)

            if not torch.isfinite(loss):
                print("[WARN] Non-finite loss. Skip step.")
                self.optimizer.zero_grad(set_to_none=True)
                self.global_iter += 1
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.cfg.clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradient(self.optimizer, self.cfg.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.clip > 0:
                    clip_gradient(self.optimizer, self.cfg.clip)
                self.optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                update_confusion_matrix(confmat, gts, pred, self.cfg.num_classes, ignore_index=self.cfg.ignore_index)

            self.global_iter += 1
            n_steps += 1
            loss_all += float(loss.detach().item())

            if i % self.cfg.log_every == 0 or i == 1:
                iou, f1, prec, rec = confmat_to_binary_stats(confmat, positive_class=1)
                pbar.set_postfix({
                    "loss": "{:.3f}".format(loss.item()),
                    "fIoU": "{:.3f}".format(float(iou.item())),
                    "lr": "{:.2e}".format(cur_lr),
                    "it": "{}/{}".format(self.global_iter, self.total_iters),
                })

        iou, f1, prec, rec = confmat_to_binary_stats(confmat, positive_class=1)
        loss_avg = loss_all / max(n_steps, 1)

        # epoch-only tensorboard logging
        self.writer.add_scalar("Train/fIoU", float(iou.item()), epoch)
        self.writer.add_scalar("Train/fF1", float(f1.item()), epoch)
        self.writer.add_scalar("Train/fPrecision", float(prec.item()), epoch)
        self.writer.add_scalar("Train/fRecall", float(rec.item()), epoch)
        self.writer.add_scalar("Train/loss_epoch", loss_avg, epoch)
        self.writer.add_scalar("Train/lr_epoch", float(self.optimizer.param_groups[0]["lr"]), epoch)

        msg = (
            "{} Epoch {:03d} Train | fIoU={:.4f} fF1={:.4f} fP={:.4f} fR={:.4f} | "
            "loss={:.4f} | lr={:.2e} | iter={}/{}"
        ).format(
            datetime.now(), epoch,
            float(iou.item()), float(f1.item()), float(prec.item()), float(rec.item()),
            loss_avg,
            float(self.optimizer.param_groups[0]["lr"]),
            self.global_iter, self.total_iters
        )
        print(msg)
        logging.info(msg)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": self._get_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_iou": self.best_iou,
            "best_epoch": self.best_epoch,
            "global_iter": self.global_iter,
        }
        torch.save(ckpt, os.path.join(self.cfg.save_path, "current_state.pth"))

    @torch.no_grad()
    def evaluate(self, epoch: int, loader, split_name: str, save_best_on_val: bool):
        self.model.eval()
        confmat = torch.zeros((self.cfg.num_classes, self.cfg.num_classes), dtype=torch.int64, device=self.device)

        pbar = tqdm(loader, desc="Eval-{}-{:03d}".format(split_name, epoch), ncols=140)
        for sample, stems in pbar:
            loss, logits, gts = self._forward_model(sample)
            pred = torch.argmax(logits, dim=1)
            update_confusion_matrix(confmat, gts, pred, self.cfg.num_classes, ignore_index=self.cfg.ignore_index)

        iou, f1, prec, rec = confmat_to_binary_stats(confmat, positive_class=1)

        self.writer.add_scalar("{}/fIoU".format(split_name), float(iou.item()), epoch)
        self.writer.add_scalar("{}/fF1".format(split_name), float(f1.item()), epoch)
        self.writer.add_scalar("{}/fPrecision".format(split_name), float(prec.item()), epoch)
        self.writer.add_scalar("{}/fRecall".format(split_name), float(rec.item()), epoch)

        msg = (
            "{} Epoch {:03d} {} | fIoU={:.4f} fF1={:.4f} fP={:.4f} fR={:.4f} | bestEpoch={} bestIoU={:.4f}"
        ).format(
            datetime.now(), epoch, split_name,
            float(iou.item()), float(f1.item()), float(prec.item()), float(rec.item()),
            self.best_epoch, self.best_iou
        )
        print(msg)
        logging.info(msg)

        if save_best_on_val and float(iou.item()) >= self.best_iou:
            self.best_iou = float(iou.item())
            self.best_epoch = epoch

            torch.save(self._get_state_dict(), os.path.join(self.cfg.save_path, split_name + "_best.pth"))
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": self._get_state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "best_iou": self.best_iou,
                "best_epoch": self.best_epoch,
                "global_iter": self.global_iter,
            }
            torch.save(best_ckpt, os.path.join(self.cfg.save_path, split_name + "best_state.pth"))
            logging.info("[SAVE] BEST on Val: epoch={}, IoU={:.4f}".format(epoch, self.best_iou))

        return float(iou.item())

    def run(self):
        print("[INFO] iters/epoch={}, total_iters={} (NO warmup, poly)".format(len(self.train_loader), self.total_iters))
        for epoch in range(self.start_epoch, self.cfg.epoch + 1):
            self.train_one_epoch(epoch)
            self.evaluate(epoch, self.val_loader, "Val", save_best_on_val=True)
            self.evaluate(epoch, self.test_loader, "Test", save_best_on_val=False)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import argparse

    def parse_ws(s):
        xs = [int(x.strip()) for x in s.split(",") if x.strip()]
        if len(xs) == 0:
            raise ValueError("Empty window set, e.g. '4,8'")
        return xs

    parser = argparse.ArgumentParser()

    # train
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--trainsize", type=int, default=512)
    parser.add_argument("--clip", type=float, default=0.5)
    parser.add_argument("--num_classes", type=int, default=2)

    # runtime
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--force_single_gpu", action="store_true")

    # optimizer
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # model cfg
    parser.add_argument("--base", type=int, default=48)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--psdp_mix", action="store_true")
    parser.add_argument("--geo_ch", type=int, default=64)
    parser.add_argument("--win8", type=str, default="2,4,8")
    parser.add_argument("--win16", type=str, default="2,4,8")
    parser.add_argument("--win32", type=str, default="2,4,8")

    # pretrained encoder
    parser.add_argument("--pre_s1", type=str, default=None)
    parser.add_argument("--pre_s2", type=str, default=None)

    # poly LR (no warmup)
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # labels
    parser.add_argument("--ignore_index", type=int, default=None)

    # resume / save
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./Experiments/FloodNet")

    # TIF dataset root
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root dir containing Train/Val/Test folders of TIF patches.")

    opt = parser.parse_args()

    # parse window sets
    opt.win8 = parse_ws(opt.win8)
    opt.win16 = parse_ws(opt.win16)
    opt.win32 = parse_ws(opt.win32)

    os.makedirs(opt.save_path, exist_ok=True)
    seed_everything(2333)

    trainer = Trainer(opt)
    trainer.run()
