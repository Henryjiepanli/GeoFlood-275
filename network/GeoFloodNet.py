# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Norm / Basic Blocks
# ============================================================
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ConvFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 3, drop: float = 0.0):
        super().__init__()
        hidden = dim * expansion
        self.pw1 = nn.Conv2d(dim, hidden, 1, bias=True)
        self.act = nn.GELU()
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.pw2 = nn.Conv2d(hidden, dim, 1, bias=True)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        x = self.act(self.pw1(x))
        x = self.act(self.dw(x))
        x = self.drop(x)
        return self.pw2(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, k=3, s=1, p=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=k, stride=s, padding=p, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ============================================================
# Window ops
# ============================================================
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, (H, W, window_size)
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, C: int, B: int) -> torch.Tensor:
    nH = H // window_size
    nW = W // window_size
    x = windows.view(B, nH, nW, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, C, H, W)


# ============================================================
# Window Self Attention
# ============================================================
class WindowSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.permute(0, 2, 1, 3) * self.scale
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(BnW, N, C)
        return self.proj_drop(self.proj(out))


# ============================================================
# Multi-Window Dynamic Fusion Block
# ============================================================
class MultiWindowTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_sizes: List[int], drop: float = 0.0):
        super().__init__()
        assert len(window_sizes) >= 1
        self.window_sizes = list(window_sizes)

        self.norm1 = LayerNorm2d(dim)
        self.attn_branches = nn.ModuleList([
            WindowSelfAttention(dim, num_heads, attn_drop=drop, proj_drop=drop)
            for _ in self.window_sizes
        ])

        K = len(self.window_sizes)
        hidden = max(dim // 2, 32)
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, K, 1, bias=True),
        )

        self.norm2 = LayerNorm2d(dim)
        self.ffn = ConvFFN(dim, expansion=3, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x
        x1 = self.norm1(x)

        g = F.adaptive_avg_pool2d(x1, 1)
        logits = self.gate_mlp(g).flatten(1)
        w = torch.softmax(logits, dim=1)

        outs = []
        for ws, attn in zip(self.window_sizes, self.attn_branches):
            assert H % ws == 0 and W % ws == 0, (H, W, ws)
            win = window_partition(x1, ws)
            win = attn(win)
            out = window_reverse(win, ws, H, W, C, B)
            outs.append(out)

        out_sum = 0.0
        for k, out in enumerate(outs):
            out_sum = out_sum + out * w[:, k].view(B, 1, 1, 1)

        x = shortcut + out_sum
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# Encoder blocks
# ============================================================
class ConvStemStride4(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(out_ch // 2),
            nn.GELU(),
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ConvStage(nn.Module):
    def __init__(self, dim: int, depth: int = 2):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks += [
                nn.Conv2d(dim, dim, 3, padding=1, bias=False),
                LayerNorm2d(dim),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class Downsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class EncoderStageMW(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, window_sizes: List[int], drop: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            MultiWindowTransformerBlock(dim=dim, num_heads=heads, window_sizes=window_sizes, drop=drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ============================================================
# Learnable PSDP
# ============================================================
class LearnablePSDP(nn.Module):
    def __init__(self, init_eps=1e-4, init_floor=1e-2, mix=True):
        super().__init__()
        self.eps_u = nn.Parameter(torch.tensor(float(init_eps)).log().view(1))
        self.floor_u = nn.Parameter(torch.tensor(float(init_floor)).log().view(1))
        self.mix = bool(mix)
        if self.mix:
            self.mixer = nn.Sequential(
                nn.Conv2d(5, 16, 1, bias=True),
                LayerNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 5, 1, bias=True),
            )

    def forward(self, s1: torch.Tensor) -> torch.Tensor:
        eps = F.softplus(self.eps_u) + 1e-12
        floor = F.softplus(self.floor_u) + 1e-12

        vv = s1[:, 0:1]
        vh = s1[:, 1:2]

        vv_abs = vv.abs().clamp_min(floor)
        vh_abs = vh.abs().clamp_min(floor)

        r = (vh_abs / (vv_abs + eps)).clamp(0.0, 10.0)
        d = (torch.log(vv_abs + eps) - torch.log(vh_abs + eps)).clamp(-5.0, 5.0)
        s = torch.sqrt(vv * vv + vh * vh + eps).clamp(0.0, 5.0)

        r = torch.tanh(r)
        d = torch.tanh(d)
        s = torch.tanh(s)

        psdp = torch.cat([vv, vh, r, d, s], dim=1)
        if self.mix:
            psdp = psdp + self.mixer(psdp)
        return psdp


# ============================================================
# MW Encoders
# ============================================================
class S2SemanticEncoderMW(nn.Module):
    def __init__(self, base=48, depths=(2, 2, 2, 2), heads=(2, 4, 8, 8),
                 win8=(4, 8), win16=(4, 8), win32=(4, 8), drop=0.0):
        super().__init__()
        self.stem4 = ConvStemStride4(4, base)
        self.stage4 = ConvStage(base, depth=2)

        self.down8 = Downsample(base, base * 2)
        self.stage8 = EncoderStageMW(base * 2, depths[1], heads[1], list(win8), drop)

        self.down16 = Downsample(base * 2, base * 4)
        self.stage16 = EncoderStageMW(base * 4, depths[2], heads[2], list(win16), drop)

        self.down32 = Downsample(base * 4, base * 8)
        self.stage32 = EncoderStageMW(base * 8, depths[3], heads[3], list(win32), drop)

    def forward(self, x):
        x4 = self.stage4(self.stem4(x))
        x8 = self.stage8(self.down8(x4))
        x16 = self.stage16(self.down16(x8))
        x32 = self.stage32(self.down32(x16))
        return [x4, x8, x16, x32]


class S1ScatteringEncoderMW(nn.Module):
    def __init__(self, base=48, depths=(2, 2, 2, 2), heads=(2, 4, 8, 8),
                 win8=(4, 8), win16=(4, 8), win32=(4, 8), drop=0.0, psdp_mix=False):
        super().__init__()
        self.psdp = LearnablePSDP(mix=psdp_mix)
        self.stem4 = ConvStemStride4(5, base)
        self.stage4 = ConvStage(base, depth=2)

        self.down8 = Downsample(base, base * 2)
        self.stage8 = EncoderStageMW(base * 2, depths[1], heads[1], list(win8), drop)

        self.down16 = Downsample(base * 2, base * 4)
        self.stage16 = EncoderStageMW(base * 4, depths[2], heads[2], list(win16), drop)

        self.down32 = Downsample(base * 4, base * 8)
        self.stage32 = EncoderStageMW(base * 8, depths[3], heads[3], list(win32), drop)

    def forward(self, x):
        x = self.psdp(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise RuntimeError("NaN/Inf after LearnablePSDP")
        x4 = self.stage4(self.stem4(x))
        x8 = self.stage8(self.down8(x4))
        x16 = self.stage16(self.down16(x8))
        x32 = self.stage32(self.down32(x16))
        return [x4, x8, x16, x32]

    


class GeoPrompt(nn.Module):
    def __init__(self, base=48, num_esa=12, num_dw=10, drop=0.0):
        super().__init__()
        self.num_esa = num_esa
        self.num_dw = num_dw
        geo_ch = max(base // 2, 24)

        self.esa_emb = nn.Embedding(num_esa, geo_ch)
        self.dw_emb = nn.Embedding(num_dw, geo_ch)

        self.slope_net = nn.Sequential(
            nn.Conv2d(1, geo_ch, 3, padding=1, bias=False),
            LayerNorm2d(geo_ch),
            nn.GELU(),
            nn.Conv2d(geo_ch, geo_ch, 3, padding=1, bias=False),
            LayerNorm2d(geo_ch),
            nn.GELU(),
        )

        self.rel_head = nn.Sequential(
            nn.Conv2d(geo_ch * 4, geo_ch, 1, bias=False),
            LayerNorm2d(geo_ch),
            nn.GELU(),
            nn.Conv2d(geo_ch, 1, 1, bias=True),
        )

        self.fuse0 = nn.Sequential(
            nn.Conv2d(geo_ch * 3, base, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(base),
            nn.GELU(),
            nn.Conv2d(base, base, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(base),
            nn.GELU(),
            nn.Dropout2d(drop),
        )

        self.down8 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(base * 2),
            nn.GELU(),
        )
        self.down16 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(base * 4),
            nn.GELU(),
        )
        self.down32 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1, bias=False),
            LayerNorm2d(base * 8),
            nn.GELU(),
        )

    def forward(self, esa: torch.Tensor, dw: torch.Tensor, slope: torch.Tensor):
        esa_i = esa.squeeze(1).clamp(0, self.num_esa - 1).long()
        dw_i = dw.squeeze(1).clamp(0, self.num_dw - 1).long()
        slope = slope.float().clamp(0.0, 1.0)

        esa_f = self.esa_emb(esa_i).permute(0, 3, 1, 2).contiguous()
        dw_f = self.dw_emb(dw_i).permute(0, 3, 1, 2).contiguous()
        sl_f = self.slope_net(slope)

        diff = (dw_f - esa_f).abs()
        r_logits = self.rel_head(torch.cat([esa_f, dw_f, diff, sl_f], dim=1))
        r = torch.sigmoid(r_logits)

        prior = r * dw_f + (1.0 - r) * esa_f
        g0 = torch.cat([prior, sl_f, diff], dim=1)

        g4 = self.fuse0(g0)
        g8 = self.down8(g4)
        g16 = self.down16(g8)
        g32 = self.down32(g16)

        return g4, g8, g16, g32, r


# ============================================================
# Geo Modulation
# ============================================================
class GeoFeatureModulation(nn.Module):
    def __init__(self, feat_dim: int, geo_dim: int):
        super().__init__()
        hidden = max(feat_dim // 2, 32)

        self.geo_proj = nn.Sequential(
            nn.Conv2d(geo_dim, hidden, 1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
        )
        self.to_scale = nn.Conv2d(hidden, feat_dim, 1, bias=True)
        self.to_bias = nn.Conv2d(hidden, feat_dim, 1, bias=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = self.geo_proj(g)
        scale = torch.tanh(self.to_scale(h))
        bias = self.to_bias(h)
        return x * (1.0 + scale) + bias


# ============================================================
# Reliability-aware Fusion
# ============================================================
class ReliabilityAwareFusion(nn.Module):
    """
    Inputs:
      f2 : modulated S2 feature
      f1 : modulated S1 feature
      g  : geo feature
      r  : reliability map at the same scale, shape (B,1,H,W)

    Output:
      fused feature with same channel dim as f2/f1
    """
    def __init__(self, feat_dim: int, geo_dim: int, drop: float = 0.0):
        super().__init__()
        hidden = max(feat_dim, 32)

        # predict branch weights
        self.weight_head = nn.Sequential(
            nn.Conv2d(feat_dim * 2 + geo_dim + 1, hidden, 1, bias=False),
            LayerNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 2, 1, bias=True),
        )

        # fuse weighted features + explicit diff
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim * 3, feat_dim, 1, bias=False),
            LayerNorm2d(feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1, bias=False),
            LayerNorm2d(feat_dim),
            nn.GELU(),
            nn.Dropout2d(drop),
        )

    def forward(self, f2: torch.Tensor, f1: torch.Tensor, g: torch.Tensor, r: torch.Tensor):
        weight_logits = self.weight_head(torch.cat([f2, f1, g, r], dim=1))
        weights = torch.softmax(weight_logits, dim=1)

        w2 = weights[:, 0:1]
        w1 = weights[:, 1:2]

        f2_w = f2 * w2
        f1_w = f1 * w1
        fd = (f2 - f1).abs()

        out = self.fuse(torch.cat([f2_w, f1_w, fd], dim=1))
        return out, w2, w1

# ============================================================
# UNet Decoder Blocks
# ============================================================
class UpUNet(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, drop: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            LayerNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(drop),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(torch.cat([x, skip], dim=1))


# ============================================================
# Baseline + GeoMod+
# ============================================================
class FloodNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        base: int = 48,
        depths=(2, 2, 2, 2),
        heads=(2, 4, 8, 8),
        drop: float = 0.0,
        win8=(4, 8),
        win16=(4, 8),
        win32=(4, 8),
        psdp_mix: bool = False,
        num_esa: int = 12,
        num_dw: int = 10,
    ):
        super().__init__()
        self.base = base
        

        self.enc_s2 = S2SemanticEncoderMW(
            base=base, depths=depths, heads=heads,
            win8=win8, win16=win16, win32=win32, drop=drop
        )
        self.enc_s1 = S1ScatteringEncoderMW(
            base=base, depths=depths, heads=heads,
            win8=win8, win16=win16, win32=win32, drop=drop, psdp_mix=psdp_mix
        )

        self.geo_prompt = GeoPrompt(
            base=base, num_esa=num_esa, num_dw=num_dw, drop=drop
        )

        C4, C8, C16, C32 = base, base * 2, base * 4, base * 8

        self.mod_s2_4 = GeoFeatureModulation(C4, C4)
        self.mod_s1_4 = GeoFeatureModulation(C4, C4)
        self.mod_s2_8 = GeoFeatureModulation(C8, C8)
        self.mod_s1_8 = GeoFeatureModulation(C8, C8)
        self.mod_s2_16 = GeoFeatureModulation(C16, C16)
        self.mod_s1_16 = GeoFeatureModulation(C16, C16)
        self.mod_s2_32 = GeoFeatureModulation(C32, C32)
        self.mod_s1_32 = GeoFeatureModulation(C32, C32)

        self.fuse4 = ReliabilityAwareFusion(C4, C4, drop=drop)
        self.fuse8 = ReliabilityAwareFusion(C8, C8, drop=drop)
        self.fuse16 = ReliabilityAwareFusion(C16, C16, drop=drop)
        self.fuse32 = ReliabilityAwareFusion(C32, C32, drop=drop)

        self.up32_16 = UpUNet(in_ch=C32, skip_ch=C16, out_ch=C16, drop=drop)
        self.up16_8 = UpUNet(in_ch=C16, skip_ch=C8, out_ch=C8, drop=drop)
        self.up8_4 = UpUNet(in_ch=C8, skip_ch=C4, out_ch=C4, drop=drop)

        self.head = nn.Sequential(
            BasicConv2d(C4, C4, k=3, s=1, p=1),
            nn.Conv2d(C4, num_classes, 1, bias=True),
        )

    # -----------------------
    # loading helpers
    # -----------------------
    @staticmethod
    def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if not any(k.startswith(prefix) for k in sd.keys()):
            return sd
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    @staticmethod
    def _load_state_any(path: str) -> Dict[str, torch.Tensor]:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if isinstance(obj, dict):
            return obj
        raise RuntimeError("Unsupported checkpoint format: {}".format(path))

    def load_pretrained_encoders(
        self,
        s2_path: Optional[str] = None,
        s1_path: Optional[str] = None,
        strict: bool = False,
        verbose: bool = True,
    ):
        if s2_path:
            sd = self._load_state_any(s2_path)
            for pref in ("module.enc_s2.", "enc_s2.", "module.enc.", "enc."):
                if any(k.startswith(pref) for k in sd.keys()):
                    sd = self._strip_prefix(sd, pref)

            missing, unexpected = self.enc_s2.load_state_dict(sd, strict=strict)
            if verbose:
                print("[LOAD S2 ENC] {} | missing={} unexpected={}".format(
                    s2_path, len(missing), len(unexpected)
                ))

        if s1_path:
            sd = self._load_state_any(s1_path)
            for pref in ("module.enc_s1.", "enc_s1.", "module.enc.", "enc."):
                if any(k.startswith(pref) for k in sd.keys()):
                    sd = self._strip_prefix(sd, pref)

            missing, unexpected = self.enc_s1.load_state_dict(sd, strict=strict)
            if verbose:
                print("[LOAD S1 ENC] {} | missing={} unexpected={}".format(
                    s1_path, len(missing), len(unexpected)
                ))

    # -----------------------
    # forward
    # -----------------------
    def forward(
        self,
        s2_pre: torch.Tensor,
        s1_post: torch.Tensor,
        slope: torch.Tensor,
        esa: torch.Tensor,
        dw: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        f2 = self.enc_s2(s2_pre)
        f1 = self.enc_s1(s1_post)

        g4, g8, g16, g32, r = self.geo_prompt(esa, dw, slope)

        r4 = F.interpolate(r, size=g4.shape[-2:], mode="bilinear", align_corners=False)
        r8 = F.interpolate(r, size=g8.shape[-2:], mode="bilinear", align_corners=False)
        r16 = F.interpolate(r, size=g16.shape[-2:], mode="bilinear", align_corners=False)
        r32 = F.interpolate(r, size=g32.shape[-2:], mode="bilinear", align_corners=False)

        s2_4 = self.mod_s2_4(f2[0], g4)
        s1_4 = self.mod_s1_4(f1[0], g4)

        s2_8 = self.mod_s2_8(f2[1], g8)
        s1_8 = self.mod_s1_8(f1[1], g8)

        s2_16 = self.mod_s2_16(f2[2], g16)
        s1_16 = self.mod_s1_16(f1[2], g16)

        s2_32 = self.mod_s2_32(f2[3], g32)
        s1_32 = self.mod_s1_32(f1[3], g32)

        x4, w2_4, w1_4 = self.fuse4(s2_4, s1_4, g4, r4)
        x8, w2_8, w1_8 = self.fuse8(s2_8, s1_8, g8, r8)
        x16, w2_16, w1_16 = self.fuse16(s2_16, s1_16, g16, r16)
        x32, w2_32, w1_32 = self.fuse32(s2_32, s1_32, g32, r32)

        d16 = self.up32_16(x32, x16)
        d8 = self.up16_8(d16, x8)
        d4 = self.up8_4(d8, x4)

        logits = self.head(d4)
        logits = F.interpolate(logits, size=s2_pre.shape[-2:], mode="bilinear", align_corners=False)

        return logits

