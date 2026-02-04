from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn


import torch.nn.functional as F
import tqdm


def conv3d(in_ch, out_ch, k=3, s=1, p=1, bias=False):
    return nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)


def conv2d(in_ch, out_ch, k=3, s=1, p=1, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)


class ResBlock3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = conv3d(ch, ch, 3, 1, 1)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = conv3d(ch, ch, 3, 1, 1)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class TemporalWeightedPool(nn.Module):
    """
    x: B,C,T,H,W -> B,C,H,W
    Learn per-pixel weights over T.
    """

    def __init__(self, ch: int, T: int):
        super().__init__()
        self.T = T
        # produce logits per time step
        self.logits = nn.Conv3d(ch, 1, kernel_size=1, bias=True)

    def forward(self, x):
        # logits: B,1,T,H,W
        a = self.logits(x)
        w = torch.softmax(a, dim=2)  # over T
        y = (x * w).sum(dim=2)  # B,C,H,W
        return y


class TurbRestore3DPlain(nn.Module):
    """
    Input:  B,T(=5),C(=3),H,W
    Output: B,3,H,W
    No encoder/decoder, constant resolution.
    """

    def __init__(self, base_ch=64, blocks=10, T=5):
        super().__init__()
        self.T = T

        self.stem = nn.Sequential(
            conv3d(3, base_ch, 3, 1, 1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
        )

        self.body = nn.Sequential(*[ResBlock3D(base_ch) for _ in range(blocks)])

        self.pool = TemporalWeightedPool(base_ch, T=T)

        self.head = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            conv2d(base_ch, base_ch, 3, 1, 1),
            nn.SiLU(),
            conv2d(base_ch, 3, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        # x: B,T,C,H,W -> B,C,T,H,W
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # central frame for residual (B,3,H,W)
        center = x[:, :, self.T // 2]

        f = self.stem(x)  # B,Ch,T,H,W
        f = self.body(f)  # B,Ch,T,H,W
        f2d = self.pool(f)  # B,Ch,H,W

        out = self.head(f2d)  # B,3,H,W
        return (out + center).clamp(0.0, 1.0)


# -------------------------
# Utils
# -------------------------


def load_checkpoint(
    model: nn.Module, ckpt_path: str, map_location: str = "cpu"
) -> None:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)


def make_weight(tile: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    2D Hann window weight map for seamless tiling.
    Returns: (1,1,tile,tile)
    """
    wy = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    w2d = wy[:, None] * wx[None, :]
    w2d = w2d / (w2d.max().clamp_min(1e-6))
    return w2d[None, None, :, :]  # 1,1,tile,tile


def tile_coords(H: int, W: int, tile: int, overlap: int) -> List[Tuple[int, int]]:
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be < tile")
    coords: List[Tuple[int, int]] = []
    ys = list(range(0, H, stride))
    xs = list(range(0, W, stride))
    for y in ys:
        y0 = min(y, H - tile)
        for x in xs:
            x0 = min(x, W - tile)
            coords.append((y0, x0))
    # remove duplicates when H-tile or W-tile causes repeated mins
    coords = list(dict.fromkeys(coords))
    return coords


# -------------------------
# Tiled inference for one temporal window
# -------------------------


@torch.inference_mode()
def infer_window_tiled(
    model: nn.Module,
    frames_t: torch.Tensor,  # (T,C,H,W) float
    tile: int,
    overlap: int,
    tile_batch: int = 8,
) -> torch.Tensor:
    """
    frames_t: (T,C,H,W) float in [0..1] on device
    returns : (C,H,W) float in [0..1]
    """
    T, C, H, W = frames_t.shape
    device = frames_t.device
    dtype = frames_t.dtype

    if tile > H or tile > W:
        # simplest: reduce tile to fit (keeps code robust on small videos)
        tile = min(tile, H, W)
        # ensure odd/valid? not required. just make overlap safe:
        overlap = min(overlap, tile - 1)

    coords = tile_coords(H, W, tile, overlap)
    w = make_weight(tile, device, dtype)  # 1,1,tile,tile

    out = torch.zeros((1, C, H, W), device=device, dtype=dtype)
    acc = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

    # batch tiles to reduce forward overhead
    for i in range(0, len(coords), tile_batch):
        batch_coords = coords[i : i + tile_batch]
        patches = []
        for y0, x0 in batch_coords:
            patch = frames_t[:, :, y0 : y0 + tile, x0 : x0 + tile]  # (T,C,tile,tile)
            patches.append(patch)
        x = torch.stack(patches, dim=0)  # (B,T,C,tile,tile)

        pred = model(x)  # (B,3,tile,tile)
        pred = pred.clamp(0, 1)

        for b, (y0, x0) in enumerate(batch_coords):
            out[:, :, y0 : y0 + tile, x0 : x0 + tile] += pred[b : b + 1] * w
            acc[:, :, y0 : y0 + tile, x0 : x0 + tile] += w

    out = out / acc.clamp_min(1e-6)
    return out[0].clamp(0, 1)  # (C,H,W)


# -------------------------
# Video stabilization (sliding window + tiling)
# -------------------------


@dataclass
class InferCfg:
    inp: str
    out: str
    ckpt: str

    window: int = 5
    base_ch: int = 64
    blocks: int = 10

    device: str = "cuda"
    half: bool = True

    tile: int = 256
    overlap: int = 64
    tile_batch: int = 8

    max_frames: Optional[int] = None
    fourcc: str = "mp4v"


def stabilize_video(cfg: InferCfg, model: nn.Module) -> None:
    if cfg.window % 2 == 0:
        raise ValueError("window must be odd (e.g. 5)")

    dev = torch.device(
        cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    model = model.to(dev)
    model.eval()

    use_half = cfg.half and dev.type == "cuda"
    if use_half:
        model = model.half()

    cap = cv2.VideoCapture(cfg.inp)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {cfg.inp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = (
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        else None
    )
    target_frames = n_frames if (n_frames and n_frames > 0) else None

    out_path = str(cfg.out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*cfg.fourcc)  # type: ignore
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for: {out_path}")

    half_w = cfg.window // 2
    buf: Deque[np.ndarray] = deque(maxlen=cfg.window)

    def read_frame_rgb01() -> Optional[np.ndarray]:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0  # H,W,3 float32

    # --- prime buffer (left replicate padding) ---
    first = read_frame_rgb01()
    if first is None:
        writer.release()
        cap.release()
        raise RuntimeError("Empty video")

    for _ in range(half_w):
        buf.append(first.copy())
    buf.append(first)

    while len(buf) < cfg.window:
        fr = read_frame_rgb01()
        if fr is None:
            buf.append(buf[-1].copy())
        else:
            buf.append(fr)

    produced = 0
    ended = False
    right_pad_left = 0
    counter = tqdm.tqdm(desc="frames processed", total=n_frames)

    try:
        while True:
            if cfg.max_frames is not None and produced >= cfg.max_frames:
                break
            if target_frames is not None and produced >= target_frames:
                break

            # Build current temporal window tensor: (T,C,H,W)
            win = np.stack(list(buf), axis=0)  # (T,H,W,3)
            x = torch.from_numpy(win).permute(0, 3, 1, 2)  # (T,3,H,W)
            x = x.to(dev, non_blocking=True)
            x = x.half() if use_half else x.float()

            # Tiled inference for this window
            y = infer_window_tiled(
                model=model,
                frames_t=x,
                tile=cfg.tile,
                overlap=cfg.overlap,
                tile_batch=cfg.tile_batch,
            )  # (3,H,W) float

            # Write frame
            y_np = y.detach().float().cpu().numpy()  # (3,H,W)
            rgb = (y_np.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)  # H,W,3
            bgr = rgb[..., ::-1]
            writer.write(bgr)
            produced += 1
            counter.update()

            # Advance window
            if not ended:
                nxt = read_frame_rgb01()
                if nxt is None:
                    ended = True
                    right_pad_left = half_w
                else:
                    buf.append(nxt)
            else:
                if right_pad_left > 0:
                    buf.append(buf[-1].copy())
                    right_pad_left -= 1
                else:
                    break

    finally:
        writer.release()
        cap.release()

    print(
        f"Saved stabilized video: {out_path}  (frames={produced}, fps={fps:.3f}, size={W}x{H})"
    )


# -------------------------
# Build your model here
# -------------------------
def build_model(base_ch: int, blocks: int, window: int) -> nn.Module:
    # You must have TurbRestore3DPlain defined/imported in your environment.
    # from your_module import TurbRestore3DPlain
    return TurbRestore3DPlain(base_ch=base_ch, blocks=blocks, T=window)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", required=True, help="Input turbulent video (mp4)"
    )
    ap.add_argument(
        "--out", dest="out", required=True, help="Output stabilized video (mp4)"
    )
    ap.add_argument("--ckpt", required=True, help="Checkpoint .pt")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--base-ch", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--tile-batch", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fourcc", type=str, default="mp4v")
    args = ap.parse_args()

    cfg = InferCfg(
        inp=args.inp,
        out=args.out,
        ckpt=args.ckpt,
        window=args.window,
        base_ch=args.base_ch,
        blocks=args.blocks,
        device=args.device,
        half=args.half,
        tile=args.tile,
        overlap=args.overlap,
        tile_batch=args.tile_batch,
        max_frames=args.max_frames,
        fourcc=args.fourcc,
    )

    model = build_model(cfg.base_ch, cfg.blocks, cfg.window)
    load_checkpoint(model, cfg.ckpt, map_location="cpu")
    stabilize_video(cfg, model)


if __name__ == "__main__":
    main()
