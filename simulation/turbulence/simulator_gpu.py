"""
PyTorch turbulence simulator (GPU-friendly)

What you get vs your NumPy/OpenCV version:
- Warp via grid_sample (bilinear) on GPU
- Blur via separable Gaussian conv (fast) on GPU
- Per-pixel blur strength via alpha map (0..1) mixing between sharp and blurred
- Temporal coherence via low-res filtered noise updated each frame (no snoise3)

Input formats:
- single frame: HxW, HxWxC, CxHxW, or 1xCxHxW
- video: TxHxWxC, TxCxHxW, or BxTxCxHxW (batch optional)

Notes:
- This does NOT reproduce simplex noise exactly; it makes a very similar smooth, temporally coherent flow field.
- For maximum speed, keep everything on GPU and process in batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy
import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class TurbTorchConfig:
    # Turbulence
    turbulence_strength_px: float = 40.0  # max displacement in pixels (roughly)
    temporal_smooth: float = 0.15  # 0..1, higher -> changes faster
    lowres_scale: int = 4  # noise computed at H/scale, W/scale (4 or 8 is typical)
    noise_sigma_px_lowres: float = 3.0  # spatial smoothness (in lowres pixels)

    # Blur
    max_blur_sigma_px: float = 40.0  # max sigma in pixels (fullres)
    blur_kernel_size: int = 31
    blur_alpha_gamma: float = 1.0

    # Scintillation
    scintillation_beta = 0.15

    # Runtime
    device: str = "cuda"  # "cuda" or "cpu"
    dtype: torch.dtype = torch.float16  # float16 on GPU, float32 on CPU
    seed: int = 123


class TurbulenceSimulatorTorch:
    def __init__(self, cfg: TurbTorchConfig):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        )
        self.dtype = cfg.dtype if self.device.type == "cuda" else torch.float32

        g = torch.Generator(device="cpu")
        g.manual_seed(cfg.seed)
        self._cpu_gen = g

        self._grid_cache: Optional[Tuple[int, int, torch.Tensor]] = (
            None  # (H,W, base_grid)
        )
        self._state_u: torch.Tensor  # lowres noise state
        self._state_v: torch.Tensor
        self._state_a: torch.Tensor
        self._state_sc: torch.Tensor

        # Precompute 1D gaussian kernel for blur (separable)
        self._k1d = self._make_gaussian_1d(
            cfg.blur_kernel_size, cfg.max_blur_sigma_px
        ).to(self.device, self.dtype)

        # For smoothing lowres noise field (cheap, 1 blur per step)
        # Build a small gaussian kernel in lowres pixel units; separable too.
        ksz_lr = max(9, int(round(cfg.noise_sigma_px_lowres * 6)) | 1)  # odd
        self._k1d_lr = self._make_gaussian_1d(ksz_lr, cfg.noise_sigma_px_lowres).to(
            self.device, self.dtype
        )

    # -------------------- Public API --------------------

    @torch.inference_mode()
    def simulate_frame(
        self, frame: Union[torch.Tensor, "numpy.ndarray"], step_index: int | None = None
    ) -> torch.Tensor:
        x = self._to_tensor_frame(frame)  # 1xCxHxW float in [0..1]
        out = self._simulate_bchw(x=x, step_index=step_index)
        return out[0]  # CxHxW

    @torch.inference_mode()
    def simulate_video(
        self,
        video: Union[torch.Tensor, "numpy.ndarray"],
        time_axis: int = 0,
        batch: int = 0,
    ) -> torch.Tensor:
        """
        Returns video in same axis layout as input if input is torch.
        For numpy input, returns torch tensor (on cfg.device).
        Supported shapes:
          - TxHxWxC
          - TxCxHxW
          - BxTxCxHxW
        If batch>0, processes T in chunks for better GPU utilization.
        """
        x, layout = self._to_tensor_video(
            video, time_axis=time_axis
        )  # -> BxTxCxHxW float [0..1]
        B, T, C, H, W = x.shape

        if batch is None or batch <= 0:
            batch = T

        outs = []
        for t0 in tqdm(range(0, T, batch)):
            chunk = x[:, t0 : t0 + batch]  # BxTcCxHxW
            # process each time step sequentially to keep temporal coherence (stateful)
            chunk_out = []
            for ti in range(chunk.shape[1]):
                y = self._simulate_bchw(chunk[:, ti], step_index=t0 + ti)  # BxCxHxW
                chunk_out.append(y)
            outs.append(torch.stack(chunk_out, dim=1))
        y = torch.cat(outs, dim=1)  # BxTxCxHxW

        return self._from_btc_hw(y, layout, time_axis=time_axis)

    # -------------------- Core --------------------

    def _ensure_base_grid(self, H: int, W: int) -> torch.Tensor:
        """
        base grid for grid_sample: 1xHxWx2 in [-1,1]
        """
        if (
            self._grid_cache is not None
            and self._grid_cache[0] == H
            and self._grid_cache[1] == W
        ):
            return self._grid_cache[2]

        ys = torch.linspace(-1.0, 1.0, H, device=self.device, dtype=self.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=self.device, dtype=self.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)[None, ...]  # 1xHxWx2
        self._grid_cache = (H, W, grid)
        return grid

    def _init_lowres_state(self, B: int, H: int, W: int):
        s = self.cfg.lowres_scale
        Hl, Wl = max(2, H // s), max(2, W // s)

        def rand_field():
            r = torch.randn(
                (B, 1, Hl, Wl),
                generator=self._cpu_gen,
                device="cpu",
                dtype=torch.float32,
            )
            r = r.to(self.device, self.dtype)
            return r

        self._state_u = rand_field()
        self._state_v = rand_field()
        self._state_a = rand_field()
        self._state_sc = rand_field()

        # smooth them initially
        self._state_u = self._smooth_lowres(self._state_u)
        self._state_v = self._smooth_lowres(self._state_v)
        self._state_a = self._smooth_lowres(self._state_a)
        self._state_sc = self._smooth_lowres(self._state_sc)

    def _smooth_lowres(self, x: torch.Tensor) -> torch.Tensor:
        # separable blur: (B,1,H,W) -> (B,1,H,W)
        x = self._gauss_blur_separable(x, self._k1d_lr)
        return x

    def _update_noise_state(self, B: int, H: int, W: int):
        if not hasattr(self, "_state_u"):
            self._init_lowres_state(B, H, W)

        # new random lowres fields
        def new_rand_like(ref: torch.Tensor):
            r = torch.randn(
                ref.shape, generator=self._cpu_gen, device="cpu", dtype=torch.float32
            )
            return r.to(self.device, self.dtype)

        u_new = self._smooth_lowres(new_rand_like(self._state_u))
        v_new = self._smooth_lowres(new_rand_like(self._state_v))
        a_new = self._smooth_lowres(new_rand_like(self._state_a))
        sc_new = self._smooth_lowres(new_rand_like(self._state_sc))
        a = float(self.cfg.temporal_smooth)
        self._state_u = (1 - a) * self._state_u + a * u_new
        self._state_v = (1 - a) * self._state_v + a * v_new
        self._state_a = (1 - a) * self._state_a + a * a_new
        self._state_sc = (1 - a) * self._state_sc + a * sc_new

    def _simulate_bchw(
        self, x: torch.Tensor, step_index: Optional[int]
    ) -> torch.Tensor:
        """
        x: BxCxHxW float in [0..1]
        """
        B, C, H, W = x.shape

        # update temporally coherent lowres fields
        self._update_noise_state(B, H, W)

        # upsample to fullres
        u = F.interpolate(
            self._state_u,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )  # Bx1xHxW
        v = F.interpolate(
            self._state_v,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )
        a = F.interpolate(
            self._state_a,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )

        # normalize u,v to roughly [-1,1]
        u = torch.tanh(u)
        v = torch.tanh(v)

        # alpha map 0..1 (per-pixel blur strength)
        a = torch.tanh(a)
        a = (a + 1.0) * 0.5
        if self.cfg.blur_alpha_gamma != 1.0:
            a = a.clamp(0, 1) ** float(self.cfg.blur_alpha_gamma)
        a = a.clamp(0, 1)

        # build sampling grid with displacement in normalized coordinates
        base = self._ensure_base_grid(H, W)  # 1xHxWx2
        # convert px displacement to normalized [-1,1] in grid_sample space
        dx_norm = (u * float(self.cfg.turbulence_strength_px)) * (2.0 / max(1, W - 1))
        dy_norm = (v * float(self.cfg.turbulence_strength_px)) * (2.0 / max(1, H - 1))

        grid = base.repeat(B, 1, 1, 1).clone()
        grid[..., 0] = grid[..., 0] + dx_norm[:, 0].to(grid.dtype)
        grid[..., 1] = grid[..., 1] + dy_norm[:, 0].to(grid.dtype)

        # warp
        warped = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )

        # blur once with max sigma and mix
        blurred = self._gauss_blur_rgb(warped, self._k1d)
        out = warped * (1.0 - a) + blurred * a
        # --- scintillation ---
        sc = F.interpolate(
            self._state_sc,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        )

        sc = torch.tanh(sc)  # [-1, 1]

        out = out * torch.exp(
            self.cfg.scintillation_beta * sc
        )  # log-normal scintillation

        return out

    # -------------------- Gaussian blur helpers --------------------

    @staticmethod
    def _make_gaussian_1d(ksize: int, sigma: float) -> torch.Tensor:
        ksize = int(ksize)
        if ksize % 2 == 0:
            ksize += 1
        sigma = float(max(1e-6, sigma))
        half = ksize // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32)
        k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        k = k / k.sum()
        return k  # (K,)

    def _gauss_blur_separable(self, x: torch.Tensor, k1d: torch.Tensor) -> torch.Tensor:
        """
        x: BxCxHxW
        """
        B, C, H, W = x.shape
        K = k1d.numel()
        # horizontal
        kh = k1d.view(1, 1, 1, K).repeat(C, 1, 1, 1)  # (C,1,1,K)
        x = F.conv2d(x, kh, padding=(0, K // 2), groups=C)
        # vertical
        kv = k1d.view(1, 1, K, 1).repeat(C, 1, 1, 1)  # (C,1,K,1)
        x = F.conv2d(x, kv, padding=(K // 2, 0), groups=C)
        return x

    def _gauss_blur_rgb(self, x: torch.Tensor, k1d: torch.Tensor) -> torch.Tensor:
        return self._gauss_blur_separable(x, k1d)

    # -------------------- Tensor IO --------------------

    def _to_tensor_frame(self, frame) -> torch.Tensor:
        # numpy -> torch
        if not isinstance(frame, torch.Tensor):
            import numpy as np

            assert isinstance(frame, np.ndarray)
            t = torch.from_numpy(frame)
        else:
            t = frame

        # Shape handling
        if t.ndim == 2:  # HxW
            t = t[None, None, :, :]
        elif t.ndim == 3:
            if t.shape[0] in (1, 3):  # CxHxW
                t = t[None, :, :, :]
            else:  # HxWxC
                t = t.permute(2, 0, 1)[None, :, :, :]
        elif t.ndim == 4:
            # assume BxCxHxW
            pass
        else:
            raise ValueError(f"Unsupported frame ndim={t.ndim}")

        t = t.contiguous()

        # normalize to [0,1] float
        if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            t = t.to(torch.float32) / 255.0
        else:
            t = t.to(torch.float32)
            # if it's likely 0..255 floats, you can uncomment:
            # t = t / 255.0

        t = t.to(self.device, dtype=self.dtype)
        return t

    def _to_tensor_video(self, video, time_axis: int = 0):
        if not isinstance(video, torch.Tensor):
            import numpy as np

            assert isinstance(video, np.ndarray)
            t = torch.from_numpy(video)
        else:
            t = video

        # Normalize axis
        time_axis = time_axis % t.ndim

        # Layout detection + convert to BxTxCxHxW
        if t.ndim == 4:
            # Could be TxHxWxC or TxCxHxW
            if time_axis != 0:
                t = t.movedim(time_axis, 0)
            # Now time is dim0
            if t.shape[-1] in (1, 3):  # TxHxWxC
                t = t.permute(0, 3, 1, 2)  # TxCxHxW
                layout = "T_HWC"
            else:
                layout = "T_CHW"
            t = t[None, ...]  # B=1
            t = t.permute(0, 1, 2, 3, 4)  # BxTxCxHxW
        elif t.ndim == 5:
            # Assume BxTxCxHxW, but allow time_axis not 1
            if time_axis != 1:
                t = t.movedim(time_axis, 1)
            layout = "B_TCHW"
        else:
            raise ValueError(f"Unsupported video ndim={t.ndim}")

        t = t.contiguous()

        if t.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            t = t.to(torch.float32) / 255.0
        else:
            t = t.to(torch.float32)

        t = t.to(self.device, dtype=self.dtype)
        return t, layout

    def _from_btc_hw(
        self, y: torch.Tensor, layout: str, time_axis: int = 0
    ) -> torch.Tensor:
        # y: BxTxCxHxW float [0..1]
        if layout == "B_TCHW":
            # put time axis back if needed
            if time_axis != 1:
                y = y.movedim(1, time_axis)
            return y
        if layout == "T_CHW":
            y = y[0]  # TxCxHxW
            if time_axis != 0:
                y = y.movedim(0, time_axis)
            return y
        if layout == "T_HWC":
            y = y[0]  # TxCxHxW
            y = y.permute(0, 2, 3, 1)  # TxHxWxC
            if time_axis != 0:
                y = y.movedim(0, time_axis)
            return y
        raise ValueError(f"Unknown layout {layout}")
