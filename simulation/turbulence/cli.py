from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .simulator_gpu import TurbulenceSimulatorTorch, TurbTorchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turbulence simulation runner")
    parser.add_argument(
        "--input-video", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--output-video", type=str, required=True, help="Path to output video"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    print("Initializing simulator")
    config = TurbTorchConfig()
    simulator = TurbulenceSimulatorTorch(config)

    print(f"Opening input video: {args.input_video}")
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found or unreadable: {args.input_video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"No frames read from video: {args.input_video}")

    height, width = frame.shape[0], frame.shape[1]
    is_color = frame.ndim == 3 and frame.shape[2] > 1
    writer = cv2.VideoWriter(
        filename=args.output_video,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        fps=fps_in,
        frameSize=(width, height),
        isColor=is_color,
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Failed to open video writer for: {args.output_video}")

    print("Applying turbulence (streaming)")
    frame_index = 0
    pbar = tqdm(total=total_frames, desc="Frames")

    while ok:
        processed = (
            (
                simulator.simulate_frame(
                    frame,
                    frame_index,
                ).clamp(0, 1)
                * 255
            )
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        writer.write(processed.astype(np.uint8))
        frame_index += 1
        pbar.update(1)
        ok, frame = cap.read()

    pbar.close()
    cap.release()
    writer.release()

    print(f"Took {time.time() - start:.2f}s to process {frame_index} frames.")


if __name__ == "__main__":
    main()
