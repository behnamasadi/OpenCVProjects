"""
Live view + capture for the HIK IR UVC camera (VID:PID 2bdf:0102).

The camera advertises YUYV 160x248, but the "YUYV" bytes are actually
big-endian 16-bit thermal values. The 248-row frame is two stacked streams:

    rows   0:120  -> AGC'd thermal preview   (narrow dynamic range)
    rows 120:248  -> raw Y16 thermal data    (full range, border metadata)

OpenCV in this env is built without V4L, so we stream raw frames from
`v4l2-ctl --stream-to=-` (v4l-utils) via a subprocess pipe.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


WIDTH, HEIGHT = 160, 248
FRAME_BYTES = WIDTH * HEIGHT * 2  # YUYV container -> 2 bytes/pixel


def spawn_stream(device: str) -> subprocess.Popen:
    if shutil.which("v4l2-ctl") is None:
        sys.exit("v4l2-ctl not found. Install with: sudo apt install v4l-utils")
    cmd = [
        "v4l2-ctl",
        f"--device={device}",
        "--stream-mmap=4",
        "--stream-count=0",   # 0 = infinite
        "--stream-to=-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


def read_frame(proc: subprocess.Popen) -> np.ndarray | None:
    buf = proc.stdout.read(FRAME_BYTES)
    if len(buf) < FRAME_BYTES:
        return None
    return np.frombuffer(buf, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)


def decode_y16(frame_u8x2: np.ndarray) -> np.ndarray:
    high = frame_u8x2[:, :, 0].astype(np.uint16)
    low = frame_u8x2[:, :, 1].astype(np.uint16)
    return (high << 8) | low


def robust_color(y16: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    a, b = np.percentile(y16, [lo, hi])
    span = max(b - a, 1)
    norm = np.clip((y16 - a) * (255.0 / span), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video2")
    parser.add_argument("--view", default="both",
                        choices=["preview", "raw", "both"],
                        help="preview = top AGC'd half, raw = bottom Y16 half")
    parser.add_argument("--outdir", default="captures")
    parser.add_argument("--scale", type=int, default=3,
                        help="integer upscale factor for display/saved PNGs")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    proc = spawn_stream(args.device)
    print(f"Streaming {args.device}  [s] save  [q] quit")

    t0 = time.time()
    n = 0
    try:
        while True:
            frame = read_frame(proc)
            if frame is None:
                print("stream ended")
                break
            y16 = decode_y16(frame)
            preview = robust_color(y16[:120])
            rawview = robust_color(y16[120:])

            if args.view == "preview":
                disp = preview
            elif args.view == "raw":
                disp = rawview
            else:
                gap = np.zeros((4, WIDTH, 3), dtype=np.uint8)
                disp = np.vstack([preview, gap, rawview])

            s = args.scale
            disp = cv2.resize(disp, (disp.shape[1] * s, disp.shape[0] * s),
                              interpolation=cv2.INTER_NEAREST)

            n += 1
            fps = n / max(time.time() - t0, 1e-6)
            cv2.putText(disp, f"{fps:.1f} fps", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("HIK IR", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                ts = int(time.time() * 1000)
                cv2.imwrite(str(outdir / f"ir_preview_{ts}.png"), preview)
                cv2.imwrite(str(outdir / f"ir_raw_{ts}.png"), rawview)
                np.save(outdir / f"ir_y16_{ts}.npy", y16)
                print(f"saved ir_preview_{ts}.png, ir_raw_{ts}.png, ir_y16_{ts}.npy")
    finally:
        cv2.destroyAllWindows()
        proc.terminate()
        proc.wait(timeout=2)


if __name__ == "__main__":
    main()
