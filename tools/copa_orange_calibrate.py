"""Check and tune the orange ball color thresholds (copaOrangeBall.cfg) on saved NAO frames.

Applies the same per-sample color rule as CopaOrangeBall.h (floor median of Cr - Cb plus a
margin, hue between yellow and red) and shows the mask; the round-blob and expected-size tests
need the camera matrix and only run on the robot. Frames: viewer screenshots ('s') or dataset
images. JPEG is decoded as full-range YCbCr, like the robot's YUYV image.

    .venv/bin/python tools/copa_orange_calibrate.py nao_lower_0000.jpg [--ball X,Y,R] [--show]

With --ball (center and radius of the real ball in pixels), prints the ball and floor
statistics and the largest minCrMinusCb / floorMargin that keep 90 % of the ball.
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

CFG = Path(__file__).resolve().parents[1] / "bhuman/files/Config/Scenarios/CopaPenaltyStriker/copaOrangeBall.cfg"


def load_cfg(path: Path) -> dict[str, float]:
    text = re.sub(r"//[^\n]*", "", path.read_text())
    return {k: float(v) for k, v in re.findall(r"^\s*(\w+)\s*=\s*([-\d.]+);", text, re.M)}


def planes(image: np.ndarray):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.int32)
    return ycrcb[..., 0], ycrcb[..., 2], ycrcb[..., 1]  # Y, Cb, Cr


def orange_mask(y, cb, cr, p: dict[str, float]):
    u, v, d = cb - 128, cr - 128, cr - cb
    floor = int(np.median(d))
    threshold = max(int(p["minCrMinusCb"]), floor + int(p["floorMargin"]))
    ratio = v / np.maximum(-u, 1)
    mask = ((y >= p["minY"]) & (y <= p["maxY"]) & (v >= p["minCr"]) & (u <= p["maxCb"]) & (d >= threshold)
            & (ratio >= p["minHueRatio"]) & (ratio <= p["maxHueRatio"]))
    return mask, floor, threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--ball", help="X,Y,R of the real ball in pixels")
    parser.add_argument("--cfg", type=Path, default=CFG)
    parser.add_argument("--show", action="store_true", help="show the mask over each image")
    args = parser.parse_args()
    params = load_cfg(args.cfg)

    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path}: cannot read")
            continue
        y, cb, cr = planes(image)
        mask, floor, threshold = orange_mask(y, cb, cr, params)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        blobs = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, count)), reverse=True)[:3]
        print(f"{path.name}: floor Cr-Cb={floor} threshold={threshold} orange px={int(mask.sum())}")
        for area, i in blobs:
            x, top, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            print(f"  blob at ({x + w / 2:.0f},{top + h / 2:.0f}) {w}x{h} px fill={area / (w * h):.2f}")

        if args.ball:
            bx, by, r = (float(s) for s in args.ball.split(","))
            yy, xx = np.mgrid[:image.shape[0], :image.shape[1]]
            inside = np.hypot(xx - bx, yy - by) <= 0.85 * r  # keep off the edge blur
            outside = np.hypot(xx - bx, yy - by) > 1.5 * r
            d = cr - cb
            for name, region in (("ball", inside), ("floor", outside)):
                print(f"  {name}: Y {np.percentile(y[region], [10, 50, 90]).round()} "
                      f"Cb {np.percentile(cb[region], [10, 50, 90]).round()} "
                      f"Cr {np.percentile(cr[region], [10, 50, 90]).round()} "
                      f"Cr-Cb {np.percentile(d[region], [10, 50, 99.9]).round()}")
            keep = int(np.percentile(d[inside], 10))
            print(f"  ball detected px: {mask[inside].mean():.0%}; 90 % of the ball needs threshold <= {keep}: "
                  f"minCrMinusCb <= {keep}, floorMargin <= {keep - floor}; "
                  f"floor 99.9 % is {int(np.percentile(d[outside], 99.9))}")

        if args.show:
            view = image.copy()
            view[mask] = (0, 255, 255)
            cv2.imshow(path.name, np.hstack([image, view]))
            cv2.waitKey(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
