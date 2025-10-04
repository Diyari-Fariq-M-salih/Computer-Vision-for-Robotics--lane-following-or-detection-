#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# -----------------------------
# Utilities & Core Components
# -----------------------------

def preprocess_lane_mask(frame):
    """
    Build a binary mask highlighting lane paint (white/yellow) and edges
    on *bright* structures, while suppressing *dark cracks*.
    """
    # Contrast (CLAHE on L in LAB) + mild gain
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L = clahe.apply(L)
    frame = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=0)

    # HLS decomposition
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    H, HL, S = cv2.split(hls)

    # Color masks (tune if needed for your region/weather)
    yellow = cv2.inRange(hls, (15,  80, 100), (35, 255, 255))
    white  = cv2.inRange(hls, ( 0, 200,   0), (255, 255, 255))

    # Dark cracks (low lightness)
    cracks = cv2.inRange(HL, 0, 80)

    # Brightness-gated edges (avoid edges on dark asphalt)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    _, sx = cv2.threshold(sobelx, 0, 255, cv2.THRESH_OTSU)
    bright = cv2.inRange(HL, 160, 255)
    sx = cv2.bitwise_and(sx, bright)

    # Combine: (color ∪ bright-edges) \ cracks
    mask = cv2.bitwise_or(yellow, white)
    mask = cv2.bitwise_or(mask, sx)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(cracks))

    # Cleanup
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def region_of_interest(binary, pts):
    """Keep only the polygonal region of interest."""
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return cv2.bitwise_and(binary, mask)


def perspective_matrices(w, h):
    """
    Symmetric trapezoid -> rectangle transform for a stable bird's-eye view.
    Adjust the fractions if your camera is different.
    """
    src = np.float32([
        [w * 0.44, h * 0.66],
        [w * 0.56, h * 0.66],
        [w * 0.12, h * 0.96],
        [w * 0.88, h * 0.96]
    ])
    dst = np.float32([
        [w * 0.25, h * 0.10],
        [w * 0.75, h * 0.10],
        [w * 0.25, h * 0.98],
        [w * 0.75, h * 0.98]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src


def sliding_window(binary, nwindows=9, margin=80, minpix=30):
    """
    Classic sliding-window lane pixel search.
    Returns left/right pixel coordinates.
    """
    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    out = np.dstack((binary, binary, binary)) * 0

    midpoint = histogram.shape[0] // 2
    leftx_base = int(np.argmax(histogram[:midpoint]))
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    window_height = binary.shape[0] // nwindows
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Tolerate dashed gaps: only shift if enough points found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        # else: keep previous x (coast)
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)

    leftx = nonzerox[left_lane_inds]; lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]; righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty


def search_around_poly(binary, left_fit, right_fit, margin=60):
    """
    Fast search around previous polynomials (good for dashed lines).
    """
    nz = binary.nonzero()
    nonzeroy, nonzerox = np.array(nz[0]), np.array(nz[1])

    left_region = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                   (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_region = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                    (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    lx, ly = nonzerox[left_region], nonzeroy[left_region]
    rx, ry = nonzerox[right_region], nonzeroy[right_region]
    return lx, ly, rx, ry


def fit_polynomial(binary, leftx, lefty, rightx, righty):
    """
    Quadratic fits for left/right lanes.
    """
    if len(leftx) < 200 or len(rightx) < 200:
        return None, None, None
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
    return left_fit, right_fit, ploty


class FitSmoother:
    """Exponential smoothing for polynomial coefficients."""
    def __init__(self, alpha=0.25):
        self.left = None
        self.right = None
        self.a = alpha

    def update(self, lf, rf):
        if lf is None or rf is None:
            return self.left, self.right
        lf = np.array(lf); rf = np.array(rf)
        if self.left is None:
            self.left, self.right = lf, rf
        else:
            self.left  = self.a*lf + (1-self.a)*self.left
            self.right = self.a*rf + (1-self.a)*self.right
        return self.left, self.right


class PID:
    """Simple PID used for a steer-like signal based on lateral error."""
    def __init__(self, kp=0.8, ki=0.0, kd=0.12):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0
        self.prev = 0.0

    def step(self, e, dt=1/30.0, windup=1.0):
        self.i = np.clip(self.i + e*dt, -windup, windup)
        d = (e - self.prev) / max(dt, 1e-6)
        self.prev = e
        return self.kp*e + self.ki*self.i + self.kd*d


def open_writer(base_path, fps, size):
    """
    Try mp4v (mp4), then XVID (avi). Returns (writer, out_path).
    """
    base = Path(base_path)
    for fourcc_str, ext in [("mp4v", ".mp4"), ("XVID", ".avi")]:
        out_path = base.with_suffix(ext)
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
        if vw.isOpened():
            print(f"[writer] Using {fourcc_str} -> {out_path.name}")
            return vw, out_path
        vw.release()
    raise RuntimeError("Failed to open VideoWriter with mp4v or XVID")


# -----------------------------
# Main Pipeline
# -----------------------------

def process_video(input_path: Path, output_path: Path, show: bool=False):
    ROOT = Path(__file__).resolve().parents[1]
    input_path = (ROOT / input_path).resolve()
    output_path = (ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"CWD     : {Path.cwd()}")
    print(f"Input   : {input_path}")
    print(f"Output  : {output_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open input video: {input_path}")

    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_cap if fps_cap and fps_cap > 1.0 else 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w == 0 or h == 0:
        cap.release()
        raise RuntimeError("Input reports 0 width/height. Re-encode the video.")

    writer, out_path = open_writer(output_path, fps, (w, h))
    print(f"Video   : {w}x{h} @ {fps:.2f} fps")

    pid = PID()
    smoother = FitSmoother(alpha=0.25)
    last_left_fit, last_right_fit = None, None

    xm_per_pix = 3.7 / 700.0  # meters per pixel (approx for bottom of image)

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = preprocess_lane_mask(frame)

        # Tight ROI (reduces asphalt noise)
        roi_pts = [
            (int(w*0.14), int(h*0.96)),
            (int(w*0.44), int(h*0.66)),
            (int(w*0.56), int(h*0.66)),
            (int(w*0.86), int(h*0.96)),
        ]
        roi = region_of_interest(mask, roi_pts)

        # Perspective (bird's-eye)
        M, Minv, src = perspective_matrices(w, h)
        warped = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_LINEAR)

        # Lane pixel search: track around last fit, else sliding windows
        if last_left_fit is not None and last_right_fit is not None:
            lx, ly, rx, ry = search_around_poly(warped, last_left_fit, last_right_fit, margin=60)
            if len(lx) < 800 or len(ry) < 800:
                lx, ly, rx, ry = sliding_window(warped)  # fallback
        else:
            lx, ly, rx, ry = sliding_window(warped)

        left_fit, right_fit, ploty = fit_polynomial(warped, lx, ly, rx, ry)

        # Geometry sanity checks (lane width & parallelism)
        valid = left_fit is not None and right_fit is not None
        if valid:
            y_eval = h - 1
            left_x  = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
            right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
            lane_width = right_x - left_x  # in pixels
            # Tune thresholds for your camera
            valid = 500 < lane_width < 900 and abs(left_fit[0] - right_fit[0]) < 1e-4

        if valid:
            left_fit, right_fit = smoother.update(left_fit, right_fit)
            last_left_fit, last_right_fit = left_fit, right_fit
        else:
            # Coast on last good fit if available
            left_fit, right_fit = last_left_fit, last_right_fit

        # Render overlay
        out_frame = frame.copy()
        if left_fit is not None and right_fit is not None:
            left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            lane_area = np.zeros((h, w), dtype=np.uint8)
            pts_left  = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right)).astype(np.int32)
            cv2.fillPoly(lane_area, pts, 255)

            color_warp = cv2.warpPerspective(lane_area, Minv, (w, h))
            overlay = frame.copy()
            overlay[color_warp > 0] = (0, 255, 0)
            out_frame = cv2.addWeighted(frame, 1.0, overlay, 0.30, 0)

            # Offset & simple steer
            lane_c = (left_fitx[-1] + right_fitx[-1]) / 2.0
            veh_c = w / 2.0
            offset_m = (veh_c - lane_c) * xm_per_pix
            err = (lane_c - veh_c) / (w/2.0)
            steer = pid.step(err)

            # HUD
            cv2.putText(out_frame, f"Offset: {offset_m:+.2f} m", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.putText(out_frame, f"Steer: {steer:+.2f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.polylines(out_frame, [np.int32(src)], True, (0, 0, 255), 2)
        else:
            cv2.putText(out_frame, "Lane lost", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        writer.write(out_frame)
        frames += 1

        if show:
            cv2.imshow("annotated", out_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Frames written: {frames}")
    print(f"Saved -> {out_path}")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Robust lane following (classical CV)")
    p.add_argument("--in",  dest="inp",  default="data/lanes/input.mp4",
                   help="Input video path (relative to repo root)")
    p.add_argument("--out", dest="outp", default="out/lanes_annotated.mp4",
                   help="Output video path (relative to repo root, extension may change by codec)")
    p.add_argument("--show", action="store_true", help="Preview while processing")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        process_video(Path(args.inp), Path(args.outp), show=args.show)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
