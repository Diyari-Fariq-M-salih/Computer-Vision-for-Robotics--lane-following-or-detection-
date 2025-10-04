#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
import argparse
import sys

# ========================
# Box shape controls
# ========================
RECT_Y_TOP     = 0.40   # how far forward the box reaches (smaller = longer box)
RECT_Y_BOTTOM  = 0.95   # how close to the bumper the box starts
RECT_TOP_SCALE = 0.80   # top width as a fraction of the bottom width (0.80..1.00)

# -----------------------------
# Utilities & Core Components
# -----------------------------

def preprocess_lane_mask(frame):
    """
    Build a binary mask highlighting lane paint (white) + bright edges,
    while suppressing dark cracks and shoulder/curb concrete.
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

    # Color masks (white-first; yellow disabled here)
    yellow = np.zeros_like(H, dtype=np.uint8)  # disable yellow detection
    white  = cv2.inRange(hls, (0, 200, 0), (255, 255, 255))  # permissive for bright roads

    # Dark cracks (low lightness)
    cracks = cv2.inRange(HL, 0, 85)

    # Brightness-gated edges (avoid edges on dark asphalt)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    _, sx = cv2.threshold(sobelx, 0, 255, cv2.THRESH_OTSU)
    bright = cv2.inRange(HL, 170, 255)
    sx = cv2.bitwise_and(sx, bright)

    # Combine: (color ∪ bright-edges) \ cracks
    mask = cv2.bitwise_or(yellow, white)
    mask = cv2.bitwise_or(mask, sx)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(cracks))

    # Cleanup
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Shoulder/curb suppression (desaturated bright concrete)
    shoulder = cv2.inRange(HL, 170, 255) & cv2.inRange(S, 0, 60)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(shoulder))
    return mask


def region_of_interest(binary, pts):
    """Keep only the polygonal region of interest."""
    mask = np.zeros_like(binary)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return cv2.bitwise_and(binary, mask)


def perspective_matrices(w, h):
    """
    Symmetric trapezoid -> rectangle transform for a stable bird's-eye view.
    Adjust fractions if your camera is different.
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
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src


def right_lane_fallback(binary, expected_lane_px=700, margin=100):
    """Find a robust right lane only, then infer the left by lane width."""
    hist = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    midpoint = hist.shape[0]//2
    right_base = int(np.argmax(hist[midpoint:]) + midpoint)

    # Single-side sliding windows
    nwindows = 9
    window_height = binary.shape[0] // nwindows
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0]); nonzerox = np.array(nonzero[1])

    rx_current = right_base
    right_inds = []

    for window in range(nwindows):
        wy_low  = binary.shape[0] - (window + 1) * window_height
        wy_high = binary.shape[0] - window * window_height
        wx_low  = rx_current - margin
        wx_high = rx_current + margin

        good = ((nonzeroy >= wy_low) & (nonzeroy < wy_high) &
                (nonzerox >= wx_low) & (nonzerox < wx_high)).nonzero()[0]
        right_inds.append(good)
        if len(good) > 20:
            rx_current = int(np.mean(nonzerox[good]))

    right_inds = np.concatenate(right_inds) if len(right_inds) else np.array([], dtype=int)
    rx, ry = nonzerox[right_inds], nonzeroy[right_inds]
    if len(rx) < 200:
        return None, None, None, None

    right_fit = np.polyfit(ry, rx, 2)

    # Infer left by shifting horizontally by expected lane width (pixels)
    y_eval = binary.shape[0] - 1
    right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    left_x_bottom  = right_x_bottom - expected_lane_px

    left_fit = np.array([right_fit[0], right_fit[1], right_fit[2] - expected_lane_px])

    ploty = np.linspace(0, binary.shape[0]-1, binary.shape[0])
    return left_fit, right_fit, ploty, (left_x_bottom, right_x_bottom)


def _window(binary, nwindows=9, margin=160, minpix=8):
    """
    Classic sliding-window lane pixel search.
    Returns left/right pixel coordinates.
    """
    histogram = np.sum(binary[binary.shape[0]//2:, :], axis=0)
    h, w = binary.shape[:2]

    midpoint = w // 2
    # Ignore extreme edges to avoid curb
    left_search  = histogram[int(w*0.15):midpoint]
    right_search = histogram[midpoint:int(w*0.85)]

    leftx_base  = int(np.argmax(left_search) + int(w*0.15))
    rightx_base = int(np.argmax(right_search) + midpoint)  # fixed (no extra +midpoint)

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

        # Adaptive widening when a window is empty (helps very long dashes)
        if len(good_left_inds) == 0:
            win_xleft_low  -= 30
            win_xleft_high += 30
        if len(good_right_inds) == 0:
            win_xright_low  -= 30
            win_xright_high += 30

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Tolerate dashed gaps: only shift if enough points found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([], dtype=int)
    right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([], dtype=int)

    leftx = nonzerox[left_lane_inds]; lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]; righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty


def search_around_poly(binary, left_fit, right_fit, margin=150):
    """Fast search around previous polynomials (good for dashed lines)."""
    nz = binary.nonzero()
    nonzeroy, nonzerox = np.array(nz[0]), np.array(nz[1])

    left_region = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) &
                   (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_region = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) &
                    (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    lx, ly = nonzerox[left_region],  nonzeroy[left_region]
    rx, ry = nonzerox[right_region], nonzeroy[right_region]
    return lx, ly, rx, ry


def fit_polynomial(binary, leftx, lefty, rightx, righty):
    """Quadratic fits for left/right lanes."""
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


class LaneRegulator:
    """
    Keeps lane width and center stable with EMA smoothing and applies
    horizontal shifts to fits so the polygon stays centered and rectangular.
    """
    def __init__(self, target_px=650, alpha_w=0.25, alpha_c=0.3):
        self.w_px = target_px     # smoothed lane width (pixels)
        self.c_px = None          # smoothed center x at bottom
        self.aw = alpha_w
        self.ac = alpha_c

    def update_measures(self, left_fit, right_fit, h):
        y = h - 1
        lx = left_fit[0]*y*y + left_fit[1]*y + left_fit[2]
        rx = right_fit[0]*y*y + right_fit[1]*y + right_fit[2]
        width = rx - lx
        center = 0.5*(lx + rx)
        self.w_px = self.aw*width + (1 - self.aw)*self.w_px
        if self.c_px is None:
            self.c_px = center
        else:
            self.c_px = self.ac*center + (1 - self.ac)*self.c_px

    def regularize(self, left_fit, right_fit, h):
        """Shift c-terms so bottom positions are centered & at smoothed width."""
        y = h - 1
        lx_d = self.c_px - self.w_px/2
        rx_d = self.c_px + self.w_px/2
        lx = left_fit[0]*y*y + left_fit[1]*y + left_fit[2]
        rx = right_fit[0]*y*y + right_fit[1]*y + right_fit[2]
        lf = left_fit.copy();  rf = right_fit.copy()
        lf[2] += (lx_d - lx)
        rf[2] += (rx_d - rx)
        return lf, rf

    def synthesize_missing(self, have_left, have_right, h):
        """If only one side exists, synthesize the other from width & center."""
        y = h - 1
        if have_left is not None and have_right is None:
            lx = have_left[0]*y*y + have_left[1]*y + have_left[2]
            c = lx + self.w_px/2 if self.c_px is None else self.c_px
            rx_bottom = c + self.w_px/2
            rf = have_left.copy()
            rf[2] += (rx_bottom - lx)
            return have_left, rf
        if have_right is not None and have_left is None:
            rx = have_right[0]*y*y + have_right[1]*y + have_right[2]
            c = rx - self.w_px/2 if self.c_px is None else self.c_px
            lx_bottom = c - self.w_px/2
            lf = have_right.copy()
            lf[2] += (lx_bottom - rx)
            return lf, have_right
        return have_left, have_right


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
    """Try mp4v (mp4), then XVID (avi). Returns (writer, out_path)."""
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
    reg = LaneRegulator(target_px=640, alpha_w=0.25, alpha_c=0.3)
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
    smoother = FitSmoother(alpha=0.18)
    last_left_fit, last_right_fit = None, None

    xm_per_pix = 3.7 / 700.0  # meters per pixel (approx for bottom of image)

    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = preprocess_lane_mask(frame)

        # Tight ROI (reduces asphalt noise) — slightly shifted right
        roi_pts = [
            (int(w*0.20), int(h*0.96)),
            (int(w*0.48), int(h*0.62)),
            (int(w*0.56), int(h*0.62)),
            (int(w*0.86), int(h*0.96)),
        ]
        roi = region_of_interest(mask, roi_pts)
        # Nuke extreme-left band (curb region)
        roi[:, :int(w*0.12)] = 0

        # Perspective (bird's-eye)
        M, Minv, src = perspective_matrices(w, h)
        warped = cv2.warpPerspective(roi, M, (w, h), flags=cv2.INTER_LINEAR)

        # Lane pixel search: track around last fit, else sliding windows
        if last_left_fit is not None and last_right_fit is not None:
            lx, ly, rx, ry = search_around_poly(warped, last_left_fit, last_right_fit, margin=100)
            # If too few points, try a one-shot wider search
            if len(lx) + len(rx) < 800:
                lx2, ly2, rx2, ry2 = search_around_poly(warped, last_left_fit, last_right_fit, margin=150)
                if len(lx2) + len(rx2) > len(lx) + len(rx):
                    lx, ly, rx, ry = lx2, ly2, rx2, ry2
            if len(lx) < 800 or len(rx) < 800:
                lx, ly, rx, ry = sliding_window(warped)  # fallback (margin=160, minpix=12)
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
            valid = 480 < lane_width < 900 and abs(left_fit[0] - right_fit[0]) < 1.2e-4

        # Fallback: right-lane-only recovery
        if not valid:
            lf_rb, rf_rb, ploty_rb, _ = right_lane_fallback(warped, expected_lane_px=640, margin=120)
            if lf_rb is not None:
                left_fit, right_fit, ploty = lf_rb, rf_rb, ploty_rb
                valid = True

        # If only one side present (rare), synthesize the other
        if left_fit is None and right_fit is not None:
            left_fit, right_fit = reg.synthesize_missing(None, right_fit, h)
            valid = right_fit is not None
        elif right_fit is None and left_fit is not None:
            left_fit, right_fit = reg.synthesize_missing(left_fit, None, h)
            valid = left_fit is not None

        # If still invalid, coast on last good fit
        if not valid:
            left_fit, right_fit = last_left_fit, last_right_fit
        else:
            # Update regulator with measured width/center and regularize fits
            reg.update_measures(left_fit, right_fit, h)
            left_fit, right_fit = reg.regularize(left_fit, right_fit, h)

        # Smoothing (after regularization for best visuals)
        if left_fit is not None and right_fit is not None:
            left_fit, right_fit = smoother.update(left_fit, right_fit)
            last_left_fit, last_right_fit = left_fit, right_fit

        # -------- Render overlay (rectangle in camera space) --------
        out_frame = frame.copy()
        if left_fit is not None and right_fit is not None:
            # Rectangle corners in warped space (top/bottom evaluations)
            y_top    = int(h * RECT_Y_TOP)
            y_bottom = int(h * RECT_Y_BOTTOM)

            def eval_x(fit, y): 
                return fit[0]*y*y + fit[1]*y + fit[2]

            # bottom & raw top from the polynomials
            x_lb = eval_x(left_fit,  y_bottom)
            x_rb = eval_x(right_fit, y_bottom)
            x_lt_raw = eval_x(left_fit,  y_top)
            x_rt_raw = eval_x(right_fit, y_top)

            # compute centers + widths, then enforce a controlled top width
            center_bottom = 0.5*(x_lb + x_rb)
            width_bottom  = (x_rb - x_lb)
            center_top = 0.5*(x_lt_raw + x_rt_raw)
            width_top  = RECT_TOP_SCALE * width_bottom

            x_lt = center_top - 0.5*width_top
            x_rt = center_top + 0.5*width_top

            def clampf(x): return float(np.clip(x, 0, w-1))
            rect_warped = np.array([
                [clampf(x_lb), float(y_bottom)],
                [clampf(x_lt), float(y_top)],
                [clampf(x_rt), float(y_top)],
                [clampf(x_rb), float(y_bottom)]
            ], dtype=np.float32)

            # Project the 4 points back to camera space and draw polygon directly
            rect_cam = cv2.perspectiveTransform(rect_warped.reshape(1, -1, 2),
                                                Minv.astype(np.float32))[0].astype(np.int32)

            overlay = frame.copy()
            cv2.fillPoly(overlay, [rect_cam], (0, 255, 0))    # crisp, straight edges
            out_frame = cv2.addWeighted(frame, 1.0, overlay, 0.30, 0)

            # Optional red outline of the same rectangle
            cv2.polylines(out_frame, [rect_cam], True, (0, 0, 255), 2)

            # Offset & simple steer from polynomial bottoms
            y_eval = h - 1
            left_x_bottom  = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
            right_x_bottom = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
            lane_c = (left_x_bottom + right_x_bottom) / 2.0
            veh_c  = w / 2.0
            offset_m = (veh_c - lane_c) * xm_per_pix
            err = (lane_c - veh_c) / (w/2.0)
            steer = pid.step(err)

            # HUD
            cv2.putText(out_frame, f"Offset: {offset_m:+.2f} m", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            cv2.putText(out_frame, f"Steer: {steer:+.2f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        else:
            cv2.putText(out_frame, "Lane lost", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
        # ------------------------------------------------------------

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
    p.add_argument("--in",  dest="inp",  default="data/lanes/test-video-(1).mp4",
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
