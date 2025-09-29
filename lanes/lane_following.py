#installed packages and imports as well as folder structure for CVR-1.0
import cv2, numpy as np, math
from collections import deque


# --- Helpers ---
def hls_threshold(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    H,L,S = cv2.split(hls)
    # White & yellow thresholds
    white = cv2.inRange(S, 0, 40)
    yellow = cv2.inRange(H, 15, 35)
    sat = cv2.inRange(S, 90, 255)
    mask = cv2.bitwise_or(yellow, sat)
    mask = cv2.bitwise_or(mask, cv2.bitwise_not(white))
    return mask


def region_of_interest(img, pts):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return cv2.bitwise_and(img, mask)


def perspective_matrices(w,h):
    src = np.float32([[w*0.43,h*0.62],[w*0.57,h*0.62],[w*0.10,h*0.95],[w*0.90,h*0.95]])
    dst = np.float32([[w*0.25,0],[w*0.75,0],[w*0.25,h],[w*0.75,h]])
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M, Minv, src


def sliding_window(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped,binary_warped,binary_warped))
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    nwindows=9; margin=80; minpix=50
    window_height = np.int32(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base; rightx_current = rightx_base
    left_lane_inds=[]; right_lane_inds=[]


    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        good_left_inds = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)).nonzero()[0]
        good_right_inds=((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xright_low)&(nonzerox<win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds)>minpix: leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds)>minpix: rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))


    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds= np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]; lefty = nonzeroy[left_lane_inds]
    rightx= nonzerox[right_lane_inds]; righty= nonzeroy[right_lane_inds]
    return leftx,lefty,rightx,righty
    


def fit_polynomial(binary_warped, leftx,lefty,rightx,righty):
    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx)>0 else None
    right_fit= np.polyfit(righty, rightx, 2) if len(rightx)>0 else None
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    return left_fit, right_fit, ploty
    
class PID:
    def __init__(self,kp=0.12,ki=0.0,kd=0.05,window=30):
        self.kp,self.ki,self.kd = kp,ki,kd
        self.errs = deque(maxlen=window)
        self.prev = None
    def step(self,err):
        self.errs.append(err)
        d = 0 if self.prev is None else err - self.prev
        self.prev = err
        i = sum(self.errs)
        return self.kp*err + self.ki*i + self.kd*d    


# --- Main ---
# --- Main (robust I/O) ---
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
IN_PATH  = ROOT / "data" / "lanes" / "input.mp4"   # change if needed
OUT_BASENAME = ROOT / "out" / "lanes_annotated.mp4"
OUT_BASENAME.parent.mkdir(parents=True, exist_ok=True)

print("CWD      :", Path.cwd())
print("Input    :", IN_PATH)
print("Output   :", OUT_BASENAME)

cap = cv2.VideoCapture(str(IN_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open input video: {IN_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
if w == 0 or h == 0:
    cap.release()
    raise RuntimeError("Input reports 0 width/height. Try another file or re-encode.")

def open_writer(base_path, fps, size):
    for fourcc_str, ext in [("mp4v", ".mp4"), ("XVID", ".avi")]:
        out_path = Path(base_path).with_suffix(ext)
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc_str), fps, size)
        if vw.isOpened():
            print(f"Opened writer with {fourcc_str} -> {out_path.name}")
            return vw, out_path
        vw.release()
    raise RuntimeError("Failed to open VideoWriter with mp4v or XVID.")

out, OUT_PATH = open_writer(OUT_BASENAME, fps, (w, h))

pid = PID()
frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]

    mask = hls_threshold(frame)
    blur = cv2.GaussianBlur(mask,(5,5),0)
    edges = cv2.Canny(blur, 50, 150)

    roi_pts = [(int(w*0.1), int(h*0.95)), (int(w*0.43), int(h*0.62)),
               (int(w*0.57), int(h*0.62)), (int(w*0.9), int(h*0.95))]
    roi = region_of_interest(edges, roi_pts)

    M, Minv, src = perspective_matrices(w,h)
    warped = cv2.warpPerspective(roi, M, (w,h), flags=cv2.INTER_LINEAR)

    leftx,lefty,rightx,righty = sliding_window(warped)
    left_fit,right_fit,ploty = fit_polynomial(warped,leftx,lefty,rightx,righty)

    lane_area = np.zeros((h,w), dtype=np.uint8)
    if left_fit is not None and right_fit is not None:
        left_fitx  = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts_left   = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right  = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])) )])
        pts        = np.hstack((pts_left, pts_right)).astype(np.int32)
        cv2.fillPoly(lane_area, pts, 255)

        y_eval = h-1
        left_x   = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
        right_x  = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
        lane_c   = (left_x + right_x)/2.0
        veh_c    = w/2.0
        xm_per_pix = 3.7/700
        offset_m = (veh_c - lane_c) * xm_per_pix

        target_err = (lane_c - veh_c) / (w/2.0)
        steer = pid.step(target_err)

        color_warp = cv2.warpPerspective(lane_area, Minv, (w,h))
        overlay = frame.copy()
        overlay[color_warp>0] = (0,255,0)
        out_frame = cv2.addWeighted(frame, 1.0, overlay, 0.3, 0)

        cv2.putText(out_frame, f"Offset: {offset_m:+.2f} m", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(out_frame, f"Steer: {steer:+.2f}", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.polylines(out_frame, [np.int32(src)], True, (0,0,255), 2)
    else:
        out_frame = frame
        cv2.putText(out_frame, "Lane lost", (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

    out.write(out_frame)
    frame_count += 1

cap.release()
out.release()
print("Frames written:", frame_count)
if frame_count == 0:
    raise RuntimeError("No frames were written — check input path/codec.")
print("Saved ->", OUT_PATH.resolve())
