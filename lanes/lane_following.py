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
    print("Saved -> out/lanes_annotated.mp4")