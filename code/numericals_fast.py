"""Fast Computer Vision numerical formulas.

Run:
    python code/numericals_fast.py
"""
from math import ceil, log


def convolution(w, k, c_in, c_out, stride=1, padding=0, dilation=1):
    k_eff = dilation * (k - 1) + 1
    output = (w - k_eff + 2 * padding) // stride + 1
    params = (k * k * c_in + 1) * c_out
    return output, params


def ransac_iterations(success, outlier_ratio, sample_size):
    return ceil(log(1 - success) / log(1 - (1 - outlier_ratio) ** sample_size))


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return intersection / (area_a + area_b - intersection)


def stereo_depth(focal_px, baseline_m, disparity_px, disparity_error_px=1):
    depth_m = focal_px * baseline_m / disparity_px
    depth_error_m = depth_m**2 / (focal_px * baseline_m) * disparity_error_px
    return depth_m, depth_error_m


def segmentation_metrics(true_positive, false_positive, false_negative):
    iou_score = true_positive / (true_positive + false_positive + false_negative)
    dice_score = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    return iou_score, dice_score


if __name__ == "__main__":
    print("Convolution:", convolution(224, 3, 64, 128, stride=2, padding=1))
    print("RANSAC:", ransac_iterations(0.99, 0.50, 4), "iterations")
    print("IoU:", round(iou((10, 10, 50, 50), (30, 20, 70, 60)), 3))
    print("Stereo depth:", tuple(round(value, 3) for value in stereo_depth(700, 0.12, 42)))
    print("Segmentation:", tuple(round(value, 3) for value in segmentation_metrics(80, 10, 20)))
