"""Session 01 — Canny edges + Harris corners on a synthetic image (no downloads).

Run:  python canny_harris_demo.py      -> prints stats, writes outputs/canny_harris.png
Exam links: Canny's 5 steps; Harris structure tensor M; R = det(M) - k*tr(M)^2;
eigenvalue classes (corner / edge / flat); rotation invariance of Harris.
"""
import os
import cv2
import numpy as np

# 1. Synthetic image: rectangle (4 corners), triangle (3 corners), circle (no corners) + noise
img = np.zeros((240, 320), np.uint8)
cv2.rectangle(img, (30, 40), (130, 140), 200, -1)
cv2.fillPoly(img, [np.array([[200, 40], [290, 140], [170, 160]], np.int32)], 60)  # low contrast
cv2.circle(img, (240, 200), 30, 255, -1)
noisy = np.clip(img + np.random.default_rng(0).normal(0, 15, img.shape), 0, 255).astype(np.uint8)

# 2. Canny: (1) Gaussian smoothing (2) Sobel gradients (3) non-max suppression
#    (4) double threshold (5) hysteresis. cv2.Canny does 2-5, so we blur first.
blur = cv2.GaussianBlur(noisy, (5, 5), 1.4)
for lo, hi in [(10, 30), (20, 60), (50, 150), (100, 250)]:  # too low = noise; too high = lose triangle
    print(f"Canny thresholds ({lo:3d},{hi:3d}) -> {np.count_nonzero(cv2.Canny(blur, lo, hi)):5d} edge px")
edges = cv2.Canny(blur, 50, 150)


def harris(gray, k=0.05, sigma=1.0):
    """Harris from scratch: returns R and the 3 structure-tensor images (Sxx, Syy, Sxy)."""
    f = gray.astype(np.float64) / 255.0
    Ix = cv2.Sobel(f, cv2.CV_64F, 1, 0, ksize=3)            # image derivatives
    Iy = cv2.Sobel(f, cv2.CV_64F, 0, 1, ksize=3)
    Sxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)          # window-weighted sums = entries of M
    Syy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
    Sxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)
    R = (Sxx * Syy - Sxy ** 2) - k * (Sxx + Syy) ** 2        # det(M) - k tr(M)^2
    return R, Sxx, Syy, Sxy


def corners(R, rel_thr=0.005):
    """Threshold + 7x7 non-maximum suppression -> list of (x, y)."""
    keep = (R > rel_thr * R.max()) & (R == cv2.dilate(R, np.ones((7, 7))))
    ys, xs = np.nonzero(keep)
    return list(zip(xs.tolist(), ys.tolist()))


R, Sxx, Syy, Sxy = harris(blur)
pts = corners(R)
print(f"Harris (from scratch): {len(pts)} corners -> {sorted(pts)}")
# R grows ~ contrast^4, so a stricter threshold drops the low-contrast triangle's corners:
print(f"  with threshold 1% of max: {len(corners(R, 0.01))} corners (Harris is NOT invariant to contrast scaling)")

# 3. Eigenvalues of M tell the story: corner = both large, edge = one large, flat = both small
for name, (x, y) in {"corner": (30, 40), "edge": (80, 40), "flat": (80, 90)}.items():
    M = np.array([[Sxx[y, x], Sxy[y, x]], [Sxy[y, x], Syy[y, x]]])
    l1, l2 = np.linalg.eigvalsh(M)
    print(f"  {name:6s} at ({x},{y}): lambda=({l1:.4f}, {l2:.4f})  R={R[y, x]:+.6f}")

# 4. Rotation invariance: rotate the image 30 deg -> same number of corners (scale would NOT be invariant)
Rot = cv2.getRotationMatrix2D((160, 120), 30, 1.0)
rot = cv2.warpAffine(blur, Rot, (320, 240), borderValue=0)
print(f"Corners after 30 deg rotation: {len(corners(harris(rot)[0]))} (vs {len(pts)})")

# 5. OpenCV's built-in for comparison (blockSize = window, ksize = Sobel aperture)
Rcv = cv2.cornerHarris(blur.astype(np.float32), blockSize=5, ksize=3, k=0.05)
print(f"cv2.cornerHarris corners: {len(corners(Rcv))}")

# 6. Save a side-by-side visualisation
vis = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
for x, y in pts:
    cv2.circle(vis, (x, y), 5, (0, 0, 255), 2)
out = np.hstack([vis, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"), exist_ok=True)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "canny_harris.png")
cv2.imwrite(path, out)
print("Saved", path)
