"""Session 02 — SIFT vs ORB matching + Lowe ratio test + RANSAC homography + panorama warp.

Run:  python sift_orb_matching_panorama.py   -> prints match/inlier stats, writes outputs/panorama_*.png
No downloads: we synthesise a textured "world", cut two overlapping views and apply a KNOWN
homography to the right view, so we can check the estimated H against ground truth.
"""
import os
import cv2
import numpy as np

rng = np.random.default_rng(1)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

# 1. Textured world (random shapes + text) -> left view and a perspective-warped right view
world = np.full((400, 900, 3), 255, np.uint8)
for _ in range(220):
    c, (x, y) = rng.integers(0, 255, 3).tolist(), rng.integers(0, [900, 400]).tolist()
    if rng.random() < 0.5:
        cv2.rectangle(world, (x, y), (x + int(rng.integers(10, 60)), y + int(rng.integers(10, 60))), c, -1)
    else:
        cv2.circle(world, (x, y), int(rng.integers(5, 30)), c, -1)
for i in range(12):
    cv2.putText(world, f"CV-{i}", (int(rng.integers(0, 820)), int(rng.integers(30, 390))), 0, 1.0, (0, 0, 0), 2)
left = world[:, :500]
Hw = np.array([[0.95, 0.05, 10], [-0.04, 1.0, 5], [5e-5, 2e-5, 1.0]])  # simulates a small camera rotation
right = cv2.warpPerspective(world[:, 300:800], Hw, (500, 400), borderValue=(255, 255, 255))
T = np.array([[1, 0, 300], [0, 1, 0], [0, 0, 1]], float)
H_true = T @ np.linalg.inv(Hw)                       # maps right-view pixels -> left/world pixels

gl, gr = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
for name, det, norm in [("SIFT", cv2.SIFT_create(), cv2.NORM_L2),          # float 128-d -> L2
                        ("ORB", cv2.ORB_create(3000), cv2.NORM_HAMMING)]:  # binary 256-bit -> Hamming
    # 2. Detect + describe
    kl, dl = det.detectAndCompute(gl, None)
    kr, dr = det.detectAndCompute(gr, None)
    # 3. 2-NN matching + Lowe's ratio test: keep only distinctive matches (d1 < 0.75 * d2)
    knn = cv2.BFMatcher(norm).knnMatch(dr, dl, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
    # 4. RANSAC homography (needs >= 4 matches; 5 px reprojection threshold)
    src = np.float32([kr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    # 5. Accuracy: where do the right view's corners land vs ground truth?
    corners = np.float32([[0, 0], [499, 0], [499, 399], [0, 399]]).reshape(-1, 1, 2)
    err = np.linalg.norm(cv2.perspectiveTransform(corners, H) - cv2.perspectiveTransform(corners, H_true), axis=2)
    print(f"{name:4s}: kps L/R={len(kl)}/{len(kr)}  2-NN pairs={len(knn)}  after ratio test={len(good)}  "
          f"RANSAC inliers={int(mask.sum())}  mean corner error={err.mean():.2f}px")
    # 6. Panorama: warp right into the left frame, then paste left (simple overwrite; real
    #    stitchers use feathering / multi-band blending + seam finding + exposure compensation)
    pano = cv2.warpPerspective(right, H, (900, 400))
    pano[:, :500] = np.where(pano[:, :500] == 0, left, (0.5 * pano[:, :500] + 0.5 * left).astype(np.uint8))
    vis = cv2.drawMatches(right, kr, left, kl, [g for g, k in zip(good, mask.ravel()) if k][:40], None)
    cv2.imwrite(os.path.join(OUT, f"panorama_{name.lower()}.png"), np.vstack([pano, vis[:, :900]]))

print("H_true (normalised):\n", np.round(H_true / H_true[2, 2], 4))
print("Saved panoramas to", OUT)
