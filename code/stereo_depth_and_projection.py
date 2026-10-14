"""Sessions 11–12 — Pinhole projection x = K[R|t]X, and depth from disparity Z = fB/d on a synthetic stereo pair.

Run:  python stereo_depth_and_projection.py
Conventions (same as the notes): X_cam = R X_world + t; lambda*[u v 1]^T = K [R|t] [X Y Z 1]^T;
rectified stereo: d = x_L - x_R (pixels), Z = f*B/d, depth error dZ ~ Z^2/(f*B) * dd.
"""
import cv2
import numpy as np

# ---------- Part A: projection with intrinsics K and extrinsics [R|t] ----------
f, cx, cy = 700.0, 320.0, 240.0
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
th = np.deg2rad(10)                                      # camera yawed 10 deg about the y axis
R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
t = np.array([[0.1], [0.0], [0.5]])
P = K @ np.hstack([R, t])                                # 3x4 projection matrix (11 DOF)
Xw = np.array([[0.0, 0.0, 4.0, 1.0], [1.0, -0.5, 6.0, 1.0]]).T  # two homogeneous world points (columns)
x = P @ Xw
uv = x[:2] / x[2]                                        # divide by lambda (= depth in camera frame)
for i in range(Xw.shape[1]):
    print(f"X={Xw[:3, i]} -> depth {x[2, i]:.3f} m -> pixel (u, v) = ({uv[0, i]:.1f}, {uv[1, i]:.1f})")
print("Camera centre C = -R^T t =", np.round((-R.T @ t).ravel(), 4))
uv_cv, _ = cv2.projectPoints(Xw[:3].T.copy(), cv2.Rodrigues(R)[0], t, K, None)  # OpenCV agrees
assert np.allclose(uv_cv.reshape(-1, 2).T, uv)

# ---------- Part B: synthetic rectified stereo pair with known disparities ----------
rng = np.random.default_rng(0)
fB = 700.0 * 0.12                                        # f = 700 px, baseline B = 0.12 m
bg = cv2.GaussianBlur(rng.integers(0, 255, (240, 400)).astype(np.uint8), (3, 3), 0)  # textured background
fg = cv2.GaussianBlur(rng.integers(0, 255, (80, 80)).astype(np.uint8), (3, 3), 0)    # textured box in front
d_bg, d_fg = 16, 32                                      # background far (small d), box near (large d)
left = bg[:, 40:360].copy()
right = bg[:, 40 + d_bg:360 + d_bg].copy()               # x_R = x_L - d  <=>  right(x) = left(x + d)
left[80:160, 150:230] = fg
right[80:160, 150 - d_fg:230 - d_fg] = fg

# Semi-global block matching (returns fixed-point disparity * 16)
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=7, P1=8 * 49, P2=32 * 49)
disp = sgbm.compute(left, right).astype(np.float32) / 16.0

for name, region, d_true in [("background", disp[20:60, 120:300], d_bg), ("box", disp[95:145, 165:215], d_fg)]:
    d_est = np.median(region[region > 0])
    print(f"{name:10s}: disparity true={d_true:2d}px est={d_est:5.2f}px -> "
          f"Z_true={fB / d_true:.3f} m  Z_est={fB / d_est:.3f} m")

Z = fB / d_bg
print(f"Depth resolution at Z={Z:.2f} m for a 1 px disparity error: dZ ~ Z^2/(fB) = {Z ** 2 / fB:.3f} m")
print(f"Max range if min usable disparity is 1 px: Z = fB/1 = {fB:.1f} m")
