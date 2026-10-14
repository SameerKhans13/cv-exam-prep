# Code: runnable demos for the practicals

Each script is short (under ~80 lines), self-contained, and runs on **synthetic images**, so no dataset downloads are needed. Every script was run and checked while building this pack. Expected output is listed below so you can check your own numericals against it.

```bash
pip install -r requirements.txt          # numpy, opencv-python, torch, torchvision, pillow
cd code
python canny_harris_demo.py              # etc. Images are written to code/outputs/
```
OpenCV ≥ 4.4 is needed so that `cv2.SIFT_create()` is in the main package (the SIFT patent expired in 2020). The first five scripts need only `numpy` + `opencv-python`. Torch is needed only for `transfer_learning_skeleton.py`, the optional `--image` mode of `dice_iou_segmentation.py`, and the optional cross-checks.

| Script | What it demonstrates | Exam topics | Notes |
|---|---|---|---|
| [`canny_harris_demo.py`](canny_harris_demo.py) | Canny threshold sweep (too low = noise, too high = lost edges). **Harris from scratch** (structure tensor → R) compared with `cv2.cornerHarris`. Eigenvalues at a corner, an edge and a flat pixel. Rotation invariance. Why R thresholds are **not** contrast-invariant (R ∝ contrast⁴). | Canny 5 steps, Harris M & R, λ-interpretation, invariances | [S01](../notes/01_image_processing_features.md) |
| [`sift_orb_matching_panorama.py`](sift_orb_matching_panorama.py) | SIFT (L2) vs ORB (Hamming) → 2-NN + **Lowe ratio test** → **RANSAC homography** → warp + blend into a panorama. Checks the estimated H against a **known ground-truth H**. | Matching, distance metrics, ratio test, RANSAC, homography, stitching | [S02](../notes/02_feature_matching_classical.md) |
| [`conv_output_and_receptive_field_calc.py`](conv_output_and_receptive_field_calc.py) | **Calculator**: output size, params, MACs, receptive field and jump for any conv/pool stack (with dilation). Cross-checked with PyTorch when installed. | ⌊(W−K+2P)/S⌋+1, (K²C_in+1)C_out, RF recurrence, transposed conv | [S03](../notes/03_cnns_under_the_hood.md) |
| [`iou_nms_ap.py`](iou_nms_ap.py) | **IoU, greedy NMS, TP/FP matching, AP (all-point and 11-point)** from scratch, with asserted hand-computed test cases. Cross-checked with `torchvision.ops`. | IoU, NMS threshold effect, P/R, AP, mAP | [S07](../notes/07_detection_one_stage.md) |
| [`dice_iou_segmentation.py`](dice_iou_segmentation.py) | IoU and Dice from TP/FP/FN, verifies **Dice = 2·IoU/(1+IoU)**. Confusion matrix → per-class IoU, **mIoU vs pixel accuracy** under class imbalance. `--image photo.jpg` runs a pretrained torchvision LR-ASPP model (the "run a pretrained segmentation model" lab). | Dice/IoU, mIoU, imbalance pitfalls, pretrained inference | [S10](../notes/10_image_segmentation.md) |
| [`stereo_depth_and_projection.py`](stereo_depth_and_projection.py) | Projects 3D points with **x = K[R\|t]X** (checked against `cv2.projectPoints`) and computes the camera centre **C = −Rᵀt**. Builds a synthetic rectified stereo pair with known disparities → **SGBM** → **Z = fB/d**. Depth resolution. | Pinhole, intrinsics/extrinsics, stereo depth, ΔZ ≈ Z²Δd/(fB) | [S11](../notes/11_camera_geometry.md), [S12](../notes/12_3d_vision_motion.md) |
| [`transfer_learning_skeleton.py`](transfer_learning_skeleton.py) | ResNet-18 with `--mode feature` (frozen backbone + BN kept in eval, head only) vs `--mode finetune` (all layers, **discriminative LRs**). Train-only augmentation, ImageNet normalisation, label smoothing, AdamW. `--data` takes a real ImageFolder. | Feature extraction vs fine-tuning, augmentation, regularisation | [S04](../notes/04_training_transfer_learning.md) |

## Expected output (verified)

- **canny_harris**: Canny edge pixels fall 9145 → 1386 → 1051 → 616 as the thresholds rise. Harris finds exactly **7 corners** (4 rectangle + 3 triangle, none on the circle), and still 7 after a 30° rotation. At a 1% threshold only 5 remain, because the low-contrast triangle drops out.
- **sift_orb**: about 127 SIFT and about 432 ORB RANSAC inliers. Mean corner error against ground-truth H is about **1 px** for both.
- **conv calc**: AlexNet conv1 227×227×3 → **55×55×96, 34,944 params**. ResNet stem → 56×56×64 with RF 11. Two stacked 3×3 convs → RF 5. Dilated 1/2/4 stack → RF 15. (Real ResNet conv1 has no bias because BN follows it, so its count is 9,408, not 9,472.)
- **iou_nms_ap**: IoU 81/119 = 0.6807. NMS@0.5 keeps [0, 2] and NMS@0.7 keeps [0, 1, 2, 4], so a **higher threshold keeps more boxes**. **AP all-point = 0.8667, 11-point = 0.8545**.
- **dice_iou**: IoU 0.8162, Dice 0.8988. The "lazy" model gets 99.5% pixel accuracy but **mIoU 0.66**.
- **stereo**: point (0,0,4) → pixel (445.3, 240.0). Disparity 16 px → 5.25 m and 32 px → 2.625 m (f = 700 px, B = 0.12 m). ΔZ at 5.25 m ≈ 0.33 m per pixel of disparity error.
- **transfer learning**: feature mode trains **5,130** params (512·10 + 10). Fine-tune mode trains all **11,181,642**. FakeData accuracy stays at chance by design, because these scripts only show the mechanics.
