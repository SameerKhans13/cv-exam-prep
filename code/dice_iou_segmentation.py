"""Session 10 — Segmentation metrics from scratch: IoU, Dice, mIoU, pixel accuracy (+ optional pretrained model).

Run:  python dice_iou_segmentation.py                    # metrics on synthetic masks, no downloads
      python dice_iou_segmentation.py --image photo.jpg  # ALSO runs torchvision LR-ASPP (downloads ~13 MB weights)
Formulas: IoU = TP/(TP+FP+FN);  Dice = 2TP/(2TP+FP+FN) = 2*IoU/(1+IoU);  IoU = Dice/(2-Dice)
"""
import argparse
import numpy as np


def binary_scores(pred, gt):
    tp = np.sum(pred & gt); fp = np.sum(pred & ~gt); fn = np.sum(~pred & gt)
    return tp, fp, fn, tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)


def confusion(pred, gt, n):
    """n x n matrix, rows = ground truth class, cols = predicted class."""
    return np.bincount(n * gt.ravel() + pred.ravel(), minlength=n * n).reshape(n, n)


def miou_and_pixel_acc(cm):
    iou = np.diag(cm) / (cm.sum(0) + cm.sum(1) - np.diag(cm))   # TP / (TP + FP + FN) per class
    return iou, np.nanmean(iou), np.trace(cm) / cm.sum()


# 1. Binary: GT circle vs a prediction shifted 8 px right
yy, xx = np.mgrid[:200, :200]
gt = (xx - 100) ** 2 + (yy - 100) ** 2 < 50 ** 2
pred = (xx - 108) ** 2 + (yy - 100) ** 2 < 50 ** 2
tp, fp, fn, iou, dice = binary_scores(pred, gt)
print(f"TP={tp} FP={fp} FN={fn}  IoU={iou:.4f}  Dice={dice:.4f}  2IoU/(1+IoU)={2 * iou / (1 + iou):.4f}")
assert abs(dice - 2 * iou / (1 + iou)) < 1e-12 and dice >= iou   # identity holds; Dice >= IoU always

# 2. Multi-class with heavy imbalance: background 0, road 1, tiny "person" class 2
gt_mc = np.zeros((200, 200), int)
gt_mc[120:, :] = 1                      # road: bottom 40%
gt_mc[60:80, 90:100] = 2                # person: 200 px = 0.5% of image
lazy = np.where(gt_mc == 2, 0, gt_mc)   # model that never predicts "person"
for name, p in [("lazy model (misses person)", lazy), ("GT-perfect model", gt_mc)]:
    per_cls, miou, acc = miou_and_pixel_acc(confusion(p, gt_mc, 3))
    print(f"{name:28s} pixel acc={acc:.4f}  per-class IoU={np.round(per_cls, 3)}  mIoU={miou:.4f}")
print("-> pixel accuracy hides the missed small class; mIoU exposes it.")


def run_pretrained(path):
    """Lab: semantic segmentation with a pretrained torchvision model (21 VOC classes)."""
    import torch
    from PIL import Image
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights).eval()            # eval(): BN/dropout in inference mode
    x = weights.transforms()(Image.open(path).convert("RGB")).unsqueeze(0)  # resize + ImageNet normalisation
    with torch.no_grad():
        logits = model(x)["out"]                                          # [1, 21, H, W] class scores
    labels = logits.argmax(1)[0]                                          # per-pixel class = argmax over classes
    names = weights.meta["categories"]
    for c, n in zip(*np.unique(labels.numpy(), return_counts=True)):
        print(f"  {names[c]:12s} {100 * n / labels.numel():5.1f}% of pixels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="optional photo to segment with a pretrained model")
    if (img := ap.parse_args().image):
        run_pretrained(img)
