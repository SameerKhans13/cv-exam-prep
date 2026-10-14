"""Session 07 — IoU, greedy NMS and Average Precision from scratch (NumPy only) + test cases.

Run:  python iou_nms_ap.py
Conventions (same as the notes): boxes are (x1, y1, x2, y2), continuous coords, area = (x2-x1)(y2-y1), no +1.
AP: VOC-style matching (each detection -> highest-IoU GT; if that GT is already taken -> FP).
"""
import numpy as np


def iou(a, b):
    """IoU of two boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, thr=0.5):
    """Greedy NMS: keep the best box, drop others with IoU > thr, repeat. Returns kept indices."""
    order, keep = list(np.argsort(scores)[::-1]), []
    while order:
        best = order.pop(0)
        keep.append(int(best))
        order = [i for i in order if iou(boxes[best], boxes[i]) <= thr]
    return keep


def match(dets, scores, gts, thr=0.5):
    """Label each detection TP (1) / FP (0), in descending score order."""
    taken, labels = set(), []
    for d in np.argsort(scores)[::-1]:
        ious = [iou(dets[d], g) for g in gts]
        j = int(np.argmax(ious)) if gts else -1
        ok = j >= 0 and ious[j] >= thr and j not in taken     # duplicates of a matched GT are FP
        if ok:
            taken.add(j)
        labels.append(int(ok))
    return np.array(labels)


def average_precision(tp, n_gt, method="all"):
    """tp: 1/0 labels sorted by score. method: 'all' (VOC2010+ all-point) or '11' (VOC2007)."""
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / (ctp + cfp)
    if method == "11":  # tolerance: linspace gives 0.6000000000000001, which would wrongly exclude recall = 0.6
        return np.mean([prec[rec >= t - 1e-9].max() if (rec >= t - 1e-9).any() else 0 for t in np.linspace(0, 1, 11)])
    r, p = np.concatenate([[0], rec, [1]]), np.concatenate([[0], prec, [0]])
    for i in range(len(p) - 2, -1, -1):                  # precision envelope: p(r) = max_{r' >= r} p(r')
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]                    # sum rectangle areas where recall increases
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


if __name__ == "__main__":
    # --- IoU: 10x10 boxes shifted by 1 px -> 81 / 119
    assert abs(iou((0, 0, 10, 10), (1, 1, 11, 11)) - 81 / 119) < 1e-9
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # --- NMS: higher threshold keeps MORE boxes
    B = [(0, 0, 10, 10), (1, 1, 11, 11), (20, 20, 30, 30), (0, 0, 10, 9), (21, 21, 31, 31)]
    S = [0.9, 0.8, 0.7, 0.6, 0.5]
    print("NMS @0.5 keeps", nms(B, S, 0.5), "| NMS @0.7 keeps", nms(B, S, 0.7))
    assert nms(B, S, 0.5) == [0, 2] and nms(B, S, 0.7) == [0, 1, 2, 4]
    # --- AP: 3 GT objects, 6 detections
    gts = [(0, 0, 10, 10), (20, 20, 30, 30), (40, 40, 50, 50)]
    dets = [(1, 1, 11, 11), (21, 19, 31, 29), (0, 1, 10, 11), (60, 60, 70, 70), (40, 42, 50, 52), (25, 25, 35, 35)]
    scores = [0.95, 0.90, 0.80, 0.70, 0.60, 0.50]
    tp = match(dets, scores, gts)
    ctp = np.cumsum(tp)
    print("TP/FP:", tp, "\nprecision:", np.round(ctp / np.arange(1, 7), 3), "\nrecall:   ", np.round(ctp / 3, 3))
    ap_all, ap_11 = average_precision(tp, 3, "all"), average_precision(tp, 3, "11")
    print(f"AP all-point = {ap_all:.4f} (hand: 1/3*1 + 1/3*1 + 1/3*0.6 = 0.8667)")
    print(f"AP 11-point  = {ap_11:.4f} (hand: (7*1 + 4*0.6)/11 = 0.8545)")
    assert list(tp) == [1, 1, 0, 0, 1, 0] and abs(ap_all - 0.8667) < 1e-4 and abs(ap_11 - 9.4 / 11) < 1e-9
    # Regression test = practice/07 Q11 (5 GT): recall hits exactly 0.6, the 11-point float edge case
    tp2 = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 1])
    assert abs(average_precision(tp2, 5, "11") - 0.7857) < 1e-4 and abs(average_precision(tp2, 5) - 0.7643) < 1e-4
    try:                                                   # optional cross-check against torchvision
        import torch
        from torchvision.ops import nms as tv_nms, box_iou
        assert tv_nms(torch.tensor(B, dtype=torch.float), torch.tensor(S), 0.5).tolist() == [0, 2]
        assert abs(box_iou(torch.tensor([B[0]], dtype=torch.float), torch.tensor([B[1]], dtype=torch.float)).item() - 81 / 119) < 1e-6
        print("torchvision cross-check passed")
    except ImportError:
        pass
    print("All tests passed.")
