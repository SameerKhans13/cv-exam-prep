# Session 07 — Object Detection II: One-Stage Detectors, NMS & Evaluation · Practice Questions

> Notes: [../notes/07_detection_one_stage.md](../notes/07_detection_one_stage.md) · Solutions: [07_solutions.md](07_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min
>
> **Box convention for every question:** boxes are $(x_1,y_1,x_2,y_2)$ in continuous pixel coordinates, $x$ = column, $y$ = row (downwards), area $=(x_2-x_1)(y_2-y_1)$ — **no +1**. $\log$/$\ln$ = natural logarithm.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
YOLOv1 on PASCAL VOC uses an $S=7$ grid, $B=2$ boxes per cell and $C=20$ classes. What is the shape of its output tensor?

- **A.** 7×7×25
- **B.** 7×7×24
- **C.** 7×7×30
- **D.** 7×7×50

### Q2 · MCQ · Medium
A detector's NMS IoU threshold is raised from 0.5 to 0.7. What is the effect on the final detections?

- **A.** Fewer boxes are suppressed, so more boxes (including possible duplicates) survive.
- **B.** More boxes are suppressed, so fewer boxes survive.
- **C.** The same boxes survive; only their scores change.
- **D.** All boxes with confidence below 0.7 are removed.

### Q3 · MCQ · Easy
The Average Precision (AP) of an object detector for one class is the area under which curve?

- **A.** The ROC curve (true-positive rate vs false-positive rate).
- **B.** Recall plotted against the IoU threshold.
- **C.** Precision plotted against the confidence threshold.
- **D.** The (interpolated) precision–recall curve.

### Q4 · MCQ · Easy
COCO's primary metric, AP = mAP@[.5:.95], averages AP over:

- **A.** 5 IoU thresholds (0.5, 0.6, 0.7, 0.8, 0.9).
- **B.** 10 IoU thresholds (0.50, 0.55, …, 0.95) and over all classes.
- **C.** the two IoU thresholds 0.5 and 0.95 only.
- **D.** 10 confidence thresholds from 0.50 to 0.95.

### Q5 · MCQ · Medium
In the focal loss $FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t$, what happens when $\gamma=0$?

- **A.** The loss becomes zero for every example.
- **B.** Only misclassified examples contribute to the loss.
- **C.** It reduces to the α-balanced cross-entropy loss.
- **D.** It becomes the smooth-L1 loss.

### Q6 · MCQ · Medium
Which statement about **FCOS** is correct?

- **A.** It needs no NMS because centre-ness guarantees exactly one prediction per object.
- **B.** It places 9 anchors at every location of every FPN level.
- **C.** It assigns each GT box to the FPN level whose anchors have the highest IoU with it.
- **D.** Each location regresses the distances $(l,t,r,b)$ to the box sides, and a centre-ness score down-weights off-centre predictions before NMS.

### Q7 · MCQ · Easy
How does **SSD** handle the very large number of negative default boxes during training?

- **A.** It sorts negatives by confidence loss and keeps the hardest ones so that negatives : positives ≤ 3 : 1.
- **B.** It uses all 8732 boxes with equal weight.
- **C.** It samples random negatives at a 1 : 1 ratio with positives.
- **D.** It down-weights easy negatives with a focal loss with $\gamma=2$.

### Q8 · MCQ · Medium
How does **CenterNet** ("Objects as Points") remove duplicate detections at inference?

- **A.** Greedy NMS with an IoU threshold of 0.5.
- **B.** It keeps only heatmap locations whose value equals their 3×3 max-pooled value (local peaks).
- **C.** Hungarian matching between predictions at inference time.
- **D.** Soft-NMS with a Gaussian score decay.

## Section B — Numericals

### Q9 · Numerical · Easy · [5 marks]
Let $A=(10,10,60,50)$.

(a) Compute IoU($A$, $B$) for $B=(30,20,90,70)$, IoU($A$, $C$) for $C=(70,60,100,90)$ and IoU($A$, $D$) for $D=(20,20,40,40)$. [3]
(b) Compute GIoU($A$, $B$) and GIoU($A$, $C$), where $\text{GIoU}=\text{IoU}-\frac{\lvert E\rvert-\lvert A\cup X\rvert}{\lvert E\rvert}$ and $E$ is the smallest box enclosing both. [1.5]
(c) Convert IoU($A$, $B$) to a Dice coefficient, and explain in one line why $1-\text{IoU}$ is a poor training loss for the pair ($A$, $C$) while $1-\text{GIoU}$ is not. [0.5]

### Q10 · Numerical · Medium · [5 marks]
A detector outputs five boxes of the **same class**:

| Box | $(x_1,y_1,x_2,y_2)$ | Score |
|---|---|---|
| $b_1$ | (100, 100, 200, 200) | 0.95 |
| $b_2$ | (110, 105, 210, 205) | 0.90 |
| $b_3$ | (300, 120, 380, 220) | 0.85 |
| $b_4$ | (150, 150, 250, 250) | 0.80 |
| $b_5$ | (310, 130, 390, 230) | 0.60 |

(a) Compute IoU($b_1,b_2$), IoU($b_1,b_4$), IoU($b_2,b_4$) and IoU($b_3,b_5$) (all other pairs do not overlap). [1.5]
(b) Run greedy NMS with threshold $\tau_{nms}=0.5$ (suppress if IoU > $\tau_{nms}$). Show each step and list the kept boxes. [1.5]
(c) Repeat with $\tau_{nms}=0.7$. [1]
(d) Instead run **linear Soft-NMS** with $N_t=0.5$ ($s_i\leftarrow s_i(1-\text{IoU})$ if IoU ≥ $N_t$). Give the final scores of $b_2$ and $b_5$. [1]

### Q11 · Numerical · Hard · [5 marks]
The class "car" has **5** ground-truth objects $G_1$–$G_5$ in the test set. The detector outputs 10 "car" detections (already sorted by score). For each, the GT with the highest IoU and that IoU are given:

| Det | Score | Best GT | IoU |
|---|---|---|---|
| $d_1$ | 0.95 | $G_1$ | 0.82 |
| $d_2$ | 0.91 | $G_2$ | 0.76 |
| $d_3$ | 0.88 | $G_1$ | 0.65 |
| $d_4$ | 0.80 | $G_3$ | 0.57 |
| $d_5$ | 0.74 | $G_4$ | 0.42 |
| $d_6$ | 0.65 | — | 0 |
| $d_7$ | 0.60 | $G_4$ | 0.71 |
| $d_8$ | 0.52 | $G_2$ | 0.58 |
| $d_9$ | 0.45 | — | 0 |
| $d_{10}$ | 0.30 | $G_5$ | 0.53 |

Matching rule: in score order, a detection is a **TP** if IoU ≥ 0.5 with its best GT **and** that GT is not already matched; otherwise it is an **FP**.

(a) Label each detection TP/FP and give the cumulative precision and recall after each detection. [1.5]
(b) Compute AP with **all-point interpolation** (area under the precision envelope). [1.5]
(c) Compute AP with **VOC2007 11-point interpolation**. [1]
(d) (i) The APs@0.5 of the other two classes are person 0.62 and bicycle 0.51. Compute mAP@0.5. (ii) The car all-point APs at IoU thresholds 0.55, 0.60, …, 0.95 are 0.664, 0.486, 0.486, 0.486, 0.400, 0.200, 0, 0, 0. Using your answer to (b) for 0.50, compute the COCO-style AP@[.5:.95] for car (all-point AP at each threshold, for simplicity) and explain why it is so much lower than AP@0.5. [1]

### Q12 · Numerical · Medium · [5 marks]
Detector "size" arithmetic.

(a) YOLOv1 with $S=7$, $B=2$: give the output tensor shape, the number of output values and the number of predicted boxes for PASCAL VOC ($C=20$); then give the tensor shape if the same design is trained on COCO ($C=80$). [1.5]
(b) YOLOv3 with a 416×416 input predicts at strides 32, 16 and 8 with 3 anchors per location at each scale and 80 classes (each anchor predicts 4 box values + 1 objectness + 80 class scores). Give the three grid sizes, the number of output channels per scale and the total number of predicted boxes. [1.5]
(c) SSD300 predicts from feature maps of size 38, 19, 10, 5, 3, 1 with 4, 6, 6, 6, 4, 4 default boxes per location. Compute the total number of default boxes, and the number of output channels of the 3×3 prediction conv on the 38×38 map for PASCAL VOC (20 classes + background, 4 offsets per box). [1.5]
(d) RetinaNet with a 640×640 input uses levels $P_3$–$P_7$ (strides 8, 16, 32, 64, 128) with 9 anchors per location. How many anchors in total? [0.5]

### Q13 · Numerical · Medium · [5 marks]
Focal loss uses $\gamma=2$, $\alpha=0.25$, with $\alpha_t=\alpha$ for foreground ($y=1$) and $\alpha_t=1-\alpha$ for background ($y=0$); $p$ is the predicted foreground probability and $p_t=p$ if $y=1$, $p_t=1-p$ if $y=0$.

(a) Compute the cross-entropy $-\ln p_t$ and the focal loss for (i) an **easy background** anchor with $p=0.1$ and (ii) an **ambiguous foreground** anchor with $p=0.5$. [2]
(b) An image produces 100 000 easy background anchors (each $p=0.1$) and 100 hard foreground anchors (each $p=0.1$). Compute the total loss contributed by each group under CE (no $\alpha$) and under focal loss, and the share of the total loss coming from the foreground anchors in each case. Comment. [1.5]
(c) **FCOS**: a GT box is $(50,40,250,200)$ and a feature location maps to image point $(110,90)$. Compute $(l^*,t^*,r^*,b^*)$, the FPN level it is assigned to (ranges: $P_3$ (0, 64], $P_4$ (64, 128], $P_5$ (128, 256], $P_6$ (256, 512], $P_7$ > 512 on $\max(l^*,t^*,r^*,b^*)$) and its centre-ness target $\sqrt{\frac{\min(l^*,r^*)}{\max(l^*,r^*)}\cdot\frac{\min(t^*,b^*)}{\max(t^*,b^*)}}$. [1.5]

## Section C — Short answer

### Q14 · Short · Medium · [4 marks]
Explain how **YOLOv1** turns detection into a single regression problem: the grid, the "responsible" cell and predictor, and what the confidence score means. Give two limitations of YOLOv1 and how later YOLO versions addressed them.

### Q15 · Short · Medium · [4 marks]
Describe the **greedy NMS** algorithm. How does the IoU threshold affect precision and recall? Why does NMS fail in crowded scenes, and how does **Soft-NMS** help?

### Q16 · Short · Medium · [4 marks]
Contrast **anchor-based** and **anchor-free** detectors: what anchors are, how training labels are assigned, and their hyper-parameters; how **FCOS** and **CenterNet** work without anchors. Does "anchor-free" imply "NMS-free"?

### Q17 · Short · Medium · [4 marks]
Explain the **foreground–background class imbalance** problem in one-stage detectors and compare three remedies: the two-stage approach (proposals + sampling), SSD's hard-negative mining, and RetinaNet's focal loss.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
**Compare two-stage and one-stage detectors.** Using Faster R-CNN and representative one-stage detectors (YOLO, SSD, RetinaNet, FCOS), compare their pipelines (with a diagram), how they generate and label candidates, how they handle class imbalance, their use of anchors and NMS, and their speed–accuracy trade-offs. Explain why one-stage detectors were historically less accurate and what closed the gap, and recommend which family to use for two different applications.

### Q19 · Long · Hard · [10 marks]
Design a detector for **drone (UAV) video** of car parks and streets that must detect vehicles and pedestrians — many objects are **small** (10–40 px) and densely packed — running at **≥ 25 fps on an embedded GPU**. Specify and justify: backbone and neck, anchor-based vs anchor-free head, label assignment and losses, how you handle class imbalance and small objects (data and architecture), duplicate removal in dense scenes, and the **evaluation protocol** (which AP variants, IoU thresholds, size breakdown, how you pick the deployment confidence and NMS thresholds).
