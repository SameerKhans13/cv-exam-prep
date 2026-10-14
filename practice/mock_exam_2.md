# Mock Exam 2 — Computer Vision · Full Paper

> Solutions: [mock_exam_2_solutions.md](mock_exam_2_solutions.md) · Formula sheet: [../00_formula_sheet.md](../00_formula_sheet.md) · Comparison tables: [../00_comparison_tables.md](../00_comparison_tables.md) · Other mock: [mock_exam_1.md](mock_exam_1.md)

**Time allowed: 3 hours · Total: 100 marks**

## Instructions to candidates

1. **Section A** (20 MCQs) and **Section B** (6 numericals) are **compulsory**.
2. **Section C**: attempt **any 5 of the 6** questions. **Section D**: attempt **any 3 of the 4** questions. If you answer more than required, only the first 5 (C) / first 3 (D) in answer-book order are marked.
3. MCQs have exactly one correct option. No negative marking.
4. Non-programmable scientific **calculators are allowed**.
5. Show all working in numericals: formula, substitution, intermediate values, final answer **with units**. Keep at least 4 significant figures in intermediate steps.
6. **State any assumptions** you make. Where a question gives a convention, use it.
7. Diagrams (sketches, block diagrams, tables) are encouraged where they make an answer clearer.

**Conventions used throughout the paper (unless a question says otherwise)**

- Image coordinates: $x$ = column (rightwards), $y$ = row (**downwards**).
- "Filter with kernel $K$" means **cross-correlation** (no kernel flip).
- Boxes are $(x_1,y_1,x_2,y_2)$ in continuous pixel coordinates; width $=x_2-x_1$, height $=y_2-y_1$, area $=(x_2-x_1)(y_2-y_1)$ — **no +1**.
- $\log$ / $\ln$ = natural logarithm.

### Suggested time plan

| Section | Suggested time | What to aim for |
|---|---|---|
| A — 20 MCQs | 25 min | ≈ 75 s per question; mark and skip anything slow |
| B — 6 numericals | 55 min | ≈ 9 min each; write the formula first |
| C — 5 of 6 short answers | 35 min | ≈ 7 min each; bullets, one diagram or equation each |
| D — 3 of 4 long answers | 60 min | ≈ 20 min each; plan for 2 min, then use headings |
| Review | 5 min | re-check units, MCQ letters, unanswered parts |
| **Total** | **180 min** | |

### Marks breakdown

| Section | Type | Questions in paper | Questions to attempt | Marks each | Section total |
|---|---|---|---|---|---|
| A | Multiple choice | 20 | all 20 | 1 | 20 |
| B | Numerical | 6 | all 6 | 5 | 30 |
| C | Short answer | 6 | any 5 | 4 | 20 |
| D | Long answer / design | 4 | any 3 | 10 | 30 |
| | | **36** | **34** | | **100** |

### Syllabus coverage

Every content session is examined. Questions spanning two sessions are listed under both. "Marks available" counts all optional questions (114 marks are on the paper; you answer 100).

| Session | Topic | Questions | Marks available (≈) |
|---|---|---|---|
| S1 | Image processing & features | A1, A2, B1 | 7 |
| S2 | Feature matching & classical CV | A3, A4, C1 | 6 |
| S3 | CNNs under the hood | A5, C2, D1 | 8 |
| S4 | Training CNNs & transfer learning | A6, A7, B5, D1 | 14 |
| S5 | Embeddings & image similarity | A8, B4, D2 | 9.5 |
| S6 | Detection I: classical & two-stage | A9, A10, B2, C3 | 6.5 |
| S7 | Detection II: one-stage, NMS, AP | A11, B2 | 3.5 |
| S8 | Detection III: transformers (DETR) | A12, A13, B6 | 7 |
| S9 + S16 | Practicals & revision (compare approaches) | B5 (practical classifier), D1–D4 (design + comparison) | — (integrated) |
| S10 | Image segmentation | A14, C3, D3 | 10 |
| S11 | Camera geometry | A15, C5, D4 | 9 |
| S12 | 3D vision & motion | A16, A17, B3, C4, D4 | 17 |
| S13 | Vision foundation models | A18, B4, D2, D3 | 10.5 |
| S14 | Vision++ with foundation models | A19, C6 | 5 |
| S15 | Generative CV | A20 | 1 |
| | | | **114** |

By syllabus block: classical CV (S1–S2) 13 · deep-learning foundations (S3–S5) 31.5 · detection (S6–S8) 17 · segmentation (S10) 10 · geometry & 3D (S11–S12) 26 · foundation & generative models (S13–S15) 16.5. Generative CV is examined only lightly in this paper — revise it with [15_questions.md](15_questions.md).

---

## Section A — Multiple choice (20 × 1 mark, compulsory)

### A1 · MCQ · Easy
A visual-SLAM system on a mobile phone must extract and match features at 30 fps. Why is ORB usually preferred to SIFT here?

- **A.** ORB detects keypoints as Difference-of-Gaussians extrema and is therefore more scale-invariant than SIFT.
- **B.** ORB descriptors are 128-dimensional floating-point vectors, which are more distinctive than SIFT's.
- **C.** ORB uses FAST corners and a 256-bit binary descriptor compared with the Hamming distance (XOR + popcount), so extraction and matching are far cheaper.
- **D.** ORB is fully invariant to affine and perspective (viewpoint) changes, unlike SIFT.

### A2 · MCQ · Hard
A SIFT candidate keypoint has a $2\times2$ Hessian of the DoG with $\operatorname{tr}(H)=20$ and $\det(H)=25$. Lowe's edge threshold is $r=10$. The keypoint is:

- **A.** rejected — $\operatorname{tr}(H)^2/\det(H)=16$ exceeds $(r+1)^2/r=12.1$, so the principal curvatures are too unequal (edge-like).
- **B.** kept — $\det(H)>0$, so both curvatures have the same sign, which is all SIFT checks.
- **C.** kept — $16$ is below $(r+1)^2=121$.
- **D.** rejected — $\operatorname{tr}(H)<\det(H)$ indicates a low-contrast extremum.

### A3 · MCQ · Medium
A homography is estimated with RANSAC from matches with an outlier ratio of 50%. How many iterations are needed for a 99% probability of drawing at least one all-inlier minimal sample?

- **A.** 17
- **B.** 35
- **C.** 71
- **D.** 72

### A4 · MCQ · Medium
A photographer walks along a street and takes overlapping photos (about 1 m between shots), then stitches them with a single homography per image pair. The distant building façades align perfectly, but nearby parked cars appear doubled ("ghosting"). The best explanation is:

- **A.** RANSAC was run with too few iterations, so the homography is wrong.
- **B.** A single homography is exact only for a planar scene or a camera rotating about its optical centre; translating the camera causes depth-dependent parallax, so near objects cannot be aligned by the same $H$ as the far façades.
- **C.** Lens distortion was not corrected, and distortion is largest for near objects.
- **D.** Lowe's ratio test was too strict and removed all matches on the cars.

### A5 · MCQ · Medium
The ResNet stem applies to a $224\times224$ input a $7\times7$ convolution (stride 2, padding 3) followed by a $3\times3$ max-pool (stride 2, padding 1). What is the spatial size of the output?

- **A.** $55\times55$
- **B.** $112\times112$
- **C.** $56\times56$
- **D.** $57\times57$

### A6 · MCQ · Easy
A CNN reaches 99% training accuracy but 71% validation accuracy, and the validation loss starts rising after epoch 8. Which action is **least** likely to help?

- **A.** Increasing the network's depth and width and training for more epochs.
- **B.** Early stopping (keeping the checkpoint with the best validation loss, around epoch 8).
- **C.** Stronger, label-preserving data augmentation.
- **D.** Adding weight decay and dropout before the classifier.

### A7 · MCQ · Medium
You fine-tune an ImageNet-pretrained ResNet-50 on 3,000 images of a new 10-class task, replacing the final layer with a randomly initialised one. Which learning-rate strategy is recommended?

- **A.** Use a large learning rate (e.g. 0.1) for all layers so the network can escape the ImageNet solution.
- **B.** Freeze the new head and train only the pretrained backbone.
- **C.** Use one learning rate for everything; with Adam the learning rate does not matter.
- **D.** First train the new head with the backbone frozen, then unfreeze and fine-tune the backbone with a smaller learning rate (e.g. 10× lower) than the head.

### A8 · MCQ · Medium
A Siamese network is trained with the contrastive loss $\mathcal L=y\,d^2+(1-y)\max(0,m-d)^2$ ($y=1$ similar, $y=0$ dissimilar) and margin $m=1.0$. A **dissimilar** pair has embedding distance $d=0.4$. Its loss is:

- **A.** 0 — dissimilar pairs contribute only when $d>m$
- **B.** 0.36
- **C.** 0.16
- **D.** 0.60

### A9 · MCQ · Easy
In the HOG descriptor, why are the cell histograms normalised over overlapping **blocks**?

- **A.** To reduce the descriptor length so that the SVM trains faster.
- **B.** To make the descriptor invariant to in-plane rotation of the person.
- **C.** To make the descriptor robust to local changes in illumination and contrast (e.g. shadows), since gradient magnitudes scale with contrast.
- **D.** To make the descriptor scale-invariant, so no image pyramid is needed.

### A10 · MCQ · Hard
Focal loss was introduced for one-stage detectors because of the extreme foreground–background imbalance among dense candidate boxes. Why do two-stage detectors such as Faster R-CNN suffer much less from this imbalance?

- **A.** The RPN (followed by NMS) reduces tens of thousands of anchors to ~1–2k proposals, discarding most easy background, and the second stage trains on mini-batches sampled with a fixed foreground fraction (≈25%).
- **B.** Two-stage detectors use deeper backbones, so background features are linearly separable.
- **C.** Two-stage detectors use a softmax classifier, which is immune to class imbalance, while one-stage detectors use sigmoids.
- **D.** NMS removes negative examples before the classification loss is computed.

### A11 · MCQ · Medium
A YOLOv1-style detector uses an $S\times S$ grid with $S=7$, $B=2$ boxes per cell and is trained on a dataset with $C=80$ classes. The output tensor has shape:

- **A.** $7\times7\times170$
- **B.** $7\times7\times90$
- **C.** $7\times7\times84$
- **D.** $7\times7\times30$

### A12 · MCQ · Medium
Which statement about Hungarian (bipartite) matching in DETR is correct?

- **A.** It is run at both training and inference time to remove duplicate boxes.
- **B.** It assigns each ground-truth object to every query whose IoU with it exceeds 0.5.
- **C.** Its matching cost uses $-\log\hat p(c)$ exactly as in the loss, so matching cost and training loss are identical.
- **D.** It is run only during training, to decide which query is supervised by which ground-truth object; at inference the $N$ predictions are simply thresholded on confidence.

### A13 · MCQ · Hard
A DETR model trained with $N=100$ object queries is applied to an aerial image of a stadium crowd containing 150 clearly visible people. What happens?

- **A.** DETR adds extra queries at inference until every person is covered.
- **B.** Each query can output several boxes, so all 150 people can still be detected.
- **C.** At most 100 people can be detected, because each query predicts at most one object; $N$ must exceed the maximum expected number of objects per image.
- **D.** NMS splits merged boxes into separate people.

### A14 · MCQ · Easy
A predicted segmentation mask has IoU = 0.60 with the ground-truth mask. Its Dice coefficient is:

- **A.** 0.375
- **B.** 0.60
- **C.** 0.43
- **D.** 0.75

### A15 · MCQ · Hard
A camera's extrinsics map world to camera coordinates by $\mathbf X_c=R\mathbf X_w+\mathbf t$, with
$R=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$ and $\mathbf t=(2,\ 0,\ 4)^\top$ m. Where is the camera centre in world coordinates?

- **A.** $(0,\ 2,\ -4)$ m
- **B.** $(-2,\ 0,\ -4)$ m
- **C.** $(0,\ -2,\ 4)$ m
- **D.** $(0,\ -2,\ -4)$ m

### A16 · MCQ · Medium
Lucas–Kanade optical flow is computed in a window that lies on a long straight edge with no texture along the edge. What can be recovered?

- **A.** The full 2-D flow, because the $2\times2$ system always has a unique solution.
- **B.** Only the flow component perpendicular to the edge (along the image gradient), because the structure matrix is (nearly) rank 1 — the aperture problem.
- **C.** Only the flow component along the edge.
- **D.** No flow at all, because brightness constancy never holds on edges.

### A17 · MCQ · Medium
A **calibrated** monocular camera is used for visual SLAM in a corridor, with no other sensors and no known object sizes. The reconstructed map and trajectory are determined only up to:

- **A.** a projective transformation (15 DOF), even though the camera is calibrated.
- **B.** nothing — the map is metric because the intrinsics $K$ are known.
- **C.** a similarity transformation: a global rotation, a translation and an unknown scale (7 DOF).
- **D.** an unknown global rotation only (3 DOF).

### A18 · MCQ · Medium
Which statement about the Segment Anything Model (SAM) is correct?

- **A.** Its SA-1B training set contains 1.1B masks labelled with 1,000 categories, so SAM outputs class-labelled masks.
- **B.** Its image encoder is lightweight and re-run for every prompt, while its mask decoder is a heavy ViT-H.
- **C.** Its main released prompt type is free-form text, and it returns a caption for each mask.
- **D.** It outputs class-agnostic masks for point, box or mask prompts; for an ambiguous single point it returns three candidate masks with predicted IoU scores.

### A19 · MCQ · Medium
To "segment every forklift" in warehouse images without any labelled forklift data, Grounding DINO is chained with SAM. Which description of this pipeline is correct?

- **A.** Grounding DINO turns the text prompt "forklift" into boxes, and each box is passed to SAM as a box prompt; SAM returns the masks.
- **B.** SAM first produces class-labelled masks of all objects; Grounding DINO then renames the labels.
- **C.** Grounding DINO is the self-supervised DINO ViT; it supplies features from which SAM reads the class name.
- **D.** SAM converts the text "forklift" into point prompts, and Grounding DINO refines the mask boundaries.

### A20 · MCQ · Medium
In a DDPM, $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ with $\epsilon\sim\mathcal N(0,I)$. If $\bar\alpha_t=0.36$, then $x_t$ equals:

- **A.** $0.36\,x_0+0.64\,\epsilon$
- **B.** $0.6\,x_0+0.8\,\epsilon$
- **C.** $0.6\,x_0+0.4\,\epsilon$
- **D.** $0.8\,x_0+0.6\,\epsilon$

---

## Section B — Numericals (6 × 5 marks, compulsory)

### B1 · Numerical · Medium · [5 marks]
**Gradients and Harris corners.** A 3×3 grayscale neighbourhood centred on pixel $p$ is (top row = row $y-1$, left column = column $x-1$):

$$
N=\begin{bmatrix}30&40&80\\30&60&120\\40&90&140\end{bmatrix}
$$

Use the Sobel kernels as **cross-correlation**, with $y$ pointing down:
$K_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$, $K_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$.

**(a)** Compute $I_x$ and $I_y$ at $p$. **[1.5]**

**(b)** Compute the gradient magnitude $\sqrt{I_x^2+I_y^2}$ and the gradient direction $\theta=\operatorname{atan2}(I_y,I_x)$ in degrees. In which direction does the edge through $p$ run? **[1]**

**(c)** Three other windows have the structure tensors below. With $k=0.04$, compute $\det M$, $\operatorname{tr}M$ and $R=\det M-k(\operatorname{tr}M)^2$ for each. **[1.5]**

$$
M_1=\begin{bmatrix}900&300\\300&400\end{bmatrix},\qquad M_2=\begin{bmatrix}1200&340\\340&100\end{bmatrix},\qquad M_3=\begin{bmatrix}12&2\\2&9\end{bmatrix}
$$

**(d)** Compute the eigenvalues of each $M_i$ and classify each window as corner / edge / flat using the rule "corner if $R>10^4$, edge if $R<0$, otherwise flat". Check that the eigenvalues tell the same story, and comment on the fact that $M_1$ and $M_2$ have the same trace. **[1]**

### B2 · Numerical · Medium · [5 marks]
**IoU, NMS and anchor encoding.** Boxes are $(x_1,y_1,x_2,y_2)$, continuous coordinates, area $=(x_2-x_1)(y_2-y_1)$ — **no +1**. A detector outputs six boxes of the **same class**:

| Box | $(x_1,y_1,x_2,y_2)$ | Score |
|---|---|---|
| $B_1$ | (100, 100, 200, 200) | 0.95 |
| $B_2$ | (110, 110, 210, 210) | 0.90 |
| $B_3$ | (300, 300, 380, 380) | 0.85 |
| $B_4$ | (140, 100, 240, 200) | 0.80 |
| $B_5$ | (310, 290, 390, 370) | 0.75 |
| $B_6$ | (150, 110, 250, 210) | 0.60 |

**(a)** Compute IoU($B_1,B_2$) and IoU($B_1,B_4$). **[1]**

**(b)** Run greedy NMS with threshold 0.5 (a box is suppressed if its IoU with an already-kept box is **> 0.5**). Show every comparison you make and list the kept boxes. **[2]**

**(c)** Repeat with threshold 0.4. Which boxes are kept now? Comment on the result. **[0.5]**

**(d)** An anchor $A=(96,96,224,160)$ is matched to the ground-truth box $G=(104,92,240,164)$. Compute IoU($A,G$) and state whether the anchor is positive under the RPN rule "positive if IoU > 0.7". Then compute the regression targets
$t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$ (centre–size form; $(x,y,w,h)$ for $G$, $(x_a,y_a,w_a,h_a)$ for $A$). **[1.5]**

### B3 · Numerical · Medium · [5 marks]
**Stereo depth and optical flow.** A rectified stereo rig has focal length $f=700$ px and baseline $B=0.12$ m. Disparity $d=x_L-x_R$ in pixels; $Z=\frac{fB}{d}$; depth error $\Delta Z\approx\frac{Z^2}{fB}\Delta d$.

**(a)** Two scene points have disparities 42 px and 7 px. Compute their depths. **[1]**

**(b)** For the far point, estimate the depth uncertainty caused by a ±1 px disparity error, using the approximation, and compare it with the exact depths obtained for $d-1$ and $d+1$. What is the uncertainty for the near point? **[1]**

**(c)** The rig is redesigned so that a point at $Z=12$ m is measured with $\Delta Z\le0.20$ m, assuming sub-pixel matching with $\Delta d=0.25$ px and the same $f$. What minimum baseline is needed? What disparity will the point at 12 m then have? **[1]**

**(d)** In a $5\times5$ window of a video, the Lucas–Kanade sums are $\sum I_x^2=40$, $\sum I_xI_y=10$, $\sum I_y^2=20$, $\sum I_xI_t=-60$, $\sum I_yI_t=-50$. Write the LK normal equations and solve for the flow $(u,v)$ in pixels/frame. **[1.5]**

**(e)** In another window the sums are $\sum I_x^2=40$, $\sum I_xI_y=20$, $\sum I_y^2=10$. Explain what goes wrong and what can still be recovered. **[0.5]**

### B4 · Numerical · Medium · [5 marks]
**CLIP zero-shot classification and InfoNCE.** Embeddings are L2-normalised; natural logarithms.

**(a)** An image is compared with the text prompts "a photo of a {dog, fox, cat, car}". The cosine similarities are 0.31, 0.29, 0.28 and 0.12 respectively. CLIP multiplies cosine similarities by a logit scale of 100 and applies a softmax. Compute the four class probabilities and the predicted class. **[2]**

**(b)** Recompute the probabilities with a logit scale of 1 (no temperature). What does this show about why CLIP learns a temperature? **[0.5]**

**(c)** During training, a batch of 4 image–caption pairs gives the cosine-similarity matrix (rows = images, columns = texts; the matching pairs are on the diagonal):

$$
S=\begin{bmatrix}0.70&0.62&0.35&0.10\\0.58&0.81&0.22&0.15\\0.30&0.25&0.77&0.40\\0.12&0.18&0.36&0.66\end{bmatrix}
$$

With temperature $\tau=0.07$, compute the image→text InfoNCE loss for **row 1**:
$\mathcal L_1=-\log\frac{\exp(S_{11}/\tau)}{\sum_{j=1}^{4}\exp(S_{1j}/\tau)}$. **[2]**

**(d)** Which negative caption dominates the gradient of $\mathcal L_1$, and why? Which entries of $S$ enter the text→image loss for caption 1? **[0.5]**

### B5 · Numerical · Medium · [5 marks]
**Transfer learning and multi-class metrics.** An ImageNet-pretrained ResNet-18 is adapted to a 3-class surface-defect dataset (**OK / Scratch / Dent**): the 1000-way FC layer is replaced by global-average-pool → FC $512\to3$ (with bias). The adapted network has **11,178,051** parameters in total. Convolutions have **no bias**; every BatchNorm layer has 2 learnable parameters per channel ($\gamma,\beta$; running statistics are not parameters). The last stage, `layer4`, contains:

| Block | Layer | Shape |
|---|---|---|
| 1 | conv1 + BN | $3\times3$, 256 → 512, stride 2 |
| 1 | conv2 + BN | $3\times3$, 512 → 512 |
| 1 | downsample conv + BN | $1\times1$, 256 → 512, stride 2 |
| 2 | conv1 + BN | $3\times3$, 512 → 512 |
| 2 | conv2 + BN | $3\times3$, 512 → 512 |

**(a)** How many parameters are trainable in **feature extraction** (backbone frozen, only the new head trained)? **[1]**

**(b)** How many are trainable when `layer4` is **also unfrozen**? What fraction of the whole network is this? **[1.5]**

**(c)** On the test set the model gives the confusion matrix below (rows = true class, columns = predicted class). Compute per-class precision, recall and F1, the overall accuracy, and the macro-averaged precision, recall and F1 (macro-F1 = mean of the per-class F1 scores). Comment on accuracy vs macro-F1. **[2.5]**

| True \ Predicted | OK | Scratch | Dent |
|---|---|---|---|
| **OK** | 80 | 6 | 4 |
| **Scratch** | 5 | 30 | 5 |
| **Dent** | 3 | 4 | 13 |

### B6 · Numerical · Hard · [5 marks]
**DETR Hungarian matching.** An image contains two ground-truth objects: $g_1$ = **cat** and $g_2$ = **dog** (they sit side by side, so their boxes overlap). Three DETR queries are shown (assume the other queries are already confidently "no object"). Classes are cat, dog and $\varnothing$ ("no object"). Boxes are normalised $(c_x,c_y,w,h)$; the table gives the L1 distance $\Vert b_j-\hat b_i\Vert_1$ between each query's box and each GT box.

| Query | $\hat p(\text{cat})$ | $\hat p(\text{dog})$ | $\hat p(\varnothing)$ | L1 to $g_1$ (cat) | L1 to $g_2$ (dog) |
|---|---|---|---|---|---|
| $q_1$ | 0.50 | 0.10 | 0.40 | 0.05 | 0.16 |
| $q_2$ | 0.65 | 0.30 | 0.05 | 0.07 | 0.10 |
| $q_3$ | 0.05 | 0.55 | 0.40 | 0.41 | 0.26 |

Matching cost (GIoU term omitted for this question, i.e. $\lambda_{giou}=0$):
$\mathcal C(q_i,g_j)=-\hat p_i(c_j)+\lambda_{L1}\Vert b_j-\hat b_i\Vert_1$ with $\lambda_{L1}=5$.

**(a)** Build the $3\times2$ cost matrix. **[2]**

**(b)** Make it square for the Hungarian algorithm. What cost do you put in the added column, and why? **[0.5]**

**(c)** Find the optimal assignment and its total cost (show that it is optimal). Which query is trained to predict "no object"? **[1.5]**

**(d)** What does a greedy "take the cheapest remaining pair" strategy give, and what is its total cost? Give the classification-loss term each of the three queries receives under the optimal assignment (cross-entropy, weight 0.1 for $\varnothing$, weight 1 otherwise). **[1]**

---

## Section C — Short answers (attempt any 5 of 6; 4 marks each)

### C1 · Short · Medium · [4 marks]
A student matches every SIFT descriptor of image A to its nearest neighbour in image B and gets many wrong matches.
**(a)** Explain **Lowe's ratio test**: the rule, a typical threshold, and why it compares against the *second*-nearest neighbour rather than using an absolute distance threshold. **[2]**
**(b)** Explain **cross-check (mutual nearest-neighbour) matching**. State which kinds of wrong match each method removes, and which kind of wrong match survives both (and how it is then removed). **[2]**

### C2 · Short · Medium · [4 marks]
**(a)** What is the **degradation problem** observed when plain CNNs are made deeper, and why is it *not* overfitting? **[1.5]**
**(b)** Explain, from an **optimisation** view and from a **gradient-flow** view, why identity shortcuts $\mathbf y=F(\mathbf x)+\mathbf x$ solve it. **[2.5]**

### C3 · Short · Medium · [4 marks]
Mask R-CNN replaced **RoI Pool** with **RoIAlign**.
**(a)** Describe the two quantisation steps in RoI Pool and the misalignment they cause; estimate its size for a stride-16 feature map. **[2]**
**(b)** How does RoIAlign avoid it? **[1]**
**(c)** Why does this misalignment matter much more for mask prediction than for box classification? **[1]**

### C4 · Short · Medium · [4 marks]
Compare the **essential matrix** $E$ and the **fundamental matrix** $F$: definition (in terms of $R$, $\mathbf t$, $K$), the coordinates each one operates on, their degrees of freedom and rank (with a one-line justification each), and the minimal number of correspondences / standard solvers for each. When must you use $F$ rather than $E$?

### C5 · Short · Medium · [4 marks]
Explain **Zhang's planar calibration method**: what is observed in each image, what is computed per view, which two constraints each view gives on the intrinsics and where they come from, how many views are needed, and how the closed-form result is refined.

### C6 · Short · Medium · [4 marks]
Compare **Grounding DINO** and **OWL-ViT** for open-vocabulary object detection: how text and image interact in each architecture, what prompts they take, and one strength and one weakness of each. Explain what "DINO" stands for in *Grounding DINO* and why this is a common trap.

---

## Section D — Long answers / design (attempt any 3 of 4; 10 marks each)

### D1 · Long · Hard · [10 marks]
**Chest X-ray classifier with little data.** A hospital network gives you **1,000 labelled frontal chest X-rays** from two hospitals (about 800 patients; some patients have several images) in **4 classes**: Normal 550, Pneumonia 250, Tuberculosis 150, Pneumothorax 50. They want a classifier that assists radiologists.

Design the solution. Cover: (i) the model choice; (ii) the transfer-learning strategy (what to freeze, learning rates, input handling); (iii) data augmentation — which augmentations are safe and which are **unsafe** for X-rays, and why; (iv) regularisation; (v) handling class imbalance; (vi) evaluation protocol and metrics; (vii) failure analysis before deployment.

### D2 · Long · Hard · [10 marks]
**Visual search for 10 million product images.** An online marketplace with **10 million** product images wants (1) "search by photo": a user uploads a phone photo and gets the 20 most similar products, and (2) "search by text": a query such as "red leather ankle boots" returns matching product images. The latency budget is 150 ms per query; thousands of new products are added daily.

Design the system. Cover: (i) the embedding model — compare a supervised ImageNet classifier, a self-supervised model (e.g. DINOv2) and CLIP, and choose; (ii) how you would fine-tune it and with which loss; (iii) indexing and approximate nearest-neighbour search, with a memory estimate; (iv) text-to-image search; (v) evaluation metrics (offline and online); (vi) keeping the index up to date and likely failure modes.

### D3 · Long · Medium · [10 marks]
**Two segmentation problems.**
*Client 1* — a hospital wants to segment **brain tumours in MRI** (about 250 annotated scans; one tumour region per scan; boundaries matter for treatment planning).
*Client 2* — a retailer wants to segment **every individual product instance on supermarket shelves** from store photos (crowded, touching, many small items; new products every month).

(a) Compare **FCN, U-Net, Mask R-CNN and SAM**: task type, architecture idea, output, data needs, strengths and weaknesses. **[4]**
(b) Choose an approach for each client and justify it, including how (if at all) you would use SAM. **[4]**
(c) Choose evaluation metrics for each client — **IoU vs Dice vs PQ** (and any others you need) — and justify. **[2]**

### D4 · Long · Hard · [10 marks]
**3D map of a building with a handheld camera.** A facilities team wants a 3D map of a 4-floor office building, captured by walking through it with a handheld camera. The building has long corridors with plain walls, glass partitions and repeated identical doors. They need a map accurate enough to measure room dimensions to within a few centimetres.

Design the pipeline. Cover: (i) camera calibration; (ii) **SfM vs visual SLAM** — how each works, including **bundle adjustment**, **loop closure** and **scale ambiguity**, and which you would use; (iii) when and why to add **stereo** or an **IMU**; (iv) how you would **evaluate** the trajectory and the map.

---

*End of paper.*
