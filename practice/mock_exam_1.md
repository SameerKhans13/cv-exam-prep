# Mock Exam 1 — Computer Vision · Full Paper

> **Solutions:** [mock_exam_1_solutions.md](mock_exam_1_solutions.md) · **Formula sheet:** [../00_formula_sheet.md](../00_formula_sheet.md) · **Comparison tables:** [../00_comparison_tables.md](../00_comparison_tables.md)
> Sit this under exam conditions: one sitting, timer on, calculator only, no notes. Mark it afterwards with the solutions file.

---

## Instructions to candidates

1. **Time allowed: 3 hours. Maximum marks: 100.**
2. **Section A** (20 multiple-choice questions, 1 mark each) is **compulsory**. Choose exactly one option per question. There is no negative marking.
3. **Section B** (6 numerical questions, 5 marks each) is **compulsory**. Show all working. Most marks go to method, so a correct final number with no working gets little credit.
4. **Section C**: attempt **any 5 of 6** questions (4 marks each). If you answer all six, only the first five in your script are marked.
5. **Section D**: attempt **any 3 of 4** questions (10 marks each).
6. Non-programmable scientific **calculators are allowed**. Give numerical answers to 3–4 significant figures unless told otherwise.
7. If a question seems ambiguous, **state your assumptions** clearly and answer under them. Reasonable stated assumptions are credited.
8. Diagrams (block diagrams, sketches, tables) are encouraged wherever they make an answer clearer.

### Conventions used throughout this paper (unless a question says otherwise)

- Image coordinates: $x$ = column (to the right), $y$ = row (**downwards**); pixel $(u,v)$ in the same order.
- Boxes are $(x_1,y_1,x_2,y_2)$ in continuous coordinates; area $=(x_2-x_1)(y_2-y_1)$, **no +1**.
- Convolution output size: $W_{out}=\left\lfloor \frac{W-K_{eff}+2P}{S}\right\rfloor+1$, with $K_{eff}=D(K-1)+1$ for dilation $D$.
- Receptive field: $r_0=1,\ j_0=1$, $r_l=r_{l-1}+(K_{eff,l}-1)\,j_{l-1}$, $j_l=j_{l-1}S_l$; pooling counts as a layer.
- RANSAC iteration counts are always **rounded up**.
- Homogeneous points are normalised by dividing by the last coordinate.
- AP is computed by **all-point interpolation** (VOC2010+) unless the question also asks for the 11-point (VOC2007) value.
- $\log$ means the natural log (any base gives the same RANSAC ratio).

### Suggested time plan

| Block | Content | Time | Marks | ≈ min per mark |
|---|---|---|---|---|
| Reading | Skim the whole paper, choose your C and D questions | (inside A) | — | — |
| Section A | 20 MCQs | 25 min | 20 | 1.25 |
| Section B | 6 numericals | 55 min | 30 | 1.8 |
| Section C | 5 of 6 short answers | 35 min | 20 | 1.75 |
| Section D | 3 of 4 long/design answers | 60 min | 30 | 2.0 |
| Review | Check arithmetic, units and unanswered MCQs | 5 min | — | — |
| **Total** | | **180 min** | **100** | |

### Marks breakdown

| Section | Type | Questions set | Attempt | Marks each | Marks counted |
|---|---|---|---|---|---|
| A | Multiple choice | 20 (A1–A20) | all 20 | 1 | **20** |
| B | Numerical | 6 (B1–B6) | all 6 | 5 | **30** |
| C | Short answer | 6 (C1–C6) | any 5 | 4 | **20** |
| D | Long answer / design | 4 (D1–D4) | any 3 | 10 | **30** |
| **Total** | | **36** | **34** | | **100** |

### Syllabus coverage

The paper sets 114 marks of questions (you answer 100). The weight of each session roughly follows the size of the syllabus: the session blocks on detection (S6–S8), geometry (S2, S11, S12), foundation models (S13–S14) and generative CV (S15) carry the most marks. D-question marks are split between the sessions they draw on.

| Session | Topic | Questions | Marks on paper (≈) |
|---|---|---|---|
| S1 | Image processing & features | A1, C1, D3 (features) | 8 |
| S2 | Feature matching & classical CV | A2, B2, D3 (matching, RANSAC, stitching) | 10 |
| S3 | CNNs under the hood | A3, A4, B1 | 7 |
| S4 | Training CNNs & transfer learning | A5, C2 | 5 |
| S5 | Embeddings & image similarity | A6, C3 | 5 |
| S6 | Detection I: two-stage | A7, D1 | 4 |
| S7 | Detection II: one-stage, IoU, NMS, AP | A8, A9, B3, D1 | 11 |
| S8 | Detection III: transformers | A10, C4, D1 | 8 |
| S9 + S16 | Revision / comparing approaches | Cross-cutting comparisons in D1, D2, D4 | — |
| S10 | Image segmentation | A11, A12, B4 | 7 |
| S11 | Camera geometry | A13, A14, B5, D3 (homography, distortion) | 10 |
| S12 | 3D vision & motion | A15, C5, D2 (depth, tracking) | 8 |
| S13 | Vision foundation models | A16, A17, C6, D2 (SAM, Depth Anything) | 10 |
| S14 | Vision++ with foundation models | A18, D2 (Grounding DINO pipeline) | 4 |
| S15 | Generative CV | A19, A20, B6, D4 | 17 |
| | | **Total** | **114** |

---

## Section A — Multiple choice (20 × 1 = 20 marks, compulsory)

### A1 · MCQ · Medium · [1 mark]
A constant 40 is added to every pixel of a grayscale image (no clipping occurs). At a fixed pixel, what happens to the Harris response $R=\det M-k(\operatorname{tr}M)^2$, computed with the same window and the same $k$?

- **A.** It increases, because brighter pixels give larger entries in $M$.
- **B.** It is unchanged, because $M$ is built only from image derivatives, and a constant offset does not change them.
- **C.** It is multiplied by $40^2$, because every entry of $M$ is a product of two intensity values.
- **D.** It becomes negative, because brightening turns corners into edges.

### A2 · MCQ · Easy · [1 mark]
SIFT descriptors are matched with Lowe's ratio test, threshold $\tau=0.8$. For one query keypoint, the nearest and second-nearest descriptor distances are $d_1=0.30$ and $d_2=0.36$. The match is:

- **A.** accepted, because $d_1$ is small in absolute terms.
- **B.** accepted, because $d_1<d_2$.
- **C.** rejected, because $d_1/d_2\approx0.83>0.8$, so the match is ambiguous.
- **D.** rejected, because $d_2-d_1<0.1$.

### A3 · MCQ · Hard · [1 mark]
A standard $3\times3$ convolution with 64 input channels and 128 output channels is replaced by a depthwise-separable convolution ($3\times3$ depthwise followed by $1\times1$ pointwise), with the same input and output channels. Ignoring biases, how many parameters does the separable version have?

- **A.** 8,768
- **B.** 73,728
- **C.** 8,192
- **D.** 1,152

### A4 · MCQ · Medium · [1 mark]
A ResNet-50 backbone ends in a $7\times7\times2048$ feature map. The classifier is **global average pooling** followed by one fully connected layer (with bias) to 1000 classes. How many parameters does this FC layer have?

- **A.** 100,353,000
- **B.** 2,048,000
- **C.** 4,097,000
- **D.** 2,049,000

### A5 · MCQ · Medium · [1 mark]
An image classifier trained on 5,000 images reaches 99% training accuracy but only 71% validation accuracy. Which change is **least** likely to reduce this gap?

- **A.** Adding random-resized-crop, horizontal-flip and colour-jitter augmentation.
- **B.** Adding weight decay and dropout before the classifier.
- **C.** Doubling the width of every layer and training for twice as many epochs.
- **D.** Starting from ImageNet-pretrained weights and early-stopping on validation loss.

### A6 · MCQ · Medium · [1 mark]
A Siamese network is trained with the contrastive loss $\mathcal L=y\,d^2+(1-y)\max(0,m-d)^2$, where $y=1$ marks a similar pair and $y=0$ a dissimilar pair, with margin $m=1.0$. A **dissimilar** pair has embedding distance $d=0.6$. What is the loss for this pair?

- **A.** 0.36
- **B.** 0, because the pair is labelled dissimilar.
- **C.** 0.40
- **D.** 0.16

### A7 · MCQ · Medium · [1 mark]
Faster R-CNN uses centre-size box regression targets $t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$. An anchor has centre $(100,100)$, $w_a=50$, $h_a=100$. Its matched ground-truth box has centre $(110,90)$, $w=100$, $h=100$. What is the target $(t_x,t_y,t_w,t_h)$?

- **A.** $(0.2,\ -0.1,\ 0.693,\ 0)$
- **B.** $(10,\ -10,\ 50,\ 0)$
- **C.** $(0.1,\ -0.1,\ 2,\ 1)$
- **D.** $(0.2,\ -0.1,\ 2,\ 1)$

### A8 · MCQ · Easy · [1 mark]
Boxes are $(x_1,y_1,x_2,y_2)$ in continuous coordinates, area $=(x_2-x_1)(y_2-y_1)$, no +1. What is the IoU of $A=(10,20,50,60)$ and $B=(30,40,90,80)$?

- **A.** 0.100
- **B.** 0.111
- **C.** 0.167
- **D.** 0.250

### A9 · MCQ · Medium · [1 mark]
Focal loss is $FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t$ with $\gamma=2$. Ignoring $\alpha_t$, by what factor is the cross-entropy loss of an easy background anchor with $p_t=0.9$ multiplied?

- **A.** 0.9
- **B.** 0.1
- **C.** 0.01
- **D.** 0.81

### A10 · MCQ · Medium · [1 mark]
The original DETR feeds only the stride-32 (C5) feature map to its transformer encoder. The main reason is that encoder self-attention over the $H\times W$ feature tokens costs:

- **A.** $O(HW)$. The cost is linear, but C5 has the richest semantics.
- **B.** $O\big((HW)^2\big)$. The cost is quadratic in the number of tokens, so higher-resolution maps are too expensive.
- **C.** $O(N^2)$ in the number of object queries $N=100$.
- **D.** $O(HW\cdot N)$, the same as decoder cross-attention.

### A11 · MCQ · Easy · [1 mark]
A street-scene model must give **every pixel** a class label (road, sky, car, person, …) and a **separate ID for each individual** car and person, while road and sky get no instance IDs. This task is:

- **A.** semantic segmentation
- **B.** instance segmentation
- **C.** object detection with bounding boxes
- **D.** panoptic segmentation

### A12 · MCQ · Hard · [1 mark]
Which statement about the **mask head of Mask R-CNN** is correct?

- **A.** For each RoI it predicts $K$ binary $28\times28$ masks (one per class) with a per-pixel sigmoid, and only the mask of the ground-truth class contributes to the mask loss.
- **B.** For each RoI it predicts a single $28\times28$ mask with a per-pixel softmax over the $K$ classes.
- **C.** It predicts a full-resolution semantic map for the whole image, which is then cropped by each detected box.
- **D.** It uses RoI pooling with quantised bins, which is fine because mask boundaries are coarse anyway.

### A13 · MCQ · Easy · [1 mark]
How many degrees of freedom does the full camera matrix $P=K[R\mid\mathbf t]$ ($3\times4$, defined up to scale, general $K$ with skew) have?

- **A.** 12, because every entry of the $3\times4$ matrix is free.
- **B.** 9: 3 rotation, 3 translation and 3 intrinsic.
- **C.** 11: 5 intrinsic and 6 extrinsic.
- **D.** 8, the same as a homography.

### A14 · MCQ · Hard · [1 mark]
In Zhang's calibration, each image of a planar checkerboard gives a homography $H\sim K[\mathbf r_1\ \mathbf r_2\ \mathbf t]$. How many constraints on the intrinsics does **each view** give, and how many views are needed in general (with skew) to solve for $K$?

- **A.** 8 constraints per view (one per homography DOF), so 1 view is enough.
- **B.** 4 constraints per view, so 2 views are enough.
- **C.** 1 constraint per view, so at least 5 views are needed.
- **D.** 2 constraints per view (from $\mathbf r_1\perp\mathbf r_2$ and $\Vert\mathbf r_1\Vert=\Vert\mathbf r_2\Vert$), so at least 3 views are needed.

### A15 · MCQ · Medium · [1 mark]
A rectified stereo rig has $f=700$ px and baseline $B=0.12$ m. A point has disparity $d=21$ px. What are its depth $Z$ and the approximate depth uncertainty for a ±1 px disparity error?

- **A.** $Z=4.0$ m, $\Delta Z\approx0.05$ m
- **B.** $Z=4.0$ m, $\Delta Z\approx0.19$ m
- **C.** $Z=0.25$ m, $\Delta Z\approx0.01$ m
- **D.** $Z=1.76$ m, $\Delta Z\approx0.19$ m

### A16 · MCQ · Easy · [1 mark]
How does CLIP perform **zero-shot classification** over a new list of class names?

- **A.** It embeds prompts such as "a photo of a {class}" with the text encoder and the image with the image encoder, then predicts the class whose text embedding has the highest cosine similarity with the image embedding.
- **B.** It fine-tunes a linear classifier on a few labelled images of each new class.
- **C.** It generates a caption with its text decoder and string-matches the caption against the class names.
- **D.** It clusters the image embeddings with k-means and assigns the clusters to class names in alphabetical order.

### A17 · MCQ · Hard · [1 mark]
Which foundation model is trained with a **purely self-supervised** objective (no class labels, no masks, no captions), by self-distillation between a student network and a momentum (EMA) teacher?

- **A.** CLIP
- **B.** Segment Anything (SAM)
- **C.** DINO / DINOv2
- **D.** OWL-ViT

### A18 · MCQ · Medium · [1 mark]
In a "Grounded-SAM" pipeline (Grounding DINO followed by SAM), the user types the prompt `dog . frisbee .`. What does SAM receive as its prompt?

- **A.** The raw text `dog . frisbee .`, passed through SAM's text encoder.
- **B.** CLIP image embeddings of each object crop.
- **C.** The class-label indices predicted by Grounding DINO.
- **D.** The bounding boxes predicted by Grounding DINO, used as box prompts.

### A19 · MCQ · Medium · [1 mark]
In a VAE the latent is sampled as $z=\mu+\sigma\odot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$. What is the **main** reason for this reparameterisation?

- **A.** It makes the KL term equal to zero.
- **B.** It moves the randomness into $\epsilon$, so gradients can flow through $\mu$ and $\sigma$ back to the encoder.
- **C.** It makes the decoder deterministic, so samples are sharper.
- **D.** It allows the latent dimension to be smaller than the input dimension.

### A20 · MCQ · Hard · [1 mark]
At the **global optimum** of the original GAN minimax game $V(D,G)=\mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z\sim p_z}[\log(1-D(G(z)))]$ (infinite capacity), what does the discriminator output, and what is the value of $V$?

- **A.** $D(x)=\tfrac12$ everywhere, and $V=-\log4$.
- **B.** $D(x)=1$ on real data and $0$ on generated data, and $V=0$.
- **C.** $D(x)=\tfrac12$ everywhere, and $V=0$.
- **D.** $D(x)=\tfrac12$ everywhere, and $V=\log2$.

---

## Section B — Numericals (6 × 5 = 30 marks, compulsory)

### B1 · Numerical · Medium · [5 marks] — CNN layer arithmetic (S3)
A $96\times96\times3$ RGB image passes through the stack below. All convolutions have a bias. Inputs and kernels are square.

| Layer | Type | Kernel $K$ | Filters | Stride $S$ | Padding $P$ | Dilation $D$ |
|---|---|---|---|---|---|---|
| L1 | conv | $5\times5$ | 32 | 2 | 2 | 1 |
| L2 | conv | $3\times3$ | 64 | 1 | 1 | 1 |
| L3 | max-pool | $2\times2$ | — | 2 | 0 | 1 |
| L4 | conv | $3\times3$ | 128 | 1 | 2 | 2 |

Use $W_{out}=\left\lfloor\frac{W-K_{eff}+2P}{S}\right\rfloor+1$ with $K_{eff}=D(K-1)+1$, and the receptive-field recurrence from the conventions box.

**(a)** Give the output shape ($H\times W\times C$) after each layer. **[2]**
**(b)** Give the number of learnable parameters in each layer and in total. **[1.5]**
**(c)** Give the theoretical receptive field (in input pixels) and the jump (effective stride) of one unit in the output of L4. **[1.5]**

### B2 · Numerical · Medium · [5 marks] — RANSAC and homography mapping (S2)
Two photos for a panorama give 500 putative SIFT matches after the ratio test. About **40%** of them are outliers ($e=0.4$). You want RANSAC to find at least one all-inlier sample with probability $p=0.99$.

**(a)** How many RANSAC iterations are needed to estimate a **homography** ($s=4$)? **[1.5]**
**(b)** How many iterations are needed to estimate a **fundamental matrix** with the 8-point algorithm ($s=8$), and with the 7-point algorithm ($s=7$)? Comment in one sentence on what this shows. **[1.5]**
**(c)** The estimated homography (image 1 → image 2) is

$$
H=\begin{bmatrix}0.9 & -0.05 & 40\\ 0.08 & 1.05 & 15\\ 0.0004 & -0.0003 & 1\end{bmatrix}.
$$

Map the image-1 pixel $\mathbf x=(300,200)$ into image 2. Give the homogeneous result and the final pixel coordinates, and explain why the division step is needed. **[2]**

### B3 · Numerical · Hard · [5 marks] — AP and mAP (S7)
A detector is evaluated on the class **car** in a validation set that contains **7 ground-truth cars** $G_1,\dots,G_7$. It outputs the 10 detections below, already sorted by confidence. For each detection, "best GT" is the ground-truth box with the highest IoU with that detection.

| Rank | Score | Best GT | IoU with best GT |
|---|---|---|---|
| 1 | 0.96 | $G_1$ | 0.82 |
| 2 | 0.93 | $G_2$ | 0.71 |
| 3 | 0.90 | $G_1$ | 0.64 |
| 4 | 0.85 | $G_3$ | 0.58 |
| 5 | 0.80 | $G_4$ | 0.43 |
| 6 | 0.74 | $G_5$ | 0.77 |
| 7 | 0.66 | none | 0.00 |
| 8 | 0.55 | $G_4$ | 0.62 |
| 9 | 0.41 | $G_2$ | 0.55 |
| 10 | 0.30 | $G_6$ | 0.51 |

**Matching rule (VOC-style, IoU threshold 0.5):** process detections in descending score. A detection is a **TP** if IoU with its best GT is $\ge0.5$ **and** that GT has not already been matched. Otherwise it is an **FP** (this includes duplicates and IoU $<0.5$). Each GT can be matched at most once.

**(a)** Label each detection TP or FP. **[1]**
**(b)** Tabulate cumulative precision and recall after each rank. **[1]**
**(c)** Compute AP with **all-point interpolation** (VOC2010+): the area under the precision envelope $p_{interp}(r)=\max_{r'\ge r}p(r')$. **[1]**
**(d)** Compute AP with **11-point interpolation** (VOC2007): the mean of $p_{interp}(r)$ for $r\in\{0,0.1,\dots,1.0\}$. Explain why it differs from (c). **[1]**
**(e)** The same detector has all-point AP@0.5 of 0.48 for **pedestrian** and 0.71 for **cyclist**. Compute the mAP@0.5 over the three classes using your all-point car AP. **[1]**

### B4 · Numerical · Medium · [5 marks] — Dice, IoU and segmentation metrics (S10)
**(a)** In a binary segmentation, the predicted mask has 5,000 foreground pixels, the ground-truth mask has 4,000, and 3,200 pixels are in both. Compute IoU and Dice, and check the identity $\text{Dice}=\frac{2\,\text{IoU}}{1+\text{IoU}}$. **[1.5]**
**(b)** A paper reports a Dice score of 0.80 on a different dataset. What IoU does this correspond to? **[0.5]**
**(c)** A 3-class semantic segmentation model is evaluated on 10,000 pixels. Its confusion matrix (rows = ground truth, columns = prediction) is:

| GT ↓ · Pred → | background | road | pedestrian |
|---|---|---|---|
| **background** | 7000 | 300 | 200 |
| **road** | 400 | 1800 | 50 |
| **pedestrian** | 150 | 50 | 50 |

Compute the pixel accuracy, the IoU of each class, and the mIoU. **[2]**
**(d)** A trivial model predicts "background" for every pixel. Compute its pixel accuracy and mIoU. Then explain, with reference to (c), why mIoU rather than pixel accuracy is reported for imbalanced data. **[1]**

### B5 · Numerical · Medium · [5 marks] — Projection with lens distortion (S11)
A camera producing $1280\times720$ images has

$$
K=\begin{bmatrix}800 & 0 & 640\\ 0 & 800 & 360\\ 0 & 0 & 1\end{bmatrix},\qquad
R=\begin{bmatrix}0.8 & 0 & 0.6\\ 0 & 1 & 0\\ -0.6 & 0 & 0.8\end{bmatrix},\qquad
\mathbf t=\begin{bmatrix}-0.7\\ 0.2\\ 0.4\end{bmatrix}\ \text{m},
$$

with world-to-camera transform $X_c=RX_w+\mathbf t$. Radial distortion is $k_1=-0.30$, $k_2=0.10$ ($k_3=p_1=p_2=0$), applied in normalised coordinates: $x_d=x(1+k_1r^2+k_2r^4)$ with $r^2=x^2+y^2$, and likewise for $y$. The world point is $X_w=(-1.0,\ 0.8,\ 5.0)^\top$ m.

**(a)** Show that $R$ is a valid rotation matrix and find the camera centre $C$ in world coordinates. **[1]**
**(b)** Compute the camera-frame point $X_c$, the normalised coordinates $(x,y)$, and the ideal (undistorted) pixel $(u,v)$. **[2]**
**(c)** Apply the radial distortion. Give the distorted normalised coordinates and the distorted pixel $(u_d,v_d)$. State the size and direction of the shift, and whether this is barrel or pincushion distortion. **[2]**

### B6 · Numerical · Medium · [5 marks] — DDPM forward process and classifier-free guidance (S15)
A toy DDPM has $T=4$ steps with noise schedule $\beta=(0.05,\ 0.10,\ 0.20,\ 0.30)$. Use $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^t\alpha_s$ and $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$.

**(a)** Compute $\alpha_t$ and $\bar\alpha_t$ for $t=1,\dots,4$. **[1]**
**(b)** For $x_0=(0.8,\ -0.4,\ 0.2)$ and $\epsilon=(0.5,\ 1.0,\ -1.5)$, compute $x_4$ directly (in one shot). For unit-variance data, what fraction of the variance of $x_4$ comes from the signal $x_0$? **[2]**
**(c)** During sampling, the network outputs $\epsilon_\theta(x_t,\varnothing)=(0.10,\ -0.20,\ 0.30)$ (unconditional) and $\epsilon_\theta(x_t,c)=(0.25,\ -0.05,\ 0.10)$ (conditioned on the prompt). Compute the classifier-free-guidance estimate $\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+w\,\big(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing)\big)$ for $w=7.5$. **[1]**
**(d)** What do $w=0$ and $w=1$ give? What is the trade-off of a large $w$? **[1]**

---

## Section C — Short answer (attempt any 5 of 6; 4 marks each = 20 marks)

### C1 · Short · Medium · [4 marks] — SIFT invariance (S1)
Explain how the SIFT descriptor achieves invariance to **(i) scale** and **(ii) in-plane rotation**, and how it is made robust to **illumination** changes. Name one transformation that SIFT is **not** invariant to.

### C2 · Short · Medium · [4 marks] — Feature extraction vs fine-tuning (S4)
You have an ImageNet-pretrained ResNet-50. For each scenario, decide between **feature extraction** (frozen backbone with a new head) and **fine-tuning** (state how much of the network), and justify your choice:
(i) 600 labelled photos of 6 dog breeds for a pet app;
(ii) 60,000 labelled grayscale chest X-rays with 14 possible findings per image.
Also give **two practical precautions** to take when fine-tuning.

### C3 · Short · Medium · [4 marks] — Siamese networks and triplet mining (S5)
Explain how a **Siamese network** trained with the **triplet loss** is used for face verification. Define **semi-hard negative mining** and explain why it is preferred over random negatives and over the hardest negatives. Illustrate with $d(a,p)=0.6$ and margin $m=0.2$.

### C4 · Short · Medium · [4 marks] — DETR without NMS (S8)
Explain why DETR needs **no non-maximum suppression**. Then describe DETR's **two main drawbacks** and the cause of each.

### C5 · Short · Medium · [4 marks] — Aperture problem and Lucas–Kanade (S12)
What is the **aperture problem** in optical flow? Use the optical-flow constraint equation to show why it arises. Explain how **Lucas–Kanade** handles it, and where LK still fails.

### C6 · Short · Medium · [4 marks] — Segment Anything (S13)
Explain **(a)** SAM's promptable segmentation task and its three-part architecture, **(b)** how SAM deals with **ambiguous** prompts, and **(c)** three **limitations** of SAM.

---

## Section D — Long answer / Design (attempt any 3 of 4; 10 marks each = 30 marks)

### D1 · Long · Hard · [10 marks] — Choosing a detector (S6–S8)
**(a)** Compare **two-stage** detectors (Faster R-CNN with FPN), **one-stage** detectors (YOLO / SSD / RetinaNet family) and **DETR-style** detectors. Cover architecture, training (label assignment and losses), speed, accuracy, small objects, and the use of anchors and NMS. A table is acceptable for part of the answer. **[6]**
**(b)** **Scenario:** a drone at 80–100 m altitude records 4K ($3840\times2160$) video of a car park. The system must count cars and detect people. Cars appear about 20–40 px wide and people about 8–15 px. Inference must run **on board on an embedded GPU at ≥ 10 fps**, and 4,000 labelled frames are available. Recommend a detector family, justify the choice against the alternatives, and give **two concrete design choices** for handling the small objects. **[4]**

### D2 · Long · Hard · [10 marks] — Text-promptable perception for dash-cam video (S12–S14)
Design a system that takes a free-text prompt such as `car . pedestrian .` and then **finds and segments** every car and pedestrian in **monocular dash-cam video** ($1920\times1080$, 30 fps) and **estimates the distance** to each one in metres. Your answer must cover:
(a) the pipeline and the role of each component, with a block diagram; **[3]**
(b) the **relative vs metric depth** problem and how you would obtain metric distances; **[3]**
(c) **temporal tracking** across frames; **[1]**
(d) a **latency** budget and how you would meet it; **[1]**
(e) **failure modes** and mitigations. **[2]**

### D3 · Long · Medium · [10 marks] — Panorama stitching (S1, S2, S11)
You take **5 overlapping handheld photos** (≈30–40% overlap between neighbours) of a city square by turning on the spot. Describe the full pipeline for stitching them into one panorama, giving the **mathematics of each step**: feature detection and description, matching, robust homography estimation, choice of reference frame and projection surface, warping, and blending. State when a homography relates two images **exactly**. Explain where and why stitching fails because of **parallax**, **moving objects** and **exposure differences**, and give a fix for each.
Marks: pipeline **[4]** · mathematics **[3]** · failure modes and fixes **[3]**.

### D4 · Long · Hard · [10 marks] — Generative models and Stable Diffusion (S15)
**(a)** Compare **autoencoders, VAEs, GANs and diffusion models** in terms of training objective, latent space, sample quality and diversity, training stability, and sampling speed. **[4]**
**(b)** Explain how **Stable Diffusion** turns a text prompt into an image. Cover the text encoder, the latent space and its VAE, the U-Net denoiser and its **cross-attention**, **classifier-free guidance**, the sampler, and decoding. **[4]**
**(c)** Explain how **ControlNet** adds spatial control (for example from an edge map or a pose skeleton) to a pretrained Stable Diffusion model, and why its **zero convolutions** matter. **[2]**

---

*End of paper. Check that you have attempted all of Sections A and B, five questions from Section C and three from Section D.*
