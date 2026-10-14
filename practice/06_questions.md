# Session 06 — Object Detection I: Classical & Two-Stage Detectors · Practice Questions

> Notes: [../notes/06_detection_two_stage.md](../notes/06_detection_two_stage.md) · Solutions: [06_solutions.md](06_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min
>
> **Box convention for every question:** boxes are $(x_1,y_1,x_2,y_2)$ in continuous pixel coordinates, $x$ = column, $y$ = row (downwards), width $=x_2-x_1$, height $=y_2-y_1$, area $=(x_2-x_1)(y_2-y_1)$ — **no +1**. $\log$ = natural logarithm.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
Which property distinguishes **object detection** from **classification + localisation**?

- **A.** Detection predicts a single class label for the whole image.
- **B.** Detection must output a variable-size set of (class, box, score) triples and therefore needs an explicit background class and duplicate removal.
- **C.** Detection uses a box-regression loss whereas classification + localisation does not.
- **D.** Detection outputs a class label for every pixel.

### Q2 · MCQ · Medium
Which HOG configuration did Dalal & Triggs find to work best for pedestrian detection?

- **A.** Gaussian smoothing ($\sigma=2$) followed by Sobel gradients, 18 signed bins over 0–360°.
- **B.** Non-overlapping 16×16-pixel blocks with L1 normalisation and 9 signed bins over 0–360°.
- **C.** Unsmoothed centred $[-1,0,1]$ gradients, 9 unsigned bins over 0–180°, 8×8-pixel cells, overlapping 2×2-cell blocks with L2-Hys normalisation.
- **D.** 4×4-pixel cells, 36 unsigned bins and no block normalisation.

### Q3 · MCQ · Easy
In the Viola–Jones face detector, the main purpose of the **attentional cascade** is to:

- **A.** reject most background windows in early stages that use very few features, so the average cost per window is tiny.
- **B.** compute any rectangle sum in constant time.
- **C.** select the most discriminative Haar features from the huge feature pool.
- **D.** make the detector invariant to in-plane rotation.

### Q4 · MCQ · Medium
The main reason **R-CNN** needs tens of seconds per image at test time is that:

- **A.** selective search alone takes tens of seconds per image.
- **B.** its per-class SVMs use expensive non-linear kernels.
- **C.** NMS over ~2000 boxes has quadratic cost.
- **D.** the CNN is run separately on each of ~2000 warped proposals, repeating heavily overlapping computation.

### Q5 · MCQ · Medium
Which statement about **Fast R-CNN** is correct?

- **A.** RoI pooling makes the whole system, including region proposals, trainable end-to-end.
- **B.** All proposals share one convolutional pass over the image, but proposals still come from an external selective-search step.
- **C.** It replaced selective search with a Region Proposal Network.
- **D.** It classifies each RoI with per-class linear SVMs trained on features cached on disk.

### Q6 · MCQ · Easy
An RPN uses $k=9$ anchors per location and, as in the original paper, a two-class softmax (object / not object) per anchor. How many output channels do the **cls** and **reg** 1×1 conv layers have, respectively?

- **A.** 18 and 36
- **B.** 2 and 4
- **C.** 18 and 4
- **D.** 36 and 18

### Q7 · MCQ · Medium
Which statement about **anchors** in Faster R-CNN is correct?

- **A.** Anchor scales and aspect ratios are learned by back-propagation together with the RPN weights.
- **B.** Each anchor is a box produced by the RPN's regression branch.
- **C.** Anchors are only placed at locations whose objectness score exceeds 0.7.
- **D.** Anchors are fixed reference boxes at every feature-map location; the network learns objectness and offsets relative to them.

### Q8 · MCQ · Medium
Where is **non-maximum suppression** used in standard Faster R-CNN inference?

- **A.** Nowhere — the RPN makes NMS unnecessary.
- **B.** Only on the final class-specific detections.
- **C.** On the RPN proposals (IoU ≈ 0.7) and again per class on the final detections.
- **D.** Only on the anchors, before the RPN is run.

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
A HOG descriptor uses 8×8-pixel cells, 9 **unsigned** orientation bins over 0–180° with bin centres at 10°, 30°, …, 170°, 2×2-cell blocks with a stride of **one cell**, and each block is normalised separately.

(a) Compute the descriptor length for a **64×128** (W×H) detection window; show the number of cells and blocks. [2]
(b) Compute the descriptor length for a **48×96** window. [1]
(c) A pixel has neighbours $I(x-1,y)=50$, $I(x+1,y)=80$, $I(x,y-1)=120$, $I(x,y+1)=60$ ($y$ increases downwards). Using centred differences $I_x=I(x+1,y)-I(x-1,y)$, $I_y=I(x,y+1)-I(x,y-1)$ and $\theta=\operatorname{atan2}(I_y,I_x)\bmod180^\circ$, compute the gradient magnitude, the unsigned orientation, and the magnitude votes cast into the two nearest bins using linear interpolation between bin centres. [2]

### Q10 · Numerical · Medium · [5 marks]
A pedestrian detector scans an **800×600** (W×H) image with a **64×128** window and a stride of **8 px** at every pyramid level. The pyramid halves the image at each level (800×600, 400×300, …) and stops as soon as the image is smaller than the window in either dimension. Number of positions per dimension $=\lfloor(\text{image}-\text{window})/\text{stride}\rfloor+1$.

(a) Number of windows at each level and in total. [2]
(b) The detector is extended to **3 window aspect ratios**; assume each ratio has the same number of positions as in (a). Total windows? [0.5]
(c) The classifier is a **10-stage cascade**; every stage has detection rate $d=0.99$ and false-positive rate $f=0.30$ (independent stages). Compute the overall detection rate $D$ and false-positive rate $F$. [1.5]
(d) Using the single-ratio window count from (a), how many false-positive windows per (person-free) image do you expect? [1]

### Q11 · Numerical · Medium · [5 marks]
Faster R-CNN with a VGG-16 backbone (total stride 16; conv5_3 has 512 channels) processes a **640×960** (H×W) image. The RPN uses 3 scales $s\in\{128,256,512\}$ × 3 aspect ratios $r=h/w\in\{0.5,1,2\}$ (area-preserving: $w=s/\sqrt r$, $h=s\sqrt r$), a 3×3 conv with 512 output channels, and two sibling 1×1 convs (two-class softmax per anchor; 4 offsets per anchor). All convolutions have biases.

(a) Feature-map size and total number of anchors. [1]
(b) Output channels of the cls and reg layers and the shapes of their output tensors. [1]
(c) Total number of learnable parameters in the RPN head (3×3 conv + both 1×1 convs). [2]
(d) Width and height of the three anchors of scale 128. [1]

### Q12 · Numerical · Hard · [5 marks]
Two ground-truth boxes: $G_1=(100,100,220,180)$, $G_2=(300,40,340,100)$. Five anchors:
$A_1=(92,96,220,192)$, $A_2=(130,90,258,186)$, $A_3=(180,140,308,236)$, $A_4=(290,38,354,102)$, $A_5=(320,50,384,114)$.
RPN labelling rules: **positive** if IoU > 0.7 with any GT **or** if it is the highest-IoU anchor for some GT; **negative** if IoU < 0.3 with **all** GTs; otherwise **ignored**.

(a) Compute the IoU of every anchor with every GT and label each anchor. [2.5]
(b) Compute the regression target $(t_x,t_y,t_w,t_h)$ of $A_1$ with respect to its matched GT, using $t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$ (centre–size form). [1.5]
(c) The RPN predicts offsets $\hat t=(0.05,-0.10,0.10,-0.20)$ for $A_1$. Decode them into a box $(x_1,y_1,x_2,y_2)$. [1]

### Q13 · Numerical · Medium · [5 marks]
Fast R-CNN uses a VGG-16 backbone (total stride 16) with **7×7 RoI pooling**; the image is resized to **800×600** (W×H). RoI corners are projected by dividing by 16 and quantised by **rounding to the nearest integer**.

(a) Size (W×H) of the conv5 feature map (state your rounding). [0.5]
(b) RoI $R_1=(64,80,400,304)$: projected coordinates, size in feature cells, and the size of each RoI-pooling bin. [1.5]
(c) RoI $R_2=(50,70,370,290)$: projected (unrounded) and quantised coordinates, size in cells, bin width, and the largest corner displacement (in **image pixels**) caused by the quantisation. [1.5]
(d) $R_1$ is a foreground RoI of class $u$ with softmax probability $p_u=0.8$; the regression residuals $t^u-v$ are $(0.4,-0.2,1.5,-0.1)$. Compute the Fast R-CNN multi-task loss $L=-\log p_u+\lambda\sum_i\operatorname{smooth}_{L_1}(t^u_i-v_i)$ with $\lambda=1$. What would the loss be if $R_1$ were a background RoI with $p_0=0.9$? [1.5]

## Section C — Short answer

### Q14 · Short · Easy · [4 marks]
Distinguish **image classification**, **classification + localisation** and **object detection** in terms of their outputs and training losses. Why can a detector not simply be a CNN with a fixed-size box-regression head?

### Q15 · Short · Medium · [4 marks]
Explain **hard-negative mining** in the HOG + linear SVM pipeline (what, why, how). Why is non-maximum suppression still needed after the SVM?

### Q16 · Short · Medium · [4 marks]
Describe how **selective search** generates region proposals. Why did region proposals make CNN-based detection (R-CNN) practical, and what is their main drawback once the CNN part becomes fast?

### Q17 · Short · Medium · [4 marks]
Explain **RoI pooling** in Fast R-CNN: what problem it solves, how it works, and how gradients flow through it. What is its **quantisation** problem and how does **RoIAlign** fix it?

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
Trace the evolution **R-CNN → Fast R-CNN → Faster R-CNN**. For each detector describe the pipeline, how it is trained and its main bottleneck. Then draw the **Faster R-CNN architecture** and explain in detail how the **Region Proposal Network** is trained (anchors, label assignment, sampling, loss) and how it produces proposals at inference. End with the "two-stage philosophy": why this design is accurate and what it costs.

### Q19 · Long · Medium · [10 marks]
A city wants to count pedestrians at a crosswalk. **Option A**: a CPU-only, low-power board inside the traffic-light cabinet (no GPU, ~10 fps required). **Option B**: stream video to a GPU server and process it there. For each option propose and justify a detector from this session (classical sliding-window or two-stage CNN), and describe: the training data and training procedure (including how negatives are handled), how you handle pedestrians from ~30 px to ~300 px tall and partial occlusion, and how you would evaluate the system. Conclude with the trade-offs between the two options.
