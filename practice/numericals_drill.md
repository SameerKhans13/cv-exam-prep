# Numericals Drill — Computer Vision (all sessions)

> **What this is:** 67 **new** exam-style numericals (D1–D67) covering every calculation type in the course, with fully worked, Python-verified solutions, a one-page answer key and an index of every other numerical in the pack.
> **Related:** [Formula sheet](../00_formula_sheet.md) · [Comparison tables](../00_comparison_tables.md) · [Mock exam 1](mock_exam_1.md) · [Mock exam 2](mock_exam_2.md) · per-session practice: [01](01_questions.md) · [02](02_questions.md) · [03](03_questions.md) · [04](04_questions.md) · [05](05_questions.md) · [06](06_questions.md) · [07](07_questions.md) · [08](08_questions.md) · [09](09_questions.md) · [10](10_questions.md) · [11](11_questions.md) · [12](12_questions.md) · [13](13_questions.md) · [14](14_questions.md) · [15](15_questions.md)

## How to use this drill

1. **Part A closed-book**, with a scientific calculator only (no notes, no formula sheet). Write down every intermediate number — in the exam, method marks come from the working.
2. **Time yourself: about 6 minutes per problem** (Easy ≈ 4 min, Medium ≈ 6 min, Hard ≈ 8 min). Exam Section B gives 6 numericals × 5 marks in roughly 50 minutes, so this is exam pace. Do one session block (4–7 problems) per sitting.
3. **Mark with Part B** (or the [Quick answer key](#quick-answer-key) for a fast check). For every miss, write one line in an error log: *which formula, which slip* (wrong convention, rounding, units, forgot +1, ...).
4. **Redo every missed problem after 2 days** without looking at the solution, then once more a week later. A problem is "learned" when you get it right twice in a row.
5. The day before the exam, re-read only the **Common slip** lines in Part B.
6. When you have finished a block, try the original practice numericals of that session — see the [Index of all numericals in the pack](#index-of-all-numericals-in-the-pack).

## Conventions box (apply to every problem unless the problem says otherwise)

| Topic | Convention used in this drill |
|---|---|
| Image coordinates | $I(x,y)$: $x$ = column (to the right), $y$ = row (**downwards**); 0-indexed. Angles $\theta=\operatorname{atan2}(I_y,I_x)$ are measured from $+x$ towards $+y$ (i.e. clockwise on screen). |
| Filtering | "Correlation" = $g(x,y)=\sum_{i,j}K(i,j)\,I(x+i,y+j)$ (no flip; $i$ = column/$x$ offset, $j$ = row/$y$ offset, so $K$ is laid on the image as printed); "convolution" flips the kernel by 180°. Sobel kernels are applied as correlation. |
| Boxes | $(x_1,y_1,x_2,y_2)$, continuous coordinates, area $=(x_2-x_1)(y_2-y_1)$ — **no +1**. DETR boxes are normalised $(c_x,c_y,w,h)$. |
| IoU / NMS | NMS suppresses a box if its IoU with an already-kept box is **> threshold**. Panoptic matching needs IoU **> 0.5** (strict). |
| AP | **Default = all-point interpolation** (VOC2010+, area under the precision envelope $p_{interp}(r)=\max_{r'\ge r}p(r')$). The 11-point (VOC2007) method is used only when stated. Retrieval AP is the non-interpolated $\frac1R\sum_k P@k\cdot\text{rel}(k)$ with $R$ = total relevant. |
| RANSAC | $N=\left\lceil\frac{\log(1-p)}{\log(1-(1-e)^s)}\right\rceil$ — always round **up** (16.008 → 17). |
| Confusion matrices | **Rows = ground truth, columns = prediction.** Precision uses column sums, recall uses row sums. |
| Logs & angles | $\log$ = natural log (nats) unless stated ($\log_2$ only where written). Angles are given and answered in **degrees** — keep the calculator in DEG mode for $\arctan$, $\arccos$, FOV (check: $\cos60^\circ=0.5$); $\log$ and $\exp$ do not depend on the mode. |
| Camera | $X_c=RX_w+\mathbf t$, $\lambda(u,v,1)^\top=K X_c$, camera centre $C=-R^\top\mathbf t$; $f$ in pixels $=f_{mm}/\text{pixel pitch}_{mm}$. |
| Rounding | Keep ≥ 5 significant figures in intermediate steps; give final answers to 3–4 significant figures (or the d.p. requested). |
| Units | Write units on every final answer (px, m, MB = $10^6$ B, GB = $10^9$ B, ms, FPS). |

## Contents

| Session | Problems |
|---|---|
| S01 Image processing & features | D1–D5 |
| S02 Feature matching & classical CV | D6–D9 |
| S03 CNNs under the hood | D10–D15 |
| S04 Training & transfer learning | D16–D20 |
| S05 Embeddings & similarity | D21–D24 |
| S06 Detection I — two-stage | D25–D28 |
| S07 Detection II — one-stage, metrics | D29–D34 |
| S08 Detection III — transformers | D35–D38 |
| S10 Segmentation | D39–D44 |
| S11 Camera geometry | D45–D49 |
| S12 3D vision & motion | D50–D56 |
| S13–14 Foundation models & pipelines | D57–D61 |
| S15 Generative CV | D62–D67 |

---

## Part A — Problems

### S01 · Image processing & features

#### D1 · Correlation vs convolution at a pixel · [S01 · Easy · 4 marks]
A 5×5 grayscale image (rows $y=0..4$ top to bottom, columns $x=0..4$ left to right) and a 3×3 kernel whose centre is the origin ($K(i,j)$: $i$ = column offset, $j$ = row offset, so $K$ is laid on the image exactly as printed):

$$
I=\begin{bmatrix}10&20&30&40&50\\15&60&90&100&55\\20&70&120&140&60\\25&75&130&150&65\\30&35&40&45&70\end{bmatrix},\qquad
K=\begin{bmatrix}0&1&2\\-1&0&1\\-2&-1&0\end{bmatrix}.
$$

**(a)** [1.5] Compute the **cross-correlation** output $g(x,y)=\sum_{i,j\in\{-1,0,1\}}K(i,j)\,I(x+i,y+j)$ at pixel $(x,y)=(2,1)$.

**(b)** [1] Compute the **convolution** output at the same pixel. How are (a) and (b) related for this particular kernel, and why?

**(c)** [1] Compute the correlation output at the corner pixel $(0,0)$ with (i) zero padding and (ii) replicate (edge) padding.

**(d)** [0.5] What is the output size if no padding is used ("valid" mode)?

#### D2 · Sobel gradient, magnitude, direction · [S01 · Medium · 4 marks]
A 3×3 neighbourhood centred on pixel $p$ (top row = row $y-1$, left column = column $x-1$, $y$ points **down**):

$$
N=\begin{bmatrix}200&180&120\\190&120&60\\130&70&40\end{bmatrix},\qquad
K_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\quad K_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}\ \text{(correlation)}.
$$

**(a)** [1] Compute $I_x$ and $I_y$ at $p$.

**(b)** [1.5] Compute the L2 and L1 gradient magnitudes and $\theta=\operatorname{atan2}(I_y,I_x)$ in degrees, in both $(-180^\circ,180^\circ]$ and $[0^\circ,360^\circ)$. In which direction does the edge through $p$ run?

**(c)** [1] Canny NMS quantises $\theta \bmod 180^\circ$ to $\{0^\circ,45^\circ,90^\circ,135^\circ\}$. Which bin is $p$ in, and which two neighbours is it compared with?

**(d)** [0.5] For a linear ramp $I=a\,x$ the Sobel $K_x$ response is $8a$. Give the gradient in intensity levels **per pixel** (divide by 8).

#### D3 · Harris response from raw gradients · [S01 · Medium · 5 marks]
Three 2×2 windows (uniform weights $w=1$) contain the following gradients $(I_x,I_y)$ at their four pixels. Use $k=0.05$ and the rule *corner if $R>1000$, edge if $R<0$, otherwise flat*.

| Window | pixel 1 | pixel 2 | pixel 3 | pixel 4 |
|---|---|---|---|---|
| W1 | (10, 2) | (8, −1) | (2, 9) | (−1, 11) |
| W2 | (10, 5) | (8, 4) | (−6, −3) | (12, 6) |
| W3 | (1, 0) | (0, 1) | (−1, 1) | (1, 1) |

**(a)** [2] Build the structure tensor $M=\sum\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}$ for each window.

**(b)** [1.5] Compute $\det M$, $\operatorname{tr}M$ and $R=\det M-k(\operatorname{tr}M)^2$ for each, and classify.

**(c)** [1] Give the eigenvalues of $M$ for W1 and W2 and the Shi–Tomasi score $\min(\lambda_1,\lambda_2)$. Why is $\det M=0$ for W2?

**(d)** [0.5] Every intensity in W1's neighbourhood is doubled. What is the new $R$?

#### D4 · Gaussian kernel size and separable cost · [S01 · Easy · 4 marks]
Use the kernel-size rule $K=2\lceil3\sigma\rceil+1$.

**(a)** [1.5] For $\sigma=1$, give $K$ and the normalised 1-D kernel weights $g(x)\propto e^{-x^2/(2\sigma^2)}$ (4 d.p.).

**(b)** [1.5] For $\sigma=1.5$ on a 1280×720 grayscale image: give $K$, the multiplications per pixel for direct 2-D filtering and for separable filtering, both totals for the image, and the speed-up.

**(c)** [1] The image is blurred with $\sigma=1.5$ and then again with $\sigma=2$. What single $\sigma$ is equivalent? Compare the separable cost per pixel of the two-pass cascade with one blur at the equivalent $\sigma$.

#### D5 · Integral image lookups · [S01 · Easy · 4 marks]
The integral image $ii(x,y)=\sum_{x'\le x,\,y'\le y}I(x',y')$ of a 5×5 image is given (rows $y=0..4$, columns $x=0..4$):

$$
ii=\begin{bmatrix}2&5&6&10&12\\7&11&18&24&29\\8&19&28&42&51\\11&24&42&57&71\\17&34&55&72&93\end{bmatrix}.
$$

**(a)** [1.5] Using exactly four lookups, find the sum and the mean of the rectangle covering columns $x=2..4$ and rows $y=1..3$ (inclusive).

**(b)** [1] Recover the single pixel value $I(3,2)$.

**(c)** [0.5] Find the sum of the bottom row ($y=4$).

**(d)** [1] A 9×9 box (mean) filter is run over a 1920×1080 image. Compare additions per pixel done directly vs with the integral image, the totals, and the speed-up (also including the ≈2 additions per pixel needed to build $ii$).

### S02 · Feature matching & classical CV

#### D6 · RANSAC iteration counts (incl. a budget) · [S02 · Medium · 5 marks]
Use $N=\lceil\log(1-p)/\log(1-(1-e)^s)\rceil$.

**(a)** [1] Line fitting ($s=2$), $e=0.5$, $p=0.99$.

**(b)** [1] Affine transform ($s=3$), $e=0.35$, $p=0.99$.

**(c)** [1] Essential matrix, 5-point solver, $e=0.45$, $p=0.999$.

**(d)** [2] A real-time stitcher can afford at most **500** homography hypotheses per frame ($s=4$) and wants $p=0.99$. What is the **largest outlier ratio** $e$ for which 500 iterations suffice? Check your answer at $e=0.69$ and $e=0.70$. What success probability do 500 iterations give if $e=0.75$?

#### D7 · Applying a homography and the inlier test · [S02 · Medium · 5 marks]
A RANSAC hypothesis maps image 1 (1280×720) to image 2:

$$
H=\begin{bmatrix}0.9&0.05&25\\-0.04&1.1&-10\\-0.0003&0.0002&1\end{bmatrix}.
$$

**(a)** [2] Map $\mathbf p_1=(320,240)$ and $\mathbf p_2=(600,80)$ into image 2 (2 d.p.).

**(b)** [1.5] The matched points detected in image 2 are $\mathbf p_1'=(343.0,\ 252.2)$ and $\mathbf p_2'=(676.6,\ 67.6)$. With an inlier threshold $t=3$ px on the transfer error $\Vert H\mathbf p-\mathbf p'\Vert$, classify each match.

**(c)** [1] Which image-1 points are mapped to infinity (the third homogeneous coordinate becomes 0)? Do any of them lie inside image 1?

**(d)** [0.5] What happens to the mapped points if $H$ is replaced by $2H$, and what does this imply for the DOF?

#### D8 · DLT matrix size and Hartley normalisation · [S02 · Medium · 4 marks]
**(a)** [1] Give the size of the DLT matrix $A$ in $A\mathbf h=\mathbf 0$ for a homography fitted to $n=4$, $6$ and $50$ correspondences. What is the rank of $A$ for noise-free, non-degenerate data, and how is $\mathbf h$ obtained?

**(b)** [0.5] For an **affine** model, how many unknowns are there and how many correspondences are minimally needed?

**(c)** [2] Hartley normalisation of the four image-1 points $(120,80)$, $(520,100)$, $(500,380)$, $(100,360)$: translate the centroid to the origin and scale so that the **mean distance to the origin is $\sqrt2$**. Give the centroid, the four distances, the scale $s$, the matrix $T$ and the normalised coordinates of the first point.

**(d)** [0.5] If $\hat H$ is estimated from points normalised by $T$ (image 1) and $T'$ (image 2), how is the final $H$ obtained?

#### D9 · Hamming distance and the ratio test for ORB · [S02 · Easy · 4 marks]
**(a)** [1] Compute the Hamming distance between the 16-bit descriptor fragments `0xB3C5` and `0x9AE1`.

**(b)** [1.5] Three ORB (256-bit) query descriptors have nearest/second-nearest Hamming distances q1: $(34, 51)$, q2: $(62, 70)$, q3: $(90, 150)$. Which matches pass Lowe's ratio test with $\tau=0.75$ (accept if $d_1/d_2<\tau$)? Comment on q3.

**(c)** [1] For two **unrelated** 256-bit descriptors (independent bits, $P(\text{differ})=0.5$), what are the mean and standard deviation of the Hamming distance? How many standard deviations below "chance" is q1's best match?

**(d)** [0.5] Storage of one ORB descriptor and of one SIFT descriptor (float32).

### S03 · CNNs under the hood

#### D10 · Output sizes through a stack (odd input) · [S03 · Medium · 4 marks]
A $225\times225\times3$ image passes through (use $W_{out}=\lfloor(W-K_{eff}+2P)/S\rfloor+1$, $K_{eff}=D(K-1)+1$):

| Layer | Operation |
|---|---|
| L1 | conv $7\times7$, 64 filters, $S=2$, $P=3$ |
| L2 | max-pool $3\times3$, $S=2$, $P=1$ |
| L3 | conv $3\times3$, 128 filters, $S=2$, $P=1$ |
| L4 | conv $3\times3$, 128 filters, $S=1$, $P=2$, dilation $D=2$ |
| L5 | conv $1\times1$, 256 filters, $S=1$, $P=0$ |
| L6 | conv $3\times3$, 256 filters, $S=2$, $P=0$ |
| L7 | global average pooling |

**(a)** [3] Give the output shape ($H\times W\times C$) after every layer.

**(b)** [1] Repeat for a $224\times224\times3$ input. Where does the floor change the result?

#### D11 · 1×1 bottleneck vs direct 3×3 — parameters · [S03 · Medium · 4 marks]
GoogLeNet's Inception (4a) module receives a $14\times14\times480$ input. Its 3×3 branch outputs 208 channels ("same" padding, stride 1, all convs with bias).

**(a)** [1] Parameters of a direct $3\times3$ conv $480\to208$.

**(b)** [2] The real module first reduces to 96 channels with a $1\times1$ conv, then applies the $3\times3$ conv $96\to208$. Give the parameters of each conv, the total and the reduction factor.

**(c)** [1] Explain why a $1\times1$ conv is "a fully connected layer applied at every pixel" and state its weight-matrix shape here.

#### D12 · MACs, FLOPs and latency · [S03 · Medium · 4 marks]
Same layers as D11 on the $14\times14$ map; ignore biases for MACs ($\text{MACs}=H_{out}W_{out}K_hK_wC_{in}C_{out}$).

**(a)** [1] MACs of the direct $3\times3$ branch.

**(b)** [1.5] MACs of the bottleneck version (both convs), and the reduction factor.

**(c)** [0.5] FLOPs ($\approx2\times$MACs) of each and the time on a GPU sustaining 2 TFLOP/s.

**(d)** [1] MACs and parameters of a fully connected layer $4096\to4096$ (with bias). Use (a)–(d) to explain why FC layers dominate the **parameter** count of VGG/AlexNet while conv layers dominate the **compute**.

#### D13 · Receptive field with pooling and dilation · [S03 · Medium · 5 marks]
Layers (padding does not affect the receptive field): L1 conv $3\times3$, $S=1$ → L2 conv $3\times3$, $S=2$ → L3 max-pool $2\times2$, $S=2$ → L4 conv $3\times3$, $S=1$, $D=2$ → L5 conv $3\times3$, $S=1$, $D=4$ → L6 conv $1\times1$.

**(a)** [3] Tabulate $K_{eff}$, $r_l$ and the jump $j_l$ after every layer ($r_0=1$, $j_0=1$).

**(b)** [1] What is the final RF if L4 and L5 are ordinary (undilated) $3\times3$ convs?

**(c)** [1] How many undilated stride-1 $3\times3$ convs after L3 would be needed to reach the RF of (a)? Compare parameters with the two dilated convs (same channel count $C$).

#### D14 · Depthwise-separable savings · [S03 · Medium · 4 marks]
A $5\times5$ conv maps $C_{in}=240$ to $C_{out}=80$ channels on a $28\times28$ map (stride 1, "same", **no bias**).

**(a)** [1] Parameters and MACs of the standard conv.

**(b)** [2] It is replaced by a depthwise $5\times5$ conv followed by a pointwise $1\times1$ conv. Give parameters and MACs of each part and of the whole.

**(c)** [1] Compute the cost ratio and check it against $\frac{1}{C_{out}}+\frac{1}{K^2}$. Which part dominates the separable layer's MACs?

#### D15 · ViT token counts · [S03 · Easy · 4 marks]
**(a)** [1.5] ViT-B/16 at $384\times384$ with a [CLS] token: number of tokens and attention-matrix entries per head per layer. By what factor does this exceed the $224\times224$ case?

**(b)** [0.5] ViT-S/8 at $224\times224$ with [CLS]: number of tokens.

**(c)** [1] ViT-B/16 on a $320\times480$ ($H\times W$) image with [CLS]: number of tokens. What must be done to the position embeddings learned at $224\times224$?

**(d)** [1] Parameters of the patch-embedding layer ($p=16$, $C=3$ channels, embedding dimension $D=768$, with bias).

### S04 · Training CNNs & transfer learning

#### D16 · Softmax, cross-entropy and label smoothing · [S04 · Medium · 4 marks]
A 4-class model (cat, dog, fox, owl) outputs logits $z=[1.2,\ 3.1,\ 0.4,\ -0.5]$ for an image whose true class is **cat**. Natural logs.

**(a)** [1.5] Compute the softmax probabilities (subtract the max logit first).

**(b)** [1] Compute the cross-entropy loss and $\partial\mathcal L/\partial z$.

**(c)** [1] With label smoothing $\varepsilon=0.1$ (PyTorch convention: every class gets $\varepsilon/K$, the true class gets $1-\varepsilon+\varepsilon/K$), give the target and the loss.

**(d)** [0.5] Which class is predicted? With label smoothing, what is the smallest loss the model could ever reach on this image?

#### D17 · BatchNorm on a conv mini-batch · [S04 · Medium · 5 marks]
One channel of a `BatchNorm2d` layer in **train mode** receives a mini-batch of $N=2$ feature maps of size $2\times2$:
image 1 $=\begin{bmatrix}1&3\\5&7\end{bmatrix}$, image 2 $=\begin{bmatrix}2&2\\4&8\end{bmatrix}$.
$\gamma=0.5$, $\beta=0.2$, $\epsilon$ negligible. Running statistics start at mean 0, variance 1; momentum $0.1$; as in PyTorch the running variance uses the **unbiased** batch variance.

**(a)** [1] Over which values are the statistics computed? Give $\mu_B$ and the (biased) $\sigma_B^2$.

**(b)** [2] Compute $\hat x$ and the outputs $y=\gamma\hat x+\beta$ for all 8 values.

**(c)** [1] Updated running mean and running variance.

**(d)** [1] In **eval mode** the layer then receives $x=4$: output? How many learnable parameters and how many buffers does `BatchNorm2d(64)` have?

#### D18 · Inverted dropout · [S04 · Easy · 3 marks]
Activations $a=[1.5,\ -0.4,\ 2.0,\ 0.8,\ 3.2,\ 0.0,\ 1.1,\ -2.5]$ pass through inverted dropout with drop probability $p=0.25$; the sampled keep-mask is $m=[1,1,0,1,0,1,1,1]$.

**(a)** [1] Training-time outputs.

**(b)** [0.5] Test-time outputs, and why the expected value matches.

**(c)** [1] With "classic" (non-inverted) dropout, what are the training output and the test output for the unit with $a=3.2$?

**(d)** [0.5] Variance of the training output of the unit with $a=3.2$ over random masks.

#### D19 · Trainable parameters: feature extraction vs fine-tuning (VGG-16) · [S04 · Medium · 5 marks]
VGG-16 has **138,357,544** parameters: a conv part of **14,714,688** (13 conv layers) and FC6 $25088\to4096$, FC7 $4096\to4096$, FC8 $4096\to1000$ (all with bias). For a 5-class flower dataset FC8 is replaced by $4096\to5$.

**(a)** [1] Parameters of FC6, FC7 and FC8; check the total.

**(b)** [0.5] Total parameters of the adapted network.

**(c)** [2] Trainable parameters and % of the adapted network for: (i) feature extraction (only new FC8), (ii) retraining FC6–FC8, (iii) unfreezing the conv5 block (three $3\times3$ convs $512\to512$) + FC6–FC8, (iv) full fine-tuning.

**(d)** [1.5] Training memory ≈ 4 B per parameter (fp32 weights, all parameters) + 12 B per **trainable** parameter (gradient + two Adam moments). Compute it for (i) and (iv) in GB ($10^9$ B), ignoring activations.

#### D20 · Confusion matrix → precision, recall, F1 (macro & micro) · [S04 · Medium · 5 marks]
A traffic-sign classifier on 200 test images (rows = ground truth, columns = prediction):

| GT \ Pred | Stop | Yield | Speed |
|---|---|---|---|
| **Stop** | 45 | 3 | 2 |
| **Yield** | 4 | 26 | 10 |
| **Speed** | 1 | 6 | 103 |

**(a)** [0.5] Accuracy.

**(b)** [2] Precision, recall and F1 of each class.

**(c)** [1.5] Macro-precision, macro-recall, macro-F1 (mean of per-class F1).

**(d)** [1] Micro-precision, micro-recall, micro-F1; show why they equal the accuracy. Also give the support-weighted F1.

### S05 · Embeddings & similarity

#### D21 · Cosine and Euclidean for unit vectors · [S05 · Easy · 4 marks]
$\mathbf a=(0.5,0.5,0.5,0.5)$, $\mathbf b=(0.7,0.1,0.1,0.7)$, $\mathbf c=(0.1,0.7,0.7,0.1)$.

**(a)** [1.5] Check that all three are unit vectors. Compute $\cos(\mathbf a,\mathbf b)$, $\Vert\mathbf a-\mathbf b\Vert$ and the angle.

**(b)** [1] Compute the same for $(\mathbf a,\mathbf c)$ and $(\mathbf b,\mathbf c)$.

**(c)** [1] A face-verification system on unit embeddings says "same person" if $\Vert\mathbf x-\mathbf y\Vert<0.9$. Give the equivalent cosine threshold and angle.

**(d)** [0.5] If $\mathbf b$ is scaled to $3\mathbf b$ (not re-normalised), give its dot product, cosine and Euclidean distance with $\mathbf a$.

#### D22 · Triplet loss with batch-all and batch-hard mining · [S05 · Medium · 5 marks]
In a mini-batch an anchor has two positives with $d(a,p_1)=0.6$, $d(a,p_2)=1.1$ and four negatives with $d(a,n_i)=0.9,\ 1.3,\ 1.6,\ 2.4$. Margin $m=0.4$, $\mathcal L=\max(0,\ d(a,p)-d(a,n)+m)$.

**(a)** [2] Compute all 8 triplet losses, and classify each negative as **easy** ($d_n>d_p+m$), **semi-hard** ($d_p<d_n<d_p+m$) or **hard** ($d_n<d_p$) with respect to each positive.

**(b)** [1.5] Batch-all loss: mean over all 8 triplets, and mean over the active (non-zero) triplets only.

**(c)** [1] Batch-hard loss (hardest positive, hardest negative).

**(d)** [0.5] Why is batch-hard mining risky early in training?

#### D23 · InfoNCE and the temperature · [S05 · Medium · 5 marks]
A query has cosine similarity $0.75$ with its positive key and $0.70,\ 0.40,\ 0.10,\ -0.30$ with four negatives. $\mathcal L=-\log\frac{\exp(s_+/\tau)}{\sum_i\exp(s_i/\tau)}$ (sum over the positive and all negatives).

**(a)** [2] For $\tau=0.1$: logits, softmax probabilities and loss.

**(b)** [1.5] Repeat for $\tau=0.5$.

**(c)** [1] The gradients are $\partial\mathcal L/\partial s_+=(p_+-1)/\tau$ and $\partial\mathcal L/\partial s_i=p_i/\tau$. Give them for both $\tau$, and the share of the total negative gradient that falls on the hardest negative (0.70).

**(d)** [0.5] What is the loss of an untrained encoder for which all five similarities are equal?

#### D24 · Retrieval AP and Recall@K · [S05 · Medium · 4 marks]
**(a)** [2.5] A query has $R=4$ relevant images in the gallery. Relevance of the top-8 results: `1 0 0 1 1 0 0 0`. Give $P@1$, $R@1$, $P@5$, $R@5$, $P@8$, $R@8$ and the non-interpolated $\text{AP}=\frac1R\sum_kP@k\cdot\text{rel}(k)$. What would a student get by dividing by the number of relevant images actually retrieved?

**(b)** [1.5] Five face-ID queries; the rank of the first correct gallery match is $1,\ 3,\ 1,\ 7,\ 2$. Compute Recall@1, @2, @5, @10 in the metric-learning sense (fraction of queries with a correct match in the top $K$) and the MRR.

### S06 · Detection I — two-stage

#### D25 · Anchor count with FPN · [S06 · Medium · 4 marks]
Faster R-CNN with FPN on a $1024\times768$ ($W\times H$) image uses levels P2–P6 (strides 4, 8, 16, 32, 64), one anchor size per level (32, 64, 128, 256, 512) and 3 aspect ratios $r=h/w\in\{0.5,1,2\}$ (area-preserving: $w=s/\sqrt r$, $h=s\sqrt r$).

**(a)** [2] Feature-map size, locations and anchors at each level; total anchors.

**(b)** [0.5] Compare with a single stride-16 map using 9 anchors per location.

**(c)** [0.5] Output channels of the RPN classification and regression convs per level ($2k$ and $4k$).

**(d)** [1] Width and height of the three P3 anchors (size 64).

#### D26 · Encoding and decoding box offsets · [S06 · Medium · 5 marks]
Proposal $P=(120,80,280,200)$ and matched ground truth $G=(130,70,310,230)$. Use $t_x=\frac{x-x_a}{w_a}$, $t_y=\frac{y-y_a}{h_a}$, $t_w=\log\frac{w}{w_a}$, $t_h=\log\frac{h}{h_a}$ (centre–size form).

**(a)** [1] IoU($P$, $G$).

**(b)** [2] Regression targets $(t_x,t_y,t_w,t_h)$, and their values after the usual normalisation by standard deviations $(0.1,0.1,0.2,0.2)$.

**(c)** [1.5] The head predicts $\hat t=(-0.05,\ 0.10,\ 0.20,\ -0.10)$ (un-normalised). Decode it into $(x_1,y_1,x_2,y_2)$ and give its IoU with $G$.

**(d)** [0.5] The image (and both boxes) are resized by ×2. What happens to the targets?

#### D27 · HOG descriptor length and orientation voting · [S06 · Easy · 4 marks]
HOG with $8\times8$-pixel cells, $2\times2$-cell blocks, 9 unsigned bins (centres $10^\circ,30^\circ,\dots,170^\circ$, linear interpolation between neighbouring centres, wrapping $170^\circ\leftrightarrow10^\circ$).

**(a)** [1.5] Descriptor length for a $96\times160$ ($W\times H$) window with block stride = 1 cell.

**(b)** [0.5] Same window with block stride = 2 cells (non-overlapping blocks).

**(c)** [0.5] Same as (a) but **signed** gradients with 18 bins over $0$–$360^\circ$.

**(d)** [1.5] Votes cast by a pixel with magnitude 50 at (i) $75^\circ$ and (ii) $175^\circ$.

#### D28 · RoI projection onto a stride-16 map and FPN level · [S06 · Medium · 4 marks]
A proposal $(x_1,y_1,x_2,y_2)=(213,117,589,405)$ (image pixels) is pooled to $7\times7$ from a stride-16 feature map.

**(a)** [1.5] Projected (unrounded) coordinates; coordinates after RoI-Pool **rounding to the nearest integer**; misalignment of each side in **image pixels**.

**(b)** [1] Size in feature cells and RoI-Pool bin size (rounded RoI) vs RoIAlign bin size (exact RoI).

**(c)** [1] In FPN an RoI of size $w\times h$ is assigned to level $k=\lfloor k_0+\log_2(\sqrt{wh}/224)\rfloor$, $k_0=4$, clamped to $[2,5]$. Which level does this RoI use? Which level does a $64\times48$ RoI use?

**(d)** [0.5] Parameters of the following FC layer $7\times7\times256\to1024$ (with bias).

### S07 · Detection II — one-stage, NMS, AP

#### D29 · IoU in three configurations · [S07 · Easy · 4 marks]
Let $A=(40,60,140,160)$.

**(a)** [1] IoU($A$, $B$) with $B=(90,110,190,210)$.

**(b)** [1] IoU($A$, $C$) with $C=(60,80,120,140)$.

**(c)** [1] IoU($A$, $D$) with $D=(140,60,200,160)$.

**(d)** [1] Compare the IoU of a $20\times20$ box with a copy shifted 10 px to the right, and of a $200\times200$ box with a copy shifted 10 px. What does this mean for small objects?

#### D30 · Per-class vs class-agnostic NMS on six boxes · [S07 · Medium · 5 marks]
NMS threshold 0.5 (suppress if IoU **> 0.5**).

| Box | $(x_1,y_1,x_2,y_2)$ | Score | Class |
|---|---|---|---|
| B1 | (50, 50, 150, 150) | 0.92 | person |
| B2 | (60, 55, 160, 150) | 0.85 | person |
| B3 | (55, 60, 145, 160) | 0.80 | dog |
| B4 | (120, 50, 220, 150) | 0.75 | person |
| B5 | (65, 70, 155, 165) | 0.60 | dog |
| B6 | (300, 300, 360, 380) | 0.55 | person |

Given: IoU(B1,B4) = 0.176, IoU(B2,B3) = 0.705, IoU(B2,B4) = 0.242, IoU(B2,B5) = 0.664, IoU(B3,B4) = 0.134, IoU(B4,B5) = 0.178; B6 overlaps nothing.

**(a)** [1.5] Compute IoU(B1,B2), IoU(B1,B3), IoU(B1,B5) and IoU(B3,B5).

**(b)** [1.5] Run **per-class** NMS. List the kept boxes.

**(c)** [1.5] Run **class-agnostic** NMS (all boxes together). List the kept boxes.

**(d)** [0.5] Comment on the difference.

#### D31 · AP: all-point and 11-point from a ranked list · [S07 · Hard · 5 marks]
A class has **6** ground-truth objects. The detector's 10 detections, sorted by score, were matched at IoU ≥ 0.5 (each GT at most once) giving: `T F T T F T F F T F` (T = TP, F = FP).

**(a)** [1.5] Tabulate cumulative TP, FP, precision and recall after each detection.

**(b)** [2] Compute AP with **all-point interpolation**.

**(c)** [1.5] Compute AP with **VOC2007 11-point interpolation**. Why does it differ from (b)?

#### D32 · YOLO output tensors and cell assignment · [S07 · Medium · 4 marks]
**(a)** [1] A YOLOv1-style detector with $S=7$, $B=3$ boxes per cell and $C=10$ classes: output shape, number of values, number of boxes.

**(b)** [1.5] A YOLOv3-style detector on a $608\times608$ input predicts at strides 32, 16, 8 with 3 anchors per location and $C=10$ (each anchor: 4 box values + 1 objectness + $C$ class scores). Grid sizes, output channels per scale and total predicted boxes.

**(c)** [0.5] Parameters of the $1\times1$ prediction conv at stride 8 (256 input channels, with bias).

**(d)** [1] An object is centred at pixel $(300,470)$ of the 608 input. Which stride-32 cell (column, row, 0-indexed) is responsible, and what are the target offsets $\sigma(t_x),\sigma(t_y)$ inside that cell?

#### D33 · SSD default boxes · [S07 · Medium · 4 marks]
**(a)** [1] SSD512 predicts from maps of size 64, 32, 16, 8, 4, 2, 1 with 4, 6, 6, 6, 6, 4, 4 default boxes per location. Total default boxes?

**(b)** [1] SSD300 scales: $s_k=s_{min}+\frac{s_{max}-s_{min}}{m-1}(k-1)$, $s_{min}=0.2$, $s_{max}=0.9$, $m=6$. Give $s_1..s_6$.

**(c)** [1.5] For map $k=2$ of SSD300 (input 300 px): size in pixels of the aspect-ratio-2 box ($w=s_k\sqrt a$, $h=s_k/\sqrt a$) and of the extra square box with $s_k'=\sqrt{s_ks_{k+1}}$.

**(d)** [0.5] Output channels of the $3\times3$ prediction conv on a map with 6 boxes per location for VOC (21 classes incl. background, 4 offsets).

#### D34 · Focal loss vs cross-entropy ratios · [S07 · Medium · 4 marks]
$FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t$, $\gamma=2$, $\alpha_t=0.25$ for foreground and $0.75$ for background.

**(a)** [2] For $p_t=0.95,\ 0.7,\ 0.2$: CE $=-\ln p_t$, the modulating factor $(1-p_t)^\gamma$, the focal loss of a **foreground** anchor, and the ratio FL/CE for foreground and for background anchors.

**(b)** [1] At what $p_t$ does the modulating factor alone reduce the loss 100× when $\gamma=2$? When $\gamma=5$?

**(c)** [1] Ignoring $\alpha$, by what factor does a hard example ($p_t=0.2$) outweigh an easy one ($p_t=0.95$) under CE and under FL?

### S08 · Detection III — transformers

#### D35 · Hungarian matching (3×3, then 4×3) · [S08 · Medium · 5 marks]
Matching costs (lower = better), rows = queries, columns = GT objects:

$$
\mathbf C=\begin{bmatrix}8&4&7\\5&2&3\\9&3&8\end{bmatrix}\qquad(\text{rows } q_1,q_2,q_3;\ \text{columns } g_1,g_2,g_3)
$$

**(a)** [2.5] Solve with the Hungarian method: row reduction, column reduction, minimum line cover, adjustment, assignment. Give the total cost.

**(b)** [1] What does a greedy "cheapest remaining pair" strategy give?

**(c)** [1.5] A fourth query $q_4$ with costs $(6,\ 7,\ 9)$ is added. Add a zero-cost "no object" column, solve the $4\times4$ problem and state which query is trained to predict $\varnothing$.

#### D36 · DETR matching cost with L1 and GIoU · [S08 · Hard · 5 marks]
One GT **dog** with normalised box $b=(c_x,c_y,w,h)=(0.50,0.50,0.40,0.30)$. Two queries:
$q_1$: $\hat p(\text{dog})=0.55$, $\hat b_1=(0.52,0.49,0.38,0.30)$; $q_2$: $\hat p(\text{dog})=0.92$, $\hat b_2=(0.62,0.55,0.40,0.36)$.
Cost $\mathcal C=-\hat p(\text{dog})+\lambda_{L1}\Vert b-\hat b\Vert_1+\lambda_{giou}\big(1-\text{GIoU}(b,\hat b)\big)$ with $\lambda_{L1}=5$, $\lambda_{giou}=2$.

**(a)** [1] L1 distance for each query.

**(b)** [2] Convert all boxes to $(x_1,y_1,x_2,y_2)$ and compute IoU and GIoU for each query.

**(c)** [1.5] Matching cost of each query; which query is matched to the dog?

**(d)** [0.5] Could $q_2$ win if its dog probability were 1.0? What if $\lambda_{giou}=0$?

#### D37 · GIoU for overlap, near-miss, far-miss and containment · [S08 · Medium · 4 marks]
GT $G=(100,100,200,180)$. Predictions: $P_1=(150,120,250,220)$, $P_2=(220,100,300,180)$, $P_3=(400,300,480,380)$, $P_4=(120,110,180,170)$.
For each prediction [1 mark each] give the IoU, the area of the smallest enclosing box $E$, $\text{GIoU}=\text{IoU}-\frac{\lvert E\rvert-\lvert G\cup P\rvert}{\lvert E\rvert}$ and the loss $1-\text{GIoU}$. Comment on $P_2$ vs $P_3$ and on $P_4$.

#### D38 · Attention-matrix sizes in DETR · [S08 · Medium · 4 marks]
DETR (ResNet-50, C5 stride 32), image $1024\times768$ ($W\times H$), 8 heads, 6 encoder + 6 decoder layers, $N=100$ queries. MB = $10^6$ B.

**(a)** [1.5] Encoder tokens; attention entries per head per layer; per layer (8 heads); memory to store all encoder attention maps (6 layers) in fp16.

**(b)** [1] Decoder self-attention and cross-attention entries per head per layer.

**(c)** [1] DETR-DC5 uses a stride-16 map: tokens, entries per head per layer, growth factor, fp16 memory for all encoder maps.

**(d)** [0.5] Decoder entries per head per layer with 300 queries.

### S10 · Segmentation

#### D39 · Dice and IoU from pixel counts, pooled vs per-image, soft Dice · [S10 · Medium · 5 marks]
Lesion segmentation on two test images: image 1: TP = 800, FP = 200, FN = 400; image 2 (a small lesion): TP = 30, FP = 10, FN = 60.

**(a)** [1.5] IoU and Dice of each image.

**(b)** [1.5] Mean of the per-image Dice vs the dataset-level ("pooled") Dice computed from summed counts. Which one hides the small-lesion failure?

**(c)** [1] Convert IoU = 0.6 to Dice and Dice = 0.9 to IoU.

**(d)** [1] Predicted foreground probabilities $p=(0.9,0.7,0.4,0.2,0.1)$, ground truth $g=(1,1,1,0,0)$. Compute the soft Dice $\frac{2\sum pg}{\sum p+\sum g}$, the soft-Dice loss, and the hard Dice after thresholding at 0.5.

#### D40 · mIoU from a 3×3 confusion matrix · [S10 · Medium · 4 marks]
Rows = GT, columns = prediction, 10,000 pixels:

| GT \ Pred | sky | building | vegetation |
|---|---|---|---|
| **sky** | 4200 | 150 | 50 |
| **building** | 100 | 3300 | 400 |
| **vegetation** | 80 | 520 | 1200 |

**(a)** [0.5] Pixel accuracy.

**(b)** [2] IoU of each class ($\frac{TP}{\text{row}+\text{col}-TP}$) and mIoU.

**(c)** [1] Mean class accuracy (mean per-class recall).

**(d)** [0.5] Frequency-weighted IoU (weights = GT pixel share). Which metric is the most pessimistic and why?

#### D41 · Transposed-convolution output sizes · [S10 · Easy · 4 marks]
Use $W_{out}=(W-1)S-2P+K+P_{out}$, where $P_{out}$ is PyTorch's `output_padding`.

**(a)** [0.5] $7\times7$ input, $K=4$, $S=2$, $P=1$.

**(b)** [1] $7\times7$ input, $K=3$, $S=2$, $P=1$, with output_padding 0 and 1.

**(c)** [0.5] $28\times28$ input, $K=2$, $S=2$, $P=0$.

**(d)** [1] FCN-32s upsamples the $16\times16$ score map of a $512\times512$ image with $K=64$, $S=32$, $P=16$. Output size?

**(e)** [1] Parameters of the layer in (b) mapping 256 to 128 channels (with bias). Which of (a)–(c) is prone to checkerboard artefacts?

#### D42 · U-Net valid-convolution size chain · [S10 · Hard · 5 marks]
A 3-level U-Net: two **unpadded** $3\times3$ convs at every level (including the bottleneck), $2\times2$ max-pool stride 2 (3 times), $2\times2$ up-convolution stride 2 (3 times), skip features **centre-cropped** to the decoder size. Input tile $188\times188$.

**(a)** [2] Sizes along the contracting path down to the bottleneck output.

**(b)** [2] After each up-convolution: size, crop applied to the skip (pixels removed per side) and size after the two decoder convs.

**(c)** [0.5] Final output size, fraction of the tile area that receives a prediction, and the border margin (overlap-tile).

**(d)** [0.5] Is an input of $190\times190$ valid? Why?

#### D43 · RoIAlign bilinear sampling · [S10 · Hard · 5 marks]
A single-channel stride-16 feature map has these values near the RoI ($F(x,y)$: $x$ = column, $y$ = row; feature values sit at integer coordinates, no half-pixel offset):

| | $x=2$ | $x=3$ | $x=4$ |
|---|---|---|---|
| $y=1$ | 12 | 18 | 20 |
| $y=2$ | 16 | 30 | 26 |
| $y=3$ | 10 | 22 | 40 |

RoIAlign: RoI $(36,20,100,76)$ in image pixels, output $2\times2$, $2\times2$ regular sampling points per bin at $(i+0.5)\times\frac{\text{bin size}}{2}$ from the bin's top-left corner, average of the samples.

**(a)** [1] RoI in feature coordinates and the bin size.

**(b)** [1] Coordinates of the four sampling points of the **top-left** bin.

**(c)** [2.5] Bilinearly interpolate each sample and give the bin's output.

**(d)** [0.5] What value would nearest-neighbour sampling give for the first sample? Why does Mask R-CNN care?

#### D44 · Panoptic quality with a borderline match · [S10 · Medium · 4 marks]
**person** (thing): 4 GT segments, 5 predicted segments; candidate same-class pairs have IoU $(p_1,g_1)=0.88$, $(p_2,g_2)=0.74$, $(p_3,g_3)=0.62$, $(p_4,g_4)=0.50$, $(p_5,g_4)=0.30$.
**sky** (stuff): 1 GT, 1 prediction, IoU 0.95. **grass** (stuff): 1 GT, 1 prediction, IoU 0.45.
A pair matches only if IoU **> 0.5**. $PQ=\frac{\sum_{TP}\text{IoU}}{\lvert TP\rvert+\frac12\lvert FP\rvert+\frac12\lvert FN\rvert}=SQ\times RQ$.

**(a)** [1] TP, FP, FN for person.

**(b)** [1.5] SQ, RQ and PQ for person.

**(c)** [1.5] PQ for sky and grass; overall PQ (mean over the 3 classes), $PQ^{Th}$ and $PQ^{St}$.

### S11 · Camera geometry

#### D45 · Projection with K[R|t] and the cheirality check · [S11 · Medium · 5 marks]
Image $1280\times720$,
$K=\begin{bmatrix}1000&0&640\\0&1010&360\\0&0&1\end{bmatrix}$, $R=\begin{bmatrix}1&0&0\\0&0.96&-0.28\\0&0.28&0.96\end{bmatrix}$, $\mathbf t=(0.2,\ -1.0,\ 4.0)^\top$ m, $X_c=RX_w+\mathbf t$.

**(a)** [0.5] Verify that $R$ is a rotation (row norms, $\det R$).

**(b)** [2] Project $X_A=(1,\ 2,\ 3)^\top$ m to pixels.

**(c)** [1.5] Project $X_B=(-1.5,\ 0.5,\ 6)^\top$ m.

**(d)** [1] For $X_C=(0,\ -2,\ -15)^\top$ m compute $X_c$ and the algebraic pixel $(u,v)$. Is the point visible?

#### D46 · Focal length in pixels and field of view · [S11 · Easy · 5 marks]
A machine-vision camera: 12 mm lens, square pixels of pitch 3.45 µm, $2448\times2048$ pixels, principal point at the centre.

**(a)** [1] $f_x$ in pixels and the sensor size in mm.

**(b)** [1.5] Horizontal, vertical and diagonal FOV.

**(c)** [1] Lens focal length needed for a 60° horizontal FOV with this sensor.

**(d)** [1] Ground sampling distance (mm per pixel) on a plane 2 m away, facing the camera; how many pixels does a 50 mm feature span?

**(e)** [0.5] With 2×2 pixel binning: new resolution, $f_x$ and HFOV.

#### D47 · Radial + tangential distortion · [S11 · Medium · 5 marks]
$f_x=f_y=900$ px, $(c_x,c_y)=(640,360)$, $(k_1,k_2,p_1,p_2,k_3)=(0.08,\ -0.01,\ 0.0012,\ -0.0008,\ 0)$. A point has **undistorted** normalised coordinates $(x,y)=(-0.45,\ 0.30)$.

**(a)** [0.5] Ideal (distortion-free) pixel.

**(b)** [3] $r^2$, the radial factor $1+k_1r^2+k_2r^4+k_3r^6$, the radial part, the tangential parts $\delta x=2p_1xy+p_2(r^2+2x^2)$, $\delta y=p_1(r^2+2y^2)+2p_2xy$, and $(x_d,y_d)$.

**(c)** [1.5] Distorted pixel, pixel shift, change in distance from the principal point; barrel or pincushion?

#### D48 · Camera centre and extrinsics · [S11 · Easy · 4 marks]
$R=\begin{bmatrix}0.8&0&-0.6\\0&1&0\\0.6&0&0.8\end{bmatrix}$, $\mathbf t=(1.0,\ -0.5,\ 3.0)^\top$ m, $X_c=RX_w+\mathbf t$.

**(a)** [1.5] Camera centre $C$ in world coordinates; check $RC+\mathbf t=\mathbf 0$.

**(b)** [0.5] Direction of the optical axis in the world frame.

**(c)** [1] Depth $Z_c$ of $X_w=(1,\ 0,\ 5)^\top$, computed two ways.

**(d)** [1] A second camera with the same $R$ is placed at $C_2=(2,\ 1.5,\ -4)^\top$. Give its $\mathbf t_2$.

#### D49 · Ground-plane (Z = 0) homography and back-projection · [S11 · Hard · 5 marks]
A dash-cam: $K=\begin{bmatrix}600&0&320\\0&600&240\\0&0&1\end{bmatrix}$. World: $X$ right, $Y$ forward, $Z$ up, ground $Z=0$. The camera is 1.5 m above the origin, $C=(0,0,1.5)$, pitched down by $\varphi$ with $\sin\varphi=0.6$, $\cos\varphi=0.8$:
$R=\begin{bmatrix}1&0&0\\0&-0.6&-0.8\\0&0.8&-0.6\end{bmatrix}$ ($X_c=RX_w+\mathbf t$).

**(a)** [1] Compute $\mathbf t=-RC$.

**(b)** [1.5] Compute the ground-to-image homography $H=K[\mathbf r_1\ \mathbf r_2\ \mathbf t]$ ($\mathbf r_i$ = columns of $R$).

**(c)** [1] Image positions of the ground points $(X,Y)=(0,2)$ and $(1,5)$ m. Explain the first result.

**(d)** [1.5] Back-project pixel $(400,300)$ onto the ground (intersect the viewing ray with $Z=0$, or use $H^{-1}$).

### S12 · 3D vision & motion

#### D50 · Depth and 3D point from disparity · [S12 · Easy · 4 marks]
Rectified stereo: $f=1050$ px, $B=0.30$ m, $(c_x,c_y)=(640,360)$ for both cameras.

**(a)** [2] A point is at $x_L=700$, $y=400$ in the left image and $x_R=679$ in the right. Disparity, depth $Z$ and $X$, $Y$ in the left-camera frame.

**(b)** [0.5] Expected disparity of a car at 45 m.

**(c)** [0.5] Both images are downsampled by 2. New $f$, new disparity of the point in (a), and its depth.

**(d)** [1] The nearest object of interest is at 2 m. Maximum disparity, and OpenCV's `numDisparities` (must be a multiple of 16).

#### D51 · Depth error and baseline design · [S12 · Medium · 5 marks]
Same rig ($f=1050$ px, $B=0.30$ m, image width 1280 px), sub-pixel matching $\Delta d=0.25$ px, $\Delta Z\approx\frac{Z^2}{fB}\Delta d$.

**(a)** [1.5] $\Delta Z$ and relative error at $Z=10$, 30 and 60 m.

**(b)** [1] Baseline needed for $\Delta Z\le1$ m at 60 m.

**(c)** [1] With that baseline and a 192-px disparity search, what is the minimum measurable depth? Compare with the original rig.

**(d)** [1.5] Alternatively keep $B=0.30$ m and use a longer lens: what $f$ (px) is needed? Compare the horizontal FOV before and after.

#### D52 · Epipolar line from F (forward motion) · [S12 · Medium · 5 marks]
A camera moves straight forward (pure translation); in pixel coordinates

$$
F=\begin{bmatrix}0&-1&300\\1&0&-400\\-300&400&0\end{bmatrix},\qquad \mathbf x'^\top F\mathbf x=0.
$$

**(a)** [1.5] Epipolar line $\mathbf l'=F\mathbf x$ in image 2 for $\mathbf x=(500,350)$; write it as $y=mx+c$.

**(b)** [1.5] Distances of the candidates $\mathbf x_1'=(560,382)$ and $\mathbf x_2'=(540,395)$ from $\mathbf l'$; accept with a 2-px threshold?

**(c)** [1] Verify $F\mathbf e=\mathbf 0$ and $F^\top\mathbf e'=\mathbf 0$ for $\mathbf e=\mathbf e'=(400,300,1)$. What is this point physically?

**(d)** [1] Rank of $F$; DOF of a general $F$.

#### D53 · Essential matrix check · [S12 · Medium · 5 marks]
$X_2=RX_1+\mathbf t$ with $R=\begin{bmatrix}0.8&0&0.6\\0&1&0\\-0.6&0&0.8\end{bmatrix}$, $\mathbf t=(1,0,0)^\top$.

**(a)** [1] Compute $E=[\mathbf t]_\times R$.

**(b)** [1] Singular values and $\det E$. Is $E$ a valid essential matrix?

**(c)** [1.5] For $X_1=(0.5,\ -0.2,\ 4)^\top$ compute $X_2$, $\hat{\mathbf x}_1$, $\hat{\mathbf x}_2$ and verify $\hat{\mathbf x}_2^\top E\hat{\mathbf x}_1=0$.

**(d)** [0.5] An 8-point estimate has singular values $(2.1,\ 1.7,\ 0.3)$. What are the singular values of the closest valid essential matrix?

**(e)** [1] Is $M=\begin{bmatrix}0&-2&0\\2&0&-1\\0&1&0\end{bmatrix}$ a valid essential matrix? If so, describe the motion.

#### D54 · Lucas–Kanade from per-pixel gradients · [S12 · Medium · 5 marks]
A 5-pixel window has $(I_x,I_y,I_t)$: $(4,1,-6)$, $(3,2,-7)$, $(-1,3,-2)$, $(2,-2,1)$, $(5,0,-8)$.

**(a)** [1.5] Compute the five sums and write the LK normal equations.

**(b)** [1.5] Solve for $(u,v)$.

**(c)** [1] Eigenvalues of $A^\top A$; is the window good to track?

**(d)** [1] Another window has $(I_x,I_y)=(2,1),(4,2),(-2,-1),(6,3)$. Compute $A^\top A$, its determinant and eigenvalues; what can be recovered?

#### D55 · Normal flow and the aperture problem · [S12 · Easy · 4 marks]
At a pixel $I_x=-6$, $I_y=8$, $I_t=20$.

**(a)** [1.5] Normal flow vector $\mathbf u_n=-\frac{I_t}{\Vert\nabla I\Vert^2}\nabla I$ and its magnitude.

**(b)** [1.5] Show that both $(-2,-4)$ and $(2.8,-0.4)$ satisfy the brightness-constancy constraint. Decompose $(-2,-4)$ into normal and tangential parts.

**(c)** [1] A neighbouring pixel with the same motion has $I_x=5$, $I_y=0$, $I_t=-4$. Solve for the full flow.

#### D56 · Bundle-adjustment problem size · [S12 · Medium · 4 marks]
SfM with 120 images from one camera: shared unknown intrinsics $(f,k_1,k_2)$, 6 pose parameters per image, 25,000 points, each seen in 5 images on average.

**(a)** [1.5] Number of unknowns, observations and scalar residuals.

**(b)** [0.5] Gauge freedom of monocular SfM and the effective number of unknowns.

**(c)** [1] Non-zeros per Jacobian row, total non-zeros, and the Jacobian density.

**(d)** [1] Size of the reduced camera system after Schur-complementing the points.

### S13–14 · Foundation models & pipelines

#### D57 · CLIP zero-shot with a logit scale · [S13 · Easy · 4 marks]
Cosine similarities of an image with "a photo of a {husky / wolf / fox}." are $0.28,\ 0.26,\ 0.21$. CLIP's logit scale is 100.

**(a)** [1.5] Softmax probabilities and prediction.

**(b)** [1] Repeat with logit scale 50.

**(c)** [1] For a 2-class problem at scale 100, what cosine gap between the classes gives 95% confidence? 90%?

**(d)** [0.5] Prompt ensembling: two unit "husky" prompt embeddings with cosine 0.9 between them have image similarities 0.28 and 0.30. The ensemble embedding is their mean, re-normalised. What is its similarity with the image?

#### D58 · InfoNCE matrix sizes at CLIP scale · [S13 · Medium · 4 marks]
Global batch $N=4096$ image–text pairs, 512-d embeddings, 8 GPUs with 512 pairs each (embeddings all-gathered). MB = $10^6$ B.

**(a)** [1.5] Entries of the logit matrix; positives; negatives per row; total negatives (one direction).

**(b)** [1] fp32 memory of the full matrix, and of one GPU's $512\times4096$ block.

**(c)** [1] MACs to compute the full similarity matrix and per GPU.

**(d)** [0.5] Loss per direction at initialisation if all logits are equal.

#### D59 · Scale-and-shift alignment of relative depth (least squares, 3 points) · [S13 · Medium · 5 marks]
Depth Anything outputs relative inverse depth $r$. Three LiDAR pixels give $(r,Z)$: $(0.9,\ 1.6\text{ m})$, $(0.5,\ 3.2\text{ m})$, $(0.2,\ 10.0\text{ m})$. Fit $1/Z=s\,r+t$ by least squares **on $1/Z$**.

**(a)** [0.5] The targets $y=1/Z$.

**(b)** [2] Normal equations $\begin{bmatrix}\sum r^2&\sum r\\\sum r&n\end{bmatrix}\begin{bmatrix}s\\t\end{bmatrix}=\begin{bmatrix}\sum ry\\\sum y\end{bmatrix}$; solve for $s,t$.

**(c)** [1] Fitted depths at the three points and their errors; comment on the far point.

**(d)** [1.5] Depth at pixels with $r=0.35$ and $r=0.06$.

#### D60 · Object size and 3D distance from a box + metric depth · [S14 · Easy · 4 marks]
$1920\times1080$ camera, $f_x=f_y=1400$ px, $(c_x,c_y)=(960,540)$. Grounding DINO returns a "person" box $(900,300,1000,720)$; metric depth at the person is 5.0 m. The person's centre pixel is $(950,510)$; a "dog" is centred at pixel $(1300,800)$ with depth 4.0 m.

**(a)** [1.5] Real height and width of the person.

**(b)** [2] 3D positions of both centres (camera frame) and the person–dog distance.

**(c)** [0.5] If the depth were 10% too large, how would the estimated height change?

#### D61 · Detection → segmentation → depth pipeline latency · [S14 · Medium · 4 marks]
On one GPU: Grounding DINO 120 ms/frame; SAM ViT-B image encoder 100 ms/frame; SAM mask decoder 10 ms per box; Depth Anything 25 ms/frame; post-processing 5 ms/frame. 4 boxes per frame.

**(a)** [1] Sequential latency and FPS.

**(b)** [1] Depth runs on a second GPU in parallel with Grounding DINO + SAM; post-processing waits for both. Latency and FPS.

**(c)** [1] Three-GPU pipeline (stage 1: Grounding DINO; stage 2: SAM encoder + decoder; stage 3: depth + post-processing, on consecutive frames). Throughput and latency.

**(d)** [1] Keyframe strategy on one GPU: the full pipeline every 10th frame; other frames run a box tracker (8 ms) + depth + post-processing. Average time per frame and FPS. Can it keep up with a 30-FPS camera?

### S15 · Generative CV

#### D62 · Noise schedule: ᾱ_t and x_t · [S15 · Medium · 5 marks]
$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$, $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$.

**(a)** [1] Constant schedule $\beta_t=0.02$: $\bar\alpha_{10}$, $\bar\alpha_{100}$ and $\mathrm{SNR}=\bar\alpha/(1-\bar\alpha)$ at both.

**(b)** [1.5] $x_{10}$ for $x_0=(0.6,\ -0.8)$, $\epsilon=(1.0,\ 0.5)$.

**(c)** [1] Smallest $t$ with $\bar\alpha_t<0.01$ (signal std below 0.1).

**(d)** [1.5] Standard linear schedule ($T=1000$, $\beta$ from $10^{-4}$ to $0.02$): using $\bar\alpha_t\approx\exp(-\sum_{s\le t}\beta_s)$, estimate $\bar\alpha_{1000}$ and $\bar\alpha_{500}$ ($\beta_{500}\approx0.01004$). The exact values are $4.04\times10^{-5}$ and $0.0786$ — comment.

#### D63 · Classifier-free guidance combinations · [S15 · Easy · 4 marks]
For four latent entries: $\epsilon_\theta(x_t,\varnothing)=(0.30,\ -0.10,\ 0.05,\ 0.20)$, $\epsilon_\theta(x_t,c)=(0.10,\ 0.20,\ 0.05,\ -0.10)$.

**(a)** [1.5] $\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+w(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing))$ for $w=5$.

**(b)** [1] The original CFG paper writes $\tilde\epsilon=(1+w')\epsilon_\theta(x_t,c)-w'\epsilon_\theta(x_t,\varnothing)$. Which $w'$ gives the same result? Verify.

**(c)** [1] A negative prompt replaces the unconditional prediction by $\epsilon_\theta(x_t,c_{neg})=(0.25,\ 0.0,\ 0.10,\ 0.15)$. Compute $\hat\epsilon$ for $w=5$.

**(d)** [0.5] U-Net evaluations for a 50-step sampler with CFG.

#### D64 · VAE KL term and reparameterised sample · [S15 · Medium · 4 marks]
Encoder output for one image: $\mu=(0.8,\ -0.3,\ 0.0)$, $\log\sigma^2=(-1.0,\ 0.5,\ 0.0)$; noise $\epsilon=(-0.5,\ 1.2,\ 0.7)$; prior $\mathcal N(0,I)$.

**(a)** [0.5] $\sigma$ per dimension.

**(b)** [2] KL per dimension $\tfrac12(\mu^2+\sigma^2-\log\sigma^2-1)$ and total (nats).

**(c)** [1] Reparameterised sample $z=\mu+\sigma\odot\epsilon$.

**(d)** [0.5] Loss of a $\beta$-VAE with $\beta=4$ if the reconstruction term is 12.4. What does dimension 3 tell you?

#### D65 · GAN optimal discriminator and value · [S15 · Medium · 5 marks]
Outcomes $\{a,b,c,d\}$: $p_{data}=(0.4,\ 0.4,\ 0.2,\ 0)$, $p_g=(0,\ 0.2,\ 0.3,\ 0.5)$. Natural logs; $0\log0=0$.

**(a)** [1] $D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$ for each outcome.

**(b)** [2] $V(D^*,G)=\sum_xp_{data}\log D^*+\sum_xp_g\log(1-D^*)$.

**(c)** [1] JSD from $V=2\,\mathrm{JSD}-\log4$.

**(d)** [1] Repeat (a)–(c) for disjoint supports $p_g=(0,0,0,1)$, and give $V$ when $p_g=p_{data}$. Why is the disjoint case bad for training $G$?

#### D66 · Latent compression factors · [S15 · Easy · 4 marks]
**(a)** [1] Stable Diffusion VAE ($f=8$, 4 latent channels) at $768\times768$ RGB: latent shape, values, compression factor.

**(b)** [1] Same VAE at $1024\times1024$.

**(c)** [1.5] A DiT works on a $32\times32\times4$ latent ($256\times256$ images). Tokens and attention entries per head for patch sizes $p=2,4,8$; compare with a pixel-space ViT with $p=8$ on $256\times256$.

**(d)** [0.5] A 16-channel $f=8$ latent at $512\times512$: compression factor.

#### D67 · FID in one dimension from samples · [S15 · Medium · 5 marks]
A toy 1-D "Inception feature". Use the sample mean and the **unbiased** sample variance ($n-1$), as `np.cov` does in FID code.
Real: 1.2, 0.8, 1.5, 0.9, 1.1, 0.7 · Model A: 1.0, 1.3, 0.6, 1.4, 0.9, 1.2 · Model B: 1.6, 1.7, 1.5, 1.6, 1.8, 1.6.

**(a)** [1] Show that in 1-D, $\mathrm{FID}=(\mu_r-\mu_g)^2+(\sigma_r-\sigma_g)^2$.

**(b)** [2] Mean, variance and standard deviation of each set.

**(c)** [1.5] FID of A and of B.

**(d)** [0.5] Interpret Model B.

---

## Part B — Solutions

### S01 · Image processing & features

#### D1 — Correlation vs convolution at a pixel

**Given:** $I$ (5×5), $K=\begin{bmatrix}0&1&2\\-1&0&1\\-2&-1&0\end{bmatrix}$, pixel $(x,y)=(2,1)$.

**Formula:**

$$
\text{corr: } g(x,y)=\sum_{i,j}K(i,j)\,I(x+i,y+j),\qquad \text{conv: } g(x,y)=\sum_{i,j}K(i,j)\,I(x-i,y-j)=\text{corr with }K\text{ rotated }180^\circ .
$$

**Working:**
- (a) Neighbourhood of $(2,1)$ = rows $y=0..2$, columns $x=1..3$: $\begin{bmatrix}20&30&40\\60&90&100\\70&120&140\end{bmatrix}$.
  Row by row: $(0\cdot20+1\cdot30+2\cdot40)=110$; $(-60+0+100)=40$; $(-140-120+0)=-260$. Sum $=110+40-260=\mathbf{-110}$.
- (b) $K$ rotated by 180° is $\begin{bmatrix}0&-1&-2\\1&0&-1\\2&1&0\end{bmatrix}=-K$, so convolution $=-(\text{correlation})=\mathbf{+110}$. For an **anti-symmetric** kernel (rot180 = −K) conv and corr differ only in sign; for a symmetric kernel (Gaussian, box, Laplacian) they are identical.
- (c) (i) zero padding: neighbourhood $\begin{bmatrix}0&0&0\\0&10&20\\0&15&60\end{bmatrix}$ → $0+(0+0+20)+(0-15+0)=\mathbf{5}$.
  (ii) replicate: $\begin{bmatrix}10&10&20\\10&10&20\\15&15&60\end{bmatrix}$ → $(0+10+40)+(-10+0+20)+(-30-15+0)=50+10-45=\mathbf{15}$.
- (d) $5-3+1=\mathbf{3}$ → a $3\times3$ output.

**Answer:** **corr $=-110$; conv $=+110$; corner: 5 (zero pad), 15 (replicate); valid output $3\times3$.**
> ✔ Verified numerically (Python, `scipy.ndimage.correlate/convolve`).
> **Common slip:** forgetting that the kernel's top-left weight multiplies the **top-left** neighbour in correlation; or flipping only rows (not rows and columns) for convolution.

#### D2 — Sobel gradient, magnitude, direction

**Given:** $N=\begin{bmatrix}200&180&120\\190&120&60\\130&70&40\end{bmatrix}$, Sobel as correlation, $y$ down.

**Formula:** $I_x=\sum K_x\odot N$, $I_y=\sum K_y\odot N$, $\Vert\nabla I\Vert=\sqrt{I_x^2+I_y^2}$, $\theta=\operatorname{atan2}(I_y,I_x)$.

**Working:**
- (a) $I_x=(\text{right column}) - (\text{left column})$ with weights 1-2-1: $(120+2\cdot60+40)-(200+2\cdot190+130)=280-710=\mathbf{-430}$.
  $I_y=(\text{bottom row})-(\text{top row})$: $(130+2\cdot70+40)-(200+2\cdot180+120)=310-680=\mathbf{-370}$.
- (b) L2: $\sqrt{430^2+370^2}=\sqrt{184900+136900}=\sqrt{321800}=\mathbf{567.27}$; L1: $430+370=\mathbf{800}$.
  $\theta=\operatorname{atan2}(-370,-430)=\mathbf{-139.29^\circ}\equiv\mathbf{220.71^\circ}$ → the gradient points **up-left** (towards the bright top-left corner; negative $I_y$ = up because $y$ points down).
  The edge is perpendicular: $220.71^\circ-90^\circ=\mathbf{130.71^\circ}$ (≡ $-49.29^\circ$), i.e. it runs **down-left ↔ up-right** (a "/" diagonal separating the bright top-left from the dark bottom-right).
- (c) $\theta\bmod180^\circ=40.71^\circ\in[22.5^\circ,67.5^\circ)$ → **45° bin**; compare with the neighbours along the gradient line: **$(x+1,y+1)$ and $(x-1,y-1)$** (bottom-right and top-left).
- (d) $I_x/8=-53.75$, $I_y/8=-46.25$, magnitude $567.27/8=\mathbf{70.91}$ levels/pixel.

**Answer:** **$I_x=-430$, $I_y=-370$; $\Vert\nabla I\Vert=567.3$ (L1 800); $\theta=-139.3^\circ=220.7^\circ$; edge along 130.7° ("/"); 45° bin, neighbours $(x\pm1,y\pm1)$; 70.9 levels/px.**
> ✔ Verified numerically (Python and `cv2.Sobel`).
> **Common slip:** using `atan` instead of `atan2` (gives $+40.7^\circ$ and loses the quadrant); or comparing NMS neighbours along the **edge** instead of along the **gradient**.

#### D3 — Harris response from raw gradients

**Given:** four $(I_x,I_y)$ per window, $w=1$, $k=0.05$.

**Formula:**

$$
M=\begin{bmatrix}\sum I_x^2&\sum I_xI_y\\\sum I_xI_y&\sum I_y^2\end{bmatrix},\qquad R=\det M-k(\operatorname{tr}M)^2=\lambda_1\lambda_2-k(\lambda_1+\lambda_2)^2 .
$$

**Working:**
- (a) W1: $\sum I_x^2=100+64+4+1=169$, $\sum I_y^2=4+1+81+121=207$, $\sum I_xI_y=20-8+18-11=19$ → $M_1=\begin{bmatrix}169&19\\19&207\end{bmatrix}$.
  W2: $\sum I_x^2=100+64+36+144=344$, $\sum I_y^2=25+16+9+36=86$, $\sum I_xI_y=50+32+18+72=172$ → $M_2=\begin{bmatrix}344&172\\172&86\end{bmatrix}$.
  W3: $\sum I_x^2=1+0+1+1=3$, $\sum I_y^2=0+1+1+1=3$, $\sum I_xI_y=0+0-1+1=0$ → $M_3=\begin{bmatrix}3&0\\0&3\end{bmatrix}$.
- (b)

| Window | $\det M$ | $\operatorname{tr}M$ | $k(\operatorname{tr}M)^2$ | $R$ | Class |
|---|---|---|---|---|---|
| W1 | $169\cdot207-19^2=34983-361=34622$ | 376 | 7068.8 | **27 553.2** | **corner** |
| W2 | $344\cdot86-172^2=29584-29584=0$ | 430 | 9245 | **−9245** | **edge** |
| W3 | $9$ | 6 | 1.8 | **7.2** | **flat** |

- (c) W1: $\lambda=\frac{376\pm\sqrt{(169-207)^2+4\cdot19^2}}{2}=\frac{376\pm53.74}{2}$ → **161.13, 214.87**; Shi–Tomasi **161.13** (both large → corner). W2: $\lambda=$ **0 and 430**, Shi–Tomasi **0**. All four W2 gradients are multiples of $(2,1)$, so $M=\sum\mathbf g\mathbf g^\top$ is a sum of rank-1 matrices in the **same** direction → rank 1, $\det=0$: an edge (gradient in one direction only).
- (d) Doubling intensities doubles every gradient → $M\times4$ → $\det\times16$, $(\operatorname{tr})^2\times16$ → $R\times16=\mathbf{440\,851.2}$. Harris is **not** invariant to intensity scaling (only to additive shifts), so a fixed threshold changes its answers.

**Answer:** **$R_1=27553.2$ (corner), $R_2=-9245$ (edge), $R_3=7.2$ (flat); $\lambda(W1)=161.1, 214.9$; $\lambda(W2)=0, 430$; doubled: $R=440851.2$.**
> ✔ Verified numerically (Python).
> **Common slip:** computing $\sum I_x\cdot\sum I_y$ instead of $\sum(I_xI_y)$ — the cross term must be summed **per pixel**.

#### D4 — Gaussian kernel size and separable cost

**Given:** $K=2\lceil3\sigma\rceil+1$; $1280\times720$ image.

**Formula:** direct 2-D: $K^2$ mult/pixel; separable: $2K$ mult/pixel; cascade $\sigma=\sqrt{\sigma_1^2+\sigma_2^2}$.

**Working:**
- (a) $\sigma=1$: $K=2\cdot3+1=\mathbf7$ ($x=-3..3$). Unnormalised $e^{-x^2/2}$: $1,\ 0.60653,\ 0.13534,\ 0.01111$ (for $\lvert x\rvert=0,1,2,3$). Sum $=1+2(0.60653+0.13534+0.01111)=2.50595$.
  Normalised: **$g(0)=0.3991$, $g(\pm1)=0.2420$, $g(\pm2)=0.0540$, $g(\pm3)=0.0044$** (sum = 1).
- (b) $\sigma=1.5$: $K=2\lceil4.5\rceil+1=\mathbf{11}$. Direct $11^2=\mathbf{121}$ mult/pixel; separable $2\cdot11=\mathbf{22}$; speed-up $121/22=\mathbf{5.5\times}$. Pixels $=1280\cdot720=921\,600$ → direct **111,513,600**, separable **20,275,200** multiplications.
- (c) $\sigma=\sqrt{1.5^2+2^2}=\sqrt{6.25}=\mathbf{2.5}$. Cascade: $K=11$ then $K=2\lceil6\rceil+1=13$ → $22+26=\mathbf{48}$ mult/pixel. One blur at $\sigma=2.5$: $K=2\lceil7.5\rceil+1=17$ → $\mathbf{34}$ mult/pixel → the single blur is cheaper (cascading is useful only when the intermediate image is needed anyway, as in a scale-space pyramid).

**Answer:** **(a) $K=7$, weights 0.3991 / 0.2420 / 0.0540 / 0.0044; (b) $K=11$, 121 vs 22 mult/px, 111.5 M vs 20.3 M, 5.5×; (c) $\sigma=2.5$; cascade 48 vs single 34 mult/px.**
> ✔ Verified numerically (Python; cascade equivalence checked with `scipy.ndimage.gaussian_filter`).
> **Common slip:** adding variances wrongly ($\sigma=1.5+2=3.5$); the **variances** add, not the σ's.

#### D5 — Integral image lookups

**Given:** $ii$ (5×5); $ii(x,y)$ with $x$ = column, $y$ = row.

**Formula:** rectangle $[x_0,x_1]\times[y_0,y_1]$:

$$
S=ii(x_1,y_1)-ii(x_0-1,y_1)-ii(x_1,y_0-1)+ii(x_0-1,y_0-1).
$$

**Working:**
- (a) $x_0=2,x_1=4,y_0=1,y_1=3$: $S=ii(4,3)-ii(1,3)-ii(4,0)+ii(1,0)=71-24-12+5=\mathbf{40}$; 9 pixels → mean $=40/9=\mathbf{4.444}$.
- (b) A single pixel is a 1×1 rectangle: $I(3,2)=ii(3,2)-ii(2,2)-ii(3,1)+ii(2,1)=42-28-24+18=\mathbf{8}$.
- (c) Row 4 = (whole image) − (rows 0–3): $ii(4,4)-ii(4,3)=93-71=\mathbf{22}$.
- (d) Direct: 81 values → **80 additions/pixel** → $80\times2\,073\,600=\mathbf{165{,}888{,}000}$. Integral image: 4 lookups → **3 add/sub per pixel** → **6,220,800**; speed-up $80/3\approx\mathbf{26.7\times}$. Including building $ii$ (≈2 additions/pixel, 4,147,200): $165.9\text{M}/10.37\text{M}=\mathbf{16\times}$ — and the cost per pixel is **independent of the box size** (the key to Viola–Jones and SURF).

**Answer:** **sum 40, mean 4.44; $I(3,2)=8$; bottom row 22; 80 vs 3 ops/px (≈26.7×, 16× incl. building $ii$).**
> ✔ Verified numerically (Python; $ii$ rebuilt from the original image with `cumsum`).
> **Common slip:** using $ii(x_0,\cdot)$ instead of $ii(x_0-1,\cdot)$ — the subtracted corners lie **outside** the rectangle.

### S02 · Feature matching & classical CV

#### D6 — RANSAC iteration counts (incl. a budget)

**Given:** $(s,e,p)$ per part.

**Formula:**

$$
N=\left\lceil\frac{\log(1-p)}{\log\left(1-(1-e)^s\right)}\right\rceil,\qquad\text{inverse: }(1-e)^s=1-(1-p)^{1/N}.
$$

**Working:**
- (a) $(1-e)^s=0.5^2=0.25$; $N=\lceil\ln0.01/\ln0.75\rceil=\lceil-4.60517/-0.28768\rceil=\lceil16.008\rceil=\mathbf{17}$.
- (b) $0.65^3=0.274625$; $\ln(0.725375)=-0.32107$; $N=\lceil4.60517/0.32107\rceil=\lceil14.34\rceil=\mathbf{15}$.
- (c) $0.55^5=0.050328$; $\ln(0.949672)=-0.051639$; $\ln(0.001)=-6.90776$; $N=\lceil133.77\rceil=\mathbf{134}$.
- (d) $(1-p)^{1/500}=0.01^{0.002}=e^{-0.0092103}=0.990832$ → $(1-e)^4\ge0.009168$ → $1-e\ge0.009168^{1/4}=0.30943$ → $e\le\mathbf{0.6906}$ (≈ 69% outliers).
  Check: $e=0.69$: $0.31^4=0.0092352$ → $N=\lceil4.60517/0.0092781\rceil=\lceil496.3\rceil=497\le500$ ✓; $e=0.70$: $0.3^4=0.0081$ → $N=\lceil566.2\rceil=567>500$ ✗.
  At $e=0.75$: $p=1-(1-0.25^4)^{500}=1-(0.99609)^{500}=\mathbf{0.859}$ — only 86% chance of ever drawing a clean sample.

**Answer:** **17; 15; 134; $e_{max}\approx0.69$ (497 vs 567 iterations at 0.69/0.70); $p=0.859$ at $e=0.75$.**
> ✔ Verified numerically (Python).
> **Common slip:** rounding 16.008 **down** to 16 — always round up; and using $e$ (outlier ratio) where $1-e$ (inlier ratio) belongs.

#### D7 — Applying a homography and the inlier test

**Given:** $H$, points, observed matches, $t=3$ px.

**Formula:** $\tilde{\mathbf x}'=H\tilde{\mathbf x}$, $\mathbf x'=\left(\frac{\tilde x_1'}{\tilde x_3'},\frac{\tilde x_2'}{\tilde x_3'}\right)$; transfer error $\Vert\mathbf x'-\mathbf p'\Vert$.

**Working:**
- (a) $\mathbf p_1$: $\tilde x_1'=0.9(320)+0.05(240)+25=288+12+25=325$; $\tilde x_2'=-0.04(320)+1.1(240)-10=-12.8+264-10=241.2$; $\tilde x_3'=-0.096+0.048+1=0.952$ → $\mathbf{(341.39,\ 253.36)}$.
  $\mathbf p_2$: $540+4+25=569$; $-24+88-10=54$; $-0.18+0.016+1=0.836$ → $\mathbf{(680.62,\ 64.59)}$.
- (b) $\mathbf p_1'-H\mathbf p_1=(1.61,\ -1.16)$ → error $\sqrt{1.61^2+1.16^2}=\mathbf{1.99}$ px $<3$ → **inlier**.
  $\mathbf p_2'-H\mathbf p_2=(-4.02,\ 3.01)$ → error $\mathbf{5.02}$ px $>3$ → **outlier**.
- (c) $\tilde x_3'=0$ ⇔ $-0.0003x+0.0002y+1=0$ ⇔ $y=1.5x-5000$. For $0\le y\le720$ this needs $3333\le x\le3813$ — far outside the 1280-px-wide image, so **no image-1 pixel maps to infinity** (that line is the image-1 line mapped to the line at infinity of image 2).
- (d) $2H$ multiplies all three homogeneous coordinates by 2 → **same** points. $H$ is defined only up to scale → 9 entries − 1 = **8 DOF**.

**Answer:** **$H\mathbf p_1=(341.39,253.36)$, $H\mathbf p_2=(680.62,64.59)$; errors 1.99 px (inlier) and 5.02 px (outlier); the line $y=1.5x-5000$ (outside the image) maps to infinity; $2H\equiv H$, 8 DOF.**
> ✔ Verified numerically (Python).
> **Common slip:** forgetting to divide by the third coordinate (0.952, 0.836) — the perspective row is exactly what makes a homography non-affine.

#### D8 — DLT matrix size and Hartley normalisation

**Given:** $n$ correspondences; four points for normalisation.

**Formula:** each correspondence gives 2 rows of $A$ → $A\in\mathbb R^{2n\times9}$; $T=\begin{bmatrix}s&0&-s\bar x\\0&s&-s\bar y\\0&0&1\end{bmatrix}$, $s=\sqrt2/\bar d$.

**Working:**
- (a) $n=4$: $\mathbf{8\times9}$; $n=6$: $\mathbf{12\times9}$; $n=50$: $\mathbf{100\times9}$. Noise-free, non-degenerate: **rank 8** (1-D null space = $\mathbf h$ up to scale). $\mathbf h$ = right singular vector of the **smallest singular value** of $A$ (minimises $\Vert A\mathbf h\Vert$ subject to $\Vert\mathbf h\Vert=1$).
- (b) Affine: 6 unknowns, each correspondence gives 2 equations → **3 correspondences** (non-collinear).
- (c) Centroid $=\left(\frac{120+520+500+100}{4},\frac{80+100+380+360}{4}\right)=\mathbf{(310,\ 230)}$.
  Offsets $(-190,-150)$, $(210,-130)$, $(190,150)$, $(-210,130)$ → distances **242.07, 246.98, 242.07, 246.98**, mean $\bar d=\mathbf{244.53}$.
  $s=\sqrt2/244.53=\mathbf{0.005783}$; $T=\begin{bmatrix}0.005783&0&-1.7929\\0&0.005783&-1.3302\\0&0&1\end{bmatrix}$.
  First point → $(0.005783\cdot120-1.7929,\ 0.005783\cdot80-1.3302)=\mathbf{(-1.0989,\ -0.8675)}$ (mean distance of all four is now exactly $\sqrt2$).
- (d) Denormalise: $\mathbf{H=T'^{-1}\hat HT}$.

**Answer:** **8×9, 12×9, 100×9; rank 8; $\mathbf h$ = last right singular vector; affine: 6 unknowns, 3 points; centroid (310, 230), $\bar d=244.53$, $s=0.005783$, first point → (−1.099, −0.868); $H=T'^{-1}\hat HT$.**
> ✔ Verified numerically (Python; full DLT with SVD recovers the true $H$ exactly from 4 points).
> **Common slip:** scaling so that the **maximum** (or RMS) distance is $\sqrt2$ — Hartley uses the **mean** distance.

#### D9 — Hamming distance and the ratio test for ORB

**Given:** hex fragments; $(d_1,d_2)$ pairs; 256-bit descriptors.

**Formula:** Hamming $=\operatorname{popcount}(a\oplus b)$; accept if $d_1/d_2<\tau$; for random bits $d\sim\text{Bin}(256,0.5)$.

**Working:**
- (a) `B3C5` = `1011 0011 1100 0101`, `9AE1` = `1001 1010 1110 0001`; XOR = `0010 1001 0010 0100` → $1+2+1+1=\mathbf5$.
- (b) q1: $34/51=0.667<0.75$ ✓; q2: $62/70=0.886$ ✗; q3: $90/150=0.600$ ✓. **Accepted: q1, q3.** q3 passes the ratio test although its best distance (90) is large — the ratio is only a *distinctiveness* test; many pipelines add an absolute cap (e.g. $d_1<64$) and/or a cross-check.
- (c) Mean $=256\times0.5=\mathbf{128}$; std $=\sqrt{256\times0.5\times0.5}=\mathbf8$. q1: $(128-34)/8=\mathbf{11.75\sigma}$ below chance → almost certainly a true match (q3: $4.75\sigma$).
- (d) ORB: 256 bits = **32 bytes**; SIFT: $128\times4=$ **512 bytes** (16× more, and L2 instead of XOR + popcount).

**Answer:** **5; accept q1 and q3 (q2 rejected); mean 128, std 8, q1 is 11.75σ below chance; 32 B vs 512 B.**
> ✔ Verified numerically (Python).
> **Common slip:** using L2 distance or the ratio test threshold the wrong way round ($d_2/d_1$).

### S03 · CNNs under the hood

#### D10 — Output sizes through a stack (odd input)

**Given:** stack L1–L7, input 225 (and 224).

**Formula:** $W_{out}=\left\lfloor\frac{W-K_{eff}+2P}{S}\right\rfloor+1$, $K_{eff}=D(K-1)+1$.

**Working (225):**

| Layer | Computation | Output |
|---|---|---|
| L1 | $\lfloor(225-7+6)/2\rfloor+1=112+1$ | $113\times113\times64$ |
| L2 | $\lfloor(113-3+2)/2\rfloor+1=56+1$ | $57\times57\times64$ |
| L3 | $\lfloor(57-3+2)/2\rfloor+1=28+1$ | $29\times29\times128$ |
| L4 | $K_{eff}=5$: $(29-5+4)/1+1$ | $29\times29\times128$ |
| L5 | $1\times1$ | $29\times29\times256$ |
| L6 | $\lfloor(29-3)/2\rfloor+1=13+1$ | $14\times14\times256$ |
| L7 | global average pool | $1\times1\times256$ |

**(b) 224:** L1 $\lfloor223/2\rfloor+1=\mathbf{112}$; L2 $\lfloor111/2\rfloor+1=\mathbf{56}$; L3 $\lfloor55/2\rfloor+1=\mathbf{28}$; L4 **28**; L5 **28**; L6 $\lfloor25/2\rfloor+1=\mathbf{13}$; L7 $1\times1\times256$. The floor discards a fractional part at **L1, L2, L3 and L6** for 224 (odd numerators), so the two inputs differ by one cell after L1 and the gap persists to $14$ vs $13$.

**Answer:** **225: 113×113×64 → 57×57×64 → 29×29×128 → 29×29×128 → 29×29×256 → 14×14×256 → 1×1×256; 224: 112 → 56 → 28 → 28 → 28 → 13 → 1.**
> ✔ Verified numerically (PyTorch forward pass).
> **Common slip:** using $K$ instead of $K_{eff}=5$ for the dilated L4 (would give 31 instead of 29).

#### D11 — 1×1 bottleneck vs direct 3×3 — parameters

**Given:** $14\times14\times480$ input, 208 output channels, reduce to 96.

**Formula:** params $=(K_hK_wC_{in}+1)\,C_{out}$.

**Working:**
- (a) $(9\cdot480+1)\cdot208=4321\cdot208=\mathbf{898{,}768}$.
- (b) $1\times1$: $(480+1)\cdot96=\mathbf{46{,}176}$; $3\times3$: $(9\cdot96+1)\cdot208=865\cdot208=\mathbf{179{,}920}$; total **226,096** → $898768/226096=\mathbf{3.98\times}$ fewer.
- (c) A $1\times1$ conv takes the $C_{in}$-vector at each pixel and multiplies it by the same $C_{out}\times C_{in}$ matrix (+ bias), i.e. a per-pixel FC layer (no spatial mixing, followed by ReLU → extra non-linearity). Here the weight matrix is **$96\times480$** (+96 biases).

**Answer:** **898,768 vs 46,176 + 179,920 = 226,096 (≈ 4× fewer); 1×1 = per-pixel FC with a 96×480 matrix.**
> ✔ Verified numerically (PyTorch parameter count).
> **Common slip:** forgetting that the $3\times3$ after the bottleneck sees only **96** input channels, not 480.

#### D12 — MACs, FLOPs and latency

**Given:** D11 layers on $14\times14$ outputs ("same" padding).

**Formula:** MACs $=H_{out}W_{out}K_hK_wC_{in}C_{out}$; FLOPs ≈ 2 MACs.

**Working:**
- (a) $196\cdot9\cdot480\cdot208=\mathbf{176{,}117{,}760}$ (≈176.1 M).
- (b) $1\times1$: $196\cdot480\cdot96=9{,}031{,}680$; $3\times3$: $196\cdot9\cdot96\cdot208=35{,}223{,}552$; total **44,255,232** (≈44.3 M) → $\mathbf{3.98\times}$ fewer.
- (c) 352.2 MFLOP → $352.2\times10^6/2\times10^{12}=\mathbf{0.176}$ ms; 88.5 MFLOP → **0.044 ms**.
- (d) FC $4096\to4096$: **16,777,216 MACs** and **16,781,312 params** — every FC weight is used **once** per image. A conv weight is reused at every output position (196× here), so convs have few parameters but many MACs; FCs have many parameters but comparatively few MACs (VGG-16: ~90% of the parameters in FC layers, >90% of the compute in conv layers).

**Answer:** **176.1 M vs 44.3 M MACs (3.98×); 0.176 ms vs 0.044 ms; FC: 16.8 M MACs = 16.8 M params.**
> ✔ Verified numerically (PyTorch, MAC count from output size × kernel volume).
> **Common slip:** using the **input** spatial size for a strided layer — MACs use $H_{out}W_{out}$.

#### D13 — Receptive field with pooling and dilation

**Given:** 6 layers; $r_0=1$, $j_0=1$.

**Formula:** $r_l=r_{l-1}+(K_{eff,l}-1)\,j_{l-1}$, $j_l=j_{l-1}S_l$.

**Working:**

| Layer | $K_{eff}$ | $S$ | $r_l$ | $j_l$ |
|---|---|---|---|---|
| L1 conv 3×3 | 3 | 1 | $1+2\cdot1=3$ | 1 |
| L2 conv 3×3, S2 | 3 | 2 | $3+2\cdot1=5$ | 2 |
| L3 pool 2×2, S2 | 2 | 2 | $5+1\cdot2=7$ | 4 |
| L4 conv 3×3, D2 | 5 | 1 | $7+4\cdot4=23$ | 4 |
| L5 conv 3×3, D4 | 9 | 1 | $23+8\cdot4=\mathbf{55}$ | 4 |
| L6 conv 1×1 | 1 | 1 | **55** | **4** |

- (b) Undilated L4, L5: $7+2\cdot4+2\cdot4=\mathbf{23}$.
- (c) After L3 each undilated $3\times3$ adds $2\times4=8$: need $(55-7)/8=\mathbf6$ convs, i.e. $6\times9C^2$ weights vs $2\times9C^2$ — **3× the parameters and MACs** (dilation enlarges the RF at no parameter cost, but samples sparsely → gridding).

**Answer:** **RF 55 px, jump 4; undilated 23 px; 6 plain convs (3× params) to match.**
> ✔ Verified numerically (Python recurrence + PyTorch gradient-footprint test with average pooling).
> **Common slip:** multiplying the dilated kernel growth by the **current** layer's stride instead of the **previous** jump $j_{l-1}$.

#### D14 — Depthwise-separable savings

**Given:** $K=5$, $C_{in}=240$, $C_{out}=80$, $28\times28$, no bias.

**Formula:** standard $K^2C_{in}C_{out}$; DS $K^2C_{in}+C_{in}C_{out}$; ratio $\frac1{C_{out}}+\frac1{K^2}$.

**Working:**
- (a) Params $25\cdot240\cdot80=\mathbf{480{,}000}$; MACs $784\cdot480{,}000=\mathbf{376{,}320{,}000}$.
- (b) Depthwise: $25\cdot240=\mathbf{6{,}000}$ params, $784\cdot6000=\mathbf{4{,}704{,}000}$ MACs. Pointwise: $240\cdot80=\mathbf{19{,}200}$ params, $\mathbf{15{,}052{,}800}$ MACs. Total **25,200** params, **19,756,800** MACs.
- (c) $25200/480000=\mathbf{0.0525}$ and $\frac1{80}+\frac1{25}=0.0125+0.04=0.0525$ ✓ → **19.05× cheaper**. The **pointwise** conv is $15.05/19.76=\mathbf{76\%}$ of the separable MACs.

**Answer:** **480,000 params / 376.3 M MACs vs 25,200 / 19.76 M (ratio 0.0525 = 1/80 + 1/25, 19×); pointwise ≈ 76% of the cost.**
> ✔ Verified numerically (PyTorch, `groups=C_in`).
> **Common slip:** writing the depthwise cost as $K^2C_{in}C_{out}$ — each depthwise filter sees **one** channel.

#### D15 — ViT token counts

**Given:** patch sizes and resolutions.

**Formula:** tokens $=(H/p)(W/p)+1$ ([CLS]); attention entries per head $=n^2$.

**Working:**
- (a) $384/16=24$ → $576+1=\mathbf{577}$ tokens; $577^2=\mathbf{332{,}929}$ entries; vs $197^2=38{,}809$ → $\mathbf{8.58\times}$ (tokens ×2.93, attention ∝ $n^2$).
- (b) $224/8=28$ → $784+1=\mathbf{785}$ tokens.
- (c) $(320/16)(480/16)=20\times30=600$ → **601** tokens. The learned $14\times14$ grid of position embeddings must be **2-D interpolated** (bicubic) to $20\times30$; the [CLS] position embedding is kept.
- (d) A linear map of each flattened $16\cdot16\cdot3=768$-d patch to 768-d (a $16\times16$ conv, stride 16): $768\cdot768+768=\mathbf{590{,}592}$.

**Answer:** **577 tokens, 332,929 entries (8.6×); 785; 601 + interpolated pos-embeddings; 590,592 params.**
> ✔ Verified numerically (Python + PyTorch `Conv2d(3,768,16,16)`).
> **Common slip:** forgetting the +1 [CLS] token, or squaring the patch count before adding it.

### S04 · Training CNNs & transfer learning

#### D16 — Softmax, cross-entropy and label smoothing

**Given:** $z=[1.2,3.1,0.4,-0.5]$, true = cat (index 0), $\varepsilon=0.1$, $K=4$.

**Formula:** $p_i=\frac{e^{z_i-\max z}}{\sum_je^{z_j-\max z}}$, $\mathcal L=-\sum_iy_i\ln p_i$, $\partial\mathcal L/\partial z=p-y$.

**Working:**
- (a) $z-3.1=[-1.9,0,-2.7,-3.6]$; $e^{(\cdot)}=[0.14957,\ 1,\ 0.06721,\ 0.02732]$, sum $=1.24410$ → $p=\mathbf{[0.1202,\ 0.8038,\ 0.0540,\ 0.0220]}$.
- (b) CE $=-\ln0.1202=\mathbf{2.1184}$; gradient $p-y=\mathbf{[-0.8798,\ 0.8038,\ 0.0540,\ 0.0220]}$.
- (c) Target $=[0.9+0.025,\ 0.025,\ 0.025,\ 0.025]=\mathbf{[0.925,0.025,0.025,0.025]}$. $-\ln p=[2.1184,\ 0.2184,\ 2.9184,\ 3.8184]$ → loss $=0.925(2.1184)+0.025(0.2184+2.9184+3.8184)=1.9595+0.1739=\mathbf{2.1334}$.
- (d) Predicted **dog** (wrong). Minimum LS loss = entropy of the smoothed target $=-0.925\ln0.925-3(0.025\ln0.025)=\mathbf{0.349}$ — label smoothing never lets the loss reach 0, so logits stop growing (better calibration).

**Answer:** **$p=[0.120,0.804,0.054,0.022]$; CE 2.118; grad $[-0.880,0.804,0.054,0.022]$; LS target $[0.925,0.025,0.025,0.025]$, loss 2.133; predicts dog; min LS loss 0.349.**
> ✔ Verified numerically (PyTorch `F.cross_entropy`, incl. `label_smoothing=0.1`).
> **Common slip:** giving the true class $1-\varepsilon=0.9$ instead of $1-\varepsilon+\varepsilon/K=0.925$.

#### D17 — BatchNorm on a conv mini-batch

**Given:** 8 values $\{1,3,5,7,2,2,4,8\}$, $\gamma=0.5$, $\beta=0.2$, momentum 0.1.

**Formula:** $\hat x=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$, $y=\gamma\hat x+\beta$; running $\leftarrow(1-0.1)\,\text{running}+0.1\,\text{batch}$.

**Working:**
- (a) `BatchNorm2d` normalises **per channel over $N\times H\times W$** = all 8 values. $\mu_B=32/8=\mathbf4$; deviations $-3,-1,1,3,-2,-2,0,4$ → squares sum $=9+1+1+9+4+4+0+16=44$ → $\sigma_B^2=44/8=\mathbf{5.5}$, $\sigma_B=2.3452$.
- (b) $\hat x=(x-4)/2.3452$ and $y=0.5\hat x+0.2$:

| | values | $\hat x$ | $y$ |
|---|---|---|---|
| image 1 | 1, 3, 5, 7 | −1.2792, −0.4264, 0.4264, 1.2792 | **−0.4396, −0.0132, 0.4132, 0.8396** |
| image 2 | 2, 2, 4, 8 | −0.8528, −0.8528, 0, 1.7056 | **−0.2264, −0.2264, 0.2000, 1.0528** |

- (c) Running mean $=0.9\cdot0+0.1\cdot4=\mathbf{0.4}$; unbiased variance $=44/7=6.2857$ → running var $=0.9\cdot1+0.1\cdot6.2857=\mathbf{1.5286}$.
- (d) Eval: $y=0.5\cdot\frac{4-0.4}{\sqrt{1.5286}}+0.2=0.5\cdot2.9118+0.2=\mathbf{1.656}$ (in train mode the same $x=4$ gave 0.2 — eval uses the running statistics, not the batch). `BatchNorm2d(64)`: **128 learnable** ($\gamma,\beta$) and **128 buffers** (running mean/var; plus a `num_batches_tracked` counter).

**Answer:** **$\mu=4$, $\sigma^2=5.5$; $y$ as in the table; running mean 0.4, var 1.5286; eval output 1.656; 128 params + 128 buffers.**
> ✔ Verified numerically (PyTorch `nn.BatchNorm2d`, train and eval mode).
> **Common slip:** normalising each image (or each spatial position) separately — conv BN pools the batch **and** the spatial dimensions.

#### D18 — Inverted dropout

**Given:** $a$, $p=0.25$, $m=[1,1,0,1,0,1,1,1]$.

**Formula:** train $y=a\odot m/(1-p)$; test $y=a$.

**Working:**
- (a) Scale $1/0.75=4/3$: $y=\mathbf{[2.0,\ -0.5333,\ 0,\ 1.0667,\ 0,\ 0,\ 1.4667,\ -3.3333]}$.
- (b) Test: identity, $y=a$. $\mathbb E[y]=(1-p)\cdot\frac{a}{1-p}+p\cdot0=a$, so train and test activations have the same expectation.
- (c) Classic dropout: train $a\cdot m$ → this unit was dropped → **0** (if kept: 3.2, unscaled); test multiplies by $(1-p)$ → $3.2\times0.75=\mathbf{2.4}$.
- (d) $\operatorname{Var}=\mathbb E[y^2]-a^2=(1-p)\frac{a^2}{(1-p)^2}-a^2=a^2\frac{p}{1-p}=10.24\cdot\frac{0.25}{0.75}=\mathbf{3.413}$.

**Answer:** **train $[2.0,-0.533,0,1.067,0,0,1.467,-3.333]$; test $=a$; classic: 0 (train, dropped) / 2.4 (test); Var = 3.41.**
> ✔ Verified numerically (Python + `nn.Dropout` scale check).
> **Common slip:** scaling by $(1-p)$ at **test** time when using inverted dropout (double-scaling).

#### D19 — Trainable parameters: feature extraction vs fine-tuning (VGG-16)

**Given:** VGG-16 totals; new FC8 $4096\to5$.

**Formula:** FC params $=(n_{in}+1)\,n_{out}$; conv $(9C_{in}+1)C_{out}$.

**Working:**
- (a) FC6 $=25088\cdot4096+4096=\mathbf{102{,}764{,}544}$; FC7 $=4096\cdot4096+4096=\mathbf{16{,}781{,}312}$; FC8 $=4096\cdot1000+1000=\mathbf{4{,}097{,}000}$. $14{,}714{,}688+102{,}764{,}544+16{,}781{,}312+4{,}097{,}000=138{,}357{,}544$ ✓.
- (b) New FC8 $=4096\cdot5+5=20{,}485$ → total $138{,}357{,}544-4{,}097{,}000+20{,}485=\mathbf{134{,}281{,}029}$.
- (c)

| Strategy | Trainable | % of 134,281,029 |
|---|---|---|
| (i) feature extraction (FC8 only) | **20,485** | **0.015%** |
| (ii) FC6–FC8 | $102{,}764{,}544+16{,}781{,}312+20{,}485=$ **119,566,341** | **89.04%** |
| (iii) conv5 + FC6–FC8 | conv5 $=3\times(9\cdot512\cdot512+512)=7{,}079{,}424$ → **126,645,765** | **94.31%** |
| (iv) full fine-tune | **134,281,029** | **100%** |

- (d) (i) $4\times134{,}281{,}029+12\times20{,}485=537{,}124{,}116+245{,}820=537{,}369{,}936$ B $=\mathbf{0.537}$ GB. (iv) $16\times134{,}281{,}029=\mathbf{2.148}$ GB.

**Answer:** **FC6 102.8 M, FC7 16.8 M, FC8 4.1 M; new total 134,281,029; trainable 20,485 (0.015%) / 119.6 M (89.0%) / 126.6 M (94.3%) / 134.3 M; memory 0.54 GB vs 2.15 GB.**
> ✔ Verified numerically (torchvision `vgg16(weights=None)` parameter counts).
> **Common slip:** assuming the conv layers hold most of VGG's parameters — 89% sit in the FC layers (FC6 alone is ≈ 74% of the original 138.4 M).

#### D20 — Confusion matrix → precision, recall, F1 (macro & micro)

**Given:** 3×3 matrix, rows = GT.

**Formula:** $P_c=\frac{TP_c}{\text{col}_c}$, $R_c=\frac{TP_c}{\text{row}_c}$, $F1=\frac{2PR}{P+R}$; micro: pool TP/FP/FN over classes.

**Working:**
- (a) Diagonal $45+26+103=174$ → accuracy $=174/200=\mathbf{0.870}$.
- (b) Column sums 50, 35, 115; row sums 50, 40, 110.

| Class | $P$ | $R$ | $F1$ |
|---|---|---|---|
| Stop | $45/50=$ **0.900** | $45/50=$ **0.900** | **0.900** |
| Yield | $26/35=$ **0.743** | $26/40=$ **0.650** | **0.693** |
| Speed | $103/115=$ **0.896** | $103/110=$ **0.936** | **0.916** |

- (c) Macro-P $=(0.9+0.7429+0.8957)/3=\mathbf{0.846}$; macro-R $=(0.9+0.65+0.9364)/3=\mathbf{0.829}$; macro-F1 $=(0.9+0.6933+0.9156)/3=\mathbf{0.836}$.
- (d) Micro: $TP=174$; every misclassified sample is **one FP** (for the predicted class) **and one FN** (for the true class) → $FP=FN=26$ → micro-P = micro-R = micro-F1 $=174/200=\mathbf{0.870}$ = accuracy (true for any single-label multi-class problem). Weighted F1 $=(50\cdot0.9+40\cdot0.6933+110\cdot0.9156)/200=\mathbf{0.867}$.

**Answer:** **acc 0.87; P/R/F1: Stop 0.90/0.90/0.90, Yield 0.743/0.65/0.693, Speed 0.896/0.936/0.916; macro P/R/F1 0.846/0.829/0.836; micro = 0.87; weighted F1 0.867.**
> ✔ Verified numerically (Python/numpy).
> **Common slip:** swapping rows and columns (precision from row sums) — always check which axis is the ground truth.

### S05 · Embeddings & similarity

#### D21 — Cosine and Euclidean for unit vectors

**Given:** $\mathbf a,\mathbf b,\mathbf c$ (4-D).

**Formula:** for unit vectors $\Vert\mathbf x-\mathbf y\Vert^2=2-2\cos\theta$.

**Working:**
- (a) $\Vert\mathbf a\Vert^2=4(0.25)=1$; $\Vert\mathbf b\Vert^2=0.49+0.01+0.01+0.49=1$; $\Vert\mathbf c\Vert^2=1$ ✓. $\cos(\mathbf a,\mathbf b)=0.5(0.7+0.1+0.1+0.7)=\mathbf{0.8}$; $\Vert\mathbf a-\mathbf b\Vert=\sqrt{2-1.6}=\sqrt{0.4}=\mathbf{0.6325}$; angle $=\arccos0.8=\mathbf{36.87^\circ}$.
- (b) $(\mathbf a,\mathbf c)$: $\cos=\mathbf{0.8}$, $d=\mathbf{0.6325}$ (a tie with $\mathbf b$). $(\mathbf b,\mathbf c)$: $\cos=0.07+0.07+0.07+0.07=\mathbf{0.28}$, $d=\sqrt{2-0.56}=\sqrt{1.44}=\mathbf{1.2}$, angle **73.74°**.
- (c) $d<0.9$ ⇔ $2-2\cos<0.81$ ⇔ $\cos>\mathbf{0.595}$ ⇔ angle $<\mathbf{53.49^\circ}$. For unit vectors a distance threshold and a cosine threshold are the same rule.
- (d) $3\mathbf b=(2.1,0.3,0.3,2.1)$: dot $=\mathbf{2.4}$; cosine **0.8** (unchanged, scale-invariant); $\Vert\mathbf a-3\mathbf b\Vert=\Vert(-1.6,0.2,0.2,-1.6)\Vert=\sqrt{5.2}=\mathbf{2.280}$ — Euclidean and dot product depend on the norm, so normalise before comparing.

**Answer:** **cos 0.8, d 0.632, 36.9°; (a,c) same; (b,c) cos 0.28, d 1.2; threshold cos > 0.595 (53.5°); scaled: dot 2.4, cos 0.8, d 2.28.**
> ✔ Verified numerically (Python).
> **Common slip:** using $\Vert\mathbf x-\mathbf y\Vert=1-\cos$ — the identity is for the **squared** distance and has a factor 2.

#### D22 — Triplet loss with batch-all and batch-hard mining

**Given:** $d_p\in\{0.6,1.1\}$, $d_n\in\{0.9,1.3,1.6,2.4\}$, $m=0.4$.

**Formula:** $\mathcal L=\max(0,\ d_p-d_n+m)$.

**Working:**
- (a)

| | $n_1$ (0.9) | $n_2$ (1.3) | $n_3$ (1.6) | $n_4$ (2.4) |
|---|---|---|---|---|
| $p_1$ (0.6) | $0.6-0.9+0.4=$ **0.1** · semi-hard | **0** · easy | **0** · easy | **0** · easy |
| $p_2$ (1.1) | $1.1-0.9+0.4=$ **0.6** · hard | $1.1-1.3+0.4=$ **0.2** · semi-hard | **0** ($-0.1$) · easy | **0** · easy |

- (b) Mean over all 8: $(0.1+0.6+0.2)/8=\mathbf{0.1125}$; mean over the 3 active triplets: $0.9/3=\mathbf{0.30}$ (averaging over active triplets stops the many easy zeros from diluting the gradient).
- (c) Hardest positive $1.1$, hardest negative $0.9$: $\max(0,1.1-0.9+0.4)=\mathbf{0.6}$.
- (d) Early on, the hardest negatives are often **label noise or near-duplicates**, and optimising only the worst case can collapse the embedding (all points mapped together); semi-hard mining (FaceNet) or batch-all is more stable at the start.

**Answer:** **losses 0.1, 0, 0, 0 / 0.6, 0.2, 0, 0; batch-all 0.1125 (all) or 0.30 (active); batch-hard 0.6.**
> ✔ Verified numerically (Python).
> **Common slip:** calling $n_1$ "hard" for $p_1$ — it is farther than the positive ($0.9>0.6$) but inside the margin, so **semi-hard**.

#### D23 — InfoNCE and the temperature

**Given:** $s=[0.75\ (+),\ 0.70,\ 0.40,\ 0.10,\ -0.30]$.

**Formula:** $p_i=\operatorname{softmax}(s/\tau)_i$, $\mathcal L=-\ln p_+$.

**Working:**
- (a) $\tau=0.1$: logits $[7.5,\ 7.0,\ 4.0,\ 1.0,\ -3.0]$; subtract 7.5 → $e^{(\cdot)}=[1,\ 0.6065,\ 0.0302,\ 0.0015,\ 0.00003]$, sum 1.6383 → $p=\mathbf{[0.6104,\ 0.3702,\ 0.0184,\ 0.0009,\ 0.00002]}$; $\mathcal L=-\ln0.6104=\mathbf{0.4936}$.
- (b) $\tau=0.5$: logits $[1.5,1.4,0.8,0.2,-0.6]$ → $p=\mathbf{[0.3576,\ 0.3236,\ 0.1776,\ 0.0975,\ 0.0438]}$; $\mathcal L=\mathbf{1.0283}$.
- (c)

| | $\partial\mathcal L/\partial s_+$ | negatives $p_i/\tau$ (0.70, 0.40, 0.10, −0.30) | share on hardest |
|---|---|---|---|
| $\tau=0.1$ | $(0.6104-1)/0.1=$ **−3.896** | **3.702, 0.184, 0.009, 0.0002** | $0.3702/0.3896=$ **95.0%** |
| $\tau=0.5$ | $(0.3576-1)/0.5=$ **−1.285** | **0.647, 0.355, 0.195, 0.088** | $0.3236/0.6424=$ **50.4%** |

  A small $\tau$ makes the loss **hardness-aware**: almost all of the push goes to the closest negative.
- (d) All equal → $p_+=1/5$ → $\mathcal L=\ln5=\mathbf{1.609}$ (in general $\ln(N+1)$).

**Answer:** **τ = 0.1: $p_+=0.610$, L = 0.494; τ = 0.5: $p_+=0.358$, L = 1.028; hardest-negative share 95% vs 50%; chance loss ln 5 = 1.609.**
> ✔ Verified numerically (Python + PyTorch autograd for the gradients).
> **Common slip:** forgetting to divide **every** similarity (including the positive) by $\tau$.

#### D24 — Retrieval AP and Recall@K

**Given:** relevance `1 0 0 1 1 0 0 0`, $R=4$; first-hit ranks $1,3,1,7,2$.

**Formula:** $P@k=\frac{\#\text{rel in top }k}{k}$, $R@k=\frac{\#\text{rel in top }k}{R}$, $\text{AP}=\frac1R\sum_kP@k\cdot\text{rel}(k)$, $\text{MRR}=\frac1Q\sum\frac1{\text{rank}_q}$.

**Working:**
- (a) $P@1=\mathbf{1}$, $R@1=\mathbf{0.25}$; top-5 has 3 relevant → $P@5=\mathbf{0.6}$, $R@5=\mathbf{0.75}$; top-8 has 3 → $P@8=\mathbf{0.375}$, $R@8=\mathbf{0.75}$.
  Relevant at ranks 1, 4, 5 with $P@k=1,\ 2/4,\ 3/5$ → $\text{AP}=\frac{1+0.5+0.6}{4}=\mathbf{0.525}$ (the 4th relevant item is never retrieved and contributes 0). Dividing by 3 instead would give **0.70** — wrong, it ignores the missed item.
- (b) Recall@1 $=2/5=\mathbf{0.4}$; @2 $=3/5=\mathbf{0.6}$; @5 $=4/5=\mathbf{0.8}$; @10 $=\mathbf{1.0}$. MRR $=\frac{1+1/3+1+1/7+1/2}{5}=\frac{2.976}{5}=\mathbf{0.595}$.

**Answer:** **P@1 1, R@1 0.25, P@5 0.6, R@5 0.75, P@8 0.375, R@8 0.75; AP 0.525 (not 0.70); Recall@1/2/5/10 = 0.4/0.6/0.8/1.0; MRR 0.595.**
> ✔ Verified numerically (Python).
> **Common slip:** mixing the two meanings of "Recall@K" — in IR it is a fraction of the $R$ relevant items; in metric-learning papers it is the hit rate. State which.

### S06 · Detection I — two-stage

#### D25 — Anchor count with FPN

**Given:** $1024\times768$, strides 4–64, 3 anchors/location.

**Formula:** locations $=(W/s)(H/s)$; anchors $=3\times$ locations.

**Working:**
- (a)

| Level | stride | map ($W\times H$) | locations | anchors |
|---|---|---|---|---|
| P2 | 4 | 256×192 | 49,152 | 147,456 |
| P3 | 8 | 128×96 | 12,288 | 36,864 |
| P4 | 16 | 64×48 | 3,072 | 9,216 |
| P5 | 32 | 32×24 | 768 | 2,304 |
| P6 | 64 | 16×12 | 192 | 576 |
| **Total** | | | **65,472** | **196,416** |

- (b) Single stride-16 map: $64\cdot48\cdot9=\mathbf{27{,}648}$ → FPN has **7.1×** more anchors (75% of them on P2, for small objects).
- (c) $k=3$: classification **6** channels ($2k$), regression **12** channels ($4k$) (the head is shared across levels).
- (d) $s=64$: $r=0.5$: $w=64/\sqrt{0.5}=\mathbf{90.51}$, $h=64\sqrt{0.5}=\mathbf{45.25}$; $r=1$: **64×64**; $r=2$: **45.25×90.51** (all have area 4096).

**Answer:** **196,416 anchors (65,472 locations) vs 27,648; 6 and 12 channels; P3 anchors 90.5×45.3, 64×64, 45.3×90.5.**
> ✔ Verified numerically (Python).
> **Common slip:** forgetting that with FPN each level has only **one** scale (3 anchors), not 9.

#### D26 — Encoding and decoding box offsets

**Given:** $P=(120,80,280,200)$, $G=(130,70,310,230)$, $\hat t=(-0.05,0.10,0.20,-0.10)$.

**Formula:** $t_x=\frac{x-x_a}{w_a}$, $t_y=\frac{y-y_a}{h_a}$, $t_w=\ln\frac w{w_a}$, $t_h=\ln\frac h{h_a}$; decode $x=x_a+t_xw_a$, $w=w_ae^{t_w}$.

**Working:**
- (a) Intersection $[130,280]\times[80,200]=150\times120=18{,}000$; areas $160\times120=19{,}200$ and $180\times160=28{,}800$; union $30{,}000$ → IoU $=\mathbf{0.600}$.
- (b) $P$: centre $(200,140)$, $160\times120$; $G$: centre $(220,150)$, $180\times160$.
  $t=\left(\frac{20}{160},\frac{10}{120},\ln\frac{180}{160},\ln\frac{160}{120}\right)=\mathbf{(0.125,\ 0.0833,\ 0.1178,\ 0.2877)}$; normalised by $(0.1,0.1,0.2,0.2)$: $\mathbf{(1.25,\ 0.833,\ 0.589,\ 1.438)}$.
- (c) $x=200-0.05\cdot160=192$; $y=140+0.10\cdot120=152$; $w=160e^{0.2}=195.42$; $h=120e^{-0.1}=108.58$ → box $\mathbf{(94.29,\ 97.71,\ 289.71,\ 206.29)}$.
  IoU with $G$: intersection $(289.71-130)(206.29-97.71)=159.71\times108.58=17{,}341.6$; areas $21{,}219.3$ and $28{,}800$; union $32{,}677.7$ → **0.531**.
- (d) All coordinates double, offsets are divided by $w_a,h_a$ and sizes enter as ratios → **targets unchanged** (scale-invariant).

**Answer:** **IoU 0.60; $t=(0.125,0.0833,0.1178,0.2877)$ → normalised (1.25, 0.833, 0.589, 1.438); decoded (94.3, 97.7, 289.7, 206.3), IoU 0.531; unchanged under scaling.**
> ✔ Verified numerically (Python + torchvision `BoxCoder` encode/decode).
> **Common slip:** using corner differences instead of **centre** differences, or $\log_{10}$ instead of $\ln$.

#### D27 — HOG descriptor length and orientation voting

**Given:** $8\times8$ cells, $2\times2$ blocks, 9 bins.

**Formula:** blocks per axis $=\frac{\text{cells}-2}{\text{block stride}}+1$; length $=\#\text{blocks}\times4\times\#\text{bins}$.

**Working:**
- (a) Cells $96/8=12$ by $160/8=20$; blocks $11\times19=209$; length $209\times36=\mathbf{7524}$.
- (b) Stride 2 cells: $(12-2)/2+1=6$ by $(20-2)/2+1=10$ → 60 blocks → $60\times36=\mathbf{2160}$.
- (c) $209\times4\times18=\mathbf{15{,}048}$.
- (d) (i) $75^\circ$ lies between centres 70° and 90°, a quarter of the way: **37.5 → 70° bin, 12.5 → 90° bin**. (ii) $175^\circ$ lies between 170° and 190° (≡ 10°), a quarter of the way: **37.5 → 170° bin, 12.5 → 10° bin** (wrap-around, because unsigned orientation 180° ≡ 0°).

**Answer:** **7524; 2160; 15,048; (i) 37.5/12.5 to 70°/90°; (ii) 37.5/12.5 to 170°/10°.**
> ✔ Verified numerically (Python and `cv2.HOGDescriptor.getDescriptorSize()`).
> **Common slip:** counting blocks as cells/2 (non-overlapping) when the block stride is one cell.

#### D28 — RoI projection onto a stride-16 map and FPN level

**Given:** RoI $(213,117,589,405)$, stride 16.

**Formula:** feature coordinate $=$ pixel$/16$; FPN level $k=\lfloor4+\log_2(\sqrt{wh}/224)\rfloor$.

**Working:**
- (a) $(213,117,589,405)/16=\mathbf{(13.3125,\ 7.3125,\ 36.8125,\ 25.3125)}$ → rounded $\mathbf{(13,\ 7,\ 37,\ 25)}$. Misalignment $=(\text{rounded}-\text{exact})\times16$: $\mathbf{(-5,\ -5,\ +3,\ -5)}$ px.
- (b) Rounded: $24\times18$ cells → RoI-Pool bins $24/7\times18/7=\mathbf{3.43\times2.57}$ cells (RoI Pool must quantise these again into integer cell ranges). Exact: $23.5\times18$ → RoIAlign bins $\mathbf{3.357\times2.571}$ (kept as floats).
- (c) $w=376$, $h=288$, $\sqrt{wh}=329.07$; $\log_2(329.07/224)=\log_2(1.469)=0.555$ → $k=\lfloor4.555\rfloor=\mathbf{4}$ (**P4**). $64\times48$: $\sqrt{3072}=55.43$, $\log_2(0.2474)=-2.015$ → $\lfloor1.985\rfloor=1$ → clamp → **P2**.
- (d) $7\cdot7\cdot256=12{,}544$ inputs → $12{,}544\cdot1024+1024=\mathbf{12{,}846{,}080}$.

**Answer:** **(13.31, 7.31, 36.81, 25.31) → (13, 7, 37, 25), shifts −5/−5/+3/−5 px; bins 3.43×2.57 (Pool) vs 3.36×2.57 (Align); P4, small RoI → P2; 12,846,080 params.**
> ✔ Verified numerically (Python).
> **Common slip:** forgetting the clamp to $[2,5]$, or using natural log instead of $\log_2$.

### S07 · Detection II — one-stage, NMS, AP

#### D29 — IoU in three configurations

**Given:** $A=(40,60,140,160)$ (area 10,000).

**Formula:** $\text{IoU}=\frac{\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert-\lvert A\cap B\rvert}$.

**Working:**
- (a) Intersection $[90,140]\times[110,160]=50\times50=2500$; union $10000+10000-2500=17500$ → **0.143**.
- (b) $C$ lies inside $A$: intersection $=\lvert C\rvert=60\times60=3600$; union $=\lvert A\rvert=10000$ → **0.36**.
- (c) $D$ touches $A$ along $x=140$: intersection width 0 → **IoU = 0** (touching is not overlapping).
- (d) $20\times20$ shifted 10: intersection $10\times20=200$, union $600$ → **0.333**; $200\times200$ shifted 10: $190\times200=38000$, union $42000$ → **0.905**. A 10-px error kills a small box's IoU but barely affects a large one → small objects are hard at strict IoU thresholds (one reason COCO reports AP$_S$ separately).

**Answer:** **0.143; 0.36; 0; 0.333 vs 0.905.**
> ✔ Verified numerically (Python).
> **Common slip:** adding +1 to widths (old VOC pixel convention) — this drill uses continuous coordinates.

#### D30 — Per-class vs class-agnostic NMS on six boxes

**Given:** six boxes, threshold 0.5 (suppress if IoU > 0.5).

**Working:**
- (a) IoU(B1,B2): $[60,150]\times[55,150]=90\times95=8550$; union $10000+9500-8550=10950$ → **0.781**.
  IoU(B1,B3): $[55,145]\times[60,150]=90\times90=8100$; union $10000+9000-8100=10900$ → **0.743**.
  IoU(B1,B5): $[65,150]\times[70,150]=85\times80=6800$; union $10000+8550-6800=11750$ → **0.579**.
  IoU(B3,B5): $[65,145]\times[70,160]=80\times90=7200$; union $9000+8550-7200=10350$ → **0.696**.
- (b) Per-class: **person** (B1, B2, B4, B6 by score): keep B1; B2 (0.781 > 0.5) suppressed; B4 (0.176) kept; B6 kept. **dog** (B3, B5): keep B3; B5 (0.696) suppressed. **Kept: B1, B3, B4, B6.**
- (c) Agnostic (B1…B6): keep B1; B2 (0.781) ✗; **B3 (0.743 with B1) ✗**; B4 (0.176 with B1) ✓; B5 (0.579 with B1) ✗; B6 ✓. **Kept: B1, B4, B6.**
- (d) Class-agnostic NMS deleted the **dog** standing in front of the person, because the two boxes overlap heavily. Per-class NMS (the default in most detectors, e.g. torchvision `batched_nms`) only lets boxes of the same class suppress each other.

**Answer:** **IoUs 0.781, 0.743, 0.579, 0.696; per-class keeps B1, B3, B4, B6; agnostic keeps B1, B4, B6 (dog lost).**
> ✔ Verified numerically (torchvision `nms` / `batched_nms`).
> **Common slip:** comparing a candidate with **suppressed** boxes — only already-kept boxes can suppress.

#### D31 — AP: all-point and 11-point from a ranked list

**Given:** `T F T T F T F F T F`, 6 GT.

**Formula:** $P=\frac{TP}{TP+FP}$, $R=\frac{TP}{6}$; all-point $\text{AP}=\sum(R_{k}-R_{k-1})\,p_{interp}(R_k)$; 11-point $=\frac1{11}\sum_{r\in\{0,0.1,\dots,1\}}p_{interp}(r)$.

**Working:**
- (a)

| Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| T/F | T | F | T | T | F | T | F | F | T | F |
| cum TP | 1 | 1 | 2 | 3 | 3 | 4 | 4 | 4 | 5 | 5 |
| cum FP | 0 | 1 | 1 | 1 | 2 | 2 | 3 | 4 | 4 | 5 |
| $P$ | 1.000 | 0.500 | 0.667 | 0.750 | 0.600 | 0.667 | 0.571 | 0.500 | 0.556 | 0.500 |
| $R$ | 0.167 | 0.167 | 0.333 | 0.500 | 0.500 | 0.667 | 0.667 | 0.667 | 0.833 | 0.833 |

- (b) Envelope at each new recall level: $R=1/6$: 1.000; $R=2/6$: $\max(0.667,0.75,\dots)=0.750$; $R=3/6$: 0.750; $R=4/6$: 0.667; $R=5/6$: 0.556; beyond $5/6$: 0.
  $\text{AP}=\frac16(1.000+0.750+0.750+0.667+0.556)=\frac{3.7222}{6}=\mathbf{0.6204}$.
- (c) $p_{interp}$ at $r=0,0.1$: 1.000; $r=0.2,0.3,0.4,0.5$: 0.750; $r=0.6$: 0.667; $r=0.7,0.8$: 0.556; $r=0.9,1.0$: 0 (max recall 0.833).
  $\text{AP}_{11}=\frac{2(1)+4(0.75)+0.667+2(0.556)}{11}=\frac{6.7778}{11}=\mathbf{0.6162}$.
  The 11-point method samples the envelope at fixed recalls (it cannot "see" recall between samples), so it slightly mis-weights each step; all-point integrates the exact area. (COCO-style 101-point gives 0.6199, close to all-point.)

**Answer:** **all-point AP = 0.620; 11-point AP = 0.616.**
> ✔ Verified numerically (Python; VOC-style envelope code, plus 101-point check).
> **Common slip:** using the raw (zig-zag) precision instead of the envelope — e.g. taking 0.667 at $R=2/6$ instead of 0.75.

#### D32 — YOLO output tensors and cell assignment

**Given:** YOLOv1 $S=7,B=3,C=10$; YOLOv3 at 608, 3 anchors, $C=10$.

**Formula:** v1: $S\times S\times(5B+C)$; v3: per scale $3(5+C)$ channels, boxes $=3\sum g^2$.

**Working:**
- (a) $5\cdot3+10=25$ → $\mathbf{7\times7\times25}$, **1225** values, $49\times3=\mathbf{147}$ boxes.
- (b) Grids $608/32=\mathbf{19}$, $608/16=\mathbf{38}$, $608/8=\mathbf{76}$; channels $3(5+10)=\mathbf{45}$ per scale; boxes $3(361+1444+5776)=1083+4332+17328=\mathbf{22{,}743}$.
- (c) $(256+1)\cdot45=\mathbf{11{,}565}$.
- (d) $300/32=9.375$, $470/32=14.6875$ → cell **(column 9, row 14)**; targets $\sigma(t_x)=\mathbf{0.375}$, $\sigma(t_y)=\mathbf{0.6875}$ (fraction of the cell).

**Answer:** **7×7×25 (1225 values, 147 boxes); grids 19/38/76, 45 channels, 22,743 boxes; 11,565 params; cell (9, 14), offsets (0.375, 0.6875).**
> ✔ Verified numerically (Python).
> **Common slip:** YOLOv1 has class scores **per cell** ($5B+C$), but YOLOv3 has them **per anchor** ($B(5+C)$).

#### D33 — SSD default boxes

**Given:** SSD512 maps; $s_{min}=0.2$, $s_{max}=0.9$, $m=6$.

**Formula:** total $=\sum f_k^2n_k$; $s_k=0.2+0.14(k-1)$; $w=s\sqrt a$, $h=s/\sqrt a$.

**Working:**
- (a) $64^2\cdot4+32^2\cdot6+16^2\cdot6+8^2\cdot6+4^2\cdot6+2^2\cdot4+1\cdot4=16384+6144+1536+384+96+16+4=\mathbf{24{,}564}$.
- (b) Step $0.7/5=0.14$: $s=\mathbf{0.20,\ 0.34,\ 0.48,\ 0.62,\ 0.76,\ 0.90}$.
- (c) $k=2$, $a=2$: $w=0.34\sqrt2=0.4808$ → **144.2 px**, $h=0.34/\sqrt2=0.2404$ → **72.1 px**. Extra square box: $s'=\sqrt{0.34\times0.48}=0.4040$ → **121.2 × 121.2 px**.
- (d) $6\times(21+4)=\mathbf{150}$ output channels.

**Answer:** **24,564 boxes; scales 0.20…0.90; 144.2×72.1 px and 121.2×121.2 px; 150 channels.**
> ✔ Verified numerically (Python; SSD300 check gives the known 8732).
> **Common slip:** forgetting the background class in $C+1$ when sizing the head.

#### D34 — Focal loss vs cross-entropy ratios

**Given:** $\gamma=2$, $\alpha_t=0.25$ (fg) / 0.75 (bg).

**Formula:** $\frac{FL}{CE}=\alpha_t(1-p_t)^\gamma$.

**Working:**
- (a)

| $p_t$ | CE | $(1-p_t)^2$ | FL (fg) | FL/CE fg | FL/CE bg |
|---|---|---|---|---|---|
| 0.95 | 0.0513 | 0.0025 | $3.21\times10^{-5}$ | **0.000625** | **0.001875** |
| 0.70 | 0.3567 | 0.09 | 0.00803 | **0.0225** | **0.0675** |
| 0.20 | 1.6094 | 0.64 | 0.2575 | **0.16** | **0.48** |

- (b) $(1-p_t)^2=0.01$ → $p_t=\mathbf{0.90}$; $(1-p_t)^5=0.01$ → $1-p_t=0.01^{0.2}=0.398$ → $p_t=\mathbf{0.602}$ (larger $\gamma$ down-weights even moderately easy examples).
- (c) CE: $1.6094/0.0513=\mathbf{31.4}$; FL: $\frac{1.6094\times0.64}{0.0513\times0.0025}=\mathbf{8033}$ — focal loss shifts the training signal ~256× further towards hard examples.

**Answer:** **ratios fg 0.000625 / 0.0225 / 0.16 (bg 0.0019 / 0.0675 / 0.48); $p_t=0.90$ (γ=2), 0.602 (γ=5); hard:easy 31× (CE) vs 8033× (FL).**
> ✔ Verified numerically (Python).
> **Common slip:** using $p$ instead of $p_t$ for background anchors ($p_t=1-p$ when $y=0$).

### S08 · Detection III — transformers

#### D35 — Hungarian matching (3×3, then 4×3)

**Given:** $\mathbf C=\begin{bmatrix}8&4&7\\5&2&3\\9&3&8\end{bmatrix}$.

**Working:**
- (a) **Row reduction** (subtract 4, 2, 3): $\begin{bmatrix}4&0&3\\3&0&1\\6&0&5\end{bmatrix}$. **Column reduction** (subtract 3, 0, 1): $\begin{bmatrix}1&0&2\\0&0&0\\3&0&4\end{bmatrix}$.
  **Cover zeros:** row 2 + column 2 = **2 lines < 3** → not optimal yet.
  **Adjust:** smallest uncovered value = 1; subtract it from uncovered entries, add it at the intersection (row 2, col 2): $\begin{bmatrix}0&0&1\\0&1&0\\2&0&3\end{bmatrix}$. Now 3 lines are needed.
  **Assign:** row 3 has a single zero → $q_3\to g_2$; row 1 → $g_1$ (col 2 taken); row 2 → $g_3$. Cost $8+3+3=\mathbf{14}$ (the other 5 permutations cost 15, 16, 17, 18, 18).
- (b) Greedy: cheapest pair $q_2$–$g_2$ (2); then from $\{q_1,q_3\}\times\{g_1,g_3\}$ the cheapest is $q_1$–$g_3$ (7); left $q_3$–$g_1$ (9) → **18** (29% worse).
- (c) $4\times4$ with a zero "no-object" column: $\begin{bmatrix}8&4&7&0\\5&2&3&0\\9&3&8&0\\6&7&9&0\end{bmatrix}$. Row mins are 0; column reduction (5, 2, 3, 0): $\begin{bmatrix}3&2&4&0\\0&0&0&0\\4&1&5&0\\1&5&6&0\end{bmatrix}$. Lines: row 2 + col 4 = 2 < 4; min uncovered = 1 → $\begin{bmatrix}2&1&3&0\\0&0&0&1\\3&0&4&0\\0&4&5&0\end{bmatrix}$. Assign: $q_1\to\varnothing$ (only zero), $q_3\to g_2$, $q_4\to g_1$, $q_2\to g_3$. Total $6+3+3=\mathbf{12}$; **$q_1$ is trained to predict $\varnothing$** (next best assignment costs 13).

**Answer:** **(a) $q_1\to g_1$, $q_2\to g_3$, $q_3\to g_2$, cost 14; (b) greedy 18; (c) $q_2\to g_3$, $q_3\to g_2$, $q_4\to g_1$, $q_1\to\varnothing$, cost 12.**
> ✔ Verified numerically (`scipy.optimize.linear_sum_assignment` + brute force over all permutations).
> **Common slip:** stopping after the reductions and assigning greedily among zeros before checking that the minimum number of covering lines equals $n$.

#### D36 — DETR matching cost with L1 and GIoU

**Given:** $b=(0.50,0.50,0.40,0.30)$; $q_1$: 0.55, $(0.52,0.49,0.38,0.30)$; $q_2$: 0.92, $(0.62,0.55,0.40,0.36)$.

**Formula:** $(c_x,c_y,w,h)\to(c_x-\frac w2,\ c_y-\frac h2,\ c_x+\frac w2,\ c_y+\frac h2)$; $\text{GIoU}=\text{IoU}-\frac{\lvert E\rvert-\lvert U\rvert}{\lvert E\rvert}$.

**Working:**
- (a) $q_1$: $0.02+0.01+0.02+0=\mathbf{0.05}$; $q_2$: $0.12+0.05+0+0.06=\mathbf{0.23}$.
- (b) $b=(0.30,0.35,0.70,0.65)$, area 0.12.
  $q_1=(0.33,0.34,0.71,0.64)$, area 0.114: $I=0.37\times0.29=0.1073$; $U=0.12+0.114-0.1073=0.1267$; IoU $=\mathbf{0.8469}$; $E=0.41\times0.31=0.1271$; GIoU $=0.8469-0.0004/0.1271=\mathbf{0.8437}$.
  $q_2=(0.42,0.37,0.82,0.73)$, area 0.144: $I=0.28\times0.28=0.0784$; $U=0.1856$; IoU $=\mathbf{0.4224}$; $E=0.52\times0.38=0.1976$; GIoU $=0.4224-0.012/0.1976=\mathbf{0.3617}$.
- (c) $\mathcal C_1=-0.55+5(0.05)+2(1-0.8437)=-0.55+0.25+0.3125=\mathbf{0.0125}$; $\mathcal C_2=-0.92+5(0.23)+2(1-0.3617)=-0.92+1.15+1.2766=\mathbf{1.5066}$ → **$q_1$ is matched** (better box beats higher confidence).
- (d) With $\hat p=1$: $\mathcal C_2=-1+2.4266=1.4266>0.0125$ → **no**. With $\lambda_{giou}=0$: $\mathcal C_1=-0.30$, $\mathcal C_2=+0.23$ → $q_1$ still wins. The class term is bounded in $[-1,0]$, the box terms are not.

**Answer:** **L1 0.05 / 0.23; IoU 0.847 / 0.422; GIoU 0.844 / 0.362; costs 0.0125 vs 1.507 → $q_1$; $q_2$ cannot win even with $p=1$.**
> ✔ Verified numerically (torchvision `generalized_box_iou`).
> **Common slip:** computing GIoU directly on $(c_x,c_y,w,h)$ — convert to corners first.

#### D37 — GIoU for overlap, near-miss, far-miss and containment

**Given:** $G=(100,100,200,180)$, area 8000.

**Working:**

| Pred | $I$ | $\lvert P\rvert$ | $U$ | IoU | Enclosing $E$ | $\lvert E\rvert$ | GIoU | $1-\text{GIoU}$ |
|---|---|---|---|---|---|---|---|---|
| $P_1$ | $50\times60=3000$ | 10000 | 15000 | **0.200** | (100,100,250,220) | 18000 | $0.2-\frac{3000}{18000}=$ **0.033** | **0.967** |
| $P_2$ | 0 | 6400 | 14400 | **0** | (100,100,300,180) | 16000 | $-\frac{1600}{16000}=$ **−0.100** | **1.100** |
| $P_3$ | 0 | 6400 | 14400 | **0** | (100,100,480,380) | 106400 | $-\frac{92000}{106400}=$ **−0.865** | **1.865** |
| $P_4$ | 3600 | 3600 | 8000 | **0.450** | $=G$ | 8000 | **0.450** | **0.550** |

$P_2$ and $P_3$ both have IoU 0 (an IoU loss gives **no gradient** for either), but GIoU ranks the near miss much better (−0.10 vs −0.86) and its gradient pulls the box towards the target. When one box contains the other, $E$ equals the larger box, so **GIoU = IoU**.

**Answer:** **GIoU = 0.033, −0.100, −0.865, 0.450 (losses 0.967, 1.100, 1.865, 0.550).**
> ✔ Verified numerically (torchvision `generalized_box_iou`).
> **Common slip:** subtracting $\lvert E\rvert-\lvert I\rvert$ instead of $\lvert E\rvert-\lvert U\rvert$.

#### D38 — Attention-matrix sizes in DETR

**Given:** $1024\times768$, stride 32, 8 heads, 6+6 layers, 100 queries.

**Formula:** self-attention $n^2$ entries per head; cross-attention $N_q\times n$.

**Working:**
- (a) $32\times24=\mathbf{768}$ tokens; $768^2=\mathbf{589{,}824}$ per head per layer; ×8 = **4,718,592** per layer; ×6 layers × 2 B = **56.6 MB**.
- (b) Decoder self-attention $100^2=\mathbf{10{,}000}$; cross-attention $100\times768=\mathbf{76{,}800}$.
- (c) DC5: $64\times48=\mathbf{3072}$ tokens; $3072^2=\mathbf{9{,}437{,}184}$ per head per layer — **16×** (4× tokens, squared); $9{,}437{,}184\times48\times2$ B = **906 MB**.
- (d) $300^2=\mathbf{90{,}000}$ self; $300\times768=\mathbf{230{,}400}$ cross.

**Answer:** **768 tokens, 589,824 entries/head, 56.6 MB; decoder 10,000 / 76,800; DC5 3072 tokens, 9.44 M entries (16×), 906 MB; 300 queries → 90,000 / 230,400.**
> ✔ Verified numerically (Python).
> **Common slip:** scaling attention cost linearly with tokens — it is **quadratic** for self-attention (the reason Deformable DETR samples only a few points).

### S10 · Segmentation

#### D39 — Dice and IoU from pixel counts, pooled vs per-image, soft Dice

**Given:** image 1 (800, 200, 400), image 2 (30, 10, 60) as (TP, FP, FN).

**Formula:** $\text{IoU}=\frac{TP}{TP+FP+FN}$, $\text{Dice}=\frac{2TP}{2TP+FP+FN}$, $\text{Dice}=\frac{2\,\text{IoU}}{1+\text{IoU}}$, $\text{IoU}=\frac{\text{Dice}}{2-\text{Dice}}$.

**Working:**
- (a) Image 1: IoU $=800/1400=\mathbf{0.571}$, Dice $=1600/2200=\mathbf{0.727}$. Image 2: IoU $=30/100=\mathbf{0.300}$, Dice $=60/130=\mathbf{0.462}$.
- (b) Mean per-image Dice $=(0.7273+0.4615)/2=\mathbf{0.594}$. Pooled: TP 830, FP 210, FN 460 → Dice $=1660/2330=\mathbf{0.712}$. Pooling is dominated by the large lesion and **hides** the small-lesion failure; the per-image mean exposes it.
- (c) IoU 0.6 → Dice $=1.2/1.6=\mathbf{0.75}$; Dice 0.9 → IoU $=0.9/1.1=\mathbf{0.818}$.
- (d) $\sum pg=0.9+0.7+0.4=2.0$, $\sum p=2.3$, $\sum g=3$ → soft Dice $=4/5.3=\mathbf{0.755}$, loss $=\mathbf{0.245}$. Thresholded at 0.5: prediction $(1,1,0,0,0)$ → TP 2, FP 0, FN 1 → Dice $=4/5=\mathbf{0.80}$.

**Answer:** **IoU/Dice 0.571/0.727 and 0.300/0.462; mean Dice 0.594 vs pooled 0.712; 0.75 and 0.818; soft Dice 0.755 (loss 0.245), hard Dice 0.80.**
> ✔ Verified numerically (Python).
> **Common slip:** averaging IoU and then converting the average to Dice — the conversion is non-linear, convert **per image** first.

#### D40 — mIoU from a 3×3 confusion matrix

**Given:** matrix, rows = GT.

**Formula:** $\text{IoU}_c=\frac{TP_c}{\text{row}_c+\text{col}_c-TP_c}$.

**Working:**
- (a) $(4200+3300+1200)/10000=\mathbf{0.870}$.
- (b) Row sums 4400, 3800, 1800; column sums 4380, 3970, 1650.
  sky $4200/(4400+4380-4200)=4200/4580=\mathbf{0.917}$; building $3300/(3800+3970-3300)=3300/4470=\mathbf{0.738}$; vegetation $1200/(1800+1650-1200)=1200/2250=\mathbf{0.533}$. mIoU $=\mathbf{0.730}$.
- (c) Recalls $4200/4400=0.955$, $3300/3800=0.868$, $1200/1800=0.667$ → mean class accuracy **0.830**.
- (d) fwIoU $=0.44(0.917)+0.38(0.738)+0.18(0.533)=\mathbf{0.780}$. **mIoU** is the most pessimistic: it weights the rare, poorly segmented vegetation class equally and penalises both FP and FN.

**Answer:** **PA 0.870; IoU 0.917 / 0.738 / 0.533, mIoU 0.730; mean class acc 0.830; fwIoU 0.780.**
> ✔ Verified numerically (Python/numpy).
> **Common slip:** using $TP/\text{row}$ (recall) as the IoU — IoU also counts the false positives in the column.

#### D41 — Transposed-convolution output sizes

**Formula:** $W_{out}=(W-1)S-2P+K+P_{out}$ ($P_{out}$ = output_padding).

**Working:**
- (a) $6\cdot2-2+4=\mathbf{14}$.
- (b) $6\cdot2-2+3=\mathbf{13}$; with output_padding 1: **14**.
- (c) $27\cdot2-0+2=\mathbf{56}$.
- (d) $15\cdot32-32+64=\mathbf{512}$ (exactly ×32).
- (e) Weight $3\times3\times256\times128=294{,}912$ + 128 biases = **295,040**. Checkerboard artefacts arise when $K$ is **not divisible by $S$** (uneven overlap) → **(b)** ($K=3$, $S=2$). (a) $K=4,S=2$ and (c) $K=2,S=2$ overlap evenly.

**Answer:** **14; 13 / 14; 56; 512; 295,040 params; (b) is checkerboard-prone.**
> ✔ Verified numerically (PyTorch `ConvTranspose2d`).
> **Common slip:** applying the *forward* conv formula ($\lfloor(W-K+2P)/S\rfloor+1$) to a transposed conv.

#### D42 — U-Net valid-convolution size chain

**Given:** input 188, 3 levels, valid $3\times3$ convs.

**Formula:** each valid conv: $-2$; pool: $\div2$; up-conv ($K=2,S=2,P=0$): $\times2$; crop per side $=(\text{skip}-\text{decoder})/2$.

**Working:**
- (a) $188\to186\to\mathbf{184}$ (skip 1) → pool **92** → $90\to\mathbf{88}$ (skip 2) → pool **44** → $42\to\mathbf{40}$ (skip 3) → pool **20** → bottleneck $18\to\mathbf{16}$.
- (b)

| Step | up-conv size | skip | crop per side | after two convs |
|---|---|---|---|---|
| up 1 | $16\times2=32$ | 40 | **4** | $30\to28$ |
| up 2 | 56 | 88 | **16** | $54\to52$ |
| up 3 | 104 | 184 | **40** | $102\to100$ |

- (c) Output $\mathbf{100\times100}$; fraction $100^2/188^2=\mathbf{28.3\%}$; margin $(188-100)/2=\mathbf{44}$ px per side (context only — overlap-tile strategy).
- (d) $190\to188\to186\to93\to91\to\mathbf{89}$: odd before the second pooling → **not valid** (pooling would drop a row/column and the skips could not be centre-cropped consistently).

**Answer:** **184, 92, 88, 44, 40, 20, 16 → 32 (crop 4) → 28 → 56 (crop 16) → 52 → 104 (crop 40) → 100; output 100×100 (28.3%, margin 44 px); 190 invalid.**
> ✔ Verified numerically (Python + PyTorch forward pass of a 3-level valid U-Net).
> **Common slip:** forgetting that the bottleneck also has two convs ($20\to16$, not 20).

#### D43 — RoIAlign bilinear sampling

**Given:** RoI $(36,20,100,76)$, stride 16, $2\times2$ output, 2×2 samples/bin.

**Formula:** bilinear at $(x,y)$ with $x_0=\lfloor x\rfloor$, $a=x-x_0$, $b=y-y_0$:

$$
f=(1-a)(1-b)F_{00}+a(1-b)F_{10}+(1-a)b\,F_{01}+ab\,F_{11}.
$$

**Working:**
- (a) $/16$ → $(2.25,\ 1.25,\ 6.25,\ 4.75)$; size $4\times3.5$ cells → bin $\mathbf{2\times1.75}$ cells.
- (b) $x=2.25+\{0.5,1.5\}\cdot1=\mathbf{2.75,\ 3.75}$; $y=1.25+\{0.5,1.5\}\cdot0.875=\mathbf{1.6875,\ 2.5625}$.
- (c)

| Sample | corners $F_{00},F_{10},F_{01},F_{11}$ | $(a,b)$ | value |
|---|---|---|---|
| (2.75, 1.6875) | 12, 18, 16, 30 | (0.75, 0.6875) | $0.0781(12)+0.2344(18)+0.1719(16)+0.5156(30)=$ **23.375** |
| (3.75, 1.6875) | 18, 20, 30, 26 | (0.75, 0.6875) | **24.656** |
| (2.75, 2.5625) | 16, 30, 10, 22 | (0.75, 0.5625) | **22.281** |
| (3.75, 2.5625) | 30, 26, 22, 40 | (0.75, 0.5625) | **31.781** |

  Bin output $=(23.375+24.656+22.281+31.781)/4=\mathbf{25.523}$.
- (d) Nearest neighbour of $(2.75,1.6875)$ → cell $(3,2)$ → **30** (vs 23.4). RoI Pool's rounding shifts features by up to half a stride (8 px) with round-to-nearest, or up to a full stride (16 px) with floor; for 28×28 masks that misalignment is visible, which is why Mask R-CNN introduced RoIAlign.

**Answer:** **RoI (2.25, 1.25, 6.25, 4.75), bins 2×1.75; samples 23.375, 24.656, 22.281, 31.781 → bin value 25.52; NN would give 30.**
> ✔ Verified numerically (torchvision `roi_align`, `aligned=False`, `sampling_ratio=2`).
> **Common slip:** placing samples at the bin corners instead of at $(i+0.5)$ fractions of the bin.

#### D44 — Panoptic quality with a borderline match

**Formula:** $SQ=\frac{\sum_{TP}\text{IoU}}{\lvert TP\rvert}$, $RQ=\frac{\lvert TP\rvert}{\lvert TP\rvert+\frac12\lvert FP\rvert+\frac12\lvert FN\rvert}$, $PQ=SQ\times RQ$.

**Working:**
- (a) Matches need IoU **> 0.5**: 0.88, 0.74, 0.62 → **TP = 3**. $(p_4,g_4)=0.50$ is **not** a match. Unmatched predictions $p_4,p_5$ → **FP = 2**; unmatched GT $g_4$ → **FN = 1**.
- (b) $SQ=(0.88+0.74+0.62)/3=\mathbf{0.747}$; $RQ=3/(3+1+0.5)=\mathbf{0.667}$; $PQ=\mathbf{0.498}$.
- (c) sky: $PQ=0.95\times1=\mathbf{0.95}$. grass: IoU 0.45 → FP 1, FN 1, TP 0 → $PQ=\mathbf0$. Overall $(0.498+0.95+0)/3=\mathbf{0.483}$; $PQ^{Th}=\mathbf{0.498}$, $PQ^{St}=(0.95+0)/2=\mathbf{0.475}$.

**Answer:** **TP 3, FP 2, FN 1; SQ 0.747, RQ 0.667, PQ 0.498; sky 0.95, grass 0; PQ 0.483 ($PQ^{Th}$ 0.498, $PQ^{St}$ 0.475).**
> ✔ Verified numerically (Python).
> **Common slip:** accepting IoU = 0.50 as a match — the rule is strictly greater than 0.5 (that is what makes the matching unique).

### S11 · Camera geometry

#### D45 — Projection with K[R|t] and the cheirality check

**Given:** $K$, $R$, $\mathbf t$; three world points.

**Formula:** $X_c=RX_w+\mathbf t$; $u=f_x\frac{X_c}{Z_c}+c_x$, $v=f_y\frac{Y_c}{Z_c}+c_y$; visible only if $Z_c>0$ and $(u,v)$ inside the image.

**Working:**
- (a) Row norms: $1$; $0.96^2+0.28^2=0.9216+0.0784=1$; $0.28^2+0.96^2=1$. Row 2 · row 3 $=0.96(0.28)+(-0.28)(0.96)=0$ (rows 1–2 and 1–3 are trivially orthogonal). $\det R=0.96\cdot0.96-(-0.28)(0.28)=1$ ✓ → a rotation about the $x$-axis by $\arctan(0.28/0.96)=16.26^\circ$.
- (b) $RX_A=(1,\ 0.96\cdot2-0.28\cdot3,\ 0.28\cdot2+0.96\cdot3)=(1,\ 1.08,\ 3.44)$; $X_c=(1.2,\ 0.08,\ 7.44)$.
  $u=1000\cdot1.2/7.44+640=161.29+640=\mathbf{801.29}$; $v=1010\cdot0.08/7.44+360=10.86+360=\mathbf{370.86}$.
- (c) $RX_B=(-1.5,\ 0.48-1.68,\ 0.14+5.76)=(-1.5,\ -1.2,\ 5.9)$; $X_c=(-1.3,\ -2.2,\ 9.9)$.
  $u=-131.31+640=\mathbf{508.69}$; $v=1010(-2.2)/9.9+360=-224.44+360=\mathbf{135.56}$.
- (d) $RX_C=(0,\ -1.92+4.2,\ -0.56-14.4)=(0,\ 2.28,\ -14.96)$; $X_c=(0.2,\ 1.28,\ -10.96)$. Algebraically $u=1000(0.2)/(-10.96)+640=\mathbf{621.75}$, $v=1010(1.28)/(-10.96)+360=\mathbf{242.04}$ — inside the image, **but $Z_c<0$: the point is behind the camera and is not visible.** Always check the sign of $Z_c$ (cheirality).

**Answer:** **$X_A\to(801.29,\ 370.86)$; $X_B\to(508.69,\ 135.56)$; $X_C$ has $Z_c=-10.96$ → behind the camera (the "pixel" (621.8, 242.0) is meaningless).**
> ✔ Verified numerically (Python + `cv2.projectPoints`).
> **Common slip:** using $f_x$ for $v$ — here $f_y=1010\ne f_x$.

#### D46 — Focal length in pixels and field of view

**Given:** $f=12$ mm, pitch 3.45 µm $=0.00345$ mm, $2448\times2048$.

**Formula:** $f_x=f/\text{pitch}$; $\text{FOV}=2\arctan\frac{N/2}{f_x}$.

**Working:**
- (a) $f_x=12/0.00345=\mathbf{3478.26}$ px. Sensor $2448\times0.00345=\mathbf{8.446}$ mm by $2048\times0.00345=\mathbf{7.066}$ mm.
- (b) HFOV $=2\arctan(1224/3478.26)=2\arctan0.3519=\mathbf{38.77^\circ}$; VFOV $=2\arctan(1024/3478.26)=\mathbf{32.81^\circ}$; DFOV $=2\arctan(1595.9/3478.26)=\mathbf{49.29^\circ}$ (half-diagonal $\sqrt{1224^2+1024^2}=1595.9$ px).
- (c) $f=\frac{8.4456/2}{\tan30^\circ}=\frac{4.2228}{0.57735}=\mathbf{7.31}$ mm.
- (d) GSD $=Z/f_x=2000/3478.26=\mathbf{0.575}$ mm/px → 50 mm spans $50/0.575=\mathbf{87}$ px.
- (e) Binning → $\mathbf{1224\times1024}$ pixels of 6.9 µm → $f_x=12/0.0069=\mathbf{1739.13}$ px; HFOV $=2\arctan(612/1739.13)=\mathbf{38.77^\circ}$ (unchanged — same sensor area, coarser sampling).

**Answer:** **$f_x=3478$ px; sensor 8.45×7.07 mm; FOV 38.8° / 32.8° / 49.3°; 7.31 mm lens for 60°; 0.575 mm/px, 87 px; binned: 1224×1024, $f_x=1739$, FOV unchanged.**
> ✔ Verified numerically (Python).
> **Common slip:** using the full width instead of the **half**-width in $\arctan$ (and forgetting the factor 2).

#### D47 — Radial + tangential distortion

**Given:** $(x,y)=(-0.45,0.30)$, $(k_1,k_2,p_1,p_2,k_3)=(0.08,-0.01,0.0012,-0.0008,0)$.

**Formula:** $x_d=x(1+k_1r^2+k_2r^4+k_3r^6)+2p_1xy+p_2(r^2+2x^2)$, $y_d=y(1+k_1r^2+k_2r^4+k_3r^6)+p_1(r^2+2y^2)+2p_2xy$.

**Working:**
- (a) $u=900(-0.45)+640=\mathbf{235}$, $v=900(0.30)+360=\mathbf{630}$.
- (b) $r^2=0.2025+0.09=\mathbf{0.2925}$, $r^4=0.085556$. Radial factor $=1+0.08(0.2925)-0.01(0.085556)=1+0.0234-0.000856=\mathbf{1.022544}$.
  Radial part: $x\cdot1.022544=-0.460145$, $y\cdot1.022544=0.306763$.
  Tangential: $\delta x=2(0.0012)(-0.135)+(-0.0008)(0.2925+0.405)=-0.000324-0.000558=\mathbf{-0.000882}$; $\delta y=0.0012(0.2925+0.18)+2(-0.0008)(-0.135)=0.000567+0.000216=\mathbf{0.000783}$.
  $x_d=\mathbf{-0.461027}$, $y_d=\mathbf{0.307546}$.
- (c) $u_d=900(-0.461027)+640=\mathbf{225.08}$, $v_d=900(0.307546)+360=\mathbf{636.79}$. Shift $(-9.92,\ +6.79)$, $\mathbf{12.0}$ px; distance from the principal point grows from 486.7 to 498.8 px → points move **outwards** → **pincushion** ($k_1>0$).

**Answer:** **ideal (235, 630); $r^2=0.2925$, factor 1.02254, $\delta=(-0.000882,\ 0.000783)$, $(x_d,y_d)=(-0.46103,\ 0.30755)$; distorted (225.08, 636.79), 12.0 px outward → pincushion.**
> ✔ Verified numerically (Python + `cv2.projectPoints` with the same coefficients).
> **Common slip:** applying distortion in **pixel** coordinates — it is applied to normalised coordinates, then $K$ is applied.

#### D48 — Camera centre and extrinsics

**Given:** $R$, $\mathbf t=(1.0,-0.5,3.0)$.

**Formula:** $C=-R^\top\mathbf t$; optical axis (world) = 3rd row of $R$; $\mathbf t=-RC$.

**Working:**
- (a) $R^\top\mathbf t=(0.8\cdot1+0.6\cdot3,\ -0.5,\ -0.6\cdot1+0.8\cdot3)=(2.6,\ -0.5,\ 1.8)$ → $C=\mathbf{(-2.6,\ 0.5,\ -1.8)}$. Check: $RC=(0.8(-2.6)-0.6(-1.8),\ 0.5,\ 0.6(-2.6)+0.8(-1.8))=(-1.0,\ 0.5,\ -3.0)=-\mathbf t$ ✓.
- (b) $\mathbf r_3=\mathbf{(0.6,\ 0,\ 0.8)}$ (the camera's $z$-axis expressed in world coordinates).
- (c) $Z_c=\mathbf r_3\cdot X_w+t_z=0.6+4.0+3.0=\mathbf{7.6}$ m; or $(X_w-C)\cdot\mathbf r_3=(3.6,\ -0.5,\ 6.8)\cdot(0.6,0,0.8)=2.16+5.44=\mathbf{7.6}$ m ✓.
- (d) $RC_2=(1.6+2.4,\ 1.5,\ 1.2-3.2)=(4.0,\ 1.5,\ -2.0)$ → $\mathbf t_2=\mathbf{(-4.0,\ -1.5,\ 2.0)}$.

**Answer:** **$C=(-2.6,0.5,-1.8)$; axis $(0.6,0,0.8)$; $Z_c=7.6$ m; $\mathbf t_2=(-4,-1.5,2)$.**
> ✔ Verified numerically (Python; also $P\tilde C=\mathbf 0$).
> **Common slip:** taking $\mathbf t$ itself as the camera position — $\mathbf t$ is the world origin expressed in camera coordinates.

#### D49 — Ground-plane (Z = 0) homography and back-projection

**Given:** $K$, $R$, $C=(0,0,1.5)$.

**Formula:** for $Z=0$: $\lambda(u,v,1)^\top=K\,[\mathbf r_1\ \mathbf r_2\ \mathbf t]\;(X,Y,1)^\top=H\,(X,Y,1)^\top$.

**Working:**
- (a) $RC=(0,\ -0.8\cdot1.5,\ -0.6\cdot1.5)=(0,-1.2,-0.9)$ → $\mathbf t=\mathbf{(0,\ 1.2,\ 0.9)}$.
- (b) $[\mathbf r_1\ \mathbf r_2\ \mathbf t]=\begin{bmatrix}1&0&0\\0&-0.6&1.2\\0&0.8&0.9\end{bmatrix}$ → $H=K[\cdot]=\begin{bmatrix}600&256&288\\0&-168&936\\0&0.8&0.9\end{bmatrix}$ (row 1 $=600\,\text{row}_1+320\,\text{row}_3$, row 2 $=600\,\text{row}_2+240\,\text{row}_3$).
- (c) $(0,2)$: $H(0,2,1)^\top=(800,\ 600,\ 2.5)$ → $\mathbf{(320,\ 240)}$ = the principal point: the optical axis hits the ground at $1.5/\tan\varphi=1.5/0.75=2$ m. $(1,5)$: $(2168,\ 96,\ 4.9)$ → $\mathbf{(442.45,\ 19.59)}$ (farther points appear higher in the image).
- (d) Ray in camera frame: $K^{-1}(400,300,1)^\top=(0.1333,\ 0.1,\ 1)$. World direction $R^\top(\cdot)=(0.1333,\ -0.6(0.1)+0.8(1),\ -0.8(0.1)-0.6(1))=(0.1333,\ 0.74,\ -0.68)$. From $C$: $Z=1.5-0.68\lambda=0$ → $\lambda=2.2059$ → ground point $\mathbf{(0.294,\ 1.632)}$ m. Check: $H(0.294,1.632,1)^\top\to(400,300)$ ✓.

**Answer:** **$\mathbf t=(0,1.2,0.9)$; $H=\begin{bmatrix}600&256&288\\0&-168&936\\0&0.8&0.9\end{bmatrix}$; (0,2) → (320, 240), (1,5) → (442.4, 19.6); pixel (400,300) → ground (0.294, 1.632) m.**
> ✔ Verified numerically (Python; $H$ vs full projection and $H^{-1}$).
> **Common slip:** using the **rows** of $R$ for $\mathbf r_1,\mathbf r_2$ — they are the first two **columns**.

### S12 · 3D vision & motion

#### D50 — Depth and 3D point from disparity

**Given:** $f=1050$, $B=0.30$ m, $c=(640,360)$.

**Formula:** $Z=\frac{fB}{d}$, $X=\frac{(x_L-c_x)Z}{f}$, $Y=\frac{(y-c_y)Z}{f}$.

**Working:**
- (a) $d=700-679=\mathbf{21}$ px; $fB=315$ px·m → $Z=315/21=\mathbf{15.0}$ m; $X=60\cdot15/1050=\mathbf{0.857}$ m; $Y=40\cdot15/1050=\mathbf{0.571}$ m.
- (b) $d=315/45=\mathbf{7}$ px.
- (c) $f=\mathbf{525}$ px, $d=\mathbf{10.5}$ px, $Z=525\cdot0.3/10.5=\mathbf{15}$ m (unchanged), but 1 px of disparity error now costs twice as much depth.
- (d) $d_{max}=315/2=\mathbf{157.5}$ px → `numDisparities` = **160**.

**Answer:** **$d=21$ px, $(X,Y,Z)=(0.857,\ 0.571,\ 15.0)$ m; 7 px; $f=525$, $d=10.5$, $Z=15$ m; 157.5 px → 160.**
> ✔ Verified numerically (Python).
> **Common slip:** using $B$ in cm (30) with $Z$ expected in m.

#### D51 — Depth error and baseline design

**Given:** $f=1050$, $B=0.30$, $\Delta d=0.25$.

**Formula:** $\Delta Z=\frac{Z^2}{fB}\Delta d$ ⇒ $B=\frac{Z^2\Delta d}{f\,\Delta Z}$, $f=\frac{Z^2\Delta d}{B\,\Delta Z}$.

**Working:**
- (a) $Z=10$: $100/315\times0.25=\mathbf{0.079}$ m (0.79%); $Z=30$: $\mathbf{0.714}$ m (2.38%); $Z=60$: $\mathbf{2.857}$ m (4.76%). Relative error grows linearly with $Z$.
- (b) $B=3600\times0.25/(1050\times1)=\mathbf{0.857}$ m.
- (c) $Z_{min}=fB/d_{max}=1050\times0.857/192=900/192=\mathbf{4.69}$ m vs $315/192=\mathbf{1.64}$ m — a wider baseline buys far-range accuracy but loses close range (and increases occlusions).
- (d) $f=3600\times0.25/(0.30\times1)=\mathbf{3000}$ px. HFOV $=2\arctan(640/1050)=\mathbf{62.7^\circ}$ → $2\arctan(640/3000)=\mathbf{24.1^\circ}$: a much narrower view.

**Answer:** **0.079 / 0.714 / 2.857 m; $B=0.857$ m; $Z_{min}$ 4.69 m (vs 1.64 m); $f=3000$ px, FOV 62.7° → 24.1°.**
> ✔ Verified numerically (Python).
> **Common slip:** treating $\Delta Z$ as linear in $Z$ — it grows with $Z^2$.

#### D52 — Epipolar line from F (forward motion)

**Given:** $F=[\mathbf e]_\times$ with $\mathbf e=(400,300,1)$.

**Formula:** $\mathbf l'=F\mathbf x$; distance of $\mathbf x'$ from $\mathbf l'=(a,b,c)$: $\frac{\lvert ax'+by'+c\rvert}{\sqrt{a^2+b^2}}$.

**Working:**
- (a) $F(500,350,1)^\top=(0-350+300,\ 500+0-400,\ -150000+140000+0)=\mathbf{(-50,\ 100,\ -10000)}$. Normalised by $\sqrt{50^2+100^2}=111.80$: $(-0.4472,\ 0.8944,\ -89.44)$. Line: $-50x+100y-10000=0$ ⇔ $\mathbf{y=0.5x+100}$.
- (b) $\mathbf x_1'$: $-50(560)+100(382)-10000=200$ → $200/111.80=\mathbf{1.79}$ px → **accept**. $\mathbf x_2'$: $-27000+39500-10000=2500$ → $\mathbf{22.36}$ px → **reject**.
- (c) $F\mathbf e=(-300+300,\ 400-400,\ -120000+120000)=\mathbf 0$ ✓; $F^\top=-F$ so $F^\top\mathbf e=\mathbf 0$ ✓. $\mathbf e$ is the **focus of expansion**: the image of the direction of travel; all epipolar lines pass through it (indeed $0.5\cdot400+100=300$), and points move away from it along these lines ($\mathbf x$ is 111.8 px from the FOE, $\mathbf x_1'$ 179.8 px).
- (d) $F$ is skew-symmetric → $\det F=0$, **rank 2** ✓. A general $F$ has **7 DOF** (this pure-translation $F$ has only 2: the epipole position).

**Answer:** **$\mathbf l'=(-50,100,-10000)$, $y=0.5x+100$; distances 1.79 px (accept) and 22.36 px (reject); epipole (400, 300) = FOE; rank 2, 7 DOF in general.**
> ✔ Verified numerically (Python).
> **Common slip:** forgetting to normalise $(a,b)$ before reading the line value as a distance in pixels.

#### D53 — Essential matrix check

**Given:** $R=R_y$ ($\cos=0.8$, $\sin=0.6$), $\mathbf t=(1,0,0)$.

**Formula:** $[\mathbf t]_\times=\begin{bmatrix}0&-t_3&t_2\\t_3&0&-t_1\\-t_2&t_1&0\end{bmatrix}$; valid $E$: singular values $(\sigma,\sigma,0)$.

**Working:**
- (a) $[\mathbf t]_\times=\begin{bmatrix}0&0&0\\0&0&-1\\0&1&0\end{bmatrix}$ → $E$: row 1 $=\mathbf 0$, row 2 $=-(\text{row 3 of }R)=(0.6,0,-0.8)$, row 3 $=\text{row 2 of }R=(0,1,0)$: $\mathbf{E=\begin{bmatrix}0&0&0\\0.6&0&-0.8\\0&1&0\end{bmatrix}}$.
- (b) $E^\top E=\begin{bmatrix}0.36&0&-0.48\\0&1&0\\-0.48&0&0.64\end{bmatrix}$ has eigenvalues 1, 1, 0 → singular values **(1, 1, 0)**; $\det E=0$ → **valid**.
- (c) $X_2=RX_1+\mathbf t=(0.4+2.4+1,\ -0.2,\ -0.3+3.2)=\mathbf{(3.8,\ -0.2,\ 2.9)}$. $\hat{\mathbf x}_1=(0.125,\ -0.05,\ 1)$, $\hat{\mathbf x}_2=(1.3103,\ -0.06897,\ 1)$. $E\hat{\mathbf x}_1=(0,\ 0.075-0.8,\ -0.05)=(0,\ -0.725,\ -0.05)$; $\hat{\mathbf x}_2^\top E\hat{\mathbf x}_1=(-0.06897)(-0.725)+1(-0.05)=0.05-0.05=\mathbf0$ ✓.
- (d) Replace the singular values by $\left(\frac{2.1+1.7}2,\frac{2.1+1.7}2,0\right)=\mathbf{(1.9,\ 1.9,\ 0)}$ (or $(1,1,0)$ since $E$ is only defined up to scale).
- (e) $M=[\mathbf t]_\times$ with $\mathbf t=(1,0,2)$ (and $R=I$); singular values $(\sqrt5,\sqrt5,0)=(2.236,2.236,0)$ → **valid**: a **pure translation** along the line $\pm(1,0,2)$ (sideways + along the optical axis; $E$ fixes $\mathbf t$ only up to sign, and with $X_2=X_1+\mathbf t$ the camera centre itself moves by $-\mathbf t$).

**Answer:** **$E=\begin{bmatrix}0&0&0\\0.6&0&-0.8\\0&1&0\end{bmatrix}$, singular values (1, 1, 0); constraint = 0 ✓; projection → (1.9, 1.9, 0); $M$ valid (pure translation along (1, 0, 2)).**
> ✔ Verified numerically (Python SVD).
> **Common slip:** computing $R[\mathbf t]_\times$ instead of $[\mathbf t]_\times R$ (different matrix unless $\mathbf t$ is along the rotation axis).

#### D54 — Lucas–Kanade from per-pixel gradients

**Formula:** $\begin{bmatrix}\sum I_x^2&\sum I_xI_y\\\sum I_xI_y&\sum I_y^2\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix}=-\begin{bmatrix}\sum I_xI_t\\\sum I_yI_t\end{bmatrix}$.

**Working:**
- (a) $\sum I_x^2=16+9+1+4+25=\mathbf{55}$; $\sum I_xI_y=4+6-3-4+0=\mathbf3$; $\sum I_y^2=1+4+9+4+0=\mathbf{18}$; $\sum I_xI_t=-24-21+2+2-40=\mathbf{-81}$; $\sum I_yI_t=-6-14-6-2+0=\mathbf{-28}$ → $\begin{bmatrix}55&3\\3&18\end{bmatrix}\begin{bmatrix}u\\v\end{bmatrix}=\begin{bmatrix}81\\28\end{bmatrix}$.
- (b) $\det=990-9=981$; $u=\frac{81\cdot18-3\cdot28}{981}=\frac{1374}{981}=\mathbf{1.401}$; $v=\frac{55\cdot28-3\cdot81}{981}=\frac{1297}{981}=\mathbf{1.322}$ px/frame.
- (c) $\lambda=\frac{73\pm\sqrt{37^2+36}}{2}=\frac{73\pm37.48}{2}$ → **55.24, 17.76**; both clearly non-zero, condition number 3.1 → a corner-like, **trackable** window.
- (d) All gradients ∥ (2, 1): $A^\top A=\begin{bmatrix}60&30\\30&15\end{bmatrix}$, $\det=900-900=\mathbf0$, eigenvalues **0 and 75** → singular: **aperture problem** — only the flow component along $(2,1)/\sqrt5$ (normal flow) is recoverable.

**Answer:** **$(u,v)=(1.401,\ 1.322)$ px/frame; eigenvalues 55.2, 17.8 (trackable); second window singular (eig 0, 75) → only normal flow.**
> ✔ Verified numerically (numpy `solve`/`eigvalsh`).
> **Common slip:** dropping the minus sign on the right-hand side ($-\sum I_xI_t$).

#### D55 — Normal flow and the aperture problem

**Given:** $\nabla I=(-6,8)$, $\Vert\nabla I\Vert=10$, $I_t=20$.

**Formula:** constraint $I_xu+I_yv+I_t=0$; $\mathbf u_n=-\frac{I_t}{\Vert\nabla I\Vert^2}\nabla I$.

**Working:**
- (a) $\mathbf u_n=-\frac{20}{100}(-6,8)=\mathbf{(1.2,\ -1.6)}$; magnitude $=\lvert I_t\rvert/\Vert\nabla I\Vert=\mathbf{2.0}$ px/frame.
- (b) $(-2,-4)$: $12-32+20=0$ ✓; $(2.8,-0.4)$: $-16.8-3.2+20=0$ ✓. Tangential direction $\perp\nabla I$: $(0.8,0.6)$. $(-2,-4)-(1.2,-1.6)=\mathbf{(-3.2,\ -2.4)}=-4\,(0.8,0.6)$ → normal $(1.2,-1.6)$ + tangential $(-3.2,-2.4)$. Every flow on this line is consistent — the tangential part is invisible (aperture problem).
- (c) Second pixel: $5u-4=0$ → $u=\mathbf{0.8}$; first: $-6(0.8)+8v+20=0$ → $v=\mathbf{-1.9}$. Two independent gradient directions fix the flow (the idea behind Lucas–Kanade).

**Answer:** **$\mathbf u_n=(1.2,-1.6)$, |·| = 2; both flows satisfy the constraint; tangential part $(-3.2,-2.4)$; full flow $(0.8,\ -1.9)$.**
> ✔ Verified numerically (Python).
> **Common slip:** normal-flow magnitude $=\lvert I_t\rvert/\Vert\nabla I\Vert$, not $\lvert I_t\rvert/\Vert\nabla I\Vert^2$.

#### D56 — Bundle-adjustment problem size

**Given:** 120 cameras, 3 shared intrinsics, 25,000 points, 5 obs/point.

**Working:**
- (a) Unknowns $=120\cdot6+3+25000\cdot3=720+3+75000=\mathbf{75{,}723}$; observations $=25000\cdot5=\mathbf{125{,}000}$; residuals $=2\times125000=\mathbf{250{,}000}$ (≈3.3 residuals per unknown).
- (b) Monocular SfM is defined up to a **similarity**: 3 rotation + 3 translation + 1 scale = **7** gauge DOF → **75,716** effective unknowns (fix them by fixing a camera and a distance, or let the solver handle the gauge).
- (c) Each residual depends on 6 pose + 3 point + 3 intrinsic parameters = **12** non-zeros → $250{,}000\times12=\mathbf{3{,}000{,}000}$ non-zeros out of $250{,}000\times75{,}723=1.89\times10^{10}$ → density **0.016%** (extremely sparse).
- (d) Eliminating the 25,000 independent $3\times3$ point blocks leaves the camera (+intrinsics) system: $\mathbf{723\times723}$.

**Answer:** **75,723 unknowns, 125,000 observations, 250,000 residuals; gauge 7 → 75,716; 12 nnz/row, 3 M nnz, 0.016% dense; reduced system 723×723.**
> ✔ Verified numerically (Python).
> **Common slip:** counting one residual per observation — each image point gives **two** ($u$ and $v$).

### S13–14 · Foundation models & pipelines

#### D57 — CLIP zero-shot with a logit scale

**Given:** sims $(0.28,0.26,0.21)$.

**Formula:** $p_c=\operatorname{softmax}(100\cdot s)_c$.

**Working:**
- (a) Logits $(28,26,21)$; relative to the max: $e^0=1$, $e^{-2}=0.13534$, $e^{-7}=0.000912$; sum $1.13625$ → $p=\mathbf{(0.880,\ 0.119,\ 0.0008)}$ → **husky**.
- (b) Logits $(14,13,10.5)$: $1,\ e^{-1}=0.3679,\ e^{-3.5}=0.0302$ → $p=\mathbf{(0.715,\ 0.263,\ 0.022)}$ — same ranking, less confident.
- (c) 2 classes: $p=\frac{1}{1+e^{-100\Delta}}=0.95$ ⇒ $\Delta=\ln19/100=\mathbf{0.0294}$; 90%: $\ln9/100=\mathbf{0.0220}$. Tiny cosine gaps give confident CLIP predictions — the logit scale does the sharpening.
- (d) $\Vert(\mathbf t_1+\mathbf t_2)/2\Vert=\sqrt{(1+1+2\cdot0.9)/4}=\sqrt{0.95}=0.9747$; similarity $=\frac{(0.28+0.30)/2}{0.9747}=\mathbf{0.2975}$ (re-normalisation raises the averaged score).

**Answer:** **(0.880, 0.119, 0.001) → husky; scale 50: (0.715, 0.263, 0.022); gaps 0.029 (95%) and 0.022 (90%); ensemble sim 0.2975.**
> ✔ Verified numerically (Python).
> **Common slip:** applying softmax to the raw cosines (scale 1) — gives nearly uniform probabilities.

#### D58 — InfoNCE matrix sizes at CLIP scale

**Given:** $N=4096$, $d=512$, 8 GPUs.

**Working:**
- (a) $N^2=\mathbf{16{,}777{,}216}$ entries; **4096** positives (diagonal); **4095** negatives per row; $N(N-1)=\mathbf{16{,}773{,}120}$ negatives per direction (the symmetric loss uses the same matrix row-wise and column-wise).
- (b) $16{,}777{,}216\times4$ B $=\mathbf{67.1}$ MB; per-GPU block $512\times4096=2{,}097{,}152$ entries $=\mathbf{8.39}$ MB.
- (c) $N^2d=\mathbf{8.59\times10^9}$ MACs; per GPU $512\cdot4096\cdot512=\mathbf{1.07\times10^9}$.
- (d) Uniform logits → $\mathcal L=\ln N=\ln4096=\mathbf{8.318}$ nats per direction (a useful sanity check for the first training step).

**Answer:** **16.8 M entries, 4096 positives, 4095 negatives/row, 16.77 M negatives; 67.1 MB (8.39 MB per GPU); 8.59 G MACs (1.07 G per GPU); initial loss 8.318.**
> ✔ Verified numerically (Python).
> **Common slip:** counting $N-1$ negatives for the **whole** batch instead of per row.

#### D59 — Scale-and-shift alignment of relative depth (least squares, 3 points)

**Given:** $(r,Z)$: $(0.9,1.6)$, $(0.5,3.2)$, $(0.2,10.0)$.

**Formula:** minimise $\sum(sr_i+t-y_i)^2$, $y=1/Z$.

**Working:**
- (a) $y=\mathbf{(0.625,\ 0.3125,\ 0.1)}$ m⁻¹.
- (b) $\sum r=1.6$, $\sum r^2=0.81+0.25+0.04=1.1$, $\sum ry=0.5625+0.15625+0.02=0.73875$, $\sum y=1.0375$, $n=3$:

$$
\begin{bmatrix}1.1&1.6\\1.6&3\end{bmatrix}\begin{bmatrix}s\\t\end{bmatrix}=\begin{bmatrix}0.73875\\1.0375\end{bmatrix},\quad \det=3.3-2.56=0.74 .
$$

  $s=\frac{3(0.73875)-1.6(1.0375)}{0.74}=\frac{0.55625}{0.74}=\mathbf{0.7517}$; $t=\frac{1.1(1.0375)-1.6(0.73875)}{0.74}=\frac{-0.04075}{0.74}=\mathbf{-0.05507}$.
- (c) Fitted $y$: $0.62145,\ 0.32078,\ 0.09527$ → $Z=\mathbf{1.609,\ 3.117,\ 10.496}$ m (errors $+0.009$, $-0.083$, $+0.496$ m). The inverse-depth residuals are all tiny ($<0.01$ m⁻¹), but $\Delta Z\approx Z^2\,\Delta(1/Z)$, so at 10 m a 0.0047 m⁻¹ residual becomes a 0.5 m depth error.
- (d) $r=0.35$: $y=0.7517(0.35)-0.05507=0.20802$ → $Z=\mathbf{4.81}$ m. $r=0.06$: $y=-0.00997<0$ → **invalid** (negative inverse depth — extrapolation beyond the calibrated range, e.g. sky); clamp to "far/infinite".

**Answer:** **$s=0.7517$, $t=-0.0551$; fitted 1.61 / 3.12 / 10.50 m; $r=0.35$ → 4.81 m; $r=0.06$ → negative inverse depth (invalid).**
> ✔ Verified numerically (numpy `lstsq` and explicit normal equations).
> **Common slip:** fitting $Z=sr+t$ directly — Depth Anything's output is affine-invariant **inverse** depth, so align in $1/Z$.

#### D60 — Object size and 3D distance from a box + metric depth

**Given:** $f=1400$, $c=(960,540)$.

**Formula:** size $=\frac{\text{pixels}\times Z}{f}$; $X=\frac{(u-c_x)Z}{f}$, $Y=\frac{(v-c_y)Z}{f}$.

**Working:**
- (a) Height $720-300=420$ px → $420\cdot5/1400=\mathbf{1.50}$ m; width 100 px → $\mathbf{0.357}$ m.
- (b) Person: $(-10\cdot5/1400,\ -30\cdot5/1400,\ 5)=\mathbf{(-0.036,\ -0.107,\ 5.0)}$; dog: $(340\cdot4/1400,\ 260\cdot4/1400,\ 4)=\mathbf{(0.971,\ 0.743,\ 4.0)}$. Difference $(1.007,\ 0.850,\ -1.0)$ → distance $\sqrt{1.014+0.723+1.0}=\mathbf{1.654}$ m.
- (c) Size $\propto Z$ → **+10%** → 1.65 m. Relative-depth models need metric calibration (D59) before sizes mean anything.

**Answer:** **1.50 m × 0.357 m; person (−0.036, −0.107, 5.0), dog (0.971, 0.743, 4.0), 1.65 m apart; 10% depth error → 10% size error (1.65 m).**
> ✔ Verified numerically (Python).
> **Common slip:** subtracting pixel coordinates of the two objects and scaling by one depth — back-project each with its **own** depth first.

#### D61 — Detection → segmentation → depth pipeline latency

**Given:** GDINO 120, SAM-enc 100, SAM-dec 10/box, depth 25, post 5 ms; 4 boxes.

**Working:**
- (a) $120+100+4\cdot10+25+5=\mathbf{290}$ ms → $1000/290=\mathbf{3.45}$ FPS.
- (b) Branch 1: $120+100+40=260$ ms; branch 2: 25 ms → $\max(260,25)+5=\mathbf{265}$ ms → **3.77 FPS** (little gain: depth was never the bottleneck).
- (c) Stage times 120, 140, 30 ms → throughput $=1/140$ ms $=\mathbf{7.14}$ FPS; latency still $120+140+30=\mathbf{290}$ ms (pipelining raises throughput, not latency).
- (d) Average $=(290+9\times38)/10=(290+342)/10=\mathbf{63.2}$ ms → **15.8 FPS** < 30 FPS → it **cannot** keep up (process every 2nd frame, use a lighter detector, or run on more GPUs).

**Answer:** **290 ms / 3.45 FPS; 265 ms / 3.77 FPS; 7.14 FPS at 290 ms latency; keyframing 63.2 ms / 15.8 FPS (not real-time at 30 FPS).**
> ✔ Verified numerically (Python).
> **Common slip:** forgetting that the SAM decoder runs **once per box**.

### S15 · Generative CV

#### D62 — Noise schedule: ᾱ_t and x_t

**Formula:** constant $\beta$: $\bar\alpha_t=(1-\beta)^t$; $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$.

**Working:**
- (a) $\bar\alpha_{10}=0.98^{10}=\mathbf{0.8171}$, SNR $=0.8171/0.1829=\mathbf{4.47}$; $\bar\alpha_{100}=0.98^{100}=\mathbf{0.1326}$, SNR $=\mathbf{0.153}$.
- (b) $\sqrt{0.81707}=0.90392$, $\sqrt{0.18293}=0.42770$: $x_{10}=(0.54235+0.42770,\ -0.72314+0.21385)=\mathbf{(0.9700,\ -0.5093)}$.
- (c) $t>\ln0.01/\ln0.98=227.95$ → $t=\mathbf{228}$ (check: $0.98^{227}=0.0102$, $0.98^{228}=0.00999$).
- (d) $\sum_{s=1}^{1000}\beta_s=1000\times\frac{0.0001+0.02}{2}=10.05$ → $\bar\alpha_{1000}\approx e^{-10.05}=\mathbf{4.32\times10^{-5}}$ (exact $4.04\times10^{-5}$). $\sum_{s\le500}\beta_s\approx500\times\frac{0.0001+0.01004}{2}=2.535$ → $\bar\alpha_{500}\approx e^{-2.535}=\mathbf{0.0793}$ (exact 0.0786). The approximation $\ln(1-\beta)\approx-\beta$ is good because every $\beta\le0.02$; it slightly **over**-estimates $\bar\alpha$ (the dropped $-\beta^2/2$ terms). At $t=1000$, $x_T$ is essentially pure noise.

**Answer:** **0.8171 (SNR 4.47), 0.1326 (SNR 0.153); $x_{10}=(0.970,\ -0.509)$; $t=228$; $\bar\alpha_{1000}\approx4.3\times10^{-5}$, $\bar\alpha_{500}\approx0.079$.**
> ✔ Verified numerically (numpy `cumprod` over the exact linear schedule).
> **Common slip:** using $\sqrt{\alpha_t}$ of a single step instead of $\sqrt{\bar\alpha_t}$ (cumulative product) in the closed form.

#### D63 — Classifier-free guidance combinations

**Formula:** $\hat\epsilon=\epsilon_u+w(\epsilon_c-\epsilon_u)$.

**Working:**
- (a) $\epsilon_c-\epsilon_u=(-0.20,\ 0.30,\ 0,\ -0.30)$ → $\hat\epsilon=(0.30-1.0,\ -0.10+1.5,\ 0.05,\ 0.20-1.5)=\mathbf{(-0.70,\ 1.40,\ 0.05,\ -1.30)}$.
- (b) $(1+w')\epsilon_c-w'\epsilon_u=\epsilon_u+(1+w')(\epsilon_c-\epsilon_u)$ → $w'=w-1=\mathbf4$. Check: $5\epsilon_c-4\epsilon_u=(0.5-1.2,\ 1.0+0.4,\ 0.25-0.2,\ -0.5-0.8)=(-0.7,\ 1.4,\ 0.05,\ -1.3)$ ✓.
- (c) $\epsilon_c-\epsilon_{neg}=(-0.15,\ 0.20,\ -0.05,\ -0.25)$ → $\hat\epsilon=(0.25-0.75,\ 0+1.0,\ 0.10-0.25,\ 0.15-1.25)=\mathbf{(-0.50,\ 1.00,\ -0.15,\ -1.10)}$ — the sample is pushed **towards** the prompt and **away from** the negative prompt.
- (d) Two evaluations per step → $\mathbf{100}$ U-Net evaluations (often run as 50 batched pairs).

**Answer:** **$(-0.70,1.40,0.05,-1.30)$; $w'=4$; negative prompt: $(-0.50,1.00,-0.15,-1.10)$; 100 evaluations.**
> ✔ Verified numerically (Python).
> **Common slip:** mixing the two CFG conventions — "$w=7.5$" in the $\epsilon_u+w(\cdot)$ form is $w'=6.5$ in the $(1+w')$ form.

#### D64 — VAE KL term and reparameterised sample

**Formula:** $\sigma=e^{\frac12\log\sigma^2}$; $\mathrm{KL}_j=\frac12(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$; $z=\mu+\sigma\odot\epsilon$.

**Working:**
- (a) $\sigma=(e^{-0.5},\ e^{0.25},\ e^0)=\mathbf{(0.6065,\ 1.2840,\ 1)}$.
- (b) dim 1: $\frac12(0.64+0.3679+1-1)=\mathbf{0.5039}$; dim 2: $\frac12(0.09+1.6487-0.5-1)=\mathbf{0.1194}$; dim 3: $\frac12(0+1-0-1)=\mathbf0$; total $\mathbf{0.6233}$ nats.
- (c) $z=(0.8-0.3033,\ -0.3+1.5408,\ 0+0.7)=\mathbf{(0.4967,\ 1.2408,\ 0.7)}$.
- (d) $12.4+4(0.6233)=\mathbf{14.893}$. Dimension 3 equals the prior exactly ($\mu=0,\sigma=1$, KL = 0): it carries **no information** about $x$ — an inactive ("collapsed") latent dimension; larger $\beta$ produces more of these.

**Answer:** **σ = (0.607, 1.284, 1); KL = 0.504 + 0.119 + 0 = 0.623 nats; $z=(0.497,\ 1.241,\ 0.7)$; β-VAE loss 14.89; dim 3 is inactive.**
> ✔ Verified numerically (Python + `torch.distributions.kl_divergence`).
> **Common slip:** treating $\log\sigma^2$ as $\log\sigma$ (gives $\sigma=e^{-1}$ instead of $e^{-0.5}$).

#### D65 — GAN optimal discriminator and value

**Formula:** $D^*=\frac{p_{data}}{p_{data}+p_g}$; $V(D^*,G)=2\,\mathrm{JSD}-\log4$.

**Working:**
- (a) $a$: $0.4/0.4=\mathbf1$; $b$: $0.4/0.6=\mathbf{0.667}$; $c$: $0.2/0.5=\mathbf{0.4}$; $d$: $0/0.5=\mathbf0$.
- (b) Data terms: $0.4\ln1+0.4\ln(2/3)+0.2\ln0.4=0-0.16219-0.18326=-0.34544$. Generator terms ($\log(1-D^*)$): $0.2\ln(1/3)+0.3\ln0.6+0.5\ln1=-0.21972-0.15325+0=-0.37297$. $V=\mathbf{-0.7184}$.
- (c) $\mathrm{JSD}=\frac{-0.7184+1.3863}{2}=\mathbf{0.3339}$ nats (checked directly from the definition).
- (d) Disjoint: $D^*=(1,1,1,0)$ → every term is $\log1=0$ → $V=\mathbf0$ → $\mathrm{JSD}=\ln2=\mathbf{0.693}$ (its maximum). It stays at $\ln2$ however $p_g$ moves while the supports stay disjoint → **no useful gradient** for $G$ (motivates the non-saturating loss and WGAN). If $p_g=p_{data}$: $D^*=\frac12$, $V=-\ln4=\mathbf{-1.386}$, JSD = 0.

**Answer:** **$D^*=(1,\ 0.667,\ 0.4,\ 0)$; $V=-0.718$; JSD = 0.334; disjoint: V = 0, JSD = ln 2 = 0.693; equal: V = −1.386.**
> ✔ Verified numerically (Python; JSD also computed directly from KL to the mixture).
> **Common slip:** writing the generator term as $p_g\log D^*$ instead of $p_g\log(1-D^*)$.

#### D66 — Latent compression factors

**Formula:** latent $=\frac Hf\times\frac Wf\times c$; factor $=\frac{3HW}{(H/f)(W/f)c}=\frac{3f^2}{c}$.

**Working:**
- (a) $96\times96\times4=\mathbf{36{,}864}$ values vs $768\cdot768\cdot3=1{,}769{,}472$ → $\mathbf{48\times}$ ($=3\cdot64/4$).
- (b) $128\times128\times4=\mathbf{65{,}536}$ vs $3{,}145{,}728$ → **48×** (the factor depends only on $f$ and $c$).
- (c) $p=2$: $16^2=\mathbf{256}$ tokens, $\mathbf{65{,}536}$ entries; $p=4$: **64**, **4096**; $p=8$: **16**, **256**. Pixel ViT, $p=8$ on 256: $32^2=\mathbf{1024}$ tokens → $1024^2/256^2=\mathbf{16\times}$ more attention than DiT with $p=2$.
- (d) $3\cdot64/16=\mathbf{12\times}$ (more channels → better reconstructions, less compression).

**Answer:** **96×96×4, 48×; 128×128×4, 48×; DiT tokens 256 / 64 / 16 (attention 65,536 / 4,096 / 256) vs 1024 tokens in pixel space (16×); 16-channel latent 12×.**
> ✔ Verified numerically (Python).
> **Common slip:** forgetting the 3 RGB channels in the pixel count (gives 16× instead of 48×).

#### D67 — FID in one dimension from samples

**Formula:** $\mathrm{FID}=\Vert\mu_r-\mu_g\Vert^2+\operatorname{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)$.

**Working:**
- (a) In 1-D, $\Sigma=\sigma^2$: $\sigma_r^2+\sigma_g^2-2\sqrt{\sigma_r^2\sigma_g^2}=\sigma_r^2+\sigma_g^2-2\sigma_r\sigma_g=(\sigma_r-\sigma_g)^2$ → $\mathrm{FID}=(\mu_r-\mu_g)^2+(\sigma_r-\sigma_g)^2$.
- (b)

| Set | $\sum x$ | $\mu$ | $\sum(x-\mu)^2$ | $s^2=\frac{\sum(x-\mu)^2}{5}$ | $s$ |
|---|---|---|---|---|---|
| Real | 6.2 | **1.0333** | 0.4333 | **0.08667** | **0.2944** |
| A | 6.4 | **1.0667** | 0.4333 | **0.08667** | **0.2944** |
| B | 9.8 | **1.6333** | 0.05333 | **0.01067** | **0.1033** |

- (c) A: $(1.0333-1.0667)^2+0=\mathbf{0.0011}$. B: $(1.0333-1.6333)^2+(0.2944-0.1033)^2=0.36+0.0365=\mathbf{0.3965}$.
- (d) B is both **shifted** (wrong mean) and **far too narrow** (std 0.10 vs 0.29) — the low-diversity, mode-collapse-like signature; FID penalises both, so B scores ≈ 357× worse than A.

**Answer:** **FID(A) = 0.0011, FID(B) = 0.397 (mean term 0.36 + spread term 0.037).**
> ✔ Verified numerically (Python; both forms of the 1-D formula).
> **Common slip:** using $(\sigma_r^2-\sigma_g^2)^2$ — the covariance term is $(\sigma_r-\sigma_g)^2$ in 1-D.

---

## Quick answer key

| # | Session | Final answer(s) |
|---|---|---|
| D1 | S01 | corr −110; conv +110; corner 5 (zero pad) / 15 (replicate); valid 3×3 |
| D2 | S01 | $I_x=-430$, $I_y=-370$; 567.3 (L1 800); θ = −139.3° = 220.7°; edge along 130.7°; 45° bin, $(x\pm1,y\pm1)$; 70.9/px |
| D3 | S01 | $R$ = 27 553.2 (corner), −9245 (edge), 7.2 (flat); λ(W1) = 161.1, 214.9; doubled → 440 851.2 |
| D4 | S01 | $K=7$: 0.3991/0.2420/0.0540/0.0044; $K=11$: 121 vs 22 (5.5×), 111.5 M vs 20.3 M; σ = 2.5, 48 vs 34 mult/px |
| D5 | S01 | sum 40, mean 4.44; $I(3,2)=8$; row 22; 80 vs 3 ops/px (26.7×; 16× incl. build) |
| D6 | S02 | 17; 15; 134; $e_{max}\approx0.69$; $p=0.859$ |
| D7 | S02 | (341.39, 253.36), (680.62, 64.59); 1.99 px inlier, 5.02 px outlier; $y=1.5x-5000$ (outside); 8 DOF |
| D8 | S02 | 8×9, 12×9, 100×9, rank 8; affine 6 unknowns / 3 pts; $s=0.005783$, first pt (−1.099, −0.868); $H=T'^{-1}\hat HT$ |
| D9 | S02 | 5; accept q1, q3; mean 128, std 8, 11.75σ; 32 B vs 512 B |
| D10 | S03 | 113 → 57 → 29 → 29 → 29 → 14 → 1×1×256; (224: 112, 56, 28, 28, 28, 13, 1) |
| D11 | S03 | 898,768 vs 226,096 (46,176 + 179,920), 3.98× |
| D12 | S03 | 176.1 M vs 44.3 M MACs; 0.176 vs 0.044 ms; FC 16.8 M MACs |
| D13 | S03 | RF 55, jump 4; undilated 23; 6 plain convs (3× params) |
| D14 | S03 | 480,000 / 376.3 M vs 25,200 / 19.76 M; ratio 0.0525; pointwise 76% |
| D15 | S03 | 577 tokens, 332,929 entries (8.6×); 785; 601; 590,592 |
| D16 | S04 | $p=(0.120,0.804,0.054,0.022)$; CE 2.118; LS loss 2.133; predicts dog; min LS loss 0.349 |
| D17 | S04 | μ = 4, σ² = 5.5; running (0.4, 1.5286); eval output 1.656; 128 params + 128 buffers |
| D18 | S04 | (2.0, −0.533, 0, 1.067, 0, 0, 1.467, −3.333); classic: 0 / 2.4; Var 3.41 |
| D19 | S04 | 134,281,029 total; 20,485 (0.015%), 119.6 M (89.0%), 126.6 M (94.3%), 134.3 M; 0.54 vs 2.15 GB |
| D20 | S04 | acc 0.87; macro P/R/F1 0.846/0.829/0.836; micro 0.87; weighted F1 0.867 |
| D21 | S05 | cos 0.8, d 0.632, 36.9°; (b,c): 0.28, 1.2; cos > 0.595; scaled: dot 2.4, d 2.28 |
| D22 | S05 | losses 0.1, 0.6, 0.2 (rest 0); batch-all 0.1125 / 0.30; batch-hard 0.6 |
| D23 | S05 | τ 0.1: L 0.494; τ 0.5: L 1.028; hardest share 95% vs 50%; ln 5 = 1.609 |
| D24 | S05 | AP 0.525; P@5 0.6, R@5 0.75; Recall@1/2/5/10 0.4/0.6/0.8/1.0; MRR 0.595 |
| D25 | S06 | 196,416 anchors vs 27,648; 6 & 12 ch; 90.5×45.3, 64×64, 45.3×90.5 |
| D26 | S06 | IoU 0.6; $t$ = (0.125, 0.0833, 0.1178, 0.2877); decoded (94.3, 97.7, 289.7, 206.3), IoU 0.531 |
| D27 | S06 | 7524; 2160; 15,048; 37.5/12.5 → 70°/90° and 170°/10° |
| D28 | S06 | (13, 7, 37, 25), shifts −5/−5/+3/−5 px; bins 3.43×2.57 vs 3.36×2.57; P4 / P2; 12,846,080 |
| D29 | S07 | 0.143; 0.36; 0; 0.333 vs 0.905 |
| D30 | S07 | 0.781, 0.743, 0.579, 0.696; per-class B1, B3, B4, B6; agnostic B1, B4, B6 |
| D31 | S07 | all-point 0.620; 11-point 0.616 |
| D32 | S07 | 7×7×25, 147 boxes; 22,743 boxes, 45 ch; 11,565; cell (9, 14), (0.375, 0.6875) |
| D33 | S07 | 24,564; 0.20…0.90; 144.2×72.1, 121.2×121.2 px; 150 ch |
| D34 | S07 | FL/CE fg 0.000625/0.0225/0.16; $p_t$ 0.90 / 0.602; 31× vs 8033× |
| D35 | S08 | cost 14 ($q_1g_1,q_2g_3,q_3g_2$); greedy 18; 4×3: cost 12, $q_1\to\varnothing$ |
| D36 | S08 | L1 0.05/0.23; GIoU 0.844/0.362; costs 0.0125 vs 1.507 → $q_1$ |
| D37 | S08 | GIoU 0.033, −0.100, −0.865, 0.450 |
| D38 | S08 | 768 tokens, 589,824/head, 56.6 MB; 10,000 / 76,800; DC5 9.44 M (16×), 906 MB; 300 queries → 90,000 / 230,400 |
| D39 | S10 | 0.571/0.727, 0.300/0.462; mean Dice 0.594 vs pooled 0.712; 0.75, 0.818; soft Dice 0.755, hard 0.80 |
| D40 | S10 | PA 0.87; IoU 0.917/0.738/0.533; mIoU 0.730; mCA 0.830; fwIoU 0.780 |
| D41 | S10 | 14; 13/14; 56; 512; 295,040 |
| D42 | S10 | …16 → 32/28 → 56/52 → 104/100; crops 4, 16, 40; 100×100 (28.3%); 190 invalid |
| D43 | S10 | samples 23.375, 24.656, 22.281, 31.781 → 25.52; NN 30 |
| D44 | S10 | TP 3, FP 2, FN 1; PQ(person) 0.498; PQ 0.483 |
| D45 | S11 | (801.29, 370.86); (508.69, 135.56); $X_C$ behind camera |
| D46 | S11 | 3478 px; 38.8°/32.8°/49.3°; 7.31 mm; 0.575 mm/px, 87 px; binned $f_x$ 1739 |
| D47 | S11 | ideal (235, 630); $(x_d,y_d)=(-0.46103, 0.30755)$; (225.08, 636.79); pincushion |
| D48 | S11 | $C=(-2.6,0.5,-1.8)$; axis (0.6, 0, 0.8); $Z_c=7.6$ m; $\mathbf t_2=(-4,-1.5,2)$ |
| D49 | S11 | $\mathbf t=(0,1.2,0.9)$; (0,2) → (320, 240); (1,5) → (442.4, 19.6); (400,300) → (0.294, 1.632) m |
| D50 | S12 | $d=21$, (0.857, 0.571, 15.0) m; 7 px; 15 m; 160 |
| D51 | S12 | 0.079/0.714/2.857 m; $B=0.857$ m; 4.69 m; $f=3000$ px (24.1°) |
| D52 | S12 | $y=0.5x+100$; 1.79 px accept, 22.36 px reject; FOE (400, 300) |
| D53 | S12 | $E$ rows (0,0,0), (0.6,0,−0.8), (0,1,0); s.v. (1,1,0); (1.9, 1.9, 0); $M$ valid |
| D54 | S12 | $(u,v)=(1.401, 1.322)$; λ = 55.2, 17.8; second window singular |
| D55 | S12 | $\mathbf u_n=(1.2,-1.6)$, 2.0; tangential (−3.2, −2.4); flow (0.8, −1.9) |
| D56 | S12 | 75,723 unknowns; 250,000 residuals; 3 M nnz (0.016%); 723×723 |
| D57 | S13 | (0.880, 0.119, 0.001); scale 50: (0.715, 0.263, 0.022); Δ = 0.029; 0.2975 |
| D58 | S13 | 16.8 M entries, 4095 neg/row; 67.1 MB; 8.59 G MACs; ln 4096 = 8.318 |
| D59 | S13 | $s=0.7517$, $t=-0.0551$; 1.61/3.12/10.50 m; 4.81 m; invalid |
| D60 | S14 | 1.50 × 0.357 m; distance 1.65 m; +10% |
| D61 | S14 | 290 ms/3.45 FPS; 265 ms/3.77 FPS; 7.14 FPS; 63.2 ms/15.8 FPS |
| D62 | S15 | 0.8171, 0.1326; $x_{10}=(0.970,-0.509)$; $t=228$; $4.3\times10^{-5}$, 0.079 |
| D63 | S15 | (−0.70, 1.40, 0.05, −1.30); $w'=4$; (−0.50, 1.00, −0.15, −1.10); 100 |
| D64 | S15 | σ (0.607, 1.284, 1); KL 0.623; $z$ (0.497, 1.241, 0.7); 14.89 |
| D65 | S15 | $D^*$ (1, 0.667, 0.4, 0); V −0.718; JSD 0.334; disjoint V 0, JSD 0.693 |
| D66 | S15 | 48×; 48×; 256/64/16 tokens; 12× |
| D67 | S15 | FID(A) 0.0011; FID(B) 0.397 |

---

## Index of all numericals in the pack

Every Section-B numerical in the per-session practice files and the two mock exams, with the drill problems that train the same skill. Solutions sit in the matching `NN_solutions.md` / `mock_exam_N_solutions.md`.

| Session | File | Numericals (question → topic) | Drill |
|---|---|---|---|
| S01 | [01_questions.md](../practice/01_questions.md) | Q9 mean/median filter, correlation vs convolution, output size · Q10 Sobel gradient, Canny NMS & hysteresis · Q11 Harris $R$, eigenvalues, Shi–Tomasi, intensity scaling · Q12 image memory, Gaussian kernel size & separable cost, cascaded σ · Q13 integral image, ORB Hamming, SIFT octave σ's | D1–D5 |
| S02 | [02_questions.md](../practice/02_questions.md) | Q9 RANSAC iterations (homography, F 8/7-pt, adaptive) · Q10 Lowe ratio test + cross-check · Q11 L1/L2/cosine/Hamming/χ² distances · Q12 applying a homography + inlier test · Q13 DOF, DLT matrix sizes, rotation-only homography | D6–D9 |
| S03 | [03_questions.md](../practice/03_questions.md) | Q9 output shapes, params, flatten vs GAP head · Q10 AlexNet conv1 size/params/MACs vs FC · Q11 receptive field with dilation · Q12 stacked 3×3 vs 5×5/7×7, ResNet bottleneck · Q13 Inception 1×1 bottleneck MACs, depthwise-separable | D10–D15 |
| S04 | [04_questions.md](../practice/04_questions.md) | Q9 ResNet-50 trainable params & ImageNet normalisation · Q10 softmax, CE, label smoothing, mixup · Q11 BatchNorm train/eval + inverted dropout · Q12 3-class confusion matrix, macro-F1, balanced accuracy · Q13 LR schedules, SGD momentum + weight decay | D16–D20 |
| S05 | [05_questions.md](../practice/05_questions.md) | Q9 dot/cosine/Euclidean, normalisation · Q10 triplet & contrastive loss, hard/semi-hard · Q11 InfoNCE with τ, SimCLR counts · Q12 retrieval P@k, R@k, AP, mAP, MRR · Q13 index memory, PQ compression, brute force vs IVF | D21–D24 |
| S06 | [06_questions.md](../practice/06_questions.md) | Q9 HOG length & orientation voting · Q10 sliding-window count, cascade rates · Q11 RPN anchors & head params · Q12 anchor labelling by IoU, box encode/decode · Q13 RoI projection & quantisation, Fast R-CNN multi-task loss | D25–D28 |
| S07 | [07_questions.md](../practice/07_questions.md) | Q9 IoU, GIoU, Dice · Q10 NMS & Soft-NMS · Q11 AP all-point / 11-point, mAP, COCO AP · Q12 YOLO/SSD/RetinaNet output sizes · Q13 focal loss, FCOS targets | D29–D34 |
| S08 | [08_questions.md](../practice/08_questions.md) | Q8 Hungarian assignment 4×3 · Q9 DETR matching cost (class + L1 + GIoU) · Q10 box conversion, IoU & GIoU · Q11 attention cost, DETR vs Deformable DETR · Q12 P/R/F1 vs score threshold | D35–D38 |
| S09+16 | [09_questions.md](../practice/09_questions.md) | Q9 confusion matrix (macro/micro/weighted F1, balanced accuracy) · Q10 retrieval AP/mAP, hit-rate, MRR · Q11 cosine vs Euclidean kNN · Q12 trainable params & memory: linear probe vs fine-tune · Q13 CLIP zero-shot accuracy & softmax | D19–D24, D57 |
| S10 | [10_questions.md](../practice/10_questions.md) | Q9 IoU/Dice from pixel counts · Q10 mIoU from a confusion matrix · Q11 U-Net valid-conv sizes · Q12 RoIPool vs RoIAlign (bilinear) · Q13 panoptic quality | D39–D44 |
| S11 | [11_questions.md](../practice/11_questions.md) | Q9 projection with $K[R\mid\mathbf t]$, camera centre · Q10 $f$ in pixels, FOV, size/distance · Q11 radial + tangential distortion · Q12 floor homography, back-projection · Q13 calibration counts, RMS reprojection error | D45–D49 |
| S12 | [12_questions.md](../practice/12_questions.md) | Q9 depth & 3D point from disparity · Q10 depth error & baseline design · Q11 essential matrix & epipolar line · Q12 normal flow, Lucas–Kanade · Q13 RANSAC 8/7/5-point, BA size | D50–D56 |
| S13 | [13_questions.md](../practice/13_questions.md) | Q9 CLIP zero-shot softmax · Q10 CLIP symmetric InfoNCE · Q11 ViT/SAM token & attention counts · Q12 Depth Anything scale & shift · Q13 DINO EMA, centring, sharpening | D15, D57–D59 |
| S14 | [14_questions.md](../practice/14_questions.md) | Q9 pipeline latency/FPS · Q10 Grounding DINO thresholds, P/R · Q11 box conversion, IoU, P/R/F1 · Q12 metric depth & size from pinhole + scale/shift · Q13 auto-labelling cost | D59–D61 |
| S15 | [15_questions.md](../practice/15_questions.md) | Q9 DDPM $\bar\alpha_t$, $x_t$, reverse step · Q10 VAE params, KL, reparameterisation · Q11 GAN $D^*$, $V$, JSD, saturating loss · Q12 latent diffusion sizes + CFG · Q13 FID | D62–D67 |
| Mock 1 | [mock_exam_1.md](../practice/mock_exam_1.md) | B1 CNN layer arithmetic (S03) · B2 RANSAC + homography mapping (S02) · B3 AP all-point/11-point, mAP (S07) · B4 Dice, IoU, mIoU (S10) · B5 projection with lens distortion (S11) · B6 DDPM forward process + CFG (S15) | D10, D6–D7, D31, D39–D40, D45–D47, D62–D63 |
| Mock 2 | [mock_exam_2.md](../practice/mock_exam_2.md) | B1 Sobel + Harris (S01) · B2 IoU, NMS, anchor encoding (S06–S07) · B3 stereo depth + Lucas–Kanade (S12) · B4 CLIP zero-shot + InfoNCE (S13) · B5 transfer-learning params + metrics (S04) · B6 DETR Hungarian matching (S08) | D2–D3, D26, D29–D30, D50–D54, D57–D58, D19–D20, D35–D36 |

**Total in the pack:** 75 per-session numericals (15 practice files × 5) + 12 mock-exam numericals + 67 drill problems = **154 numericals**.
