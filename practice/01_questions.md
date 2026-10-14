# Session 01 — Course Structure, Image Processing & Features · Practice Questions

> Notes: [../notes/01_image_processing_features.md](../notes/01_image_processing_features.md) · Solutions: [01_solutions.md](01_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 120 min

Conventions used throughout: image coordinates have $x$ = column (right), $y$ = row (**down**); "filter with kernel $K$" means **cross-correlation** unless the question says convolution.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
Which of the following filters is **non-linear**?

- **A.** Gaussian filter
- **B.** Box (mean) filter
- **C.** Median filter
- **D.** Sobel filter

### Q2 · MCQ · Medium
A "convolution" layer in a CNN (and OpenCV's `cv2.filter2D`) actually computes:

- **A.** True convolution — the kernel is flipped by 180° before sliding
- **B.** Cross-correlation — the kernel is slid over the image without flipping
- **C.** A median of the products under the kernel
- **D.** Convolution in the Fourier domain only, never in the spatial domain

### Q3 · MCQ · Medium
At a pixel the Harris structure tensor $M$ has eigenvalues $\lambda_1=500$ and $\lambda_2=2$ ($k=0.04$). The pixel is best described as:

- **A.** A corner, because $R$ is large and positive
- **B.** A flat region, because $\lambda_2$ is small
- **C.** An edge, but with $R>0$
- **D.** An edge, with $R<0$

### Q4 · MCQ · Medium
Which change to the image can make a Harris corner **stop being detected** (with the same window size and threshold) even though the physical corner is still fully visible?

- **A.** Zooming in by a factor of 4
- **B.** Rotating the image by 30°
- **C.** Adding 40 to every pixel intensity
- **D.** Translating the image by 25 pixels

### Q5 · MCQ · Medium
Canny is run with $T_{low}=50$ and $T_{high}=150$. After non-maximum suppression a pixel has gradient magnitude 90. In the final edge map this pixel is:

- **A.** Always kept, because 90 > $T_{low}$
- **B.** Always discarded, because 90 < $T_{high}$
- **C.** Kept only if it is connected (8-connectivity, through other weak pixels) to a pixel with magnitude ≥ 150
- **D.** Kept only if its gradient direction is quantised to 0°

### Q6 · MCQ · Easy
The standard SIFT descriptor has dimension:

- **A.** 64 = 4×4 subregions × 4 Haar sums
- **B.** 128 = 4×4 cells × 8 orientation bins
- **C.** 256 = number of binary intensity tests
- **D.** 36 = number of orientation-histogram bins

### Q7 · MCQ · Hard
SIFT detects keypoints as extrema of $D=L(x,y,k\sigma)-L(x,y,\sigma)$. This is justified because $D$ approximates:

- **A.** The first derivative of a Gaussian, the optimal step-edge detector
- **B.** The Harris response $\det M-k(\operatorname{tr}M)^2$ at scale $\sigma$
- **C.** The un-normalised Laplacian $\nabla^2G$, whose response grows at coarse scales
- **D.** The scale-normalised Laplacian of Gaussian $\sigma^2\nabla^2G$, up to the constant factor $(k-1)$

### Q8 · MCQ · Medium
Which statement about ORB is **correct**?

- **A.** Its 256-bit rBRIEF descriptor is matched with the Hamming distance
- **B.** Its keypoints are extrema of a Difference-of-Gaussians scale space
- **C.** It is patented and available only in OpenCV's non-free contrib module
- **D.** Its descriptor is 64 floating-point Haar-wavelet sums

## Section B — Numericals

### Q9 · Numerical · Easy · [5 marks]
**(a)** [2] A 3×3 neighbourhood of a grayscale image (8-bit) is

$$
A=\begin{bmatrix}12&14&20\\12&255&24\\14&16&26\end{bmatrix}.
$$

Compute the output at the centre pixel for (i) a 3×3 mean filter and (ii) a 3×3 median filter. What type of noise is the centre value, and which filter handles it better?

**(b)** [2] For the neighbourhood $B=\begin{bmatrix}10&20&30\\40&50&60\\70&80&90\end{bmatrix}$ and kernel $K=\begin{bmatrix}1&2&3\\0&0&0\\0&0&0\end{bmatrix}$, compute the output at the centre pixel using (i) cross-correlation and (ii) convolution. For which kernels would the two answers always be equal?

**(c)** [1] A 512×512 image is filtered with a 5×5 kernel, stride 1. Give the output size with no padding and with padding $P=2$.

### Q10 · Numerical · Medium · [5 marks]
A 3×3 grayscale neighbourhood centred on pixel $p$ is

$$
N=\begin{bmatrix}20&40&60\\30&50&90\\40&80&120\end{bmatrix}
$$

(top row = row $y-1$). Use the Sobel kernels as cross-correlation, with $y$ pointing down:
$K_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix}$, $K_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}$.

**(a)** [1] Compute $I_x$ and $I_y$ at $p$.
**(b)** [1.5] Compute the gradient magnitude (L2 and L1 versions) and direction $\theta=\operatorname{atan2}(I_y,I_x)$ in degrees. What is the direction of the edge itself?
**(c)** [1.5] Canny NMS quantises $\theta$ to {0°, 45°, 90°, 135°}. Which bin does $p$ fall in, and which two neighbours is $p$ compared with? Those neighbours have (L2) magnitudes 250 and 275. Does $p$ survive NMS?
**(d)** [1] With $T_{low}=100$, $T_{high}=250$ (L2 magnitude), classify $p$. What is the high:low ratio, and is it in Canny's recommended range?

### Q11 · Numerical · Medium · [5 marks]
Three windows give the following Harris structure tensors. Use $k=0.04$.

$$
M_A=\begin{bmatrix}400&20\\20&380\end{bmatrix},\quad M_B=\begin{bmatrix}500&10\\10&8\end{bmatrix},\quad M_C=\begin{bmatrix}6&1\\1&5\end{bmatrix}
$$

**(a)** [2] Compute $\det M$, $\operatorname{tr}M$ and $R=\det M-k(\operatorname{tr}M)^2$ for each.
**(b)** [2] Compute the eigenvalues of each matrix and classify each window as corner / edge / flat (rule: corner if $R>10^4$, edge if $R<0$, otherwise flat). Give the Shi–Tomasi score for each.
**(c)** [1] If every pixel intensity in window A's neighbourhood is doubled, what is the new $R_A$? What does this say about Harris's photometric invariance?

### Q12 · Numerical · Medium · [5 marks]
**(a)** [1] How much memory does one uncompressed 1920×1080 RGB image with 8 bits per channel need? Give bytes, MB ($10^6$ B) and MiB ($2^{20}$ B).
**(b)** [1] How much for one minute of such uncompressed video at 30 fps (GB, $10^9$ B)?
**(c)** [2] The frame is blurred with a Gaussian of $\sigma=2$ using kernel size $K=2\lceil3\sigma\rceil+1$. Give $K$, the multiplications per pixel per channel for (i) direct 2-D filtering and (ii) separable filtering, the speed-up, and the total multiplications for the whole RGB frame in each case.
**(d)** [1] Repeat the per-pixel count for $\sigma=4$. A student instead blurs twice with $\sigma=2$ "to get $\sigma=4$". What $\sigma$ do they actually get?

### Q13 · Numerical · Medium · [5 marks]
**(a)** [2] For the image (rows top to bottom, 0-indexed)

$$
I=\begin{bmatrix}3&1&4&1\\5&9&2&6\\5&3&5&8\\9&7&9&3\end{bmatrix},
$$

compute the integral image $ii(x,y)=\sum_{x'\le x,\,y'\le y}I(x',y')$, then use **exactly four lookups** to find the sum of the rectangle covering rows 1–2 and columns 1–3 (inclusive). Check by direct summation.
**(b)** [1] Two ORB descriptor bytes are `10110100` and `10011101`. Compute their Hamming distance. How many bytes does a full ORB descriptor occupy?
**(c)** [2] For the first octave of SIFT with $\sigma_0=1.6$ and $s=3$ intervals, list the $\sigma$ of every Gaussian image, state how many DoG images there are, on which DoG levels extrema are searched, and how many neighbours each candidate is compared with.

## Section C — Short answer

### Q14 · Short · Medium · [4 marks]
State Canny's three criteria for an optimal edge detector and explain which step(s) of the Canny algorithm address each one. How does increasing $\sigma$ affect the result?

### Q15 · Short · Medium · [4 marks]
Explain why the Harris detector is rotation-invariant but not scale-invariant. Describe how SIFT achieves scale and rotation invariance.

### Q16 · Short · Easy · [4 marks]
Choose between a Gaussian, a median and a bilateral filter for each case and justify: (i) mild Gaussian sensor noise before computing gradients, (ii) an image with 5% salt-and-pepper pixels, (iii) smoothing skin in a portrait while keeping eye and face outlines sharp. Which of these filters are linear?

### Q17 · Short · Medium · [4 marks]
Distinguish a keypoint detector from a descriptor and list four properties of a good local feature. Explain how ORB fixes the shortcomings of plain FAST and plain BRIEF.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
Describe the complete SIFT pipeline from input image to descriptor. For each stage give the key parameters and explain **why** it is needed (which invariance or robustness it provides). Conclude with the invariances SIFT does and does not have, and when you would choose ORB instead.

### Q19 · Long · Hard · [10 marks]
**Design question.** A factory photographs machined metal plates on a conveyor with a fixed overhead grayscale camera. The images contain Gaussian sensor noise and occasional dead/hot pixels (salt-and-pepper); lighting drifts during the day. Plates arrive at **random in-plane rotations**, and two production lines use cameras at **different heights** (plates appear at two different scales). You must (1) extract the plate outline to measure its dimensions, and (2) find distinctive points to align each image with a CAD template.
Design the classical pipeline: pre-processing, edge extraction (with parameter choices), feature detection/description for alignment, and how you handle rotation, scale and lighting. Discuss failure cases and how you would evaluate the system.
