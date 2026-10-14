# Session 02 — Feature Matching & Classical CV · Practice Questions

> Notes: [../notes/02_feature_matching_classical.md](../notes/02_feature_matching_classical.md) · Solutions: [02_solutions.md](02_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 120 min

Conventions: RANSAC $N=\left\lceil\frac{\log(1-p)}{\log(1-(1-e)^s)}\right\rceil$ (always round up); homogeneous points $\tilde{\mathbf x}=(x,y,1)^\top$; pixel coordinates $x$ = column, $y$ = row.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
You match ORB descriptors with `cv2.BFMatcher`. Which norm should you use?

- **A.** `NORM_L2`
- **B.** `NORM_L1`
- **C.** `NORM_HAMMING`
- **D.** Cosine distance after converting the descriptors to float

### Q2 · MCQ · Medium
Lowe's ratio test rejects a putative match when:

- **A.** $d_1/d_2$ is close to 1, i.e. the best and second-best candidates are almost equally good
- **B.** $d_1$ exceeds a fixed absolute distance threshold
- **C.** The match is not mutual (cross-check fails)
- **D.** $d_2$ is much larger than $d_1$

### Q3 · MCQ · Easy
A planar homography has how many degrees of freedom, and how many point correspondences are needed at minimum to estimate it?

- **A.** 9 DOF, 5 correspondences
- **B.** 6 DOF, 3 correspondences
- **C.** 8 DOF, 3 correspondences
- **D.** 8 DOF, 4 correspondences with no three collinear

### Q4 · MCQ · Medium
In which situation are two photographs related **exactly** by a single homography?

- **A.** The camera translates sideways along a street with buildings at different depths
- **B.** The camera rotates about its optical centre while photographing a 3-D scene
- **C.** Any two images of a rigid, static scene
- **D.** The camera moves forward through a furnished room

### Q5 · MCQ · Medium
How many RANSAC iterations are needed to estimate a homography with probability $p=0.99$ of drawing at least one outlier-free sample when half of the matches are outliers?

- **A.** 17
- **B.** 71
- **C.** 72
- **D.** 1177

### Q6 · MCQ · Medium
Which FLANN index is appropriate for matching large sets of ORB descriptors?

- **A.** Randomised KD-trees (`FLANN_INDEX_KDTREE`, `trees=5`)
- **B.** Brute force with `NORM_L2`
- **C.** Hierarchical k-means tree with L2 distance
- **D.** Locality-sensitive hashing (`FLANN_INDEX_LSH`)

### Q7 · MCQ · Medium
What is the main purpose of multi-band (Laplacian-pyramid) blending in panorama stitching?

- **A.** To remove ghosting caused by parallax
- **B.** To blend low frequencies over a wide transition and high frequencies over a narrow one, giving seamless yet sharp results
- **C.** To estimate the homographies between images more accurately
- **D.** To correct radial lens distortion

### Q8 · MCQ · Hard
In which scenario is a classical SIFT + ratio test + RANSAC pipeline **most likely to fail**?

- **A.** Matching two views of a plain white, textureless wall
- **B.** Stitching two photos of a textured mountain range taken by rotating the camera
- **C.** Recognising the same book cover rotated by 45° and scaled by 2×
- **D.** Tracking a textured planar poster under a moderate lighting change

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
Use $p=0.99$ unless stated.
**(a)** [1] Homography ($s=4$), outlier ratio $e=0.5$: number of iterations $N$.
**(b)** [1] Same, but a better matcher reduces the outlier ratio to $e=0.3$.
**(c)** [1.5] Fundamental matrix with $e=0.5$ using (i) the 8-point algorithm and (ii) the 7-point algorithm. Comment on the comparison with (a).
**(d)** [1.5] Adaptive RANSAC for a homography on 100 matches starts with $N=\infty$. The first good hypothesis has 60 inliers. What is the updated $N$? What would $N$ be for $e=0.5$ but $p=0.999$?

### Q10 · Numerical · Medium · [5 marks]
SIFT descriptors of 6 query keypoints were matched against a second image (L2 distances). For each query the table gives the nearest ($d_1$) and second-nearest ($d_2$) distances, the nearest train keypoint, and that train keypoint's own nearest query (reverse match).

| Query | $d_1$ | $d_2$ | Nearest train | Reverse NN of that train |
|---|---|---|---|---|
| q1 | 120 | 300 | t4 | q1 |
| q2 | 200 | 230 | t7 | q2 |
| q3 | 150 | 190 | t2 | q3 |
| q4 | 90 | 95 | t9 | q4 |
| q5 | 250 | 420 | t1 | q6 |
| q6 | 180 | 400 | t1 | q6 |

**(a)** [2] Compute $d_1/d_2$ for each query and list the matches accepted by the ratio test with $\tau=0.75$ and with $\tau=0.8$.
**(b)** [1.5] Apply the cross-check (mutual nearest neighbour) on top of the ratio test. Which matches remain for each $\tau$?
**(c)** [1.5] A student instead uses an absolute threshold $d_1<160$. Which matches does that accept? Explain, using q4 and q6, why the ratio test is better.

### Q11 · Numerical · Easy · [5 marks]
**(a)** [2] For $\mathbf a=(1,2,0,3)$ and $\mathbf b=(2,0,1,3)$ compute the L1 distance, L2 distance, cosine similarity and cosine distance.
**(b)** [1] Normalise both vectors to unit length and compute their L2 distance. Verify it with $\Vert\hat{\mathbf a}-\hat{\mathbf b}\Vert^2=2-2\cos$.
**(c)** [1] Compute the Hamming distance between the 16-bit binary descriptors `1011001011100001` and `1001011010100111`.
**(d)** [1] Compute the χ² distance $\tfrac12\sum_i\frac{(h_i-g_i)^2}{h_i+g_i}$ between the histograms $h=(0.2,0.5,0.3)$ and $g=(0.3,0.3,0.4)$. Which metric would you use for SIFT, ORB and colour histograms respectively?

### Q12 · Numerical · Medium · [5 marks]
A RANSAC hypothesis maps image 1 to image 2 with

$$
H=\begin{bmatrix}1.2&0.1&30\\0.05&1.1&20\\0.0005&0.0002&1\end{bmatrix}.
$$

**(a)** [2] Map the points $\mathbf p_1=(100,50)$ and $\mathbf p_2=(200,150)$ into image 2 (give 2 decimals). Where does the origin $(0,0)$ go?
**(b)** [2] The matched points observed in image 2 are $\mathbf p'_1=(147,76)$ and $\mathbf p'_2=(260,170)$. With an inlier threshold of $t=5$ px on the transfer error, classify each correspondence as inlier or outlier.
**(c)** [1] Is $H$ an affine transformation? What happens to the mapped point if $H$ is replaced by $2H$?

### Q13 · Numerical · Medium · [5 marks]
**(a)** [2] Give the DOF and the minimum number of point correspondences for: translation, Euclidean (rigid), similarity, affine, projective (homography).
**(b)** [1.5] For a homography, 25 putative matches are available and RANSAC finds 18 inliers. Give the size of the DLT matrix $A$ for (i) one minimal RANSAC sample, (ii) the final refit on the inliers, (iii) a (non-robust) fit on all matches. How many unknowns does $\mathbf h$ have, and what is the rank of $A$ for a noise-free, non-degenerate minimal sample?
**(c)** [1.5] A tripod panorama is shot by pure rotation and the intrinsics $K$ are known, so $H=KRK^{-1}$. How many DOF does this model have and how many correspondences are minimally needed? With $e=0.6$ and $p=0.99$, compare the RANSAC iterations for this model with those for a general homography.

## Section C — Short answer

### Q14 · Short · Medium · [4 marks]
Explain why Lowe's ratio test works better than an absolute distance threshold for filtering descriptor matches. What values of $\tau$ are typical, how does the test behave on repetitive structures, and how does it differ from a cross-check?

### Q15 · Short · Medium · [4 marks]
Derive the RANSAC iteration formula. How should the inlier threshold be chosen, and what is adaptive RANSAC? State two limitations of RANSAC and one variant that addresses a limitation.

### Q16 · Short · Hard · [4 marks]
**(a)** [2] Show how one point correspondence gives two linear equations in the entries of $H$, and explain how the DLT solves for $H$ from $n$ correspondences, including Hartley normalisation and degenerate configurations.
**(b)** [2] When is a homography the correct model between two images? Explain parallax and what model should be used when the homography assumption fails.

### Q17 · Short · Medium · [4 marks]
Where does classical feature-based computer vision work well, and where does it fail? Give at least three failure cases with reasons, and briefly explain how learned matchers (SuperPoint + SuperGlue/LightGlue, LoFTR) address them.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
**Design a panorama app.** A phone app lets users sweep the camera to capture 6–10 overlapping photos and produces a single panorama. Describe the full pipeline (features, matching, robust estimation, global alignment, projection, compositing), justify the key choices (detector/descriptor, thresholds, projection type, blending), and explain the main failure cases (e.g. parallax, moving people, exposure changes, textureless sky, wide field of view) and how your design mitigates each.

### Q19 · Long · Hard · [10 marks]
**SIFT vs ORB for mobile AR.** You are building a real-time (30 fps) mobile AR app that recognises a printed poster and overlays 3-D content on it, running on a mid-range phone CPU. Compare SIFT and ORB for this application (detection, description, matching, speed, memory, invariances, licensing) and recommend a design for the full recognition-and-tracking loop, including how you would estimate the poster pose each frame and handle failure cases.
