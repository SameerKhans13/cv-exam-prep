# Session 12 — 3D Vision & Motion · Practice Questions

> Notes: [../notes/12_3d_vision_motion.md](../notes/12_3d_vision_motion.md) · Solutions: [12_solutions.md](12_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min

Conventions used throughout: rectified stereo with $d=x_L-x_R$ and $Z=fB/d$ ($f$ in pixels). Two-view motion is $\mathbf X_2=R\mathbf X_1+\mathbf t$, with $E=[\mathbf t]_\times R$ and $\hat{\mathbf x}_2^\top E\,\hat{\mathbf x}_1=0$ on normalised coordinates. Optical flow $(u,v)$ is in pixels per frame. RANSAC: $N=\lceil\log(1-p)/\log(1-(1-e)^s)\rceil$.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
In a rectified stereo pair, point P has a larger disparity than point Q. What can you conclude?

- **A.** P is farther from the cameras than Q
- **B.** P and Q are at the same depth, because disparity depends only on the baseline
- **C.** P is nearer to the cameras than Q
- **D.** P lies closer to the epipole than Q

### Q2 · MCQ · Easy
How many degrees of freedom does the fundamental matrix $F$ have?

- **A.** 8
- **B.** 7
- **C.** 9
- **D.** 5

### Q3 · MCQ · Medium
Why does the essential matrix $E=[\mathbf t]_\times R$ have **5** degrees of freedom rather than 6?

- **A.** $R$ has 3 DOF and $\mathbf t$ has 3, but the overall scale of $\mathbf t$ cannot be recovered from image correspondences
- **B.** The rank-2 constraint removes one DOF from the 6 of $R$ and $\mathbf t$
- **C.** $E$ has 9 entries minus the 4 parameters of $K$
- **D.** Only 5 point correspondences are ever available between two views

### Q4 · MCQ · Medium
Lucas–Kanade is applied to a window that contains a single straight **vertical** edge: intensity varies only along $x$, so $I_y\approx0$ everywhere in the window. What happens?

- **A.** Both $u$ and $v$ are recovered accurately
- **B.** Only the vertical component $v$ (along the edge) can be recovered
- **C.** Neither component can be recovered, because $I_t=0$ on edges
- **D.** Only the horizontal component $u$ (normal to the edge) can be recovered, because $A^\top A$ is nearly singular

### Q5 · MCQ · Medium
The $2\times2$ matrix $A^\top A=\begin{bmatrix}\sum I_x^2&\sum I_xI_y\\\sum I_xI_y&\sum I_y^2\end{bmatrix}$ in Lucas–Kanade is the same as which matrix?

- **A.** The Hessian used in SIFT's edge-response test
- **B.** The Harris second-moment (structure) matrix $M$
- **C.** The fundamental matrix restricted to the window
- **D.** The covariance matrix of SORT's Kalman filter

### Q6 · MCQ · Medium
Incremental SfM is run on photographs alone (calibrated cameras, no GPS, no known lengths). What can it recover?

- **A.** A metric 3D model and camera poses in metres
- **B.** Only the camera rotations, with no 3D structure
- **C.** Structure and poses up to an unknown similarity transform (global scale, rotation and translation)
- **D.** Structure only up to an arbitrary projective transformation, even though $K$ is known

### Q7 · MCQ · Medium
In a SLAM system, what does **loop closure** do that **relocalisation** does not?

- **A.** It detects a revisit of an already-mapped place during normal tracking and corrects the accumulated drift through pose-graph optimisation
- **B.** It recovers the camera pose after tracking has been lost, without modifying the map
- **C.** It initialises the map by triangulating points from the first two frames
- **D.** It selects which frames become keyframes for local bundle adjustment

### Q8 · MCQ · Hard
A stereo rig has a depth uncertainty of 5 cm for objects at 5 m. With the same rig and the same disparity (matching) accuracy, what uncertainty do you expect at 15 m?

- **A.** 15 cm
- **B.** 5 cm
- **C.** 135 cm
- **D.** 45 cm

## Section B — Numericals

### Q9 · Numerical · Easy · [5 marks]
A rectified stereo rig has $f=700$ px, baseline $B=12$ cm and principal point $(c_x,c_y)=(320,240)$ (identical for both cameras).

(a) Compute the depth of points with disparities $d=84$, $21$ and $7$ px. **[2]**

(b) A point appears at $(x_L,y)=(450,300)$ in the left image and at $x_R=415$ in the right image. Compute its disparity and its 3D coordinates $(X,Y,Z)$ in the left-camera frame. **[3]**

### Q10 · Numerical · Medium · [5 marks]
The rig of Q9 ($f=700$ px, $B=0.12$ m) matches disparities with a sub-pixel accuracy of $\Delta d=0.25$ px.

(a) Estimate the depth uncertainty $\Delta Z$ at $Z=10$ m. **[1]**

(b) What baseline would be needed for $\Delta Z\le0.10$ m at $Z=10$ m (same $f$ and $\Delta d$)? **[2]**

(c) For the **original** rig: what is the maximum range if the smallest reliable disparity is 1 px? What is the minimum measurable depth if the disparity search range is $0$–$128$ px? Then give the minimum depth for the **new** baseline from (b) with the same search range, and comment on the trade-off. **[2]**

### Q11 · Numerical · Medium · [5 marks]
Two calibrated views are related by $\mathbf X_2=R\mathbf X_1+\mathbf t$ with
$R=R_z(90^\circ)=\begin{bmatrix}0&-1&0\\1&0&0\\0&0&1\end{bmatrix}$ and $\mathbf t=(1,\ 0,\ 0)^\top$ (unit-norm translation).

(a) Compute $E=[\mathbf t]_\times R$. **[1]**

(b) A point has camera-1 coordinates $\mathbf X_1=(1,\ 2,\ 4)^\top$. Compute $\mathbf X_2$, the normalised image points $\hat{\mathbf x}_1,\hat{\mathbf x}_2$, and verify that $\hat{\mathbf x}_2^\top E\,\hat{\mathbf x}_1=0$. **[2]**

(c) Compute the epipolar line $\mathbf l_2=E\hat{\mathbf x}_1$ in image 2 (normalised coordinates), write it as an equation, and check that $\hat{\mathbf x}_2$ lies on it. **[1]**

(d) Give the singular values of $E$ and the epipole $\mathbf e_2$ in image 2 ($E^\top\mathbf e_2=\mathbf 0$). What does each tell you? **[1]**

### Q12 · Numerical · Medium · [5 marks]
(a) At a pixel, $I_x=3$, $I_y=4$ and $I_t=-10$ (intensity units per pixel and per frame). Using only this pixel, what flow can be recovered? Give the normal-flow vector and its magnitude, and check it against the optical flow constraint. **[2]**

(b) Over a 5×5 Lucas–Kanade window: $\sum I_x^2=20$, $\sum I_xI_y=4$, $\sum I_y^2=10$, $\sum I_xI_t=-28$, $\sum I_yI_t=-1$. Solve for $(u,v)$. **[2]**

(c) Compute the eigenvalues of $A^\top A$ in (b). Compare it with a second window where $\sum I_x^2=20$, $\sum I_xI_y=0$, $\sum I_y^2=0.01$. Which window is reliable to track, and why? **[1]**

### Q13 · Numerical · Medium · [5 marks]
(a) Geometric verification between two images must succeed with probability $p=0.99$ when half the putative matches are outliers ($e=0.5$). How many RANSAC iterations are needed with the 8-point, the 7-point and the 5-point solver? **[3]**

(b) A bundle adjustment has 50 cameras with fixed, known intrinsics (6 pose parameters each) and 10 000 3D points, each observed in 4 images on average. How many unknown parameters, observations and scalar residuals are there? What is the size of the reduced camera system after eliminating the points with the Schur complement? **[2]**

## Section C — Short answer

### Q14 · Short · Medium · [4 marks]
Derive $Z=fB/d$ for a rectified stereo pair using similar triangles. Then explain why depth precision degrades with distance (give $\Delta Z$), and describe the trade-offs in choosing the baseline.

### Q15 · Short · Medium · [4 marks]
State the three assumptions behind differential optical flow and derive the optical flow constraint equation. Explain the aperture problem, and how Lucas–Kanade and Horn–Schunck each deal with it. Why is a pyramid needed?

### Q16 · Short · Medium · [4 marks]
Compare the essential and fundamental matrices (coordinates, formula, DOF, rank and singular values). Then list the steps of the **normalised** 8-point algorithm, explaining why normalisation and rank-2 enforcement are needed.

### Q17 · Short · Medium · [4 marks]
Describe the front end and back end of a visual SLAM system. Explain how loop closure works (three steps), how it differs from relocalisation, and why monocular SLAM suffers from scale drift.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
A heritage agency gives you 300 tourist photographs of a historic building, taken with different phones from all around it. Design an **incremental Structure-from-Motion** pipeline to recover the camera poses and a 3D model. Cover:
feature extraction and matching; geometric verification (which model and why RANSAC); initial-pair selection; two-view reconstruction (E, decomposition, cheirality, triangulation); image registration; bundle adjustment (objective, size, solver); the scale ambiguity and how to obtain metres; densification; and likely failure modes with mitigations.

### Q19 · Long · Hard · [10 marks]
Design the perception system for a warehouse robot that must (i) measure the depth of obstacles from 0.5 m to 8 m and (ii) track the people walking around it. A stereo camera with $f=800$ px and a matching accuracy of $\Delta d=0.25$ px is available, and the requirement is $\Delta Z\le5$ cm at $Z=4$ m. Cover:
the choice of baseline and disparity search range (with calculations), and the depth uncertainty at 8 m; calibration and rectification; the choice of stereo-matching algorithm and post-processing; problem surfaces in a warehouse and fixes; how to track people over time (motion model, association, identity through occlusion); and how you would evaluate the system.
