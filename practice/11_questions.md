# Session 11 — Camera Geometry · Practice Questions

> Notes: [../notes/11_camera_geometry.md](../notes/11_camera_geometry.md) · Solutions: [11_solutions.md](11_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min

Conventions used throughout: camera frame $x$ right, $y$ down, $z$ forward (OpenCV). World → camera is $\mathbf X_c=R\mathbf X_w+\mathbf t$. $\lambda\tilde{\mathbf x}=K[R\mid\mathbf t]\tilde{\mathbf X}$. Distortion is applied in normalised coordinates with OpenCV order $(k_1,k_2,p_1,p_2,k_3)$.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
How many degrees of freedom does a general intrinsic matrix $K$ (including skew) have?

- **A.** 4
- **B.** 6
- **C.** 5
- **D.** 11

### Q2 · MCQ · Easy
A calibrated camera is moved from one tripod position to another and rotated. Zoom and focus are unchanged. Which parameters must be re-estimated?

- **A.** The intrinsic matrix $K$ only
- **B.** The rotation $R$ and translation $\mathbf t$ only
- **C.** $K$, $R$ and $\mathbf t$
- **D.** The distortion coefficients and $K$

### Q3 · MCQ · Medium
Which property of the scene is preserved by ideal perspective (pinhole) projection?

- **A.** Parallelism of lines
- **B.** Ratios of lengths along a line (e.g. midpoints)
- **C.** Angles between lines
- **D.** Straight lines map to straight lines (ignoring lens distortion)

### Q4 · MCQ · Medium
A wide-angle lens makes straight lines near the image border bow outward. In the model $x_d=x(1+k_1r^2+k_2r^4+k_3r^6)$, what is the most likely sign of $k_1$, and how do image points move relative to the ideal pinhole image?

- **A.** $k_1<0$; points are pulled towards the image centre
- **B.** $k_1>0$; points are pushed away from the image centre
- **C.** $k_1<0$; points are pushed away from the image centre
- **D.** $k_1>0$; points are pulled towards the image centre

### Q5 · MCQ · Medium
The $3\times4$ camera projection matrix $P=K[R\mid\mathbf t]$ has how many degrees of freedom?

- **A.** 12
- **B.** 6
- **C.** 8
- **D.** 11

### Q6 · MCQ · Medium
In which situation are two photographs related **exactly** by a single $3\times3$ homography?

- **A.** A camera translates 1 m sideways while viewing a cluttered room
- **B.** Two cameras with a 50 cm baseline view a 3D statue
- **C.** A camera rotates about its optical centre while viewing a cluttered room
- **D.** Any two images, once lens distortion has been removed

### Q7 · MCQ · Hard
In Zhang's calibration with a planar checkerboard and no assumption on skew, why are at least **three** views needed?

- **A.** Each view's homography has 8 DOF, 6 of which are used by the board pose. That leaves 2 constraints per view on the 5 unknowns of $B=K^{-\top}K^{-1}$.
- **B.** Each view gives one constraint, and $K$ has 3 unknowns.
- **C.** Three views are needed to triangulate the board corners in 3D before calibrating.
- **D.** Each view gives 4 constraints, and the 12 entries of $P$ need three views.

### Q8 · MCQ · Hard
A camera has $f_x=f_y=800$ px and $(c_x,c_y)=(320,240)$. A family of parallel 3D lines has direction $\mathbf D=(1,0,1)$ in camera coordinates. Where is their vanishing point in the image?

- **A.** $(320,\ 240)$
- **B.** $(1120,\ 240)$
- **C.** $(885.7,\ 240)$
- **D.** At infinity: parallel lines never meet in the image

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
A camera has $K=\begin{bmatrix}800&0&320\\0&800&240\\0&0&1\end{bmatrix}$ (pixels) and no distortion.

(a) With $R=I$ and $\mathbf t=(0.1,\ 0,\ 1)^\top$ m, project the world point $\mathbf X_w=(0.3,\ -0.2,\ 3)^\top$ m to pixels. **[2]**

(b) The camera is now rotated by $+90^\circ$ about the $y$-axis, $R=R_y(90^\circ)=\begin{bmatrix}0&0&1\\0&1&0\\-1&0&0\end{bmatrix}$, with $\mathbf t=(0,\ 0,\ 5)^\top$ m. Project $\mathbf X_w=(1,\ 0.6,\ 1.2)^\top$ m. **[2]**

(c) For the pose in (b), give the camera centre $C$ in world coordinates and the direction of the optical axis in the world frame. **[1]**

### Q10 · Numerical · Easy · [5 marks]
A smartphone camera has a lens focal length $f=4.2$ mm, square pixels of pitch $1.4\ \mu$m, and an image of $4000\times3000$ pixels. The principal point is at the image centre and there is no distortion.

(a) Compute $f_x$ in pixels. **[1]**

(b) Compute the horizontal and vertical fields of view. **[2]**

(c) A person 1.8 m tall stands upright, facing the camera, 12 m away (the camera axis is horizontal). How tall is the person in the image, in pixels? **[1]**

(d) A car known to be 1.5 m tall appears 75 px tall. Estimate its distance. **[1]**

### Q11 · Numerical · Medium · [5 marks]
A camera has $f_x=f_y=800$ px, $(c_x,c_y)=(320,240)$ and OpenCV distortion coefficients $(k_1,k_2,p_1,p_2,k_3)=(-0.25,\ 0.05,\ 0.001,\ -0.002,\ 0)$. A 3D point projects to the **undistorted** normalised coordinates $(x,y)=(0.3,\ 0.2)$.

(a) Where would the point appear in an ideal (distortion-free) camera, in pixels? **[1]**

(b) Compute the distorted normalised coordinates $(x_d,y_d)$, showing the radial and tangential parts separately. **[3]**

(c) Convert to pixels, give the pixel shift caused by distortion, and state whether this is barrel or pincushion distortion. **[1]**

### Q12 · Numerical · Medium · [5 marks]
A camera with $K=\begin{bmatrix}800&0&320\\0&800&240\\0&0&1\end{bmatrix}$ views a flat floor. The world frame lies on the floor ($Z_w=0$), with
$R=\begin{bmatrix}1&0&0\\0&0.8&-0.6\\0&0.6&0.8\end{bmatrix}$ and $\mathbf t=(0,\ 0,\ 5)^\top$ m.

(a) Compute the floor-to-image homography $H=K[\mathbf r_1\ \mathbf r_2\ \mathbf t]$. **[2]**

(b) Use $H$ to find the pixel of the floor point $(X,Y)=(1,\ 2)$ m, and check it by projecting the 3D point $(1,2,0)$ with the full camera model. **[2]**

(c) Separately, a depth sensor reports that pixel $(u,v)=(500,\ 150)$ has camera-frame depth $Z_c=2.5$ m. Back-project it to $(X_c,Y_c,Z_c)$. **[1]**

### Q13 · Numerical · Medium · [5 marks]
(a) Calibration by DLT with a 3D rig: what is the minimum number of non-coplanar 3D–2D correspondences needed to estimate $P$, and what is the size of the matrix $A$ in $A\mathbf p=\mathbf 0$ for that minimum? **[1]**

(b) Zhang's method: how many constraints on $B=K^{-\top}K^{-1}$ does each board view give? What is the minimum number of views if (i) all 5 intrinsics are unknown, (ii) skew $=0$ is imposed, (iii) skew $=0$ and the principal point is known? **[2]**

(c) An OpenCV calibration uses 15 views of a board with $9\times6$ inner corners. It estimates $f_x,f_y,c_x,c_y$ (zero skew), 5 distortion coefficients and the extrinsics of every view. How many parameters and how many scalar residuals does the final Levenberg–Marquardt problem have? **[1]**

(d) Four detected corners have reprojection error vectors (pixels) $(0.3,0.4)$, $(-0.6,0.8)$, $(0,0.5)$, $(0.5,-1.2)$. Compute the RMS reprojection error. **[1]**

## Section C — Short answer

### Q14 · Short · Easy · [3 marks]
What are homogeneous coordinates? Give three reasons why camera geometry is written in homogeneous coordinates.

### Q15 · Short · Medium · [4 marks]
Distinguish intrinsic from extrinsic camera parameters: list each set, give its DOF, and explain what it physically represents. For each of the following, say which parameters change: (i) the camera is mounted on a moving car, (ii) the lens is zoomed in, (iii) every image is resized to half resolution.

### Q16 · Short · Medium · [4 marks]
Explain radial and tangential lens distortion: the model, barrel vs pincushion, and where each comes from. Why can a homography **not** correct radial distortion? How does OpenCV undistort a whole video stream efficiently?

### Q17 · Short · Hard · [5 marks]
(a) Show that the image of a world plane is related to the plane by a homography $H=K[\mathbf r_1\ \mathbf r_2\ \mathbf t]$. **[2]**
(b) Show that two images taken by a camera rotating purely about its optical centre are related by $H=KRK^{-1}$ for **any** scene. **[2]**
(c) Why is there no single homography between two views of a general 3D scene when the camera translates? **[1]**

## Section D — Long answer / Design

### Q18 · Long · Medium · [10 marks]
You must calibrate the camera of a mobile robot so that it can measure distances in metres. Design the full calibration procedure using Zhang's method. Cover:
the model being estimated (intrinsics, distortion, extrinsics); why a planar target gives a homography per view; why at least three views are needed; the step-by-step algorithm, including the non-linear refinement; how you would do it in OpenCV; how you would judge calibration quality; and four practical pitfalls.

### Q19 · Long · Hard · [10 marks]
A city wants to measure vehicle speeds with a **single fixed camera** mounted 6 m above a straight, flat road, looking down the road at an angle. Design a system that converts image tracks into speeds in km/h using camera geometry and a road-plane homography. Cover:
calibration (intrinsics and distortion, and how to obtain the image↔road mapping); the bird's-eye-view / IPM transform; which image point of each vehicle to track and why; the speed computation (worked example: a tracked ground point moves 2.5 m on the road plane in 5 frames at 25 fps); error sources, including what happens if you track a point on the vehicle **roof** (1.5 m high) instead of the road contact point; and how you would validate the system.
