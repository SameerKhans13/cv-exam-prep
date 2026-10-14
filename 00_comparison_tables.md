# Comparison Tables — every "X vs Y" in one place

> Organised by session; cross-session tables sit in the session where the comparison is usually examined. **Bold** = the decisive difference. Each table ends with an **Exam verdict** (when to use which).
> Conventions and formulas: [Formula sheet](00_formula_sheet.md) · Source notes: [notes/](notes/) (section 6 of each file).

## Contents

- **[S01 Image processing and features](#s01-image-processing-and-features)**: [Correlation vs convolution](#correlation-vs-convolution) · [Box vs Gaussian vs median vs bilateral](#box-vs-gaussian-vs-median-vs-bilateral) · [Edge operators](#edge-operators) · [Harris vs Shi-Tomasi vs FAST](#harris-vs-shi-tomasi-vs-fast) · [SIFT vs SURF vs ORB](#sift-vs-surf-vs-orb) · [Float vs binary descriptors](#float-vs-binary-descriptors)
- **[S02 Feature matching and classical CV](#s02-feature-matching-and-classical-cv)**: [Distance metrics L1 L2 Hamming cosine chi-square](#distance-metrics-l1-l2-hamming-cosine-chi-square) · [Brute force vs FLANN](#brute-force-vs-flann) · [Match filtering strategies](#match-filtering-strategies) · [Transform hierarchy](#transform-hierarchy) · [RANSAC variants](#ransac-variants) · [Panorama projections and blending](#panorama-projections-and-blending) · [Feature-based vs learned matching](#feature-based-vs-learned-matching)
- **[S03 CNNs under the hood](#s03-cnns-under-the-hood)**: [Conv vs FC layer](#conv-vs-fc-layer) · [Equivariance vs invariance](#equivariance-vs-invariance) · [Max vs avg vs global pooling](#max-vs-avg-vs-global-pooling) · [Kernel design choices](#kernel-design-choices) · [LeNet to ViT architecture evolution](#lenet-to-vit-architecture-evolution) · [Plain vs residual network](#plain-vs-residual-network) · [Basic vs bottleneck block](#basic-vs-bottleneck-block) · [CNN vs ViT](#cnn-vs-vit)
- **[S04 Training CNNs and transfer learning](#s04-training-cnns-and-transfer-learning)**: [Regularisers L1 vs L2 vs dropout vs BN vs early stopping](#regularisers-l1-vs-l2-vs-dropout-vs-bn-vs-early-stopping) · [Overfitting vs underfitting diagnosis](#overfitting-vs-underfitting-diagnosis) · [Feature extraction vs fine-tuning](#feature-extraction-vs-fine-tuning) · [Transfer learning decision matrix](#transfer-learning-decision-matrix) · [Augmentation types](#augmentation-types) · [Cutout vs Mixup vs CutMix](#cutout-vs-mixup-vs-cutmix) · [SGD vs Adam vs AdamW](#sgd-vs-adam-vs-adamw) · [LR schedules](#lr-schedules) · [Metrics under imbalance](#metrics-under-imbalance) · [BatchNorm train vs eval mode](#batchnorm-train-vs-eval-mode)
- **[S05 Embeddings and image similarity](#s05-embeddings-and-image-similarity)**: [Euclidean vs cosine vs dot product](#euclidean-vs-cosine-vs-dot-product) · [Siamese contrastive vs triplet vs InfoNCE](#siamese-contrastive-vs-triplet-vs-infonce) · [SimCLR vs MoCo vs BYOL vs DINO](#simclr-vs-moco-vs-byol-vs-dino) · [Sources of embeddings](#sources-of-embeddings) · [Exact vs approximate nearest-neighbour search](#exact-vs-approximate-nearest-neighbour-search) · [Verification vs identification vs retrieval](#verification-vs-identification-vs-retrieval)
- **[S06 Detection I two-stage](#s06-detection-i-two-stage)**: [Classification vs localisation vs detection vs segmentation](#classification-vs-localisation-vs-detection-vs-segmentation) · [R-CNN vs Fast R-CNN vs Faster R-CNN](#r-cnn-vs-fast-r-cnn-vs-faster-r-cnn) · [Proposal methods](#proposal-methods) · [HOG-SVM vs Viola-Jones vs DPM](#hog-svm-vs-viola-jones-vs-dpm)
- **[S07 Detection II one-stage](#s07-detection-ii-one-stage)**: [YOLO vs SSD vs RetinaNet vs FCOS vs CenterNet](#yolo-vs-ssd-vs-retinanet-vs-fcos-vs-centernet) · [Anchor-based vs anchor-free](#anchor-based-vs-anchor-free) · [NMS vs Soft-NMS vs NMS-free](#nms-vs-soft-nms-vs-nms-free) · [AP at 0.5 vs COCO AP](#ap-at-05-vs-coco-ap)
- **[S08 Detection III transformers](#s08-detection-iii-transformers)**: [Two-stage vs one-stage vs DETR](#two-stage-vs-one-stage-vs-detr) · [DETR vs Deformable DETR](#detr-vs-deformable-detr) · [Matching cost vs Hungarian loss](#matching-cost-vs-hungarian-loss) · [Error types in detector analysis](#error-types-in-detector-analysis) · [Modern detector families](#modern-detector-families)
- **[S09 Revision and practicals](#s09-revision-and-practicals)**: [Hand-crafted vs learned vs foundation features](#hand-crafted-vs-learned-vs-foundation-features) · [Supervised vs self-supervised vs language-supervised](#supervised-vs-self-supervised-vs-language-supervised) · [Classification metrics](#classification-metrics) · [Retrieval metrics](#retrieval-metrics) · [PCA vs t-SNE vs UMAP](#pca-vs-t-sne-vs-umap)
- **[S10 Image segmentation](#s10-image-segmentation)**: [Semantic vs instance vs panoptic](#semantic-vs-instance-vs-panoptic) · [FCN vs U-Net vs Mask R-CNN vs DeepLab vs Mask2Former vs SAM](#fcn-vs-u-net-vs-mask-r-cnn-vs-deeplab-vs-mask2former-vs-sam) · [Other modern segmentation models](#other-modern-segmentation-models) · [Skip connections addition vs concatenation](#skip-connections-addition-vs-concatenation) · [RoI Pool vs RoIAlign](#roi-pool-vs-roialign) · [IoU vs Dice vs pixel accuracy](#iou-vs-dice-vs-pixel-accuracy)
- **[S11 Camera geometry](#s11-camera-geometry)**: [Perspective vs weak-perspective vs orthographic](#perspective-vs-weak-perspective-vs-orthographic) · [Intrinsics vs extrinsics](#intrinsics-vs-extrinsics) · [Radial vs tangential distortion](#radial-vs-tangential-distortion) · [Calibration methods DLT vs Zhang vs PnP](#calibration-methods-dlt-vs-zhang-vs-pnp) · [Homography cases](#homography-cases)
- **[S12 3D vision and motion](#s12-3d-vision-and-motion)**: [Homography vs F vs E vs P](#homography-vs-f-vs-e-vs-p) · [Stereo matching methods](#stereo-matching-methods) · [Larger vs smaller baseline](#larger-vs-smaller-baseline) · [LK vs Horn-Schunck vs RAFT](#lk-vs-horn-schunck-vs-raft) · [KLT vs SORT vs DeepSORT](#klt-vs-sort-vs-deepsort) · [SfM vs SLAM vs VO](#sfm-vs-slam-vs-vo) · [Incremental vs global SfM](#incremental-vs-global-sfm) · [SLAM sensor setups](#slam-sensor-setups) · [Stereo vs monocular depth vs SfM vs SLAM](#stereo-vs-monocular-depth-vs-sfm-vs-slam)
- **[S13 Vision foundation models](#s13-vision-foundation-models)**: [CLIP vs DINOv2 vs SAM vs Depth Anything vs OWL-ViT vs Grounding DINO](#clip-vs-dinov2-vs-sam-vs-depth-anything-vs-owl-vit-vs-grounding-dino) · [Adaptation modes](#adaptation-modes) · [CLIP vs DINOv2 features](#clip-vs-dinov2-features) · [SAM vs Mask R-CNN](#sam-vs-mask-r-cnn)
- **[S14 Vision++ with foundation models](#s14-vision-with-foundation-models)**: [Self-supervised DINO vs Grounding DINO's DINO](#self-supervised-dino-vs-grounding-dinos-dino) · [OWL-ViT vs Grounding DINO](#owl-vit-vs-grounding-dino) · [Closed-set vs open-vocabulary detection](#closed-set-vs-open-vocabulary-detection) · [Zero-shot vs few-shot vs fine-tuned](#zero-shot-vs-few-shot-vs-fine-tuned) · [Three ways to get instance masks for a class](#three-ways-to-get-instance-masks-for-a-class) · [Ways to get metric distance](#ways-to-get-metric-distance) · [Pipeline stages and failure modes](#pipeline-stages-and-failure-modes)
- **[S15 Generative CV](#s15-generative-cv)**: [AE vs VAE vs GAN vs diffusion](#ae-vs-vae-vs-gan-vs-diffusion) · [Generative learning trilemma](#generative-learning-trilemma) · [DDPM vs DDIM](#ddpm-vs-ddim) · [Classifier guidance vs CFG](#classifier-guidance-vs-cfg) · [Pixel vs latent diffusion](#pixel-vs-latent-diffusion) · [pix2pix vs CycleGAN](#pix2pix-vs-cyclegan) · [Editing and personalisation methods](#editing-and-personalisation-methods) · [FID vs IS](#fid-vs-is)

---

## S01 Image processing and features

> Notes: [01_image_processing_features.md](notes/01_image_processing_features.md#6-comparison-tables)

### Correlation vs convolution

| | Correlation | Convolution |
|---|---|---|
| Formula | $\sum K(u,v)\,I(x+u,y+v)$ | $\sum K(u,v)\,I(x-u,y-v)$ |
| Kernel | used as is | **flipped 180°** |
| Commutative / associative | no | **yes** (LSI theory, FFT) |
| Used by | template matching, `cv2.filter2D`, **CNN layers** | signal-processing theory, `scipy.ndimage.convolve` |
| Same result when | kernel symmetric under 180° (Gaussian, box, Laplacian); Sobel only flips sign | ← same |

**Exam verdict:** CNN "convolution" is really correlation (the flip is irrelevant because weights are learned); state the flip whenever you hand-compute with an asymmetric kernel.

### Box vs Gaussian vs median vs bilateral

| Filter | Linear? | Edge-preserving? | Best for | Cost | Weakness |
|---|---|---|---|---|---|
| Box / mean | yes | no | fastest blur | O(1)/px with integral image | ringing, anisotropic |
| Gaussian | yes | no | **Gaussian noise**, pre-smoothing, scale space | $2K$/px (separable) | blurs edges |
| Median | **no** | mostly (steps) | **salt-and-pepper** | sort per pixel | removes thin lines, rounds corners |
| Bilateral | **no** | **yes** | denoise while keeping edges | expensive, not separable | cartoon/staircase look; two parameters $\sigma_s,\sigma_r$ |

**Exam verdict:** Gaussian for Gaussian noise and before derivatives; median for impulse noise; bilateral when edges must survive; box only when raw speed matters.

### Edge operators

| Operator | Order | Output | Noise robustness | Notes |
|---|---|---|---|---|
| Prewitt / Sobel | 1st | $I_x,I_y$ | low–medium / medium | Sobel = [1,2,1] smoothing × [−1,0,1], separable |
| Laplacian | 2nd | zero-crossings | **very low** | isotropic, no direction, double edges |
| LoG | 2nd | zero-crossings / blobs | good (σ) | Marr–Hildreth; blob detector at $\sigma=r/\sqrt2$ |
| DoG | ≈ 2nd | blobs | good | cheap approximation of scale-normalised LoG (SIFT) |
| Canny | 1st + logic | **thin, connected edges** | **best** | Gaussian → Sobel → NMS → hysteresis (σ + 2 thresholds) |

**Exam verdict:** Canny for clean edge maps; Sobel when you need gradient magnitude/direction (HOG, Harris); LoG/DoG for blobs and scale selection.

### Harris vs Shi-Tomasi vs FAST

| Detector | Response | Rotation inv. | Scale inv. | Speed | Typical use |
|---|---|---|---|---|---|
| Harris | $\det M-k(\operatorname{tr}M)^2$ | yes | **no** | fast | calibration corners |
| Shi–Tomasi | $\min(\lambda_1,\lambda_2)$ | yes | no | fast | **KLT tracking** (`goodFeaturesToTrack`) |
| FAST-9 | segment test: 9 contiguous of 16 circle px | no orientation | no | **fastest** | real-time SLAM (inside ORB) |
| DoG (SIFT) | scale-space extrema | yes (with orientation) | **yes** | slow | SIFT keypoints |
| Fast-Hessian (SURF) | $\det H_{approx}$ via box filters | yes | yes | medium | SURF keypoints |

**Exam verdict:** tracking → Shi–Tomasi (same $M$ as Lucas–Kanade); real-time → FAST (ORB adds Harris ranking, pyramid, orientation); scale changes → DoG / Fast-Hessian. None of Harris/Shi–Tomasi/FAST is scale invariant on its own.

### SIFT vs SURF vs ORB

| | SIFT | SURF | ORB |
|---|---|---|---|
| Year / authors | 1999/2004, Lowe | 2006, Bay et al. | 2011, Rublee et al. |
| Detector | DoG extrema in scale space | Fast-Hessian (box filters, integral image) | oFAST: FAST-9 + Harris ranking + pyramid (8 levels × 1.2) |
| Orientation | 36-bin gradient histogram | Haar wavelets | intensity centroid |
| Descriptor | float gradient histograms, **128-d** (512 B) | float Haar sums, **64-d** (256 B) | **binary rBRIEF, 256 bits = 32 B** |
| Match metric | L2 | L2 | **Hamming** |
| Speed | slowest | several × faster than SIFT | **fastest** (real-time on CPU) |
| Scale invariance | strong | strong | moderate (pyramid) |
| Distinctiveness | **highest** | high | lower |
| Licence | patent expired 2020 (main OpenCV) | **patented / non-free** | free (BSD) |

**Exam verdict:** accuracy, offline matching, SfM → SIFT; real-time, mobile, SLAM → ORB; SURF is a legacy speed/accuracy compromise held back by its licence.

### Float vs binary descriptors

| | Float (SIFT, SURF) | Binary (BRIEF, ORB, BRISK) |
|---|---|---|
| Distance | L2 (or L1) | **Hamming** (XOR + popcount) |
| Memory | 256–512 B | **32–64 B** |
| Matching speed | slower | **very fast** |
| ANN index | KD-tree (FLANN) | **LSH** |
| Robustness | **higher** | lower under large changes |

**Exam verdict:** binary when memory/speed dominate (embedded, SLAM); float when robustness and distinctiveness matter.

---

## S02 Feature matching and classical CV

> Notes: [02_feature_matching_classical.md](notes/02_feature_matching_classical.md#6-comparison-tables)

### Distance metrics L1 L2 Hamming cosine chi-square

| Metric | Formula | Use for | OpenCV |
|---|---|---|---|
| L1 | $\sum\lvert a_i-b_i\rvert$ | histograms; less outlier-sensitive than L2 | `NORM_L1` |
| L2 | $\sqrt{\sum(a_i-b_i)^2}$ | **SIFT, SURF, RootSIFT** | `NORM_L2` |
| Hamming | $\operatorname{popcount}(\mathbf a\oplus\mathbf b)$ | **binary**: ORB, BRIEF, BRISK, AKAZE | `NORM_HAMMING` (`NORM_HAMMING2` for ORB WTA_K 3/4) |
| Cosine | $1-\cos\theta$ | **deep embeddings** (CNN, CLIP); ignores magnitude | — |
| χ² | $\frac12\sum\frac{(h_i-g_i)^2}{h_i+g_i}$ | **histograms** (colour, LBP, BoW) | `compareHist` (conventions differ by constants) |

**Exam verdict:** match the metric to the descriptor — float → L2, binary → Hamming, histograms → χ² / intersection, embeddings → cosine (equivalent to L2 on unit vectors).

### Brute force vs FLANN

| | Brute force | FLANN KD-tree | FLANN LSH |
|---|---|---|---|
| Exact? | **yes** | approximate | approximate |
| Descriptors | any (choose norm) | **float** | **binary** |
| Cost | $O(N_1N_2D)$ | ~logarithmic per query | ~constant per query (hashing) |
| Best for | two images, ≤ a few thousand features | large sets, databases | large binary sets |
| OpenCV | `BFMatcher(NORM_L2 or NORM_HAMMING)` | `FlannBasedMatcher(algorithm=1)` | `FlannBasedMatcher(algorithm=6)` |

**Exam verdict:** brute force for an image pair (exact, simple, supports cross-check); FLANN once the database is large — KD-tree for float, LSH for binary.

### Match filtering strategies

| Strategy | Rule | Removes | Caveat |
|---|---|---|---|
| Absolute threshold | $d_1 < T$ | very poor matches | threshold not transferable; keeps ambiguous matches |
| **Ratio test** | $d_1/d_2 < \tau$ (0.7–0.8) | **ambiguous** matches | needs $k=2$; discards true matches on repetitive structure |
| Cross-check (mutual NN) | $j=NN(i)$ and $i=NN(j)$ | many-to-one matches | both directions; in OpenCV only with $k=1$ |
| Symmetric ratio test | ratio both ways + mutual | both of the above | strictest, fewest matches |
| Geometric verification | RANSAC inliers | **anything inconsistent with $H$ / $F$** | needs a valid global model |

**Exam verdict:** standard pipeline = ratio test, then RANSAC; only geometric verification removes matches that look good but are wrong.

### Transform hierarchy

| Transform | Matrix | DOF | Min. pts | Preserves |
|---|---|---|---|---|
| Translation | $[I\ \ \mathbf t]$ | 2 | 1 | everything but position |
| Euclidean (rigid) | $[R\ \ \mathbf t]$ | 3 | 2 | lengths, angles, areas |
| Similarity | $[sR\ \ \mathbf t]$ | 4 | 2 | angles, length ratios (shape) |
| Affine | $[A\ \ \mathbf t]$ | 6 | 3 (non-collinear) | **parallelism**, area ratios |
| Homography | $H$ (3×3, up to scale) | **8** | **4** (no 3 collinear) | **collinearity**, cross-ratio only |

**Exam verdict:** use the simplest model the scene allows (fewer DOF → smaller RANSAC sample, more stable); a homography is exact only for a **planar scene or pure rotation**.

### RANSAC variants

| Variant | Key change | Benefit |
|---|---|---|
| RANSAC | count inliers below $t$ | simple, robust |
| MSAC | truncated quadratic score | better model ranking |
| PROSAC | sample best-quality matches first | **much faster** when scores are informative |
| LO-RANSAC | local optimisation of new best models | more accurate, fewer iterations |
| MLESAC / MAGSAC++ | likelihood / threshold-marginalised scoring | **less sensitive to $t$** |

**Exam verdict:** derive and use plain RANSAC; name PROSAC (speed) and MAGSAC++ (threshold robustness) as improvements.

### Panorama projections and blending

| Choice | Pros | Cons |
|---|---|---|
| Planar | straight lines stay straight; simple | extreme stretching for wide FOV; cannot reach 180° |
| Cylindrical | **360° horizontal**; rotation → translation | needs $f$; bends straight lines |
| Spherical | full 360° × 180° | strong line distortion |
| Feathering | simple, fast | ghosting if misaligned; exposure seams |
| Multi-band | **seamless and sharp** | more compute; cannot fix parallax |

**Exam verdict:** few images → planar; wide or 360° → cylindrical/spherical; multi-band blending for quality. Parallax can only be avoided by rotating about the optical centre.

### Feature-based vs learned matching

| | Classical (SIFT/ORB + NN + ratio) | Learned (SuperPoint + SuperGlue/LightGlue, LoFTR) |
|---|---|---|
| Training data | **none** | large datasets |
| Hardware | CPU, embedded | GPU preferred |
| Textureless / repetitive scenes | poor | **better** (context, dense matching) |
| Large illumination/viewpoint change | degrades | **much more robust** |
| Interpretability | **high** | lower |
| Geometric precision | excellent with RANSAC + BA | excellent; LoFTR refines sub-pixel |

**Exam verdict:** classical when compute, predictability or "no training data" matter; learned for hard appearance changes and low texture. Both still need RANSAC.

---

## S03 CNNs under the hood

> Notes: [03_cnns_under_the_hood.md](notes/03_cnns_under_the_hood.md#6-comparison-tables)

### Conv vs FC layer

| Aspect | Fully connected | Convolutional |
|---|---|---|
| Connectivity | every input to every output | **local** $K\times K\times C_{in}$ window |
| Weight sharing | none | **same filter at all positions** |
| Params | $(N_{in}+1)N_{out}$ — **grows with image size** | $(K^2C_{in}+1)C_{out}$ — independent of image size |
| Spatial structure | destroyed by flattening | preserved (feature maps) |
| Translation | no built-in property | **equivariant** (stride 1) |
| Input size | fixed | any (fully convolutional) |
| 224×224×3 example | → 1000 units: 150.5M params | 3×3, 64 filters: 1,792 params |

**Exam verdict:** convs for images (locality + weight sharing → few params, equivariance); FC only in the head, and even there GAP + a small linear layer is the modern default.

### Equivariance vs invariance

| | Equivariance | Invariance |
|---|---|---|
| Definition | $f(T_tI)=T_tf(I)$ — output **shifts with** input | $g(T_tI)=g(I)$ — output **unchanged** |
| Provided by | stride-1 convolution (and ReLU, BN) | max/avg pooling (local), **global pooling** (global) |
| Needed for | detection, segmentation ("where") | classification ("what") |
| Caveat | broken by striding/aliasing, borders | only approximate in practice |

**Exam verdict:** convolution gives "where" (equivariance), pooling gives "what" (invariance) — dense tasks keep the former, classifiers collapse to the latter.

### Max vs avg vs global pooling

| Operator | Learnable params | What it keeps | Typical use |
|---|---|---|---|
| Max-pool 2×2/s2 | 0 | **strongest response** per window | VGG (AlexNet and the ResNet stem use overlapping 3×3/s2) |
| Avg-pool 2×2/s2 | 0 | mean response | LeNet "subsampling", DenseNet transitions |
| **Global average pool** | 0 | per-channel mean → **one vector, any input size** | GoogLeNet, ResNet, EfficientNet heads |
| Stride-2 conv | $(K^2C_{in}+1)C_{out}$ | **learned** downsampling, can change channels | ResNet stages, All-Conv Net |
| Dilated conv | same as undilated | **full resolution**, larger RF | segmentation (DeepLab) |

**Exam verdict:** max-pool inside feature extractors (local invariance), GAP at the head (replaces huge FC layers), dilation when resolution must be preserved.

### Kernel design choices

| Option | RF | Weights ($C$ in/out, no bias) | Non-linearities |
|---|---|---|---|
| one 5×5 | 5 | $25C^2$ | 1 |
| **two 3×3** | 5 | $18C^2$ (−28 %) | 2 |
| one 7×7 | 7 | $49C^2$ | 1 |
| **three 3×3** | 7 | $27C^2$ (−45 %) | 3 |
| 3×3, $D=2$ | 5 | $9C^2$ | 1 (sparse sampling) |
| depthwise-separable 3×3 | 3 | $9C+C^2$ | – |

**Exam verdict:** stacked 3×3 (VGG) beats one big kernel: same RF, fewer weights, more non-linearity; depthwise-separable when compute is the constraint (MobileNet).

### LeNet to ViT architecture evolution

| Model | Year | Key idea | Depth | Params |
|---|---|---|---|---|
| LeNet-5 | 1998 | conv → avg-pool → FC, tanh, digits | 7 | ≈ 60K |
| AlexNet | 2012 | **ReLU, dropout, GPUs**, augmentation, LRN, overlapping max-pool | 8 | ≈ 60M |
| VGG-16 | 2014 | **only 3×3 convs**, uniform deep design | 16 | ≈ 138M (mostly FC) |
| GoogLeNet / Inception-v1 | 2014 | **Inception module, 1×1 bottlenecks, GAP** | 22 | ≈ 6.8M |
| ResNet-50 | 2015 | **residual $y=F(x)+x$**, BN, bottlenecks; fixes **degradation** | 50 | ≈ 25.6M |
| DenseNet-121 | 2017 | concatenate all previous maps (feature reuse) | 121 | ≈ 8.0M |
| MobileNetV2 | 2018 | depthwise-separable, inverted residuals | – | ≈ 3.5M |
| EfficientNet-B0 | 2019 | **compound scaling** of depth/width/resolution, MBConv | – | ≈ 5.3M |
| ViT-B/16 | 2020 | **16×16 patches as tokens**, global self-attention | 12 | ≈ 86M |

**Exam verdict:** the trend is deeper (ReLU → BN → residuals), cheaper (1×1, depthwise, GAP), principled scaling (EfficientNet), then weaker inductive bias + more data (ViT). ResNet fixes **optimisation (degradation)**, not overfitting.

### Plain vs residual network

| | Plain deep net | ResNet |
|---|---|---|
| Block output | $H(x)$ | $F(x)+x$ |
| Identity mapping | must be learned by non-linear layers (hard) | **$F\to0$ (easy)** |
| Backward Jacobian | product of layer Jacobians | $I+\partial F/\partial x$ (**direct gradient path**) |
| Depth behaviour | degradation: deeper → higher **training** error | deeper → lower error |
| Extra params | – | 0 (identity); 1×1 conv for projections |

**Exam verdict:** residuals make depth trainable; if the question shows deeper = worse **training** error, the answer is degradation, fixed by skip connections.

### Basic vs bottleneck block

| | Basic | Bottleneck |
|---|---|---|
| Layers | 3×3 → 3×3 | **1×1 reduce → 3×3 → 1×1 expand** |
| Used in | ResNet-18/34 | ResNet-50/101/152 |
| Weights at 256 channels | $2\cdot9\cdot256^2=1{,}179{,}648$ | **69,632** |

**Exam verdict:** bottlenecks let ResNet-50+ go deeper at similar cost.

### CNN vs ViT

| | CNN (ResNet-50) | ViT-B/16 |
|---|---|---|
| Basic op | local convolution | **global self-attention** over patch tokens |
| Inductive bias | **strong** (locality, equivariance, hierarchy) | weak (patching + position embeddings) |
| Receptive field | grows with depth | **global from layer 1** |
| Data needs | works on ImageNet-1k or less | **needs large pretraining** or strong augmentation/distillation |
| Params | ≈ 25.6M | ≈ 86M |
| Cost vs resolution | linear in pixels | **quadratic in tokens** |

**Exam verdict:** small/medium data or tight compute → CNN; large-scale pretraining available → ViT (the backbone of CLIP, DINOv2, SAM).

---

## S04 Training CNNs and transfer learning

> Notes: [04_training_transfer_learning.md](notes/04_training_transfer_learning.md#6-comparison-tables)

### Regularisers L1 vs L2 vs dropout vs BN vs early stopping

| Technique | Mechanism | Train vs test | Key hyper-param | Note |
|---|---|---|---|---|
| L2 / weight decay | shrink weights every step | same | $\lambda$ ($10^{-4}$ SGD, $10^{-2}$ AdamW) | smoother function; default |
| L1 | push weights **exactly to 0** | same | $\lambda$ | **sparsity** / pruning |
| Dropout | random zeroing → implicit ensemble | train: drop + scale $1/(1-p)$; **test: identity** | $p$ (0.2–0.5) | big FC layers |
| BatchNorm | normalise per channel with batch stats | train: batch stats; **test: running stats** | momentum, $\epsilon$ | **mainly an optimisation aid** (higher LR); mild regulariser |
| Early stopping | stop at the **validation** minimum | – | patience | ≈ implicit L2; saves compute |
| Label smoothing | soft targets | training targets only | $\varepsilon$ (0.1) | calibration |
| Data augmentation | more, varied data | **train only** (optional TTA) | policy / magnitude | encodes invariances |
| Mixup / CutMix | mix images **and labels** | train only | $\alpha$ | calibration / localisation |

**Exam verdict:** first line of defence = more/augmented data + pretrained weights + weight decay + early stopping; dropout for large FC heads; BN is not primarily a regulariser.

### Overfitting vs underfitting diagnosis

| Curve pattern | Diagnosis | Remedies |
|---|---|---|
| Train ↓, val ↓ then **↑**, gap widening | **overfitting** (high variance) | early stopping, augmentation, weight decay, dropout, more data, pretrained / smaller model |
| Train and val both high and close | **underfitting** (high bias) | bigger model, train longer, better LR, less regularisation |
| Both low, small stable gap | good fit | stop; test once |
| Loss explodes / NaN | LR too high | lower LR, warm-up, gradient clipping |
| Loss stuck at $\ln K$ | uniform outputs — nothing learns | check `optimizer.step()`, frozen params, LR, labels |
| Val slightly **above** train accuracy | dropout/augmentation only in training | usually fine; check leakage if large |

**Exam verdict:** read the **gap** (variance) and the **level** (bias) separately; fix bias with capacity, variance with data/regularisation.

### Feature extraction vs fine-tuning

| | Feature extraction | Fine-tuning |
|---|---|---|
| Trainable | **new head only** (ResNet-50, $K=10$: 20,490 params) | head + some/all backbone blocks |
| Learning rate | normal for head ($10^{-3}$) | **low** for backbone ($10^{-4}$–$10^{-5}$), discriminative per layer |
| Data needed | very little | moderate to large |
| Compute | low (features can be cached) | higher (full backprop) |
| Domain adaptation | **none** | **yes** |
| Risks | underfits distant domains | overfitting, catastrophic forgetting, BN drift |
| Must remember | keep frozen BN in `eval()` | warm up the head first; unfreeze top-down |

**Exam verdict:** see the 2×2 matrix below — data size and domain distance decide.

### Transfer learning decision matrix

| | **Similar to ImageNet** (dog breeds, products) | **Different domain** (X-ray, satellite, microscopy) |
|---|---|---|
| **Small data** (hundreds–few thousand) | **Feature extraction**: freeze backbone, train a linear head | hardest: freeze early layers, **fine-tune top blocks** with low LR + strong augmentation; or use earlier-layer features / domain-specific weights |
| **Large data** (tens of thousands+) | **fine-tune** most/all layers with low LR | **fine-tune the whole network** (pretrained init still helps); from scratch becomes viable |

**Exam verdict:** small + similar → freeze; large → fine-tune; small + different → partial fine-tune with heavy regularisation.

### Augmentation types

| Type | Examples | Builds robustness to | Label-preservation caveat |
|---|---|---|---|
| Geometric | flip, RandomResizedCrop, rotation, affine, elastic | position, scale, viewpoint | 6 ↔ 9, text, turn-left signs, medical laterality; **transform boxes/masks too** |
| Photometric | colour jitter, grayscale, blur, noise, PCA lighting | illumination, camera | colour-diagnostic tasks (skin lesions, ripeness) |
| Occlusion | Cutout, Random Erasing | occlusion; forces many cues | small objects can be erased |
| Mixing | Mixup (blend), CutMix (paste box) | smoother boundaries, calibration | **labels must be mixed too** |
| Policies | AutoAugment (RL search), RandAugment ($N$, $M$) | broad | AutoAugment search is expensive |
| Test-time (TTA) | average over flips/crops | – | $n\times$ inference cost; not for model selection |

**Exam verdict:** augment the **training set only**, with label-preserving transforms; val/test use a deterministic resize + centre-crop.

### Cutout vs Mixup vs CutMix

| | Cutout / Random Erasing | Mixup | CutMix |
|---|---|---|---|
| Image | random region removed | **global blend** of two images | box from image $j$ pasted into $i$ |
| Label | unchanged | $\lambda y_i+(1-\lambda)y_j$ | mixed **by area ratio** |
| Realism | partly occluded | ghosted overlay | locally natural |
| Main benefit | occlusion robustness | smoother boundaries, calibration | both + better localisation |

**Exam verdict:** CutMix when localisation matters (detection backbones); Mixup for calibration; Cutout for occlusion robustness.

### SGD vs Adam vs AdamW

| | SGD + momentum | Adam | AdamW |
|---|---|---|---|
| Step size | global LR | per-parameter adaptive | per-parameter adaptive |
| Weight decay | L2 ≡ weight decay | L2 rescaled by $1/\sqrt{\hat v}$ (**weak**) | **decoupled** (correct) |
| Typical LR | 0.1 from scratch; $10^{-3}$–$10^{-2}$ fine-tune | $10^{-3}$ | $10^{-3}$–$10^{-5}$ |
| Strength | best final accuracy for CNNs | fast, little tuning | **standard for fine-tuning and Transformers** |

**Exam verdict:** CNN from scratch → SGD + momentum + step/cosine; fine-tuning or ViTs → AdamW.

### LR schedules

| Schedule | Shape | Notes |
|---|---|---|
| Step decay | constant, ×0.1 every $s$ epochs | classic ResNet (every 30) |
| Cosine annealing | smooth half-cosine $\eta_{max}\to\eta_{min}$ | no step-time hyper-params |
| Linear warm-up | 0 → $\eta_{max}$ over first epochs | stabilises large LR/batch, Transformers; combined with cosine |
| ReduceLROnPlateau | decay when val metric stalls | adaptive; needs patience |

**Exam verdict:** warm-up + cosine is the modern default; step decay for the classic ResNet recipe.

### Metrics under imbalance

| Metric (900 neg, 100 pos, model predicts all negative) | Value | Lesson |
|---|---|---|
| Accuracy | **90 %** | looks good, is useless |
| Recall (positive class) | **0 %** | reveals the failure |
| Macro-F1 | ≈ 0.47 | penalises ignoring the rare class |
| Balanced accuracy | 50 % | chance level |

**Exam verdict:** with imbalance report per-class recall/precision, macro-F1 or balanced accuracy — never accuracy alone.

### BatchNorm train vs eval mode

| | `model.train()` | `model.eval()` |
|---|---|---|
| Statistics used | current mini-batch | **running** mean/var |
| Running stats updated? | **yes** (even if `requires_grad=False`) | no |
| Output depends on other images? | yes | no |
| Dropout | active (scaled) | off (identity) |

**Exam verdict:** always `eval()` for inference, and keep frozen BN layers in eval mode during fine-tuning.

---

## S05 Embeddings and image similarity

> Notes: [05_embeddings_similarity.md](notes/05_embeddings_similarity.md#6-comparison-tables)

### Euclidean vs cosine vs dot product

| | Euclidean $\Vert a-b\Vert$ | Cosine $\frac{a\cdot b}{\Vert a\Vert\Vert b\Vert}$ | Dot product $a\cdot b$ |
|---|---|---|---|
| Type | distance (lower = closer) | similarity (higher = closer) | similarity |
| Range | $[0,\infty)$; $[0,2]$ for unit vectors | $[-1,1]$ | $(-\infty,\infty)$ |
| Magnitude-sensitive? | yes | **no** (direction only) | yes (rewards large norms) |
| Unit vectors | $\sqrt{2-2\cos\theta}$ | $\cos\theta$ | $=\cos\theta$ |
| Typical use | triplet/contrastive losses, k-NN | retrieval, CLIP, SSL losses | MIPS, attention scores |

**Exam verdict:** L2-normalise embeddings and the three give the **same ranking**; without normalisation prefer cosine for retrieval.

### Siamese contrastive vs triplet vs InfoNCE

| | Contrastive (Siamese pairs) | Triplet | InfoNCE / NT-Xent |
|---|---|---|---|
| Unit | pair $(x_1,x_2,y)$ | $(a,p,n)$ | query + 1 positive + $N$ negatives |
| Constraint | **absolute**: $d_{pos}\to0$, $d_{neg}\ge m$ | **relative**: $d(a,n)\ge d(a,p)+m$ | positive must **win a softmax** |
| Hyper-param | margin $m$ | margin $m$ | temperature $\tau$, number of negatives |
| Negatives per step | 1 | 1 | **many** (batch / queue) |
| Mining | helpful | **essential** (semi-hard, batch-hard) | implicit (softmax weights hard negatives) |
| Typical use | signature/face verification | FaceNet, re-ID, product search | SimCLR, MoCo, CLIP |

**Exam verdict:** pairs → verification with labels; triplets → metric learning with identity labels (mine hard); InfoNCE → large-scale self-supervised / image–text training.

### SimCLR vs MoCo vs BYOL vs DINO

| | SimCLR | MoCo | BYOL | DINO |
|---|---|---|---|---|
| Negatives | rest of batch ($2N-2$) | **queue** (65,536) | **none** | **none** |
| Second branch | same encoder (shared) | momentum (EMA) key encoder | EMA target | EMA teacher |
| Collapse prevented by | negatives | negatives | predictor + stop-gradient + EMA | **centering + sharpening** + EMA |
| Batch-size dependence | **high** | low | low | low |
| Loss | NT-Xent | InfoNCE | MSE of normalised vectors | CE student vs teacher distributions |

**Exam verdict:** SimCLR needs huge batches; MoCo decouples negatives from batch size; BYOL/DINO drop negatives entirely (DINO → DINOv2 foundation features).

### Sources of embeddings

| | Classification features | Metric learning | Self-supervised | Language-supervised (CLIP) |
|---|---|---|---|---|
| Labels needed | class labels | same/different pairs | **none** | image–text pairs (web) |
| Positive pairs | implicit via classes | same identity | two augmentations of one image | image + its caption |
| Optimises distances? | no (decision boundaries) | **yes** | yes (instance level) | yes (cross-modal) |
| Unseen classes | moderate | yes | yes | **yes, via text (zero-shot)** |
| Weakness | ignores within-class detail | label cost, mining | false negatives, augmentation design | web bias, weak fine-grained/counting |

**Exam verdict:** need text queries → CLIP; no labels → self-supervised; identity-level retrieval (faces, products) → metric learning.

### Exact vs approximate nearest-neighbour search

| Method | Idea | Memory | Speed | Recall |
|---|---|---|---|---|
| Flat (brute force) | compare with all $N$ | $ND\cdot4$ B | $O(ND)$ | **exact** |
| IVF | k-means cells, probe `nprobe` of `nlist` | ≈ flat | ≈ nprobe/nlist of flat | high, tunable |
| PQ / IVF-PQ | sub-vectors → 1-byte codes | **$m$ B per vector** | fast (lookups) | lower (quantisation) |
| HNSW | layered proximity graph | flat + links | **very fast** | **very high** |
| LSH | random-projection hashing | small codes | fast | moderate |

**Exam verdict:** small DB → flat; millions with RAM → HNSW; billions or tight memory → IVF-PQ.

### Verification vs identification vs retrieval

| | Verification | Identification | Retrieval |
|---|---|---|---|
| Question | "Is A the same as B?" (**1:1**) | "Who is A?" (**1:N**, closed set) | "Show items like A" (ranked list) |
| Decision | $d < t$ threshold | nearest neighbour in gallery (+ reject threshold) | top-$k$ by similarity |
| Metric | ROC, TAR@FAR, EER | rank-1 accuracy, CMC | P@k, R@k, mAP, MRR |

**Exam verdict:** identify the task first — the metric follows from it.

---

## S06 Detection I two-stage

> Notes: [06_detection_two_stage.md](notes/06_detection_two_stage.md#6-comparison-tables)

### Classification vs localisation vs detection vs segmentation

| | Classification | Cls + localisation | Detection | Semantic seg. | Instance seg. | Panoptic seg. |
|---|---|---|---|---|---|---|
| Output | 1 label | 1 label + 1 box | **variable set** of (label, box, score) | label per pixel | set of (label, mask, score) | (class, instance id) per pixel |
| Background class | no | no | **yes** ($K+1$) | yes (a class) | yes | stuff classes |
| Separates instances | – | – (one object) | yes (boxes) | **no** | yes | yes (things) |
| Duplicate removal | – | – | NMS / set prediction | – | NMS / set prediction | merge, no overlaps |
| Label cost | lowest | low | medium (boxes) | high (masks) | high | **highest** |
| Metric | accuracy, top-5, macro-F1 | accuracy + IoU ≥ 0.5 | **AP / mAP** | **mIoU**, Dice | mask AP | **PQ** |
| Example | ResNet, ViT | ResNet + box head | Faster R-CNN, YOLO, DETR | FCN, U-Net, DeepLab | Mask R-CNN | Panoptic FPN, Mask2Former |

**Exam verdict:** pick the **coarsest output that answers the question** — counting/locating → detection; area or shape → segmentation (instance if objects must be told apart, panoptic for a complete scene map).

### R-CNN vs Fast R-CNN vs Faster R-CNN

| Aspect | R-CNN (2014) | Fast R-CNN (2015) | Faster R-CNN (2015) |
|---|---|---|---|
| Proposals | selective search (~2000, CPU) | selective search (~2000, CPU) | **RPN** (learned, ~300 at test, GPU, shared backbone) |
| CNN passes per image | **~2000** (one per warped crop) | **1** | **1** |
| Per-region features | warp crop → fc7 | **RoI pooling** 7×7 | RoI pooling 7×7 (RoIAlign later) |
| Classifier | per-class linear SVMs | softmax $K+1$ | softmax $K+1$ |
| Box refinement | separate regressors | joint smooth-L1 head | anchor → proposal → final box |
| Training | multi-stage, features on disk | single-stage multi-task | RPN + detector (alternating / approx. joint) |
| Bottleneck | 2000 CNN passes | **selective search** | per-RoI head, two stages |
| Speed | tens of s / image | ~seconds | **~5 fps** (VGG-16) |

**Exam verdict:** each step removes the previous bottleneck — shared computation (Fast), then learned proposals (Faster).

### Proposal methods

| Method | Learned? | Boxes | Speed | Notes |
|---|---|---|---|---|
| Sliding windows | no | $10^4$–$10^6$ | needs cheap classifier | fixed shapes, pyramid for scale |
| Selective search | no | ~2000 | seconds (CPU) | hierarchical grouping, high recall |
| EdgeBoxes | no | ~$10^3$ | fraction of a second | contours enclosed vs straddling |
| Objectness / BING | partly | ~$10^3$ | fast | generic object score |
| **RPN** | **yes** | ~300 | ~10 ms (GPU, shared) | anchors + objectness + offsets |

**Exam verdict:** RPN wins because it is learned and shares the backbone — proposals become almost free.

### HOG-SVM vs Viola-Jones vs DPM

| | HOG + linear SVM (Dalal–Triggs) | Viola–Jones | DPM |
|---|---|---|---|
| Feature | HOG (3780-d for 64×128) | Haar-like rectangles (integral image) | HOG root + parts |
| Classifier | linear SVM | **boosted cascade** | latent SVM |
| Handles deformation | no (rigid template) | no | **yes** (parts + deformation cost) |
| Speed trick | shared HOG cells | **early rejection**, 4-lookup sums | distance transforms |
| Typical use | pedestrians | **faces** (near-frontal) | general VOC classes |

**Exam verdict:** faces in real time → Viola–Jones; rigid upright objects → HOG + SVM; articulated objects → DPM. All superseded by CNN detectors.

RoI Pool vs RoIAlign: see [S10](#roi-pool-vs-roialign).

---

## S07 Detection II one-stage

> Notes: [07_detection_one_stage.md](notes/07_detection_one_stage.md#6-comparison-tables) · One-stage vs two-stage: see the three-way table in [S08](#two-stage-vs-one-stage-vs-detr).

### YOLO vs SSD vs RetinaNet vs FCOS vs CenterNet

| | YOLOv1 | SSD300 | RetinaNet | FCOS | CenterNet |
|---|---|---|---|---|---|
| Candidates | 7×7 cells × 2 = **98** | **8732** default boxes, 6 maps | ~$10^5$ anchors, 9/location, P3–P7 | 1 point/location, P3–P7 | 1 heatmap peak/object (stride 4) |
| Priors | none | default boxes | anchors (3 scales × 3 ratios) | **points** + size range per level | **points** |
| Cls loss | SSE (+ $\lambda_{noobj}$) | softmax CE + hard-neg. mining 3:1 | **focal loss** | focal loss | penalty-reduced focal |
| Box loss | SSE on $x,y,\sqrt w,\sqrt h$ | smooth-L1 | smooth-L1 | IoU/GIoU | L1 size + offset |
| Multi-scale | no | yes (pyramidal maps) | FPN | FPN | single high-res map |
| Duplicate removal | NMS | NMS | NMS | NMS (+ centre-ness) | **3×3 max-pool** |

**Exam verdict:** YOLOv1 = fastest idea but weak on small/crowded objects; SSD adds multi-scale; RetinaNet's focal loss closes the accuracy gap; FCOS/CenterNet remove anchors.

### Anchor-based vs anchor-free

| | Anchor-based | Anchor-free |
|---|---|---|
| Reference | boxes of fixed scales/ratios | **points** or keypoints |
| Regression target | offsets $(t_x,t_y,t_w,t_h)$ w.r.t. anchor | distances $(l,t,r,b)$ (FCOS) or size $(w,h)$ (CenterNet) |
| Label assignment | IoU(anchor, GT) thresholds | point inside box/centre region + scale range; Gaussian heatmap |
| Hyper-params | **many** (scales, ratios, IoU thresholds) | fewer |
| Outputs per location | $A$ (e.g. 9) | 1 |
| NMS | yes | FCOS **yes**; CenterNet max-pool only |
| Examples | Faster R-CNN, SSD, RetinaNet, YOLOv2–v5 | CornerNet, CenterNet, FCOS, YOLOX, YOLOv8 |

**Exam verdict:** anchor-free = fewer hand-tuned priors and outputs; it does **not** automatically mean NMS-free.

### NMS vs Soft-NMS vs NMS-free

| Method | Rule | Pros | Cons |
|---|---|---|---|
| Greedy NMS | delete boxes with IoU > $\tau$ vs a higher-scoring box | simple, fast, standard | hard threshold; **fails in crowds** |
| Soft-NMS | **decay** scores by overlap (linear / Gaussian) | better recall in crowds, drop-in | still heuristic; needs final score threshold |
| Max-pool peaks (CenterNet) | keep 3×3 heatmap maxima | very cheap, no IoU | close centres can merge |
| Set prediction (DETR) | **one-to-one Hungarian matching** in training | **no NMS at all** | slow convergence |

**Exam verdict:** crowded scenes → Soft-NMS or set prediction; otherwise greedy NMS at 0.5. Lower $\tau$ = more aggressive suppression.

### AP at 0.5 vs COCO AP

| Variant | Recall points | IoU threshold(s) | Used by |
|---|---|---|---|
| 11-point interpolated | 0, 0.1, …, 1 (11) | 0.5 | PASCAL VOC 2007 |
| **All-point** interpolated | every recall change (exact area under envelope) | 0.5 | VOC 2010+ (**pack default**) |
| 101-point interpolated | 0, 0.01, …, 1 | **0.50:0.05:0.95 (10), averaged** | COCO AP; also AP50, AP75 |
| Size-specific | 101-point | 0.50:0.05:0.95 | COCO AP_S (< 32²), AP_M, AP_L (> 96²) |

**Exam verdict:** mAP@0.5 rewards finding objects; COCO AP also rewards **tight boxes** — a loose-box detector can be high on the first and low on the second. Always state the method.

---

## S08 Detection III transformers

> Notes: [08_detection_transformers.md](notes/08_detection_transformers.md#6-comparison-tables)

### Two-stage vs one-stage vs DETR

| Aspect | Two-stage (Faster R-CNN + FPN) | One-stage (YOLO, SSD, RetinaNet, FCOS) | DETR / Deformable DETR |
|---|---|---|---|
| Pipeline | backbone → RPN → RoIAlign → per-RoI head | backbone (+FPN) → dense head | backbone → Transformer enc–dec → $N$ queries |
| Proposals | ~300–2000 RoIs | **none** (dense grid/anchors/points) | **none** ($N=100$ learned queries) |
| Anchors | yes (9/location) | YOLOv2–v5, SSD, RetinaNet; FCOS/CenterNet anchor-free | **no** |
| Label assignment | many-to-one, IoU thresholds | many-to-one | **one-to-one Hungarian** |
| Class imbalance | proposals + fixed fg:bg sampling | $\lambda_{noobj}$, hard-neg. mining, **focal loss** | $\varnothing$ class weight 0.1 |
| Box refinement | twice (anchor → proposal → box) | once | direct set prediction (iterative refinement in Deformable DETR) |
| NMS | yes (twice) | yes | **no** |
| Training (COCO) | short (~12–36 epochs) | short–moderate | **~500 epochs**; Deformable ~50 |
| Inference speed | slowest | **fastest (real time)** | moderate (RT-DETR real time) |
| Small objects | good (FPN + RoIAlign) | good with FPN | **weak** in DETR; fixed by Deformable |
| Large objects / context | good / local | good / local | **very good / global** |
| Crowded scenes | NMS deletes true overlaps | same | no NMS (max $N$ outputs) |
| Best for | accuracy-critical, offline, small objects | real time, edge, video | end-to-end simplicity, SOTA (DINO-style) |

**Exam verdict:** real time → one-stage; maximum accuracy on small/crowded objects with classic tooling → two-stage; no hand-designed anchors/NMS → DETR family (Deformable DETR for practical training time).

### DETR vs Deformable DETR

| Aspect | DETR | Deformable DETR |
|---|---|---|
| Attention | dense: every query attends to all $HW$ keys | **sparse**: $K$ sampled points per head per level around a reference point |
| Feature scales | single (C5, stride 32) | **multi-scale** (4 levels), no FPN needed |
| Encoder cost | $O((HW)^2)$ | $O(HW\cdot MLK)$ — **linear** |
| Training epochs | ~500 | **~50 (≈ 10× fewer)** |
| Small objects | weak | much better |
| Where to look | learned slowly from near-uniform attention | offsets predicted directly from the query |

**Exam verdict:** Deformable DETR fixes DETR's two weaknesses — slow convergence and small objects — via sparse multi-scale attention.

### Matching cost vs Hungarian loss

| | Matching cost (choose pairs) | Hungarian loss (train network) |
|---|---|---|
| Class term | $-\hat p(c_i)$ — **probability** | $-\log\hat p(c_i)$ — **cross-entropy** |
| $\varnothing$ predictions | cost 0 (padding) | CE towards $\varnothing$, weight 0.1 |
| Box term | $5\,L_1+2\,\mathcal L_{GIoU}$ | same, matched pairs only |
| Gradient | none | yes |
| At inference? | no | no |

**Exam verdict:** matching decides **which** query learns which object; the loss then trains it — neither is used at inference.

### Error types in detector analysis

| Error (TIDE-style, $t_f=0.5$, $t_b=0.1$) | Condition | Typical fix |
|---|---|---|
| Classification | IoU ≥ $t_f$, wrong class | more data for confused classes |
| Localisation | right class, $t_b\le$ IoU $< t_f$ | better box regression (GIoU), resolution |
| Both | wrong class and $t_b\le$ IoU $< t_f$ | inspect hard cases |
| Duplicate | right class, IoU ≥ $t_f$, GT already matched | **tune NMS**; one-to-one training |
| Background | IoU $< t_b$ with all GT | hard-negative mining; check missing labels |
| Missed | GT never detected | multi-scale, more data, raise max detections |

**Exam verdict:** diagnose before fixing — duplicates point to NMS, localisation errors to the box head, misses to scale/resolution.

### Modern detector families

| Family | Examples | Key idea | NMS? |
|---|---|---|---|
| Multi-scale CNN | Faster R-CNN + FPN, RetinaNet | feature pyramid, focal loss | yes |
| Anchor-free | FCOS, CenterNet, YOLOX | points/centres instead of anchors | yes |
| End-to-end transformer | DETR, Deformable DETR, DINO (detector) | queries + Hungarian matching | **no** |
| Real-time | YOLO family, RT-DETR | speed/accuracy trade-off | YOLO mostly yes; RT-DETR no |
| Open-vocabulary | OWL-ViT, Grounding DINO | **text-conditioned classes** | OWL-ViT optional; Grounding DINO no |

**Exam verdict:** pick the family by constraint — latency (YOLO/RT-DETR), accuracy (DINO-style), new classes without retraining (open-vocabulary).

---

## S09 Revision and practicals

> Notes: [09_revision_practicals.md](notes/09_revision_practicals.md#6-comparison-tables). Its cross-session tables are merged into: [task types](#classification-vs-localisation-vs-detection-vs-segmentation) · [two-stage vs one-stage vs DETR](#two-stage-vs-one-stage-vs-detr) · [segmentation models](#fcn-vs-u-net-vs-mask-r-cnn-vs-deeplab-vs-mask2former-vs-sam) · [H vs F vs E vs P](#homography-vs-f-vs-e-vs-p) · [depth methods](#stereo-vs-monocular-depth-vs-sfm-vs-slam) · [closed-set vs open-vocabulary](#closed-set-vs-open-vocabulary-detection) · [generative families](#ae-vs-vae-vs-gan-vs-diffusion) · [adaptation modes](#adaptation-modes).

### Hand-crafted vs learned vs foundation features

| | Hand-crafted (Harris, SIFT, ORB, HOG) | Learned CNN/ViT (supervised) | Foundation (CLIP, DINOv2) |
|---|---|---|---|
| Invariance comes from | **design** (scale space, orientation, gradient histograms) | labelled data + augmentation | web-scale captions / self-supervision |
| Training data | **none** | thousands+ labelled images (or transfer) | none for you; a probe needs a few labels |
| Matches | the **same instance** (local patches) | **categories** | categories and instances; CLIP also text |
| Geometric precision | **excellent** (sub-pixel → RANSAC, BA) | weak (coarse maps) | weak (patch grid /14–/32) |
| Robustness to big appearance change | low | medium–high | **high** |
| Interpretability | high | low | low |
| Compute | CPU, real time (ORB) | GPU for training | heavy encoders, GPU |
| Typical use | panoramas, SfM, SLAM, calibration | product/plant classifiers, detectors | zero-shot, retrieval, auto-labelling |

**Exam verdict:** geometry and instance matching → SIFT/ORB (+ RANSAC); category recognition with labels → learned features; few/no labels or text queries → foundation features.

### Supervised vs self-supervised vs language-supervised

| | Supervised | Self-supervised | Language-supervised |
|---|---|---|---|
| Examples | ImageNet ResNet/ViT | SimCLR, MoCo, BYOL, DINO/DINOv2, MAE | CLIP |
| Labels | manual class labels | **none** | image–caption pairs from the web |
| Training signal | CE on fixed classes | agreement between augmented views / masked prediction | symmetric InfoNCE image ↔ text |
| Scale (reference) | 1.28M images (ImageNet-1k) | up to 142M curated images (DINOv2) | 400M pairs |
| Zero-shot by text | no | no | **yes** |
| Typical evaluation | fine-tune / transfer | k-NN, linear probe, dense heads | zero-shot, linear probe, retrieval |
| Weakness | label cost, class-biased features | augmentation design, compute | web bias, fine-grained, counting, typographic attacks |

**Exam verdict:** labels plentiful → supervised; lots of unlabelled in-domain images → self-supervised pretraining + linear probe; recognition by text without labels → CLIP.

### Classification metrics

| Metric | Definition | Under imbalance |
|---|---|---|
| Accuracy | $\sum_c TP_c/N$ | **misleading** — dominated by the majority class |
| Balanced accuracy | mean per-class recall | chance $=1/K$; robust |
| Macro-F1 | mean of per-class F1 | rare classes count equally — **preferred** |
| Weighted-F1 | per-class F1 weighted by support | close to accuracy; hides rare classes |
| Micro-F1 | F1 of pooled TP/FP/FN | **= accuracy** for single-label multi-class |
| Top-$k$ accuracy | true class among the $k$ highest scores | confusable / fine-grained classes |

**Exam verdict:** with imbalance report macro-F1 or balanced accuracy plus the confusion matrix (rows = true class); accuracy alone loses marks.

### Retrieval metrics

| Metric | Definition | Rewards | Note |
|---|---|---|---|
| P@k | relevant in top $k$ / $k$ | clean first page | ignores how many relevant exist |
| R@k | relevant in top $k$ / $R$ | coverage | bounded by $k/R$ |
| Hit-rate@k ("Recall@K" in metric-learning papers) | 1 if any relevant in top $k$ | at least one good result | **state which "recall" you mean** |
| AP / mAP | mean precision at relevant ranks ÷ $R$ | relevant items **early** | standard single-number summary |
| MRR | mean of 1/rank of first relevant | "one right answer" tasks | ignores later hits |

**Exam verdict:** one correct answer (face ID, landmark) → MRR / Hit@k; full ranked quality → mAP; user sees one page → P@k.

### PCA vs t-SNE vs UMAP

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Type | **linear** projection | non-linear, neighbour-probability matching | non-linear, neighbour-graph layout |
| Deterministic | **yes** | no (seed) | no (seed) |
| Preserves | global variance directions | **local** neighbourhoods | local + more global structure |
| Inter-cluster distances meaningful? | yes | **no** | not reliably |
| New points | project directly | re-run | `transform` with fitted model |
| Speed | fastest | slowest | fast |

**Exam verdict:** PCA for real dimensionality reduction (e.g. to ~50-d first) and variance analysis; t-SNE/UMAP only for qualitative cluster plots — never read distances between clusters.

---

## S10 Image segmentation

> Notes: [10_image_segmentation.md](notes/10_image_segmentation.md#6-comparison-tables)

### Semantic vs instance vs panoptic

| | Semantic | Instance | Panoptic |
|---|---|---|---|
| Label per pixel | class | (class, id) for object pixels only | (class, id) for things, class for stuff |
| Separates instances? | **no** | yes | yes (things) |
| Covers stuff (sky, road)? | yes | **no** | yes |
| Overlapping masks? | no | possible | **no** |
| Metric | mIoU | mask AP | **PQ = SQ × RQ** |
| Example model | FCN, U-Net, DeepLab | Mask R-CNN | Panoptic FPN, Mask2Former |

**Exam verdict:** counting objects → instance; scene layout (road, sky) → semantic; both with a complete non-overlapping map → panoptic.

### FCN vs U-Net vs Mask R-CNN vs DeepLab vs Mask2Former vs SAM

| Aspect | FCN | U-Net | Mask R-CNN | DeepLab v3+ | Mask2Former | SAM |
|---|---|---|---|---|---|---|
| Task | semantic | semantic (medical) | **instance** | semantic | **universal** (sem / inst / pan) | **promptable, class-agnostic** |
| Core idea | conv-ised VGG + upsampling | symmetric encoder–decoder | Faster R-CNN + mask head + RoIAlign | **atrous conv + ASPP** | **queries predict (class, mask)**, masked attention | ViT-H encoder + prompt encoder + light decoder |
| Skip fusion | **addition** of score maps | **concatenation** at every level | FPN (addition) | concat one low-level map | multi-scale pixel decoder | – |
| Output | $C\times H\times W$ | $C\times H'\times W'$ (572 → 388) | boxes + 28×28 mask per RoI | $C\times H\times W$ | $N$ (class, mask) pairs | 3 masks + predicted IoU |
| Class labels | closed set | closed set | closed set | closed set | closed set | **none** |
| Strength | first end-to-end dense CNN | small data, sharp boundaries | separates instances | context, high mIoU | one model for all tasks | zero-shot to new objects/domains |
| Weakness | coarse boundaries | no instances | 28×28 masks, slower | no instances | more complex (queries + Hungarian matching) | no classes; heavy encoder |

**Exam verdict:** medical/small data → U-Net; driving scenes (semantic) → DeepLab/SegFormer; per-object masks → Mask R-CNN; one architecture for everything → Mask2Former; labelled masks for new classes → SAM + a detector (Grounded-SAM).

### Other modern segmentation models

| Model | Task | Key idea |
|---|---|---|
| DeepLab v1/v2 | semantic | atrous conv; ASPP (v2); dense CRF post-processing |
| PSPNet | semantic | pyramid pooling (1, 2, 3, 6 bins) for global context |
| SegFormer | semantic | hierarchical Transformer, **no positional encodings**, all-MLP decoder |
| Panoptic FPN | panoptic | Mask R-CNN + FPN + semantic branch, merged |

**Exam verdict:** global context is the recurring theme — ASPP, pyramid pooling, or attention.

### Skip connections addition vs concatenation

| | Addition (FCN, FPN, ResNet) | Concatenation (U-Net, DenseNet) |
|---|---|---|
| Operation | element-wise sum, shapes must match | stack along channels |
| Channels after | unchanged | $C_1+C_2$ |
| What is combined | FCN: class **scores** | full **feature maps** |
| Flexibility | fixed equal weighting | **following convs learn how to mix** |
| Cost | cheaper | more memory/compute |

**Exam verdict:** concatenation (U-Net) keeps more detail for precise boundaries; addition is cheaper.

### RoI Pool vs RoIAlign

| | RoI Pool (Fast/Faster R-CNN) | RoIAlign (Mask R-CNN) |
|---|---|---|
| RoI coordinates | **rounded** to feature cells | exact floating point |
| Bin boundaries | rounded | exact |
| Values read | integer cells in each bin | **bilinear interpolation** at sample points |
| Aggregation | max | average (or max) |
| Misalignment | up to $s/2$ px per quantisation (8 px at stride 16) | **none** |
| Effect | fine for classification | large gain for **masks/keypoints**, small for boxes |

**Exam verdict:** any pixel-accurate output (masks, keypoints) needs RoIAlign.

### IoU vs Dice vs pixel accuracy

| Metric | Formula | Good for | Pitfall |
|---|---|---|---|
| Pixel accuracy | correct / all pixels | balanced data | **dominated by background** ("all background" scores high) |
| Mean class accuracy | mean per-class recall | imbalance-aware | ignores false positives |
| IoU (Jaccard) | $TP/(TP+FP+FN)$ | per-class overlap | insensitive to boundary detail of large objects |
| mIoU | mean IoU over classes | **semantic benchmark** | rare classes noisy |
| Dice (F1) | $2TP/(2TP+FP+FN)$ | **medical, small structures** | same ranking as IoU per pair; averages can differ |
| Boundary IoU / Hausdorff | contour-based | boundary quality | Hausdorff sensitive to outliers |
| Mask AP | AP with mask IoU | instance seg. | needs scores |
| PQ = SQ × RQ | IoU > 0.5 matching | panoptic | one matching threshold |

**Exam verdict:** report mIoU (or Dice in medical work), never pixel accuracy alone; Dice ≥ IoU always, and $\text{Dice}=2\,\text{IoU}/(1+\text{IoU})$.

---

## S11 Camera geometry

> Notes: [11_camera_geometry.md](notes/11_camera_geometry.md#6-comparison-tables)

### Perspective vs weak-perspective vs orthographic

| | Perspective | Weak perspective | Orthographic |
|---|---|---|---|
| Equation | $x=fX/Z$ | $x=(f/Z_0)X$ | $x=X$ |
| Parallel lines | **converge** (vanishing points) | stay parallel | stay parallel |
| Size vs depth | $\propto1/Z$ **per point** | one scale for the whole object | no change |
| Linear in $\mathbf X$? | only in homogeneous coords | yes (affine) | yes (affine) |
| Use | general cameras | distant objects, telephoto | telecentric metrology, CAD |

**Exam verdict:** weak perspective is fine when depth variation ≪ distance (→ affine models in S02); otherwise use full perspective.

### Intrinsics vs extrinsics

| | Intrinsics | Extrinsics |
|---|---|---|
| Parameters | $f_x,f_y,s,c_x,c_y$ (+ distortion $k_1,k_2,p_1,p_2,k_3$) | $R$ (3), $\mathbf t$ (3) |
| DOF | **5** (+ distortion) | **6** |
| Describes | lens + sensor | **camera pose** in the world |
| Changes when | zoom, focus, resolution/crop | camera (or world frame) moves |
| Estimated | once per camera setting (calibration) | per image (calibration views, PnP, SLAM) |

**Exam verdict:** $P=K[R\mid\mathbf t]$ has 5 + 6 = **11 DOF**; resizing the image changes $K$, moving the camera changes $[R\mid\mathbf t]$.

### Radial vs tangential distortion

| | Radial | Tangential |
|---|---|---|
| Cause | lens curvature; magnification varies with $r$ | lens **decentred/tilted** relative to sensor |
| Coefficients | $k_1,k_2,k_3$ | $p_1,p_2$ |
| Symmetry | **symmetric** about principal point | asymmetric |
| Visual effect | barrel ($k_1 < 0$) / pincushion ($k_1 > 0$) | slight skewed warping |
| Size | usually **dominant** | usually small |

**Exam verdict:** straight lines bowing outward/inward → radial; OpenCV order is $(k_1,k_2,p_1,p_2,k_3)$.

### Calibration methods DLT vs Zhang vs PnP

| | DLT (3D rig) | Zhang (planar board) | PnP |
|---|---|---|---|
| Target | known **non-coplanar** 3D points | **planar checkerboard**, several orientations | known 3D points |
| Unknowns | full $P$ (11 DOF) | $K$ (5) + distortion + 6 per view | **pose only** (6), $K$ known |
| Minimum data | 6 points | 3 views (2 if $s=0$), ≥ 4 corners each | 3 points (P3P) + 1 to disambiguate |
| Linear step | SVD of $2n\times12$ | per-view $H$, then SVD for $B$ | P3P / EPnP |
| Practicality | rig hard to build | **cheap, accurate → default** | AR, localisation, SfM registration |

**Exam verdict:** calibrate intrinsics with Zhang; once $K$ is known, get per-image pose with PnP.

### Homography cases

| Situation | $H$ | Exact? |
|---|---|---|
| Plane $Z=0$ → image | $K[\mathbf r_1\ \mathbf r_2\ \mathbf t]$ | yes |
| Two views of a plane | $K'(R-\mathbf t\mathbf n^\top/d)K^{-1}$ (plane $\mathbf n^\top\mathbf X+d=0$) | yes |
| Pure rotation, any scene | $K'RK^{-1}$ | yes |
| Translation + general 3D scene | – | **no** (parallax → epipolar geometry) |
| Distant scene, small baseline | ≈ any of the above | approximate |

**Exam verdict:** homography ⇔ plane or rotation; with parallax switch to $F$/$E$ (S12). Sign convention: with the plane written $\mathbf n^\top\mathbf X=d$ (notes 02) the middle term becomes $R+\mathbf t\mathbf n^\top/d$.

---

## S12 3D vision and motion

> Notes: [12_3d_vision_motion.md](notes/12_3d_vision_motion.md#6-comparison-tables)

### Homography vs F vs E vs P

| | Affine $A$ | Homography $H$ | Fundamental $F$ | Essential $E$ | Camera $P$ |
|---|---|---|---|---|---|
| Maps | 2D point → 2D point | 2D point → **2D point** | point → **epipolar line** | normalised point → line | **3D point → pixel** |
| Size / constraint | 2×3 (last row 0 0 1) | 3×3, up to scale | 3×3, rank 2 | 3×3, rank 2, singular values $(\sigma,\sigma,0)$ | 3×4, up to scale |
| DOF | 6 | **8** | **7** | **5** | **11** |
| Equation | $\mathbf x'=A\tilde{\mathbf x}$ | $\tilde{\mathbf x}'\sim H\tilde{\mathbf x}$ | $\mathbf x'^\top F\mathbf x=0$ | $\hat{\mathbf x}'^\top E\hat{\mathbf x}=0$ | $\lambda\tilde{\mathbf x}=P\tilde{\mathbf X}$ |
| Needs $K$? | no | no | **no** (pixels) | **yes** | contains $K$ |
| Minimal data | 3 points | 4 (no 3 collinear) | 7 (8 linear) | 5 | 6 (3D–2D, DLT); 3 + 1 if $K$ known (P3P) |
| RANSAC $N$ ($e=0.5$, $p=0.99$) | 35 | 72 | 588 / 1177 | 146 | – |
| Valid when | weak perspective, far planar patch | **plane or pure rotation** | any rigid scene, uncalibrated | any rigid scene, calibrated | calibration / pose |
| Gives | warp | warp, panorama, rectification; with $K$: $R$, $\mathbf t/d$, $\mathbf n$ (≤ 4 solutions) | epipolar lines; pose only up to projective ambiguity | **$R$, $\mathbf t$ up to scale** (4 candidates + cheirality) | projection, pose, calibration |

**Exam verdict:** panorama/document (plane or rotation) → $H$; general two-view geometry → $E$ if calibrated (then recover pose), else $F$; known 3D points → $P$ (or PnP with known $K$).

### Stereo matching methods

| | Block matching (SAD/SSD/NCC) | SGM | Learned (cost volume, RAFT-Stereo) |
|---|---|---|---|
| Optimisation | local, winner-takes-all | semi-global (1D DP along 8–16 paths) | learned aggregation / iterative refinement |
| Textureless regions | poor | better (smoothness $P_1,P_2$) | **best** |
| Depth edges | fattening | good | good |
| Speed | very fast (CPU) | real time (CPU/FPGA) | GPU needed |
| Data needed | none | none | training data, domain gap |

**Exam verdict:** SGM is the classical default; learned stereo wins on hard regions if a GPU and in-domain training data are available.

### Larger vs smaller baseline

| Larger baseline $B$ | Smaller baseline $B$ |
|---|---|
| ✅ better depth resolution ($\Delta Z\propto1/B$), longer range | ✅ easier matching, fewer occlusions |
| ❌ more occlusion, harder matching, larger $Z_{min}$ | ❌ poor depth precision far away |

**Exam verdict:** choose $B$ from the required $\Delta Z$ at the working range: $B=Z^2\Delta d/(f\Delta Z)$.

### LK vs Horn-Schunck vs RAFT

| | Lucas–Kanade | Horn–Schunck | RAFT (deep) |
|---|---|---|---|
| Output | **sparse** (features) | **dense** | dense |
| Constraint | constant flow in a window (local) | **global smoothness** | learned (all-pairs correlation + GRU) |
| Aperture problem | fails on edges/flat (needs corners) | filled in by smoothness | learned context |
| Large motion | needs a pyramid | needs a pyramid | **all-pairs correlation** |
| Motion boundaries | OK locally | oversmoothed | **sharp** |
| Cost | very cheap | moderate (iterative) | GPU |

**Exam verdict:** track a few corners cheaply → (pyramidal) LK; dense classical flow → Horn–Schunck; best accuracy → RAFT.

### KLT vs SORT vs DeepSORT

| | KLT | SORT | DeepSORT |
|---|---|---|---|
| Tracks | feature points | object boxes | object boxes |
| Motion model | LK flow | Kalman, constant velocity | Kalman, constant velocity |
| Association | implicit (local search) | **Hungarian on IoU** | cascade: **appearance (cosine)** + Mahalanobis gate |
| Occlusion | track lost | ID switches | **re-identifies** after occlusion |
| Speed | very fast | very fast | slower (re-ID CNN) |

**Exam verdict:** points → KLT; multi-object tracking-by-detection → SORT, or DeepSORT when occlusions cause ID switches.

### SfM vs SLAM vs VO

| | SfM | SLAM | Visual odometry |
|---|---|---|---|
| Input | unordered image set (or video) | sequential live stream | sequential live stream |
| Timing | **offline**, batch | **online, real time** | online, real time |
| Output | poses + 3D model (sparse → MVS) | trajectory + **consistent map** | trajectory only (local) |
| Optimisation | full BA (incremental/global) | local BA + pose graph | windowed / local |
| Drift handling | global BA | **loop closure** | **none** — drift accumulates |
| Typical use | 3D reconstruction, photogrammetry, NeRF/3DGS poses | robots, AR/VR, drones | odometry front end |

**Exam verdict:** offline reconstruction → SfM; real-time pose + map that must stay consistent → SLAM; VO is SLAM without loop closure.

### Incremental vs global SfM

| | Incremental (COLMAP) | Global |
|---|---|---|
| Strategy | add images one by one, BA repeatedly | rotation averaging → translation averaging → one BA |
| Robustness | **high** | sensitive to bad pairs |
| Speed | slower | **faster** |
| Drift | possible on long chains | error spread evenly |

**Exam verdict:** incremental for robustness (default), global for speed on clean data.

### SLAM sensor setups

| Sensor | Scale | Notes |
|---|---|---|
| Monocular | **unobservable** (up to scale, drifts) | cheapest; needs translation to initialise; pure rotation degenerate |
| Stereo | metric (baseline) | range limited by $fB$ |
| RGB-D | metric (depth sensor) | dense, mainly indoors, struggles in sunlight |
| Visual-inertial | metric (accelerometer) + gravity | robust to fast motion; needs IMU calibration and sync |

**Exam verdict:** need metric scale with one camera → add an IMU (VIO) or use stereo/RGB-D.

### Stereo vs monocular depth vs SfM vs SLAM

| | Stereo | Monocular foundation (Depth Anything) | SfM | SLAM |
|---|---|---|---|---|
| Input | 2 calibrated, synced cameras | **1 image** | many images | live stream (+ IMU/depth) |
| Principle | triangulation $Z=fB/d$ (block matching / SGM) | **learned priors** (DINOv2 + DPT) | features, $E$/$F$, triangulation, PnP, bundle adjustment | tracking/VO, local BA, loop closure + pose graph |
| Output | dense depth per frame | dense **relative inverse** depth | poses + sparse 3D (then MVS) | trajectory + map |
| Scale | **metric** (baseline) | relative (needs alignment/fine-tune) | up to scale (mono) | mono: up to scale; stereo/RGB-D/VI: metric |
| Timing | real time | one forward pass per image (GPU) | **offline** | **online** |
| Fails on | textureless, occlusion, far range ($\Delta Z\propto Z^2$) | out-of-distribution, reflective/transparent, scale | low overlap, repetitive texture | fast motion, low texture, pure-rotation init |

**Exam verdict:** metric depth with two cameras → stereo; one camera, dense relative depth → Depth Anything (+ scale-shift alignment for metric); offline 3D model → SfM; real-time localisation and map → SLAM.

---

## S13 Vision foundation models

> Notes: [13_vision_foundation_models.md](notes/13_vision_foundation_models.md#6-comparison-tables)

### CLIP vs DINOv2 vs SAM vs Depth Anything vs OWL-ViT vs Grounding DINO

| Model | Supervision / data | Input | Output | Typical use | Key limitation |
|---|---|---|---|---|---|
| **CLIP** | 400M web image–text pairs; symmetric InfoNCE, learned $\tau$ | image and/or text | **global embeddings** in a shared space | **zero-shot classification**, retrieval, text encoder for diffusion | fine-grained, counting, spatial relations; no localisation |
| **DINO / DINOv2** | self-distillation; **no labels, no text** (LVD-142M for v2) | image | [CLS] + **dense patch features** | frozen backbone: k-NN, linear probe, seg/depth heads | no text → no prompt-based zero-shot |
| **SAM** | SA-1B: 11M images, 1.1B masks | image + points/box/mask | **3 masks + predicted IoU** | interactive / automatic segmentation | **no class labels**; heavy encoder |
| **Depth Anything** | ~1.5M labelled + ~62M pseudo-labelled; DINOv2 + DPT | 1 RGB image | **relative** (affine-invariant inverse) depth | robotics, AR, depth conditioning | no metric scale without alignment / fine-tune |
| **OWL-ViT** | CLIP-style pre-training + detection fine-tune (bipartite matching) | image + text or image queries | **boxes** + per-query scores | open-vocabulary / one-shot detection | box quality < closed-set on common classes; no masks |
| **Grounding DINO** | detection + grounding + caption data; Swin + BERT, deep fusion | image + text phrase | **boxes** + per-phrase scores | text-grounded detection; + SAM = Grounded-SAM | slower; no masks |

**Exam verdict:** classify by text → CLIP; best frozen visual features → DINOv2; masks from prompts → SAM; depth from one image → Depth Anything; boxes from text → OWL-ViT (fast) or Grounding DINO (better phrase grounding).

### Adaptation modes

| Mode | Labels | Trainable params (ResNet-50, 50 classes, where given) | Use when | Example |
|---|---|---|---|---|
| Zero-shot / prompting | **none** | 0 | new classes on the fly, no data | CLIP prompts, SAM clicks, OWL-ViT queries |
| k-NN on frozen features | some | 0 (stores $N\times D$ embeddings) | fast baseline, class list changes often | DINOv2 k-NN |
| Linear probe | some–many | $(2048+1)\cdot50=102{,}450$ | standard frozen-feature evaluation | CLIP / DINOv2 + logistic regression |
| PEFT (adapters, LoRA) | moderate | ~1–5 % of the model | limited GPU, many tasks share a backbone | LoRA on ViT |
| Partial fine-tune (`layer4` + head) | many | 15,067,186 | domain shift, more data | ResNet-50 practical (S09) |
| Full fine-tune | **many** | 23,610,482 | large or very different dataset, max accuracy | Depth Anything → metric (NYUv2) |

**Exam verdict:** climb the ladder only as far as labels and domain shift require — zero-shot first, then k-NN/linear probe (zero-shot CLIP can beat 1–4-shot probes), then PEFT or fine-tuning.

### CLIP vs DINOv2 features

| | CLIP | DINOv2 |
|---|---|---|
| Supervision | language (captions) | none (self-distillation) |
| Zero-shot by text | **yes** | no |
| Dense / patch features | weaker (global matching) | **strong** (iBOT patch objective) |
| Fine-grained visual similarity | moderate | **strong** |
| Typical role | classifier from prompts, text retrieval, text encoder | frozen backbone for dense heads, retrieval |

**Exam verdict:** text in the loop → CLIP; pure visual similarity or dense prediction → DINOv2.

### SAM vs Mask R-CNN

| | Mask R-CNN | SAM |
|---|---|---|
| Classes | fixed closed set (80 COCO) | **none** (class-agnostic) |
| Input | image only | image + **prompt** |
| Training data | COCO (~118k images) | SA-1B (11M images, 1.1B masks) |
| Output | boxes + class + mask per instance | 3 masks + IoU per prompt |
| Generalisation | in-distribution | **zero-shot** to new objects/domains |
| Speed | one pass per image | heavy encoder once + light decoder per prompt |

**Exam verdict:** known classes with labels and speed → Mask R-CNN; arbitrary objects or interactive annotation → SAM (add a detector for class names).

---

## S14 Vision++ with foundation models

> Notes: [14_vision_plus_plus.md](notes/14_vision_plus_plus.md#6-comparison-tables)

### Self-supervised DINO vs Grounding DINO's DINO

| | DINO / DINOv2 (Session 13) | "DINO" inside Grounding DINO |
|---|---|---|
| Name | self-**di**stillation with **no** labels | **D**ETR with **I**mproved de**N**oising anch**O**r boxes |
| What it is | **self-supervised representation learning** (ViT backbone) | a **DETR-family object detector** |
| Training signal | student matches EMA teacher (centering + sharpening), no labels | box + class supervision (+ text grounding in Grounding DINO) |
| Output | features ([CLS] + patches) | boxes + class/phrase scores |

**Exam verdict:** same acronym, unrelated methods — Grounding DINO extends the **detector** DINO with text fusion; it is not built on self-supervised DINO.

### OWL-ViT vs Grounding DINO

| | OWL-ViT / OWLv2 | Grounding DINO |
|---|---|---|
| Base | CLIP-style ViT dual encoder, box + class embedding per image token | DINO detector (DETR family) + Swin image + BERT text backbones |
| Image–text fusion | **late** — dot product of token embeddings with query embeddings | **early and deep** — feature enhancer, language-guided query selection, cross-modality decoder |
| Predictions | one box per patch token (576 for B/32 at 768²) | 900 decoder queries |
| Query types | text **or example image** (one-shot) | text: category list or referring expression |
| Text encoding | independent of the image → cache once, huge vocabularies cheap | re-fused with every image; ≤ 256 tokens per pass |
| Thresholds | one score threshold per query | `box_threshold` + `text_threshold` |
| Strength | simple, faster, image-conditioned search | **higher zero-shot accuracy** (≈ 52.5 COCO AP zero-shot), phrase grounding, referring expressions |
| Weakness | weaker localisation; scores not comparable across queries | slower, label merging, prompt-length limit |

**Exam verdict:** large fixed vocabularies or example-image search → OWL-ViT; complex phrases ("the dog on the left") and best zero-shot boxes → Grounding DINO (first stage of Grounded-SAM).

### Closed-set vs open-vocabulary detection

| | Closed-set (YOLO, Faster R-CNN) | Open-vocabulary (OWL-ViT, Grounding DINO, YOLO-World) |
|---|---|---|
| Vocabulary | **fixed at training** | any text (or example image) at inference |
| Classifier weights | learned per class | **text embeddings** of the queries |
| New class needs | hundreds of labelled boxes + retraining | **a prompt** (zero-shot) or one example |
| Accuracy on trained classes | **highest** (in-domain) | lower, but non-zero on unseen classes |
| Speed | **real time** on edge | usually 100s of ms on GPU (YOLO-World is the real-time exception) |
| Calibration | one global threshold usually fine | per-prompt thresholds; scores depend on wording |
| Best use | production with fixed classes | prototyping, long-tail search, **auto-labelling** to train the closed-set model |

**Exam verdict:** fixed classes + latency limits → closed-set; long-tail or changing classes → open-vocabulary, then distil into a closed-set model for deployment.

### Zero-shot vs few-shot vs fine-tuned

| | Zero-shot | Few-shot | Fine-tuned |
|---|---|---|---|
| Labels | **none** | 1–10 per class | hundreds–thousands |
| Time to first result | minutes | hours | days–weeks |
| In-domain accuracy | lowest | medium | **highest** |
| Domain-shift handling | poor for niche domains | better | best |
| Inference compute | large foundation model | usually same backbone | **can be small and fast** |
| Maintenance | edit prompts | add examples | re-label and retrain |
| Typical use | prototype, rare/changing classes, auto-labelling | custom classes quickly | fixed classes, production, edge |

**Exam verdict:** prototype zero-shot → few-shot for custom classes → fine-tune (or distil) for production and edge. "Zero-shot" means no labelled target examples, not that the concept was never seen in pre-training.

### Three ways to get instance masks for a class

| | Mask R-CNN (closed-set) | Grounded-SAM (text → boxes → masks) | SAM everything + CLIP (masks → labels) |
|---|---|---|---|
| Labels needed | many masks per class | **none** | none |
| Vocabulary | fixed | any text prompt | any text list |
| Mask quality | good (28×28 head, upsampled) | **high** (SAM boundaries) | high, but part/whole duplicates |
| Recall limited by | training data | Grounding DINO detections | SAM grid + CLIP accuracy on small crops |
| Speed | **fast** (one pass) | slow (detector + SAM) | slowest (hundreds of masks + CLIP crops) |
| Best for | production, fixed classes | prompted search, auto-labelling | exploring "what is in this image" |

**Exam verdict:** known classes and speed → Mask R-CNN; "find and segment X" with no labels → Grounded-SAM; open-ended discovery → SAM everything + CLIP.

### Ways to get metric distance

| Method | Needs | Pros | Cons |
|---|---|---|---|
| Metric fine-tuned depth model | model for the right domain (indoor/outdoor) | no extra input | domain-specific; error grows with range |
| Known-size reference ($Z=fH/h_{px}$) | intrinsics $f$ + a known object | cheap, single image | one reference → scale only ($t=0$ assumed) |
| Sparse LiDAR / stereo points | extra sensor or second camera | **accurate**, robust least-squares fit of $s,t$ | hardware cost, calibration |
| Camera height + ground plane | calibrated mounting | works for objects on the ground | flat-ground assumption |

**Exam verdict:** relative depth needs at least one metric cue; ≥ 2 references (or LiDAR/stereo) to fit both scale and shift of $1/Z=s\,r+t$.

### Pipeline stages and failure modes

| Stage | Typical failure | Effect downstream | Mitigation |
|---|---|---|---|
| Grounding DINO | missed object / FP / merged label | **nothing downstream can recover a miss** | prompt engineering, thresholds, synonyms, CLIP check |
| SAM | mask leaks into background or covers a part | wrong depth sample, wrong size | box + point prompts, predicted-IoU filter, erosion |
| Depth Anything | reflective/transparent surfaces, far range, flicker | distance error | median over mask, temporal smoothing, metric model |
| Scale calibration | wrong or noisy reference | **all** distances scaled wrongly | ≥ 2 references, robust fit, size sanity check |
| Tracker | ID switches, drift | wrong per-object history | periodic re-detection, mask-IoU association |

**Exam verdict:** end-to-end success ≈ product of stage success rates — evaluate every stage and the whole pipeline; composition costs latency (~1–5 FPS on a server GPU), so distil for real-time use.

---

## S15 Generative CV

> Notes: [15_generative_cv.md](notes/15_generative_cv.md#6-comparison-tables)

### AE vs VAE vs GAN vs diffusion

| Aspect | Autoencoder | VAE | GAN | Diffusion (DDPM / LDM) |
|---|---|---|---|---|
| Objective | reconstruction (MSE/BCE) | **ELBO** = reconstruction + KL | **adversarial minimax** (≈ JSD) or Wasserstein | **denoising**: predict $\epsilon$ (MSE, weighted ELBO) |
| Latent | deterministic, unstructured, holes | low-dim Gaussian, smooth | low-dim $z$; no encoder | same size as image (or VAE latent in LDM) |
| Sampling | not a sampler | fast (1 pass) | **fast (1 pass)** | **slow** (20–1000 passes, ×2 with CFG) |
| Sample quality | – | **blurry** | sharp | **state of the art** |
| Mode coverage | – | good | **weak (mode collapse)** | good |
| Training stability | very stable | stable | **unstable** | stable (regression loss) |
| Likelihood | none | lower bound | none (implicit) | lower bound |
| Typical use | compression, denoising, anomaly detection | representation, LDM latent space | real-time generation, translation, faces | text-to-image, editing, inpainting, video |

**Exam verdict:** need speed → GAN; need likelihood/smooth latent → VAE; need quality + diversity + controllability → diffusion (latent diffusion to make it affordable).

### Generative learning trilemma

| | High quality | Diversity / coverage | Fast sampling |
|---|---|---|---|
| GAN | ✅ | ❌ (mode collapse) | ✅ |
| VAE | ❌ (blurry) | ✅ | ✅ |
| Diffusion | ✅ | ✅ | ❌ (mitigated by DDIM, solvers, distillation, latents) |

**Exam verdict:** each family gives up one corner — name which one in any "compare generative models" answer.

### DDPM vs DDIM

| | DDPM sampler | DDIM sampler |
|---|---|---|
| Process | Markovian, **stochastic** ($\sigma_t > 0$) | non-Markovian, **deterministic** for $\eta=0$ |
| Steps | all $T$ (1000) | **sub-sequence** (e.g. 50) |
| Training | ε-prediction loss | **same model, no retraining** |
| Same seed → same image? | no | **yes**; supports inversion and interpolation |

**Exam verdict:** DDIM is a faster, deterministic sampler for the same trained model — use it for speed, inversion and editing.

### Classifier guidance vs CFG

| | Classifier guidance | Classifier-free guidance |
|---|---|---|
| Extra model | separate **noise-aware classifier** $p_\phi(y\mid x_t)$ | **none** |
| Training | diffusion model + classifier on noisy images | one model; drop the condition ~10 % of the time |
| Sampling | $\epsilon_\theta-s\sqrt{1-\bar\alpha_t}\nabla\log p_\phi(y\mid x_t)$ | $\epsilon_\varnothing+w(\epsilon_c-\epsilon_\varnothing)$ |
| Cost per step | 1 U-Net + classifier forward/backward | **2 U-Net passes** (batched) |
| Conditions | class labels the classifier knows | **anything** (text, images) |
| Used in | ADM ("Diffusion models beat GANs") | Stable Diffusion, Imagen |

**Exam verdict:** CFG is the default for text-to-image; higher $w$ = more prompt adherence, less diversity.

### Pixel vs latent diffusion

| | Pixel diffusion (DDPM, Imagen base) | Latent diffusion (Stable Diffusion) |
|---|---|---|
| Denoised tensor | 512×512×3 pixels | **64×64×4 latents** ($f=8$, 48× fewer values) |
| Extra component | – (often cascaded super-resolution) | pretrained **VAE** encoder/decoder |
| Cost | very high | **much lower** — consumer GPUs |
| Weakness | compute | fine detail capped by the autoencoder |

**Exam verdict:** latent diffusion trades a little fine detail for ~48× less work per step — why Stable Diffusion runs on consumer GPUs.

### pix2pix vs CycleGAN

| | pix2pix | CycleGAN |
|---|---|---|
| Data | **paired**, pixel-aligned $(x,y)$ | **unpaired** domains $X$, $Y$ |
| Networks | 1 U-Net generator + 1 PatchGAN $D$ | **2 generators** $G,F$ + 2 discriminators |
| Key loss | cGAN + $\lambda L_1$ to ground truth ($\lambda=100$) | GAN × 2 + **cycle-consistency** $L_1$ ($\lambda=10$) |
| Typical tasks | labels → photo, edges → photo, map ↔ aerial | horse ↔ zebra, summer ↔ winter, photo ↔ painting |
| Limitation | needs aligned pairs | weak at geometric change |

**Exam verdict:** paired data → pix2pix; only unpaired collections → CycleGAN.

### Editing and personalisation methods

| Method | Input | What is trained | Typical use |
|---|---|---|---|
| img2img / SDEdit | image + prompt + strength | **nothing** (sampling trick) | restyle, sketch → image |
| Inpainting (SD-inpainting) | image + mask + prompt | U-Net fine-tuned with 9-channel input | remove/replace objects |
| RePaint | image + mask | nothing (resampling trick) | inpainting with any diffusion model |
| ControlNet | edge/depth/pose/seg map + prompt | trainable encoder copy + zero convs (base frozen) | **precise spatial layout** |
| Textual inversion | 3–5 subject images | **one new token embedding** | lightweight concept/style |
| DreamBooth | 3–5 images + class-prior images | whole U-Net (± text encoder) | high-fidelity subject |
| LoRA | subject/style images | **low-rank $BA$ adapters** | small, swappable fine-tunes |

**Exam verdict:** layout/pose control → ControlNet; new subject → DreamBooth or LoRA; quick edits without training → img2img / inpainting.

### FID vs IS

| | FID | IS |
|---|---|---|
| Uses real images? | **yes** (compares feature statistics) | no |
| Better is | **lower** (0 = match) | **higher** (max = number of classes) |
| Detects mode collapse? | **yes** (covariance term) | poorly |
| Features | Inception-v3 2048-d pool | Inception-v3 class posterior |

**Exam verdict:** report FID (it compares against real data and sees diversity); IS is a weaker, real-data-free sanity check.
