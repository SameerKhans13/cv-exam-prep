# Formula Sheet — Computer Vision (S01–S15)

> **Conventions:** boxes $(x_1,y_1,x_2,y_2)$, continuous coords, area $=(x_2-x_1)(y_2-y_1)$, **no +1** · **AP = all-point interpolation** unless stated · RANSAC $N$: log base irrelevant (it is a ratio), always round **up** · homogeneous $\tilde{\mathbf x}=(x,y,1)^\top$, $\sim$ = equal up to scale · confusion matrices: **rows = GT, columns = prediction** · $x$ = column, $y$ = row (down) · $\log$ = natural unless stated.
> Companion pages: [Comparison tables](00_comparison_tables.md) · [Numericals drill](practice/numericals_drill.md) · S09 (revision) adds no new formulas — its high-yield list is in [notes 09 §3](notes/09_revision_practicals.md#3-key-formulas).

## [S01 · Image processing & features](notes/01_image_processing_features.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $\text{bytes}=W\,H\,C\,b/8$ | $b$ bits/channel, $2^b$ levels; 1920×1080×3 at 8 bit = 6.22 MB |
| $(K\otimes I)(x,y)=\sum_{u,v}K(u,v)\,I(x+u,y+v)$ | **correlation** (what CNN layers compute) |
| $(K\ast I)(x,y)=\sum_{u,v}K(u,v)\,I(x-u,y-v)$ | **convolution** = correlation with kernel rotated 180°; commutative, associative |
| $W_{out}=\lfloor (W-K+2P)/S\rfloor+1$ | $K$ kernel, $P$ padding per side, $S$ stride |
| $G_\sigma(x,y)=\frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/2\sigma^2}=g_\sigma(x)\,g_\sigma(y)$; $\ G_{\sigma_1}\ast G_{\sigma_2}=G_{\sqrt{\sigma_1^2+\sigma_2^2}}$ | separable: $2K$ vs $K^2$ mults/px; size $K=2\lceil 3\sigma\rceil+1$; cascade (σ 3 then 4 ≡ σ 5) |
| $BF(\mathbf p)=\frac{1}{W_{\mathbf p}}\sum_{\mathbf q}G_{\sigma_s}(\Vert\mathbf p-\mathbf q\Vert)\,G_{\sigma_r}(\lvert I_{\mathbf p}-I_{\mathbf q}\rvert)\,I_{\mathbf q}$ | bilateral: spatial × range weight → edge-preserving, non-linear |
| $K_x=[1,2,1]^\top[-1,0,1]$, $\ K_y=K_x^\top$ | Sobel = smoothing × difference, separable ($y$ down) |
| $\Vert\nabla I\Vert=\sqrt{I_x^2+I_y^2}$, $\ \theta=\operatorname{atan2}(I_y,I_x)$ | magnitude, direction; edge ⟂ gradient; `cv2.Canny` default L1 $\lvert I_x\rvert+\lvert I_y\rvert$ |
| $\nabla^2I=I_{xx}+I_{yy}$; $\ G_{k\sigma}-G_\sigma\approx(k-1)\,\sigma^2\nabla^2G_\sigma$ | Laplacian; DoG ≈ scale-normalised LoG; blob of radius $r$ peaks at $\sigma=r/\sqrt2$ |
| $T_{high}:T_{low}\approx 2{:}1$ to $3{:}1$ | Canny: Gaussian → Sobel → NMS → double threshold → hysteresis |
| $M=\sum_{\mathcal W}w\begin{bmatrix}I_x^2 & I_xI_y\\ I_xI_y & I_y^2\end{bmatrix}$ | Harris structure tensor; window $\mathcal W$, weights $w$ |
| $R=\det M-k(\operatorname{tr}M)^2=\lambda_1\lambda_2-k(\lambda_1+\lambda_2)^2$ | $k\in[0.04,0.06]$; **corner** both λ large, $R\gg0$; **edge** one large, $R < 0$; **flat** both small, $\lvert R\rvert$ small |
| $R_{ST}=\min(\lambda_1,\lambda_2)$; Harris: $I\to aI\Rightarrow R\to a^4R$ | Shi–Tomasi (KLT); Harris invariant to rotation & additive shift, partial to gain, **not to scale** |
| ≥ 9 contiguous of 16 circle px all $> I_p+t$ or all $< I_p-t$ | FAST-9 (Bresenham circle r = 3); no orientation, no scale |
| $\sum_{\text{rect}}I=ii(x_1,y_1)-ii(x_0-1,y_1)-ii(x_1,y_0-1)+ii(x_0-1,y_0-1)$ | integral image $ii(x,y)=\sum_{x'\le x,\,y'\le y}I$; 4 lookups for any box size |
| $k=2^{1/s}$, $s=3$, $\sigma_0=1.6$; $s+3$ Gaussians, $s+2$ DoGs per octave | SIFT scale space: σ = 1.6, 2.016, 2.540, 3.2, 4.032, 5.080 |
| $D=L(x,y,k\sigma)-L(x,y,\sigma)$; reject $\lvert D(\hat{\mathbf x})\rvert < 0.03$ | DoG; sub-pixel $\hat{\mathbf x}=-(\partial^2D/\partial\mathbf x^2)^{-1}\partial D/\partial\mathbf x$; image in [0, 1] |
| $\operatorname{tr}(H)^2/\det H < (r+1)^2/r$, $\ r=10\Rightarrow 12.1$ | SIFT edge rejection; $H$ = 2×2 Hessian of $D$ |
| 36 bins (10°), extra keypoint ≥ 80 % of peak; $4\times4\times8=\mathbf{128}$-d | SIFT orientation; 16×16 window; normalise → clip 0.2 → renormalise |
| $\det H_{approx}=D_{xx}D_{yy}-(0.9D_{xy})^2$; $\ 4\times4\times4=\mathbf{64}$-d | SURF box-filter Hessian; descriptor $(\sum d_x,\sum d_y,\sum\lvert d_x\rvert,\sum\lvert d_y\rvert)$ per subregion |
| $m_{pq}=\sum x^py^qI(x,y)$, $\ \theta=\operatorname{atan2}(m_{01},m_{10})$ | ORB intensity-centroid orientation; rBRIEF 256 tests = 256 bits = 32 B |
| $d_H=\operatorname{popcount}(\mathbf a\oplus\mathbf b)$ | Hamming distance (binary descriptors ORB/BRIEF) |

## [S02 · Feature matching & classical CV](notes/02_feature_matching_classical.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $d_{L1}=\sum_i\lvert a_i-b_i\rvert$, $\ d_{L2}=\sqrt{\sum_i(a_i-b_i)^2}$ | L2 for SIFT/SURF |
| $d_{\cos}=1-\frac{\mathbf a\cdot\mathbf b}{\Vert\mathbf a\Vert\Vert\mathbf b\Vert}$; unit vectors: $\Vert\mathbf a-\mathbf b\Vert^2=2-2\cos\theta$ | cosine distance (deep embeddings) |
| $\chi^2=\frac12\sum_i\frac{(h_i-g_i)^2}{h_i+g_i}$ | histograms (colour, LBP, BoW) |
| accept if $d_1/d_2 < \tau$, $\ \tau\approx0.7$–$0.8$ | Lowe ratio test ($d_1,d_2$ = 1st/2nd-NN distance); Lowe 0.8, OpenCV 0.75 |
| $N=\left\lceil\frac{\log(1-p)}{\log(1-(1-e)^s)}\right\rceil$ | RANSAC; $p$ success prob., $e$ outlier ratio, $s$ minimal sample size |
| $s$: line/similarity 2, affine 3, $H$ 4, $E$ 5, $F$ 7 (8-pt: 8) | $N$ at $p=0.99,\ e=0.5$: 17, 35, **72**, 146, 588 (1177) |
| $t^2=\chi^2_{m,0.95}\,\sigma^2$: $3.84\sigma^2$ ($m=1$), $5.99\sigma^2$ ($m=2$) | inlier threshold; $m$ = codimension (epipolar 1, homography 2) |
| $(u,v,w)^\top=H(x,y,1)^\top$, $\ x'=u/w,\ y'=v/w$ | apply homography — **divide by $w$**; $H$ and $\lambda H$ same map |
| DOF / min pts: translation 2/1, rigid 3/2, similarity 4/2, affine 6/3, homography **8/4** | min pts $=\lceil\text{DOF}/2\rceil$; $H$: no 3 collinear |
| $\mathbf a_x=(x,y,1,0,0,0,-x'x,-x'y,-x')$, $\ \mathbf a_y=(0,0,0,x,y,1,-y'x,-y'y,-y')$ | DLT: 2 rows/match → $A$ is $2n\times9$; $A\mathbf h=\mathbf 0$, $\Vert\mathbf h\Vert=1$ → right singular vector of smallest singular value |
| centroid → 0, mean distance → $\sqrt2$; $\ H=T'^{-1}\tilde HT$ | Hartley normalisation (conditioning) |
| $r_i=\Vert\mathbf x_i'-\pi(H\tilde{\mathbf x}_i)\Vert$ | transfer error; $\pi$ = divide by 3rd coordinate |
| $\theta=\arctan(x/f)$, $\ h=y/\sqrt{x^2+f^2}$; $\ x'=f\theta,\ y'=fh$ | cylindrical warp ($x,y$ relative to principal point, $f$ px) |
| $I=\sum_iw_iI_i/\sum_iw_i$; $\ L_k^{bl}=G_k(M)L_k(A)+(1-G_k(M))L_k(B)$ | feathering; multi-band (Laplacian-pyramid) blending |

## [S03 · CNNs under the hood](notes/03_cnns_under_the_hood.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $W_{out}=\lfloor (W-K+2P)/S\rfloor+1$ | conv & pooling (use $K_{eff}$ if dilated); "same" at $S=1$: $P=(K-1)/2$ |
| $K_{eff}=D(K-1)+1$ | dilation $D$ (3×3, $D=2$ → 5); params unchanged |
| $\text{Params}=(K_hK_wC_{in}+1)\,C_{out}$; FC: $(N_{in}+1)\,N_{out}$ | with bias; AlexNet conv1 $(11^2\cdot3+1)\cdot96=34{,}944$; FC grows with image size |
| $\text{MACs}=H_{out}W_{out}K_hK_wC_{in}C_{out}$ | FLOPs ≈ 2 × MACs |
| $K^2C_{in}+C_{in}C_{out}$; ratio $\frac{1}{C_{out}}+\frac{1}{K^2}$ | depthwise-separable params (+ biases) |
| $W_{out}=(W-1)S-2P+K$ (+ output_padding) | transposed conv (learned upsampling) |
| $r_0=1,\ j_0=1$; $\ r_l=r_{l-1}+(K_l-1)\,j_{l-1}$, $\ j_l=j_{l-1}S_l$ | receptive field; $j$ = jump; pooling counts as a layer; $n$ stride-1 $K\times K$ convs: $n(K-1)+1$ |
| two 3×3: $18C^2$ vs one 5×5: $25C^2$; three 3×3: $27C^2$ vs 7×7: $49C^2$ | same RF, fewer weights, more ReLUs (VGG) |
| $y=F(x)+x$; $\ \partial y/\partial x=I+\partial F/\partial x$ | residual; projection $W_sx$ (1×1 conv) if shapes differ; fixes **degradation** |
| $N=(H/p)(W/p)$ (+1 [CLS]) | ViT tokens; B/16 at 224: 196 + 1 = 197 |
| $d=\alpha^\phi,\ w=\beta^\phi,\ r=\gamma^\phi$, $\ \alpha\beta^2\gamma^2\approx2$ | EfficientNet compound scaling (FLOPs ≈ $2^\phi$) |
| LeNet-5 60K · AlexNet 60M · VGG-16 138M · GoogLeNet 6.8M · ResNet-50 25.6M · EffNet-B0 5.3M · ViT-B/16 86M | reference parameter counts |

## [S04 · Training CNNs & transfer learning](notes/04_training_transfer_learning.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $p_k=e^{z_k}/\sum_je^{z_j}$ | softmax over $K$ logits $z$ |
| $\mathcal L_{CE}=-\sum_ky_k\log p_k=-\log p_c$; $\ \partial\mathcal L/\partial z_k=p_k-y_k$ | cross-entropy; uniform output → $\ln K$ |
| $q_k=(1-\varepsilon)y_k+\varepsilon/K$ | label smoothing ($\varepsilon=0.1$) |
| $J=\mathcal L+\frac{\lambda}{2}\Vert\theta\Vert_2^2\ \Rightarrow\ \theta\leftarrow(1-\eta\lambda)\theta-\eta\nabla\mathcal L$ | L2 = weight decay (plain SGD); L1 $\lambda\sum\lvert\theta_i\rvert$ → sparsity |
| $g_t=\nabla\mathcal L+\lambda\theta_t$, $\ v_{t+1}=\mu v_t+g_t$, $\ \theta_{t+1}=\theta_t-\eta v_{t+1}$ | SGD + momentum (PyTorch form), $\mu=0.9$, $v_0=0$ |
| $\theta\leftarrow\theta-\eta\big(\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta\big)$ | AdamW (decoupled decay); $\hat m=m/(1-\beta_1^t)$, $\hat v=v/(1-\beta_2^t)$ |
| train $\tilde y=r\,y/(1-p)$, $r\sim\text{Bern}(1-p)$; test $\tilde y=y$ | inverted dropout, $p$ = drop prob.; $\mathbb E[\tilde y]=y$ |
| $\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, $\ y=\gamma\hat x+\beta$; $\ \hat\mu\leftarrow(1-\rho)\hat\mu+\rho\,\mu_B$ | BN per channel over $N\times H\times W$; $2C$ learnable + $2C$ buffers; running stats updated in train mode ($\rho=0.1$), used in eval |
| $\tilde x=\lambda x_i+(1-\lambda)x_j$, $\ \tilde y=\lambda y_i+(1-\lambda)y_j$, $\ \lambda\sim\text{Beta}(\alpha,\alpha)$ | mixup; CutMix $\lambda=1-w_bh_b/(WH)$ |
| $\eta_0\gamma^{\lfloor t/s\rfloor}$; $\ \eta_{min}+\frac12(\eta_{max}-\eta_{min})(1+\cos\frac{\pi t}{T})$; $\ \eta_{max}t/T_w$ | step; cosine; linear warm-up |
| $w_c=N/(K\,n_c)$ | class weights for imbalance |
| $P=\frac{TP}{TP+FP}$, $R=\frac{TP}{TP+FN}$, $F1=\frac{2PR}{P+R}$; macro-F1 = mean over classes | $FP_c$ = column sum − $TP_c$, $FN_c$ = row sum − $TP_c$; balanced acc. = mean per-class recall; micro-F1 = accuracy (single-label) |
| trainable $=(d+1)K$ | frozen backbone + linear head; ResNet-50 ($d=2048$), $K=10$ → 20,490 |
| $x'_c=(x_c/255-\text{mean}_c)/\text{std}_c$ | ImageNet input normalisation |

## [S05 · Embeddings & image similarity](notes/05_embeddings_similarity.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $d=\Vert\mathbf a-\mathbf b\Vert_2$, $\ \mathrm{sim}=\cos\theta=\frac{\mathbf a\cdot\mathbf b}{\Vert\mathbf a\Vert\Vert\mathbf b\Vert}$ | Euclidean distance, cosine similarity |
| $\Vert\hat{\mathbf a}-\hat{\mathbf b}\Vert^2=2-2\cos\theta$ | unit vectors: same ranking; $d\in[0,2]$; orthogonal → $\sqrt2$ |
| $\mathcal L=y\,d^2+(1-y)\max(0,m-d)^2$ | contrastive (Siamese pairs), $y=1$ similar, margin $m$ |
| $\mathcal L=\max(0,\ d(a,p)-d(a,n)+m)$ | triplet; semi-hard $d(a,p) < d(a,n) < d(a,p)+m$ |
| $\mathcal L=-\log\frac{\exp(\mathrm{sim}(q,k^+)/\tau)}{\sum_{i=0}^{N}\exp(\mathrm{sim}(q,k_i)/\tau)}$ | InfoNCE = CE over $N+1$ candidates; lower $\tau$ → sharper; collapse gives $\log(N+1)$ |
| $N$ images → $2N$ views → $2N-2$ negatives per anchor | SimCLR NT-Xent (in-batch negatives) |
| $\theta_k\leftarrow m\,\theta_k+(1-m)\,\theta_q$ | EMA / momentum encoder (MoCo $m=0.999$; BYOL, DINO) |
| $P@k=\frac1k\sum_{i\le k}\text{rel}(i)$, $\ R@k=\frac1R\sum_{i\le k}\text{rel}(i)$, $\ \text{AP}=\frac1R\sum_kP@k\cdot\text{rel}(k)$ | retrieval ($R$ = relevant items in DB); mAP = mean AP over queries; MRR = mean $1/\text{rank}$ of first hit; Hit@k = any hit in top $k$ |
| memory $=N\,D\,b$; PQ ≈ $N\,m$ bytes; brute force $N D$ MACs/query | index cost ($10^6\times512\times4$ B = 2.048 GB) |

## [S06 · Detection I: two-stage](notes/06_detection_two_stage.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $m=\sqrt{I_x^2+I_y^2}$, $\ \theta=\operatorname{atan2}(I_y,I_x)\bmod180^\circ$ | HOG: $[-1,0,1]$ gradients, 9 unsigned bins × 20°, bilinear voting |
| $N_{desc}=(n_x-b+1)(n_y-b+1)\,b^2\,n_{bins}$ | HOG length; 64×128, 8-px cells, 2×2 blocks → $7\cdot15\cdot36=\mathbf{3780}$; L2-Hys: normalise → clip 0.2 → renormalise |
| $\left(\lfloor\frac{W_l-w}{s}\rfloor+1\right)\left(\lfloor\frac{H_l-h}{s}\rfloor+1\right)$ per pyramid level | sliding-window count |
| $D=\prod_id_i$, $\ F=\prod_if_i$ | cascade detection / false-positive rate |
| $t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$ | box **encode** (centre-size; $a$ = anchor/proposal); corner → centre $x=\frac{x_1+x_2}{2}$, $w=x_2-x_1$ (no +1) |
| $\hat x=x_a+\hat t_xw_a,\ \hat y=y_a+\hat t_yh_a,\ \hat w=w_ae^{\hat t_w},\ \hat h=h_ae^{\hat t_h}$ | box **decode** |
| $\operatorname{smooth}_{L1}(z)=0.5z^2$ if $\lvert z\rvert < 1$, else $\lvert z\rvert-0.5$ | quadratic near 0, linear (robust) for outliers |
| $L=-\log p_u+\lambda[u\ge1]\sum_i\operatorname{smooth}_{L1}(t_i^u-v_i)$ | Fast R-CNN multi-task loss; no box loss for background ($u=0$) |
| $L=\frac{1}{N_{cls}}\sum_iL_{cls}+\lambda\frac{1}{N_{reg}}\sum_ip_i^{\ast}L_{reg}$ | RPN loss; $N_{cls}=256$, $N_{reg}\approx2400$, $\lambda=10$ |
| $w_a=s/\sqrt r$, $h_a=s\sqrt r$ ($r=h/w$); $\ N=\frac{H}{S_f}\cdot\frac{W}{S_f}\cdot k$ | anchors: $k=9$ (3 scales × 3 ratios), area $s^2$; RPN outputs $2k$ scores + $4k$ offsets |
| $\operatorname{round}(x/S_f)$; bins $\frac{w'}{7}\times\frac{h'}{7}$ | RoI Pool 7×7: two quantisations → misalignment (RoIAlign fixes) |
| $k=\lfloor4+\log_2(\sqrt{wh}/224)\rfloor$, clipped to $[2,5]$ | FPN pyramid level for an RoI |

## [S07 · Detection II: one-stage, IoU, NMS, AP](notes/07_detection_one_stage.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $\text{IoU}=\frac{I}{\lvert A\rvert+\lvert B\rvert-I}$, $\ I=\max(0,\min x_2-\max x_1)\cdot\max(0,\min y_2-\max y_1)$ | box IoU |
| $\text{GIoU}=\text{IoU}-\frac{\lvert C\rvert-\lvert A\cup B\rvert}{\lvert C\rvert}\in(-1,1]$; $\ \mathcal L_{GIoU}=1-\text{GIoU}$ | $C$ = smallest enclosing box; gradient even with no overlap |
| keep top-score $M$; delete $b_i$ if $\text{IoU}(M,b_i) > \tau_{nms}$; repeat | greedy NMS, per class; $\tau_{nms}\approx0.5$ (RPN 0.7) |
| $s_i\leftarrow s_i(1-\text{IoU})$ if IoU ≥ $N_t$; $\ s_i\leftarrow s_ie^{-\text{IoU}^2/\sigma}$ | Soft-NMS linear / Gaussian (decay instead of delete) |
| $P_k=\frac{TP_k}{TP_k+FP_k}$, $\ R_k=\frac{TP_k}{N_{GT}}$, $\ F_1=\frac{2PR}{P+R}$ | cumulative down the ranked list; TP if IoU ≥ τ **and** GT unmatched; duplicates = FP; $F_1$ picks deployment threshold |
| $AP_{all}=\sum_k(R_k-R_{k-1})\,p_{interp}(R_k)$, $\ p_{interp}(r)=\max_{r'\ge r}p(r')$ | **default** (VOC2010+): area under the precision envelope |
| $AP_{11}=\frac{1}{11}\sum_{r\in\lbrace0,0.1,\dots,1\rbrace}p_{interp}(r)$; $AP_{101}$ same on $\lbrace0,0.01,\dots,1\rbrace$ | VOC2007; COCO |
| $\text{mAP}=\frac1K\sum_cAP_c$; $\ AP_{COCO}=\frac{1}{10}\sum_{\tau=0.50}^{0.95}\text{mAP}_\tau$ | VOC mAP@0.5 vs COCO mAP@[.5:.95] (step 0.05, 10 thresholds) |
| $S\times S\times(5B+C)$ | YOLOv1: $7\times7\times30$ (VOC), 98 boxes; score $=\Pr(\text{cls}\mid\text{obj})\cdot\text{conf}$ |
| $\lambda_{coord}=5$, $\lambda_{noobj}=0.5$, SSE on $\sqrt w,\sqrt h$ | YOLOv1 loss (√ penalises small-box errors more) |
| $b_x=\sigma(t_x)+c_x$, $\ b_w=p_we^{t_w}$; k-means $d=1-\text{IoU}$ | YOLOv2/v3 decode; anchor clustering |
| $s_k=s_{min}+\frac{s_{max}-s_{min}}{m-1}(k-1)$; $\ N=\sum_kf_k^2n_k$ | SSD: $s\in[0.2,0.9]$, $m=6$ maps; SSD300 = **8732** boxes |
| $\mathcal L=\frac1N(\mathcal L_{conf}+\alpha\mathcal L_{loc})$, neg:pos ≤ 3:1 | SSD loss with hard-negative mining |
| $FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t$ | $\gamma=2$, $\alpha=0.25$; $p_t=p$ if $y=1$ else $1-p$; easy $p_t=0.9$ → ×100 smaller |
| $b=-\log\frac{1-\pi}{\pi}$ ($\pi=0.01$ → −4.595) | RetinaNet prior-probability bias init |
| $(l^{\ast},t^{\ast},r^{\ast},b^{\ast})$; centre-ness $=\sqrt{\frac{\min(l,r)}{\max(l,r)}\cdot\frac{\min(t,b)}{\max(t,b)}}$ | FCOS (anchor-free) distances to box sides |
| box $=R\,(\hat x+\delta\hat x\mp\hat w/2,\ \hat y+\delta\hat y\mp\hat h/2)$ | CenterNet: peak + offset + size, stride $R=4$; 3×3 max-pool replaces NMS |

## [S08 · Detection III: transformers](notes/08_detection_transformers.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $\hat\sigma=\arg\min_{\sigma\in\mathfrak S_N}\sum_i\mathcal L_{match}(y_i,\hat y_{\sigma(i)})$ | bipartite matching, Hungarian $O(N^3)$; $N=100$ queries, GT padded with $\varnothing$ |
| $\mathcal L_{match}=-\mathbb 1_{\lbrace c_i\neq\varnothing\rbrace}\hat p_{\sigma(i)}(c_i)+\mathbb 1_{\lbrace c_i\neq\varnothing\rbrace}\mathcal L_{box}$ | matching cost uses the **probability** |
| $\mathcal L_{box}=\lambda_{L1}\Vert b-\hat b\Vert_1+\lambda_{giou}\mathcal L_{GIoU}$ | $\lambda_{L1}=5$, $\lambda_{giou}=2$; normalised $(c_x,c_y,w,h)$ |
| $\mathcal L_H=\sum_i\big[-w_{c_i}\log\hat p_{\hat\sigma(i)}(c_i)+\mathbb 1_{\lbrace c_i\neq\varnothing\rbrace}\mathcal L_{box}\big]$ | set loss uses **log-prob**; $w_\varnothing=0.1$; aux loss at every decoder layer |
| $\hat b=\operatorname{sigmoid}(\text{MLP}(h))$; $\ x_1=(c_x-\frac w2)W,\ y_1=(c_y-\frac h2)H$ | DETR box → pixels |
| $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$ | attention; encoder self-attn $O((HW)^2d)$; decoder cross-attn $N\times HW$ |
| samples per query $=M\cdot L\cdot K$ ($8\cdot4\cdot4=128$) | deformable attention, linear in $HW$; ~50 vs ~500 epochs (≈10× fewer) |
| $N!/(N-M)!$ | ways to assign $M$ objects to $N$ queries (why Hungarian) |

## [S10 · Image segmentation](notes/10_image_segmentation.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $\mathcal L_{CE}=-\frac{1}{\lvert\Omega\rvert}\sum_{p\in\Omega}\sum_cy_{p,c}\log\hat y_{p,c}$ | pixel-wise CE ($\Omega$ = labelled pixels, void excluded) |
| $w(\mathbf x)=w_c+w_0\exp(-(d_1+d_2)^2/2\sigma^2)$ | U-Net border weight ($w_0=10$, $\sigma\approx5$ px) |
| valid conv $W-K+1$; pool $\lfloor W/2\rfloor$; up-conv $(W-1)S-2P+K$ | U-Net 572 → 388; FCN-32s ($K=64,S=32,P=16$): 7 → 224 |
| $\text{IoU}=\frac{TP}{TP+FP+FN}$, $\ \text{Dice}=\frac{2TP}{2TP+FP+FN}=\frac{2\lvert A\cap B\rvert}{\lvert A\rvert+\lvert B\rvert}$ | Jaccard; Dice = F1 |
| $\text{Dice}=\frac{2\,\text{IoU}}{1+\text{IoU}}$, $\ \text{IoU}=\frac{\text{Dice}}{2-\text{Dice}}$ | Dice ≥ IoU, monotone (IoU 0.5 ↔ Dice 0.667) |
| $\text{PA}=\frac{\sum_cn_{cc}}{\sum_{i,j}n_{ij}}$; mean class acc. $=\frac1C\sum_c\frac{n_{cc}}{\sum_jn_{cj}}$ | pixel accuracy; $n_{ij}$ = pixels of GT $i$ predicted $j$; mean class acc. = mean per-class recall |
| $\text{IoU}_c=\frac{n_{cc}}{\sum_jn_{cj}+\sum_in_{ic}-n_{cc}}$, $\ \text{mIoU}=\frac1C\sum_c\text{IoU}_c$ | row sum = GT pixels, column sum = predicted pixels |
| $\mathcal L_{Dice}=1-\frac{2\sum p_ig_i+\epsilon}{\sum p_i+\sum g_i+\epsilon}$ | soft Dice loss |
| $PQ=\frac{\sum_{TP}\text{IoU}}{\lvert TP\rvert+\frac12\lvert FP\rvert+\frac12\lvert FN\rvert}=SQ\times RQ$ | match if IoU > 0.5 (unique); SQ = mean matched IoU, RQ = F1 |
| $f=(1-a)(1-b)f_{00}+a(1-b)f_{10}+(1-a)bf_{01}+abf_{11}$ | bilinear sampling (RoIAlign); $a=x-x_0$, $b=y-y_0$ |
| error $=\lvert[x/s]-x/s\rvert\cdot s\le s/2$ | RoI Pool rounding (≤ 8 px at stride 16) |
| $\mathcal L=\mathcal L_{cls}+\mathcal L_{box}+\mathcal L_{mask}$; mask head $K\times28\times28$ | Mask R-CNN; per-pixel sigmoid BCE on the **GT-class channel only** |
| $K_{eff}=D(K-1)+1$ | atrous conv (DeepLab): larger RF, same params, no downsampling |

## [S11 · Camera geometry](notes/11_camera_geometry.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $x=fX/Z$, $\ y=fY/Z$; weak persp. $x=(f/Z_0)X$; orthographic $x=X$ | perspective (camera frame); the other two are affine approximations |
| $f_x=f/\rho_x$ | focal length mm → px ($\rho_x$ = pixel pitch) |
| $\text{FOV}_h=2\arctan\frac{W}{2f_x}$ | horizontal field of view ($W$ px) |
| $\lambda\,(u,v,1)^\top=K[R\mid\mathbf t]\,(X,Y,Z,1)^\top$ | pinhole; $\lambda=Z_c$; $P=K[R\mid\mathbf t]$ is 3×4 |
| $K=\begin{bmatrix}f_x&s&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}$ | intrinsics: focal (px), skew, principal point |
| $u=f_x\frac{X_c}{Z_c}+c_x$, $\ v=f_y\frac{Y_c}{Z_c}+c_y$ | zero skew |
| $X_c=RX_w+\mathbf t$; $\ C=-R^\top\mathbf t$ | world → camera; camera centre in world |
| $X_c=\frac{(u-c_x)Z_c}{f_x}$; $\ X_w=R^\top(X_c-\mathbf t)$ | back-projection at known depth |
| DOF: $K$ 5 · $[R\mid\mathbf t]$ 6 · $P$ **11** · $H$ 8 · $B$ 5 · distortion 5 | $P$, $H$, $B$ defined up to scale |
| $h=f_yH_o/Z\iff Z=f_yH_o/h$ | pinhole object size ↔ distance ($H_o$ real height, $h$ px) |
| $\tilde{\mathbf v}\sim K\mathbf D$ | vanishing point of 3-D direction $\mathbf D$ (camera frame; world direction $\mathbf D_w$: $K R\mathbf D_w$) |
| $x_d=x(1+k_1r^2+k_2r^4+k_3r^6)$ | radial ($r^2=x^2+y^2$, normalised coords); barrel $k_1 < 0$, pincushion $k_1 > 0$ |
| $x_d \mathrel{+}= 2p_1xy+p_2(r^2+2x^2)$, $\ y_d \mathrel{+}= p_1(r^2+2y^2)+2p_2xy$ | tangential (decentred lens) |
| $\text{RMS}=\sqrt{\frac1N\sum_k\Vert\mathbf x_k-\hat{\mathbf x}_k\Vert^2}$ | reprojection error (good: well under 1 px, ≈ 0.1–0.5) |
| $H=\lambda K[\mathbf r_1\ \mathbf r_2\ \mathbf t]$; $\ \mathbf h_1^\top B\mathbf h_2=0$, $\ \mathbf h_1^\top B\mathbf h_1=\mathbf h_2^\top B\mathbf h_2$ | Zhang (plane $Z=0$); $B=K^{-\top}K^{-1}$; 2 constraints/view → ≥ 3 views (2 if $s=0$) |
| $H=K'RK^{-1}$ | pure rotation, any scene ($KRK^{-1}$ for one camera) |
| $H=K'(R-\mathbf t\mathbf n^\top/d)K^{-1}$ for plane $\mathbf n^\top\mathbf X+d=0$ | two views of a plane, cam-1 frame, $\mathbf X_2=R\mathbf X_1+\mathbf t$; **if plane is $\mathbf n^\top\mathbf X=d$ (S02 form): $R+\mathbf t\mathbf n^\top/d$** |
| DLT: $2n\times12$, ≥ 6 pts; PnP: ≥ 3 pts (+1) | full $P$ (11 DOF) vs pose only (6 DOF, $K$ known) |

## [S12 · 3D vision & motion](notes/12_3d_vision_motion.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $Z=fB/d$, $\ d=x_L-x_R$ | rectified stereo; $f$ px, $B$ m, $d$ px |
| $X=(x_L-c_x)Z/f$, $\ Y=(y-c_y)Z/f$ | 3-D point from disparity (left camera frame) |
| $\Delta Z\approx\frac{Z^2}{fB}\Delta d$; $\ Z_{max}=\frac{fB}{d_{min}}$, $\ Z_{min}=\frac{fB}{d_{max}}$ | depth error ∝ $Z^2$; larger $B$ → better far, harder matching |
| $\hat{\mathbf x}'^\top E\hat{\mathbf x}=0$, $\ E=[\mathbf t]_\times R$ | essential; $\hat{\mathbf x}=K^{-1}\tilde{\mathbf x}$; **5 DOF**, rank 2, singular values $(\sigma,\sigma,0)$ |
| $\mathbf x'^\top F\mathbf x=0$, $\ F=K'^{-\top}EK^{-1}$ | fundamental (pixels); **7 DOF**, rank 2 |
| $\mathbf l'=F\mathbf x$, $\ \mathbf l=F^\top\mathbf x'$; $\ F\mathbf e=\mathbf 0$, $\ F^\top\mathbf e'=\mathbf 0$ | epipolar lines; epipoles |
| $[x'x,\ x'y,\ x',\ y'x,\ y'y,\ y',\ x,\ y,\ 1]\,\mathbf f=0$ | 8-point row; enforce rank 2 (zero smallest σ); $F=T'^\top\hat FT$; RANSAC ($e=0.5$): 8-pt 1177, 7-pt 588, 5-pt 146 |
| rows $x\mathbf p^{3\top}-\mathbf p^{1\top}$, $\ y\mathbf p^{3\top}-\mathbf p^{2\top}$ (both views) → $A\tilde{\mathbf X}=\mathbf 0$ | linear triangulation (4×4, SVD); $\mathbf p^{i\top}$ = row $i$ of $P$ |
| $I_xu+I_yv+I_t=0$ | brightness constancy: 1 equation, 2 unknowns (aperture problem) |
| $\Vert\mathbf u_\perp\Vert=\lvert I_t\rvert/\Vert\nabla I\Vert$ | normal flow (only the component along the gradient) |
| $\begin{bmatrix}\sum I_x^2&\sum I_xI_y\\ \sum I_xI_y&\sum I_y^2\end{bmatrix}\begin{bmatrix}u\\ v\end{bmatrix}=-\begin{bmatrix}\sum I_xI_t\\ \sum I_yI_t\end{bmatrix}$ | Lucas–Kanade; matrix = Harris $M$ → track corners; pyramid level $L$: motion $/2^L$ |
| $E=\iint(I_xu+I_yv+I_t)^2+\alpha^2(\Vert\nabla u\Vert^2+\Vert\nabla v\Vert^2)$ | Horn–Schunck (dense, global smoothness $\alpha$) |
| $G=\Sigma^-C^\top(C\Sigma^-C^\top+R_m)^{-1}$, $\ \mathbf s=\mathbf s^-+G(\mathbf z-C\mathbf s^-)$ | Kalman update (SORT); symbols renamed as in notes 12 |
| $\min\sum_{i,j}\Vert\mathbf x_{ij}-\pi(K_i,R_i,\mathbf t_i,\mathbf X_j)\Vert^2$ | bundle adjustment (total reprojection error) |
| params $=6n_{cams}+3n_{pts}$ (+ intrinsics); residuals $=2n_{obs}$ | BA size (10 cams, 1000 pts seen 3× → 3060 / 6000) |
| monocular SfM/SLAM: up to a similarity (7 DOF) | scale unobservable; loop closure estimated in Sim(3) |

## [S13 · Vision foundation models](notes/13_vision_foundation_models.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $N_{tok}=\frac Hp\cdot\frac Wp$ (+1); attention entries $N_{tok}^2$ per head per layer | ViT-B/16 at 224 → 197; SAM ViT-H/16 at 1024 → 64×64 = 4096 |
| $S_{ij}=\cos(\mathbf v_i,\mathbf t_j)$, logits $\ell_{ij}=S_{ij}/\tau$ | CLIP $N\times N$ matrix; $1/\tau$ learned (≤ 100); $N$ positives (diagonal), $N-1$ negatives per row |
| $\mathcal L=\frac12(\mathcal L_{I\to T}+\mathcal L_{T\to I})$, $\ \mathcal L_{I\to T}=-\frac1N\sum_i\log\frac{e^{\ell_{ii}}}{\sum_je^{\ell_{ij}}}$ | symmetric InfoNCE (rows + columns); 400M pairs |
| $p(c\mid x)=\frac{\exp(\cos(\mathbf v,\mathbf t_c)/\tau)}{\sum_{c'}\exp(\cos(\mathbf v,\mathbf t_{c'})/\tau)}$ | zero-shot: prompt "a photo of a {c}."; ensembling = average normalised $\mathbf t_c$, renormalise |
| $P_t\propto\exp((g_t-c)/\tau_t)$, $\ P_s\propto\exp(g_s/\tau_s)$; $\ \mathcal L=\sum H(P_t(x),P_s(x'))$ | DINO: centering + sharpening ($\tau_t\approx0.04$–$0.07 < \tau_s=0.1$), cross-view, $x\neq x'$ |
| $\theta_t\leftarrow\lambda\theta_t+(1-\lambda)\theta_s$, $\lambda\in[0.996,1)$; $\ c\leftarrow mc+(1-m)\bar g_t$ | DINO EMA teacher; running centre ($m\approx0.9$) |
| $\mathcal L_{koleo}=-\frac1n\sum_i\log\min_{j\neq i}\Vert\mathbf x_i-\mathbf x_j\Vert$ | DINOv2 regulariser: spread features |
| embedding $256\times64\times64$; mask loss $20\cdot\text{focal}+1\cdot\text{Dice}$ | SAM: encode once, light decoder per prompt; 3 masks + predicted IoU |
| $1/Z\approx s\,r+t$; least squares over ≥ 2 pixels of known depth | Depth Anything output $r$ = **relative inverse** depth; scale-shift alignment |
| $\hat s=\frac{n\sum r_iy_i-\sum r_i\sum y_i}{n\sum r_i^2-(\sum r_i)^2}$, $\ \hat t=\frac{\sum y_i-\hat s\sum r_i}{n}$, $\ y_i=1/Z_i$ | closed-form fit (larger $r$ = closer) |
| $\text{logit}_{i,q}=\alpha_i(\mathbf e_i\cdot\mathbf t_q)+\beta_i$, score $=\sigma(\text{logit})$ | OWL-ViT: one box per image token, scored per text/image query |

## [S14 · Vision++ with foundation models](notes/14_vision_plus_plus.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $p(c\mid\mathbf f)=\frac{\exp(\cos(\mathbf f,\mathbf t_c)/\tau)}{\sum_{c'}\exp(\cos(\mathbf f,\mathbf t_{c'})/\tau)}$ vs $\ s_c=\sigma(\mathbf f\cdot\mathbf t_c)$ | open-vocab scoring: CLIP softmax (classes compete) vs detector sigmoid (independent per phrase) |
| keep query $q$ if $\max_j\sigma(\ell_{q,j})$ > `box_threshold`; phrase = tokens $j$ with $\sigma(\ell_{q,j})$ > `text_threshold` | Grounding DINO (defaults 0.35 / 0.25; 900 queries, ≤ 256 text tokens) |
| $T_{seq}=\sum_kT_k$; $\ \text{FPS}=1000/T_{\text{ms}}$; pipelined throughput $=1000/\max_kT_k$ | pipeline latency (100 + 400 + 2×10 + 60 = 580 ms → 1.7 FPS) |
| $1/Z=s\,r+t$; one reference ($t=0$ assumed): $s=(1/Z_{ref})/r_{ref}$ | relative → metric depth; $r$ = median inverse depth inside the mask |
| $Z_{ref}=fH_{ref}/h_{px}$, $\ H_{obj}=Z\,h_{px}/f$ | pinhole: known size ↔ distance (S11 $Z=f_yH_o/h$) |
| $X=Z(u-c_x)/f_x$, $\ Y=Z(v-c_y)/f_y$; range $=\sqrt{X^2+Y^2+Z^2}$ | depth $Z$ (optical axis) vs straight-line distance |
| $P(\text{success})\approx\prod_kp_k$ | error propagation through stages ($0.90\cdot0.95\cdot0.95\approx0.81$) |
| $\text{AbsRel}=\frac1n\sum\lvert d-d^{\ast}\rvert/d^{\ast}$; $\ \delta_1$ = % pixels with $\max(d/d^{\ast},d^{\ast}/d) < 1.25$ | depth-accuracy metrics |
| $\frac{Nt_{model}}{3600}c_{GPU}+\frac{\rho Nt_{review}}{3600}c_{human}$ vs $\frac{Nn_{obj}t_{manual}}{3600}c_{human}$ | auto-labelling (+ QA fraction $\rho$) vs manual labelling cost |

## [S15 · Generative CV](notes/15_generative_cv.md#3-key-formulas)
| Formula | Meaning / symbols |
|---|---|
| $\mathcal L_{AE}=\frac1N\sum_i\Vert x_i-g_\theta(f_\phi(x_i))\Vert^2$ | autoencoder; anomaly = high reconstruction error |
| $\log p_\theta(x)\ge\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]-\mathrm{KL}(q_\phi(z\mid x)\Vert p(z))$ | VAE ELBO; loss = −ELBO = reconstruction + KL |
| $\log p_\theta(x)=\text{ELBO}+\mathrm{KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))$ | gap = KL to the true posterior |
| $z=\mu+\sigma\odot\epsilon$, $\ \epsilon\sim\mathcal N(0,I)$ | reparameterisation (backprop through $\mu,\sigma$) |
| $\mathrm{KL}=\frac12\sum_j(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1)$ | diagonal Gaussian vs $\mathcal N(0,I)$; 0 iff $\mu=0,\sigma=1$ |
| $\min_G\max_D\ \mathbb E_x[\log D(x)]+\mathbb E_z[\log(1-D(G(z)))]$ | GAN minimax |
| $D^{\ast}(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}$ | optimal discriminator |
| $V(D^{\ast},G)=2\,\mathrm{JSD}(p_{data}\Vert p_g)-\log4$ | JSD ∈ [0, log 2]; at $p_g=p_{data}$: $D^{\ast}=\frac12$, $V=-\log4\approx-1.386$ |
| $\mathcal L_G^{NS}=-\mathbb E_z[\log D(G(z))]$ | non-saturating generator loss (strong gradient when G is losing) |
| $\mathcal L_{critic}=\mathbb E[f(\tilde x)]-\mathbb E[f(x)]+\lambda\,\mathbb E[(\Vert\nabla_{\hat x}f(\hat x)\Vert-1)^2]$ | WGAN-GP, $\lambda=10$ |
| pix2pix $\mathcal L_{cGAN}+100\Vert y-G(x)\Vert_1$; CycleGAN $2\times\text{GAN}+10\,\text{cycle-}L_1$ | paired vs unpaired translation |
| $\mathrm{FID}=\Vert\mu_r-\mu_g\Vert^2+\operatorname{Tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ | Inception-v3 2048-d; lower better; 1-D: $(\mu_r-\mu_g)^2+(\sigma_r-\sigma_g)^2$ |
| $\mathrm{IS}=\exp(\mathbb E_x\,\mathrm{KL}(p(y\mid x)\Vert p(y)))$ | higher better, max = number of classes |
| $q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_tI)$, $\ \alpha_t=1-\beta_t$, $\ \bar\alpha_t=\prod_{s=1}^t\alpha_s$ | DDPM forward; $T=1000$, linear $\beta$: $10^{-4}\to0.02$; $\bar\alpha_{500}\approx0.079$, $\bar\alpha_{1000}\approx4\times10^{-5}$ |
| $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ | closed form (jump to any $t$); SNR $=\bar\alpha_t/(1-\bar\alpha_t)$ |
| $\mathcal L=\mathbb E_{t,x_0,\epsilon}\Vert\epsilon-\epsilon_\theta(x_t,t)\Vert^2$ | simple loss: predict the noise |
| $\mu_\theta=\frac{1}{\sqrt{\alpha_t}}\big(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\big)$; $\ x_{t-1}=\mu_\theta+\sigma_tz$ | reverse mean / sampling step; $\sigma_t^2\in\lbrace\beta_t,\tilde\beta_t\rbrace$, $z=0$ at last step |
| $\hat x_0=(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta)/\sqrt{\bar\alpha_t}$ | clean-image estimate |
| $x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\epsilon_\theta+\sigma_tz$ | DDIM; $\eta=0$ deterministic; skip steps (e.g. 50), no retraining |
| $\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+w(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing))$ | CFG; $w=1$ plain conditional, $w > 1$ extrapolates (≈ 7.5); 2 passes/step |
| $\hat\epsilon=\epsilon_\theta(x_t)-s\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y\mid x_t)$ | classifier guidance (noise-aware classifier $p_\phi$) |
| compression $=\frac{H\cdot W\cdot3}{(H/f)(W/f)\,c}$ | latent diffusion; SD $f=8$, $c=4$: 512²×3 → 64²×4 = **48×** |
| steps $=\lfloor sN\rfloor$; $\ W'=W+BA$, LoRA params $r(d+k)$ | img2img strength $s$ of $N$ steps; LoRA rank $r$ |
