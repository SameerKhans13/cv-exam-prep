# Session 13 — Vision Foundation Models · Practice Questions

> Notes: [../notes/13_vision_foundation_models.md](../notes/13_vision_foundation_models.md) · Solutions: [13_solutions.md](13_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
Which statement best describes a **foundation model**?

- **A.** A model pre-trained on broad data at scale (mostly self- or weakly-supervised) that can be adapted to many downstream tasks
- **B.** Any model with more than one billion parameters
- **C.** A model trained on a single large labelled dataset to solve one task as well as possible
- **D.** A model that cannot be fine-tuned and can only be used through prompts

### Q2 · MCQ · Medium
What is CLIP's pre-training objective?

- **A.** Cross-entropy over the 1,000 ImageNet classes, followed by fine-tuning on captions
- **B.** Autoregressive generation of each image's caption, word by word
- **C.** A symmetric contrastive (InfoNCE) loss over the N × N image–text cosine-similarity matrix, with a learned temperature
- **D.** Mean-squared error between each image embedding and its caption embedding, with no negatives

### Q3 · MCQ · Medium
In CLIP zero-shot classification, **prompt ensembling** means:

- **A.** averaging the image embeddings of several random crops of the test image
- **B.** averaging the normalised text embeddings of several prompt templates per class (then re-normalising) to form each class embedding
- **C.** training a separate linear classifier for each prompt template and taking a majority vote
- **D.** choosing, for each class, the single template that gives the highest accuracy on the test set

### Q4 · MCQ · Medium
How does DINO avoid representation collapse?

- **A.** By contrasting each image against 65,536 negatives stored in a queue
- **B.** By using ImageNet labels as the teacher's targets
- **C.** By reconstructing masked pixels with a decoder
- **D.** By centering and sharpening the outputs of an EMA teacher (with stop-gradient)

### Q5 · MCQ · Medium
A user clicks once on a person's shirt and SAM returns **three** masks, each with a predicted IoU. Why?

- **A.** SAM returns one mask per predicted class, e.g. shirt, person and background
- **B.** The click is ambiguous (sub-part, part or whole object), so SAM returns several valid masks ranked by predicted IoU
- **C.** The three masks come from three image-pyramid scales of the image encoder
- **D.** They are the outputs of three decoder layers that are used only for deep supervision

### Q6 · MCQ · Easy
What does the (non-fine-tuned) **Depth Anything** model output for a single RGB image?

- **A.** A relative, affine-invariant inverse-depth map that must be scaled and shifted to obtain metres
- **B.** Metric depth in metres for any camera, without calibration
- **C.** A disparity map that requires a second, rectified image as input
- **D.** Surface normals and a semantic label for every pixel

### Q7 · MCQ · Medium
How does **OWL-ViT** detect a category it never saw labelled boxes for?

- **A.** It trains a new classification head with labelled boxes for every new category
- **B.** It segments everything with SAM and then classifies each mask with CLIP
- **C.** Each image token predicts a box and a class embedding, which is scored against the embeddings of the text (or example-image) queries
- **D.** It generates a caption for the image and searches the caption for the category name

### Q8 · MCQ · Medium
Which statement correctly describes how **DINOv2** differs from the original DINO?

- **A.** DINOv2 adds a text encoder so that it can classify images from text prompts
- **B.** DINOv2 is trained with ImageNet-22k labels instead of self-supervision
- **C.** DINOv2 replaces self-distillation with a pixel-reconstruction (MAE) loss only
- **D.** DINOv2 trains on a curated 142M-image dataset and adds a masked-patch (iBOT) objective and a KoLeo regulariser to the DINO loss

## Section B — Numericals

### Q9 · Numerical · Easy · [5 marks]
CLIP zero-shot classification of one image. The cosine similarities between the image embedding and the prompt embeddings "a photo of a {dog / cat / fox / car}." are **0.32, 0.29, 0.30, 0.15**. The logit scale is $1/\tau=100$ ($\tau=0.01$, given).

(a) Compute the softmax probability of each class and the predicted class. [3]
(b) Recompute the probabilities with logit scale 1 (no temperature). What does this show? [1]
(c) A fifth prompt "a photo of a wolf." is added with cosine similarity 0.32. What is the new probability of "dog"? What does this tell you about CLIP's zero-shot probabilities? [1]

### Q10 · Numerical · Medium · [5 marks]
A toy CLIP batch of $N=3$ pairs has the image–text cosine-similarity matrix (rows = images $i$, columns = texts $j$; the matching pairs are on the diagonal):

$$
S=\begin{bmatrix}0.30 & 0.10 & 0.05\\ 0.12 & 0.28 & 0.15\\ 0.02 & 0.20 & 0.25\end{bmatrix}
$$

Use a logit scale of 20 (given) and natural logs.

(a) Compute the image→text loss for each row and $\mathcal L_{I\to T}$ (mean over rows). [2]
(b) Compute the text→image loss for each column and $\mathcal L_{T\to I}$. [2]
(c) Give the symmetric CLIP loss. For CLIP's real batch size $N=32{,}768$, how many similarity entries, how many negatives per row, and how much memory (float32, GB = $10^9$ B) does the full logit matrix need? [1]

### Q11 · Numerical · Medium · [5 marks]
Token counts and attention cost for ViT-based foundation models (one [CLS] token where stated).

(a) ViT-B/16 at 224×224 with a [CLS] token: number of tokens, attention-matrix entries per head per layer, and total entries for 12 heads × 12 layers. [1.5]
(b) ViT-L/14 at 336×336 with [CLS]: number of tokens and the ratio of attention entries per head compared with (a). [1]
(c) SAM's image encoder: 1024×1024 input, patch 16, **no** [CLS] token. Give the token grid, the number of tokens and the attention entries per head for **global** attention. SAM mostly uses **windowed** attention: assume the grid is padded to 70×70 and split into non-overlapping 14×14 windows. How many entries per head now, and what is the reduction factor? [1.5]
(d) SAM's cached image embedding is 256×64×64 float32 values: size in MB ($10^6$ B)? With an encoder time of 0.45 s and a prompt-encoder + decoder time of 0.05 s per click (given), how long do 10 interactive clicks take if the embedding is cached, versus re-running the encoder each click? [1]

### Q12 · Numerical · Medium · [5 marks]
Depth Anything returns a relative inverse-depth map $r$ (larger = closer). Assume $1/Z=s\,r+t$ ($Z$ in metres). A LiDAR gives two reference pixels: $r_1=0.8$ at $Z_1=2.0$ m and $r_2=0.3$ at $Z_2=10.0$ m.

(a) Solve for $s$ and $t$. [2]
(b) Estimate the metric depth of pixels with $r=0.5$ and $r=0.2$. [1]
(c) A third reference $r_3=0.6$ at $Z_3=2.5$ m is added. Fit $s,t$ by least squares on $1/Z$ using all three points and re-estimate the depths at $r=0.5$ and $r=0.2$. Comment on the far pixel. [1.5]
(d) A student instead assumes $Z=k\,r$ and fits $k$ from the first reference only. What depth does this give at $r_2=0.3$, and why is it wrong? [0.5]

### Q13 · Numerical · Medium · [5 marks]
DINO with a toy $K=3$-dimensional head.

(a) A teacher weight is $\theta_t=0.40$ and the corresponding student weight is $\theta_s=0.60$. Compute the teacher weight after one EMA update with $\lambda=0.996$. [1]
(b) The current centre is $c=(2.2,\ 0.1,\ 1.0)$. The teacher's raw output for global view 1 is $(3,\ 1,\ 2)$. Compute the centred logits and the teacher distribution $P_t$ with teacher temperature $\tau_t=0.05$. [1.5]
(c) The student's raw output for a local view of the same image is $(0.2,\ 0.1,\ 0.3)$ with $\tau_s=0.1$. Compute $P_s$ and the cross-entropy $H(P_t,P_s)=-\sum_k P_t^{(k)}\ln P_s^{(k)}$. [1.5]
(d) The teacher outputs for the batch (two global views) are $(3,1,2)$ and $(5,1,0)$. Update the centre with momentum $m=0.9$. What would $P_t$ in (b) be with $\tau_t=1$ (no sharpening), and why would that be a problem? [1]

## Section C — Short answer

### Q14 · Short · Easy · [4 marks]
Define a foundation model. Describe four ways of adapting one to a downstream task, ordered by cost, and give two opportunities and two risks of building many applications on the same foundation model (homogenisation).

### Q15 · Short · Medium · [4 marks]
Explain step by step how CLIP performs zero-shot image classification, including prompt templates and prompt ensembling. List four limitations of CLIP.

### Q16 · Short · Medium · [4 marks]
Explain DINO's self-distillation. Cover the student and teacher, how the teacher is updated, multi-crop, and how centering and sharpening prevent collapse. Mention one emergent property. A diagram is expected.

### Q17 · Short · Medium · [4 marks]
Describe SAM's architecture and explain why it is split into a heavy image encoder and a light prompt encoder + mask decoder. How does SAM handle ambiguous prompts, how was SA-1B built, and what is SAM's most important limitation for real applications?

## Section D — Long answer / Design

### Q18 · Long · Medium · [10 marks]
(a) Compare CLIP, DINOv2, SAM, Depth Anything and OWL-ViT in a table: training data, training objective, input, output, a typical use and a key limitation. [6]
(b) For each scenario, choose the most suitable model(s) and justify briefly: (i) search a personal photo library with free-text queries like "birthday cake on a beach"; (ii) build a 30-class plant-disease classifier from 20 labelled leaf images per class; (iii) an AR app that must place a virtual object behind real furniture; (iv) find every forklift in warehouse images without any labelled forklift boxes. [4]

### Q19 · Long · Hard · [10 marks]
An online marketplace has **2 million unlabelled** product photos and **5,000 labelled** photos across **50 categories**; new categories are added every month. It needs (1) automatic category tagging of new listings, (2) "visually similar items" retrieval, and (3) clean product cut-outs (background removal) for listing thumbnails. Design a solution based on foundation models. Compare CLIP zero-shot, DINOv2 (k-NN / linear probe) and full fine-tuning for tagging; explain how SAM is prompted for cut-outs; describe evaluation, cost/latency and risks.
