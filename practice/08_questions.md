# Session 08 — Object Detection III: Transformers · Practice Questions

> Notes: [../notes/08_detection_transformers.md](../notes/08_detection_transformers.md) · Solutions: [08_solutions.md](08_solutions.md)
> **Mix:** 7 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min

Conventions for this paper: boxes in pixels are $(x_1,y_1,x_2,y_2)$ in continuous coordinates, area $=(x_2-x_1)(y_2-y_1)$ — **no +1**. DETR boxes are normalised $(c_x,c_y,w,h)\in[0,1]^4$. DETR matching cost $=-\hat p(c_i)+\lambda_{L1}\Vert b_i-\hat b\Vert_1+\lambda_{giou}(1-\text{GIoU})$ with $\lambda_{L1}=5$, $\lambda_{giou}=2$. Natural logarithms.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
In DETR, the classification head applied to each object query outputs a probability distribution over:

- **A.** the $C$ object classes only, with a separate sigmoid objectness score
- **B.** $C+1$ classes, the extra class being "no object" ($\varnothing$)
- **C.** two classes (object / background), followed by a second-stage classifier
- **D.** the $C$ object classes, with background handled by NMS

### Q2 · MCQ · Medium
A training image contains 7 ground-truth objects and DETR uses $N=100$ object queries. After Hungarian matching, how many queries contribute an L1 + GIoU **box** loss for this image (per decoder layer)?

- **A.** 100 — every query regresses a box
- **B.** 93 — the queries matched to "no object"
- **C.** 7 — only the queries matched to real objects
- **D.** however many queries have IoU > 0.5 with some ground-truth box

### Q3 · MCQ · Medium
Which statement about DETR's object queries is correct?

- **A.** They are learned embedding vectors, identical for every image; the decoder turns them into image-specific outputs through cross-attention to the encoder memory.
- **B.** They are region proposals produced by a region proposal network on the backbone features.
- **C.** They are 100 fixed anchor boxes of different scales and aspect ratios placed on the image.
- **D.** They are the 100 highest-scoring flattened encoder tokens of the current image.

### Q4 · MCQ · Medium
What is the main reason DETR does not need non-maximum suppression?

- **A.** A confidence threshold of 0.7 at inference removes all duplicate boxes.
- **B.** It uses only 100 queries, far fewer than the anchors of a one-stage detector.
- **C.** The GIoU loss pushes overlapping predicted boxes away from each other.
- **D.** One-to-one bipartite matching in training penalises duplicate predictions (they are matched to "no object"), and decoder self-attention lets queries coordinate.

### Q5 · MCQ · Medium
Deformable DETR reaches better accuracy than DETR with about 10× fewer training epochs. The key change is that:

- **A.** Hungarian matching is replaced by IoU-threshold assignment to anchors.
- **B.** the ResNet backbone uses deformable convolutions instead of standard convolutions.
- **C.** each query attends to only a few sampling points (learned offsets around a reference point) per head and per level of a multi-scale feature pyramid.
- **D.** the transformer encoder is removed and the decoder attends directly to the backbone.

### Q6 · MCQ · Easy
In the open-vocabulary detector **Grounding DINO**, "DINO" refers to:

- **A.** self-DIstillation with NO labels — the self-supervised ViT training method
- **B.** DETR with Improved deNoising anchOr boxes — a DETR-style detector
- **C.** Detection In Natural Open-world — a dataset of open-world images
- **D.** the DINOv2 backbone, which is the only pretrained component of the model

### Q7 · MCQ · Medium
In a TIDE-style error analysis ($t_f=0.5$, $t_b=0.1$), a detection labelled "dog" has IoU 0.35 with a ground-truth "dog" box and IoU < 0.1 with every other ground-truth box. This false positive is counted as a:

- **A.** classification error
- **B.** duplicate detection error
- **C.** background error
- **D.** localisation error

## Section B — Numericals

### Q8 · Numerical · Medium · [5 marks]
During DETR training on one image there are 4 queries $q_1..q_4$ and 3 ground-truth objects $g_1,g_2,g_3$. The matching costs (class + box terms already combined, lower = better) are:

$$
\mathbf C=\begin{bmatrix} 2 & 3 & 7\\ 7 & 9 & 8\\ 1 & 5 & 6\\ 5 & 1 & 2\end{bmatrix}\qquad(\text{rows } q_1..q_4,\ \text{columns } g_1,g_2,g_3)
$$

(a) Make the matrix square by adding a "no object" ($\varnothing$) column; state and justify the cost you put in it. [1]
(b) Solve the assignment with the Hungarian method, showing row reduction, column reduction, line covering and any adjustment step. [2]
(c) Give the optimal assignment and its total cost. Which query is trained to predict $\varnothing$? [1]
(d) What total cost does a greedy "take the cheapest remaining pair" strategy give? Comment. [1]

### Q9 · Numerical · Hard · [5 marks]
An image has two ground-truth objects (normalised $(c_x,c_y,w,h)$):
$g_1$ = **cat**, $b_1=(0.30,0.40,0.20,0.30)$; $g_2$ = **dog**, $b_2=(0.70,0.60,0.30,0.40)$.
Three DETR queries (only these three are shown; classes are cat, dog and $\varnothing$):

| Query | $\hat p(\text{cat})$ | $\hat p(\text{dog})$ | $\hat p(\varnothing)$ | box $\hat b$ |
|---|---|---|---|---|
| $q_1$ | 0.60 | 0.10 | 0.30 | (0.31, 0.41, 0.20, 0.28) |
| $q_2$ | 0.15 | 0.70 | 0.15 | (0.68, 0.62, 0.30, 0.38) |
| $q_3$ | 0.85 | 0.05 | 0.10 | (0.40, 0.46, 0.24, 0.34) |

Pre-computed GIoU values (rows $q_1..q_3$, columns $g_1,g_2$): $\begin{bmatrix}0.84 & -0.48\\ -0.51 & 0.79\\ 0.19 & -0.31\end{bmatrix}$.

(a) Compute the L1 distance $\Vert b_i-\hat b_j\Vert_1$ for all six pairs. [1]
(b) Build the $3\times2$ matching-cost matrix ($\lambda_{L1}=5$, $\lambda_{giou}=2$). [2]
(c) Find the optimal matching and its cost. Is the query with the highest cat probability matched to the cat? Explain. [1]
(d) Give the classification-loss term (cross-entropy, $\varnothing$ weight 0.1) that each of the three queries receives. [1]

### Q10 · Numerical · Medium · [5 marks]
DETR runs on a $640\times480$ image ($W\times H$). Two queries output normalised boxes
$\hat b_1=(0.30,0.35,0.20,0.25)$ and $\hat b_2=(0.75,0.35,0.20,0.25)$ in $(c_x,c_y,w,h)$ format.
The ground-truth box is $G=(160,120,288,240)$ in pixel $(x_1,y_1,x_2,y_2)$ (no +1 in areas).

(a) Convert $\hat b_1$ and $\hat b_2$ to pixel $(x_1,y_1,x_2,y_2)$. [1]
(b) Compute IoU and GIoU between $\hat b_1$ and $G$. [2]
(c) Compute IoU and GIoU between $\hat b_2$ and $G$, and the GIoU loss $1-\text{GIoU}$ for both predictions. Why is GIoU (not IoU) used as a loss here? [2]

### Q11 · Numerical · Medium · [5 marks]
An $800\times1216$ ($H\times W$) image is fed to DETR (ResNet-50, C5 stride 32, encoder with 8 heads and 6 layers).

(a) How many tokens enter the encoder? How many attention weights does **one head of one encoder layer** compute? How many megabytes are needed to store the attention weights of **one layer (all 8 heads)** in fp32 (4 bytes, 1 MB = $10^6$ bytes)? [2]
(b) If the stride-8 map (C3) were used instead, how many tokens and attention weights per head per layer would there be, and by what factor does the attention cost grow? [1]
(c) Deformable DETR uses $M=8$ heads, $L=4$ levels and $K=4$ sampling points. The four levels for this image are $100\times152$, $50\times76$, $25\times38$ and $13\times19$. Compute the total number of encoder tokens, the number of sampled points per query, the total number of sampled points per encoder layer, and compare with dense multi-head (8 heads) attention over the same tokens. [2]

### Q12 · Numerical · Medium · [5 marks]
A detector's outputs for one class on a validation set (sorted by confidence) are matched to 8 ground-truth objects at IoU ≥ 0.5 (T = true positive, F = false positive):

| Score | 0.97 | 0.93 | 0.88 | 0.81 | 0.76 | 0.69 | 0.62 | 0.55 | 0.41 | 0.33 | 0.28 | 0.12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Match | T | T | F | T | T | T | F | T | F | F | T | F |

A detection is kept if score ≥ $\tau$.

(a) For $\tau\in\{0.9,0.7,0.5,0.3,0.1\}$ compute precision, recall and $F_1$. [3]
(b) Which threshold maximises $F_1$? [1]
(c) A safety application requires recall ≥ 0.85. Which of these thresholds would you deploy, and what precision do you pay? [1]

## Section C — Short answer

### Q13 · Short · Medium · [4 marks]
Explain why object detection can be treated as a **set prediction** problem and why this requires a **permutation-invariant** loss. How does DETR's bipartite matching provide such a loss, and why does it discourage duplicate detections?

### Q14 · Short · Medium · [4 marks]
State the role of each of the following in DETR: (a) encoder self-attention, (b) object queries, (c) decoder self-attention, (d) decoder cross-attention.

### Q15 · Short · Hard · [5 marks]
Explain why the original DETR (i) needs ~500 training epochs and (ii) performs poorly on small objects. Describe how **Deformable DETR** addresses both problems.

### Q16 · Short · Medium · [4 marks]
A detector evaluated on COCO-style data gives: $AP=38$, $AP_{50}=65$, $AP_{75}=37$, $AP_S=12$, $AP_M=42$, $AP_L=55$. A TIDE analysis shows the largest $\Delta AP$ for **localisation** and **missed** errors. Interpret these numbers and propose three concrete improvements. How would you choose the confidence threshold for deployment?

## Section D — Long answer / Design

### Q17 · Long · Hard · [10 marks]
**Compare two-stage, one-stage and DETR detectors.** Describe the pipeline of each (with a named example), and compare them on: region proposals, anchors, label assignment, NMS, hand-designed components, training time, inference speed, small objects and crowded scenes. Conclude with when you would choose each.

### Q18 · Long · Hard · [10 marks]
**Design question.** A warehouse robot must detect products on shelves from an onboard camera. Requirements: ≥ 30 FPS on an embedded GPU; many **small** items; **crowded**, overlapping items; new product categories are added every month. Propose a detection solution: choose and justify the detector family, explain how you handle small objects, crowded scenes and new categories, and describe how you would evaluate and tune the system before deployment.
