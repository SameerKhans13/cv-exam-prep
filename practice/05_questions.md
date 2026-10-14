# Session 05 — Embeddings & Image Similarity · Practice Questions

> Notes: [../notes/05_embeddings_similarity.md](../notes/05_embeddings_similarity.md) · Solutions: [05_solutions.md](05_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
In a Siamese network used for signature verification, the two branches:

- **A.** are two independently trained CNNs whose outputs are concatenated and classified
- **B.** share the same weights, so both inputs are mapped into the same embedding space
- **C.** share weights only in the first convolutional layer and are free afterwards
- **D.** are trained with cross-entropy over all known signers and never compare distances

### Q2 · MCQ · Easy
All gallery embeddings and the query embedding are L2-normalised. Ranking the gallery by **increasing Euclidean distance** to the query gives:

- **A.** the same order as ranking by increasing cosine similarity
- **B.** a different order in general, because Euclidean distance is sensitive to vector magnitude
- **C.** the same order only when the embedding dimension is small (D ≤ 3)
- **D.** the same order as ranking by decreasing cosine similarity

### Q3 · MCQ · Medium
In an InfoNCE / NT-Xent loss, the temperature is lowered from τ = 0.5 to τ = 0.05 with everything else unchanged. The softmax over the candidates becomes:

- **A.** sharper, so the loss concentrates on the hardest (most similar) negatives
- **B.** softer, so all negatives are weighted more equally
- **C.** unchanged, because τ cancels between numerator and denominator
- **D.** sharper for the positive only; the weights of the negatives are unaffected

### Q4 · MCQ · Medium
SimCLR is trained with a batch of N = 1024 images (two augmented views each). How many **negatives** does each anchor view have in its NT-Xent denominator?

- **A.** 1023
- **B.** 2047
- **C.** 2046
- **D.** 1024

### Q5 · MCQ · Medium
What is the main purpose of the **momentum (key) encoder** in MoCo?

- **A.** It computes gradients twice as fast, so training needs half the iterations
- **B.** It changes slowly (EMA of the query encoder), so keys stored in a large queue stay consistent and the number of negatives is decoupled from the batch size
- **C.** It removes the need for negatives entirely, exactly as in BYOL
- **D.** It is a separately trained network whose output is used as the downstream representation

### Q6 · MCQ · Medium
For an anchor–positive pair, d(a, p) = 0.40 and the triplet margin is m = 0.30. Which negative distance d(a, n) makes the triplet **semi-hard**?

- **A.** 0.30
- **B.** 0.80
- **C.** 0.20
- **D.** 0.60

### Q7 · MCQ · Medium
After SimCLR pre-training, which representation is normally used for downstream linear evaluation?

- **A.** The encoder output h, taken before the projection head
- **B.** The projection-head output z, because the loss was computed on it
- **C.** The concatenation of z from the two augmented views
- **D.** The row of the 2N × 2N similarity matrix belonging to the image

### Q8 · MCQ · Medium
Which approximate-nearest-neighbour technique mainly reduces **memory** by replacing each stored vector with a few bytes of codebook indices?

- **A.** HNSW (hierarchical navigable small-world graph)
- **B.** IVF (inverted file) storing the full float vectors
- **C.** Product quantisation (PQ)
- **D.** A brute-force flat index on a GPU

## Section B — Numericals

### Q9 · Numerical · Easy · [5 marks]
Embeddings $\mathbf u=(2,1,2)$ and $\mathbf v=(1,4,8)$.

(a) Compute the dot product, the cosine similarity and the Euclidean distance between $\mathbf u$ and $\mathbf v$. [2]
(b) L2-normalise both vectors. Compute the Euclidean distance between the unit vectors directly and check it with $\Vert\hat{\mathbf u}-\hat{\mathbf v}\Vert^2=2-2\cos\theta$. [1]
(c) A query $\mathbf q=(1,0)$ is compared with two **un-normalised** gallery vectors $\mathbf x_1=(10,1)$ and $\mathbf x_2=(0.5,0.5)$. Which gallery item is ranked first by cosine similarity, and which by Euclidean distance? Explain the disagreement. [1]
(d) Two unit-norm embeddings are at Euclidean distance 0.8. What are their cosine similarity and the angle between them? [1]

### Q10 · Numerical · Medium · [5 marks]
2-D embeddings (Euclidean distance $d$, not squared): anchor $a=(1,1)$, positive $p=(1.3,1.4)$, negatives $n_1=(1.6,1.8)$, $n_2=(1.36,0.52)$, $n_3=(0.76,1.32)$.

(a) Compute $d(a,p)$ and $d(a,n_i)$ for $i=1,2,3$. [1.5]
(b) Using the triplet loss $\mathcal L=\max(0,\ d(a,p)-d(a,n)+m)$ with $m=0.2$, compute the loss for each of the three triplets $(a,p,n_i)$ and classify each negative as **easy**, **semi-hard** or **hard**. [2]
(c) Using the contrastive loss $\mathcal L=y\,d^2+(1-y)\max(0,m-d)^2$ with $m=1.0$, compute the loss for the pair $(a,p)$ ($y=1$) and for each pair $(a,n_i)$ ($y=0$). [1.5]

### Q11 · Numerical · Medium · [5 marks]
A query embedding has cosine similarity 0.8 with its positive key and similarities 0.6, 0.2 and −0.4 with three negative keys. Use
$\mathcal L=-\log\frac{\exp(\mathrm{sim}(q,k^+)/\tau)}{\sum_{i=0}^{N}\exp(\mathrm{sim}(q,k_i)/\tau)}$ (natural log).

(a) Compute the probability assigned to the positive and the loss for $\tau=0.5$. [1.5]
(b) Repeat for $\tau=0.1$. [1.5]
(c) For each $\tau$, what fraction of the total probability assigned to the **negatives** falls on the hardest negative (sim 0.6)? What does this say about the role of $\tau$? [1]
(d) SimCLR with a batch of $N=256$ images: how many views, how many negatives per anchor, and how many entries in the similarity matrix? If the encoder **collapsed** so that all similarities were equal, what would the NT-Xent loss per anchor be? [1]

### Q12 · Numerical · Medium · [5 marks]
Two queries are run against an image database. 1 = relevant, 0 = not relevant (rank 1 first).

| Query | Ranked results | Total relevant images in DB ($R$) |
|---|---|---|
| A | 1, 0, 1, 1, 0, 0, 1, 0, 0, 0 (top-10) | 5 |
| B | 0, 1, 0, 0, 1 (top-5) | 2 |

Use **non-interpolated retrieval AP**: $\text{AP}=\frac1R\sum_k P@k\cdot\text{rel}(k)$ (relevant images never retrieved contribute 0).

(a) For query A compute $P@5$, $R@5$, $P@10$, $R@10$. [1.5]
(b) Compute $\text{AP}_A$ and $\text{AP}_B$, and the mAP over the two queries. [2.5]
(c) Compute the MRR over the two queries. [1]

### Q13 · Numerical · Medium · [5 marks]
A retrieval service stores **1,000,000** image embeddings of dimension **512**.

(a) Memory of a flat float32 index (in GB, 1 GB = $10^9$ B) and of a float16 index. [1]
(b) Product quantisation with $m=64$ sub-quantisers, 8 bits each: memory of the codes, compression ratio vs float32, and the size of the codebooks (each sub-quantiser has 256 float32 centroids of dimension 512/64). [1.5]
(c) Brute-force search: multiply-accumulates (MACs) per query, and the time per query and queries/s on a CPU sustaining 50 GMAC/s. [1.5]
(d) An IVF index has `nlist = 1024` cells and searches `nprobe = 8`. Assuming equally sized cells, approximately how many vectors are compared per query (include the comparisons with the centroids), and what is the speed-up over brute force? [1]

## Section C — Short answer

### Q14 · Short · Easy · [4 marks]
Prove that for unit vectors $\Vert\hat{\mathbf a}-\hat{\mathbf b}\Vert^2=2-2\cos\theta$. Give three reasons why embeddings are usually L2-normalised before retrieval or before computing a contrastive loss.

### Q15 · Short · Medium · [4 marks]
Compare four ways of obtaining image embeddings: penultimate-layer features of an ImageNet classifier, metric learning (Siamese/triplet), self-supervised contrastive learning, and language supervision (CLIP). For each state where its "positive pairs"/supervision come from, one strength and one weakness.

### Q16 · Short · Medium · [4 marks]
Explain SimCLR. Include a diagram, the role of the augmentations (which ones matter most and why), the projection head, the NT-Xent loss, and why it needs large batches.

### Q17 · Short · Hard · [4 marks]
What is **representation collapse** in self-supervised learning? Explain why the InfoNCE loss with negatives does not have collapse as a minimum, and how MoCo and BYOL each avoid collapse.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
Design a **visual product search** ("shop the look") system for an online retailer with **10 million** catalogue images. A user uploads a phone photo of a clothing item; the system must return the 20 most similar catalogue products in under 200 ms. Discuss: choice and training of the embedding model (data, loss, mining), the index (with a memory estimate), re-ranking, evaluation metrics, how the catalogue is kept up to date, and likely failure modes.

### Q19 · Long · Medium · [10 marks]
An office wants **face verification** at the entrance: an employee badge identifies the claimed person and a camera image must be verified against that person's enrolled photo(s). New employees join every week and must be added **without retraining**. Design the system using Siamese/triplet networks: architecture and embedding, training data and loss, negative mining, threshold selection and evaluation, enrolment of new employees, and risks/limitations. Compare the contrastive (pair) loss with the triplet loss for this use case.
