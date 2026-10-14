# Session 09 + 16 — Practicals, Revision & Exam Preparation · Practice Questions

> Notes: [../notes/09_revision_practicals.md](../notes/09_revision_practicals.md) · Solutions: [09_solutions.md](09_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min
>
> Conventions (as in the notes): confusion matrices have **rows = true class, columns = predicted class**; cosine similarity $\mathrm{sim}(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\Vert\mathbf a\Vert\,\Vert\mathbf b\Vert}$; in retrieval a gallery item is **relevant** if it has the query's class, and **AP** is the non-interpolated mean of the precisions at the ranks of the relevant items **divided by the total number $R$ of relevant gallery items**; logs are natural logs. Most questions span several sessions — this set is cross-course revision.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
A start-up must detect 12 fixed classes of road damage in dash-cam video at **≥ 30 fps on an embedded GPU**. It has 15,000 labelled boxes. Which detector family is the most appropriate first choice?

- **A.** R-CNN with selective-search proposals and per-class linear SVMs.
- **B.** A one-stage detector from the YOLO family, fine-tuned from pretrained weights.
- **C.** The original DETR (ResNet-50, 100 object queries) trained from scratch on the 12 classes.
- **D.** Grounding DINO prompted with the 12 class names, run on every frame.

### Q2 · MCQ · Easy
A leaf-disease classifier is tested on 1,000 images: 950 healthy and 50 diseased. Which evaluation best shows how useful the classifier is?

- **A.** Overall accuracy, because it uses all 1,000 test images.
- **B.** Top-5 accuracy, because it is more forgiving than top-1.
- **C.** The confusion matrix with per-class precision and recall, summarised by macro-F1 or balanced accuracy.
- **D.** Mean IoU between the predicted and true labels.

### Q3 · MCQ · Medium
A **calibrated** handheld camera ($K$ known) takes two photos while walking around a statue (a general 3-D scene; the camera translates). You want the relative rotation and translation direction from SIFT matches. Which model and minimal RANSAC sample should you use?

- **A.** A homography, estimated from 4-point samples.
- **B.** An affine transform, estimated from 3-point samples.
- **C.** A fundamental matrix from 8-point samples, then read $R$ and $\mathbf t$ directly from $F$ without using $K$.
- **D.** An essential matrix from 5-point samples, then decompose it into $R$ and $\mathbf t$ (up to scale) with a cheirality check.

### Q4 · MCQ · Medium
A dataset has 6,000 skin-lesion photos from 1,500 patients (several photos per lesion). Which protocol gives an **honest** test estimate?

- **A.** Split by patient ID (stratified by class) so all photos of a patient fall in one split, and fit any normalisation statistics on the training split only.
- **B.** Shuffle all 6,000 images and split 70/15/15 at image level, stratified by class.
- **C.** Make 5 augmented copies of every image, then split the enlarged set 70/15/15.
- **D.** Train on all the data and report the final-epoch accuracy on the training set as the test result.

### Q5 · MCQ · Medium
In the embedding practical a student uses **every gallery image as a query against the full gallery** and reports P@1 = 100% and mAP ≈ 0.98 for frozen ResNet-50 features on 50 plant species. What is the most likely explanation?

- **A.** ImageNet ResNet-50 features separate plant species perfectly, so the result is expected.
- **B.** Each query retrieves itself at rank 1 with cosine similarity 1; the self-match must be excluded before computing the metrics.
- **C.** L2-normalising the embeddings always inflates P@1.
- **D.** mAP should have been computed with 11-point interpolation, which would lower it to a realistic value.

### Q6 · MCQ · Medium
Which statement about a **t-SNE** plot of image embeddings is correct?

- **A.** t-SNE is a linear, deterministic projection, so two runs always give the same plot.
- **B.** If two classes appear as well-separated clusters, a linear probe on the original features must reach 100% accuracy on them.
- **C.** The gap between two clusters in the plot measures how different the two classes are in the original embedding space.
- **D.** Cluster sizes and gaps can change with perplexity and the random seed, so the plot is only a qualitative check and metrics must be computed in the original space.

### Q7 · MCQ · Medium
Which model–usage pairing works **out of the box**, with no extra model, labelled data or calibration?

- **A.** CLIP: classify images into new categories using only text prompts of the class names.
- **B.** DINOv2: zero-shot classification by comparing image features with class-name prompts.
- **C.** SAM: output a class label for every mask it produces in a street scene.
- **D.** Depth Anything: report the distance to a pedestrian in metres from a single image.

### Q8 · MCQ · Medium
A fruit-picking robot must outline **each individual apple** (apples often touch and overlap) and know which masks are apples and which are leaves. You have 3,000 images with instance masks for both classes. Which model fits best?

- **A.** U-Net trained for semantic segmentation.
- **B.** FCN-8s trained for semantic segmentation.
- **C.** Mask R-CNN fine-tuned on the instance masks.
- **D.** SAM in automatic "segment everything" mode, with no other model.

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
A leaf-disease classifier is evaluated on 120 test leaves (rows = true, columns = predicted):

| True \ Pred | Healthy | Rust | Blight |
|---|---|---|---|
| **Healthy** | 70 | 5 | 5 |
| **Rust** | 4 | 12 | 4 |
| **Blight** | 2 | 3 | 15 |

(a) Overall accuracy. **[0.5]**
(b) Precision, recall and F1 for each class. **[2]**
(c) Macro-precision, macro-recall, macro-F1 and micro-F1. Explain why micro-F1 equals the accuracy here. **[1.5]**
(d) Weighted-F1 (weights = class support) and balanced accuracy. **[0.5]**
(e) A trivial model predicts "Healthy" for every leaf. Give its accuracy, macro-F1 and balanced accuracy, and state what this comparison shows. **[0.5]**

### Q10 · Numerical · Medium · [5 marks]
Two queries are run against a gallery of plant photos using frozen DINOv2 embeddings and cosine kNN (each query is **not** in the gallery). Relevance of the top-10 results (1 = same species as the query):

| Query | Relevant items in the whole gallery $R$ | Ranks 1 → 10 |
|---|---|---|
| A (fern) | 4 | 1 0 1 1 0 0 0 1 0 0 |
| B (moss) | 3 | 0 0 1 0 1 0 0 0 0 0 |

(a) P@5, R@5, P@10 and R@10 for each query. **[1.5]**
(b) AP for each query and the mAP. **[2]**
(c) Hit-rate@1, hit-rate@5 (fraction of queries with at least one relevant item in the top $k$) and MRR over the two queries. **[1]**
(d) A classmate divides each query's sum of precisions by the number of relevant items **actually retrieved** in the top 10. What mAP would they report, and why is it wrong? **[0.5]**

### Q11 · Numerical · Medium · [5 marks]
Toy 2-D embeddings: a query leaf $\mathbf q=(3,4)$ and a labelled gallery:

| Item | Embedding | Class |
|---|---|---|
| m1 | (6, 8) | Maple |
| m2 | (1, 1) | Maple |
| m3 | (9, 13) | Maple |
| o1 | (4, 3) | Oak |
| o2 | (2, 6) | Oak |
| b1 | (5, 0) | Birch |

(a) L2-normalise $\mathbf q$ and compute the cosine similarity between $\mathbf q$ and every gallery item. **[2]**
(b) Classify $\mathbf q$ with a $k=3$ cosine kNN (majority vote). **[1]**
(c) Classify $\mathbf q$ with a $k=3$ kNN using Euclidean distance on the **raw, unnormalised** vectors. **[1]**
(d) Explain the disagreement and state which result you would trust for CNN embeddings. Show that Euclidean distance between **L2-normalised** vectors reproduces the cosine ranking (compute it for o1). **[1]**

### Q12 · Numerical · Medium · [5 marks]
For a 50-species plant classifier you compare adaptation strategies. An ImageNet ResNet-50 **without its fc layer** has 23,508,032 parameters and outputs a 2048-d feature; its last stage `layer4` has 14,964,736 parameters. A ViT-B/16 **without its head** has 85,798,656 parameters and a 768-d [CLS] feature. All new heads are linear layers with bias.

(a) Linear probe on ResNet-50: trainable parameters and trainable percentage of the whole model. **[1]**
(b) Fine-tuning `layer4` + head: trainable parameters and percentage. **[1]**
(c) Full fine-tuning: trainable parameters. **[0.5]**
(d) Training with AdamW in fp32 stores, per **trainable** parameter, the weight, its gradient and two moment buffers (4 bytes each); a **frozen** parameter needs only its weight. Estimate the memory for parameters + gradients + optimiser state (ignore activations) for full fine-tuning and for the linear probe with the frozen backbone on the GPU. **[1.5]**
(e) Trainable parameters of a linear probe on the ViT-B/16 [CLS] feature and of a kNN classifier; memory to store a kNN gallery of 14,000 ViT [CLS] embeddings in float32. **[1]**

### Q13 · Numerical · Medium · [5 marks]
CLIP zero-shot classification of 5 flower photos into [rose, tulip, daisy, sunflower] with prompts "a photo of a {class}, a type of flower." The cosine similarities between each image embedding and each class prompt embedding are:

| Image (true class) | rose | tulip | daisy | sunflower |
|---|---|---|---|---|
| I1 (rose) | 0.31 | 0.28 | 0.22 | 0.20 |
| I2 (tulip) | 0.29 | 0.27 | 0.21 | 0.19 |
| I3 (daisy) | 0.20 | 0.22 | 0.30 | 0.26 |
| I4 (sunflower) | 0.19 | 0.21 | 0.27 | 0.25 |
| I5 (daisy) | 0.24 | 0.23 | 0.22 | 0.28 |

(a) Predicted class of each image; top-1 and top-2 accuracy. **[1.5]**
(b) Per-class recall and balanced accuracy. **[1]**
(c) Softmax class probabilities for I2 using CLIP's logit scale of 100. Comment on the result. **[1.5]**
(d) The same probabilities for I2 with a logit scale of 1. What does this show about the learned temperature? **[0.5]**
(e) Give two ways to improve accuracy on this task without fine-tuning the CLIP encoders. **[0.5]**

## Section C — Short answer

### Q14 · Short · Medium · [4 marks]
Compare **ImageNet-supervised ResNet-50**, **self-supervised DINOv2** and **language-supervised CLIP** features as the embedding for a plant-photo similarity-search tool. Cover the supervision signal, labels needed, support for text queries, fine-grained similarity and one weakness of each, and end with a recommendation.

### Q15 · Short · Medium · [4 marks]
A delivery robot needs the distance to obstacles in front of it. Compare **stereo depth**, **monocular foundation depth (Depth Anything)**, **Structure from Motion** and **SLAM** for this need (input, scale, timing, failure modes) and recommend a sensing design.

### Q16 · Short · Medium · [4 marks]
Choose between **FCN**, **U-Net**, **Mask R-CNN** and **SAM** for each scenario and justify in one or two lines:
(i) segmenting tumours in 200 labelled MRI slices;
(ii) counting and outlining touching apples (with class labels) given 3,000 instance-labelled images;
(iii) an annotation tool in which a person clicks once on any object to get its mask;
(iv) why FCN is rarely the first choice today, although it is historically important.

### Q17 · Short · Medium · [4 marks]
Compare **autoencoders, VAEs, GANs and diffusion models** on training objective, sample quality, diversity, sampling speed and training stability. Then pick one for each task and justify briefly: (i) detecting defective parts when only images of good parts are available; (ii) real-time 4× super-resolution on a phone; (iii) text-to-image generation for marketing images.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
A retailer wants to recognise which of **2,000 packaged products** are on supermarket shelves from staff phone photos, and to flag empty gaps. Compare three approaches —
(1) a **classical local-feature pipeline** (SIFT/ORB matching against reference packshots with RANSAC verification),
(2) a **CNN-based pipeline** (detector + fine-tuned classifier or embedding),
(3) a **foundation-model pipeline** (open-vocabulary detection and/or SAM, with CLIP or DINOv2 embeddings) —
in terms of **accuracy, data needs, compute, robustness and explainability**. Recommend and justify a final system, including how you would evaluate it.

### Q19 · Long · Hard · [10 marks]
You must build (1) a **classifier** and (2) a **similarity-search tool** for **20,000 images of 50 plant species**; **30% (6,000) of the images are unlabelled**. The images come from about 2,000 individual plants, each photographed several times. Design the full solution: data splitting, model choice, how you use the unlabelled images, the training procedure, how the search index is built, evaluation metrics for **both** tools, and an error-analysis plan. Include a pipeline diagram.
