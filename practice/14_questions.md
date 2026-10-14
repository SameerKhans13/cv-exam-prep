# Session 14 — Vision++ with Foundation Models · Practice Questions

> Notes: [../notes/14_vision_plus_plus.md](../notes/14_vision_plus_plus.md) · Solutions: [14_solutions.md](14_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
In **Grounding DINO**, the word "DINO" refers to:

- **A.** the self-supervised self-distillation method whose features form the image backbone
- **B.** DINOv2, used as a frozen joint image–text encoder
- **C.** DETR with Improved deNoising anchOr boxes, a DETR-family object detector
- **D.** a distillation step that compresses Grounding DINO into a YOLO model

### Q2 · MCQ · Easy
An **open-vocabulary** object detector is one that:

- **A.** can detect categories specified at inference time as free text, because it classifies regions by comparing them with text embeddings
- **B.** detects only the 80 COCO classes but with higher recall than a standard detector
- **C.** finds every object in the image but cannot name any of them
- **D.** requires retraining a classification head whenever a new category is added

### Q3 · MCQ · Medium
In Grounding DINO's standard inference, what does **`box_threshold`** control?

- **A.** The IoU threshold used for NMS between overlapping boxes
- **B.** Which prompt tokens are attached to a kept box as its phrase label
- **C.** The minimum box area in pixels
- **D.** Which query predictions are kept, based on each query's highest similarity to any prompt token

### Q4 · MCQ · Medium
In a **Grounded-SAM** pipeline, what does SAM receive from Grounding DINO?

- **A.** Class labels that condition SAM's mask decoder on the category
- **B.** Bounding boxes, used as box prompts to produce one mask per detected object
- **C.** Text embeddings that replace SAM's image encoder
- **D.** Depth maps used to separate overlapping objects

### Q5 · MCQ · Medium
A Grounded-SAM + Depth Anything (standard relative-depth model) pipeline must report each dog's distance **in metres**. What else is required?

- **A.** Nothing — the Depth Anything output is already in metres
- **B.** A higher `box_threshold`, so that only confident dogs receive a depth
- **C.** A metric reference (a known-size object, calibrated depth points, or a metric fine-tuned depth model) to fix the scale and shift
- **D.** Running SAM with multi-mask output so that three depth values are available per dog

### Q6 · MCQ · Medium
Grounding DINO misses a dog that is partly hidden behind a sofa. What happens to that dog in a Grounded-SAM pipeline?

- **A.** It is missed by the whole pipeline — SAM only segments what it is prompted with
- **B.** SAM segments it anyway, because SAM always segments every object in the image
- **C.** SAM detects it and labels it "dog" from its mask shape
- **D.** SAM re-scores the image and raises the detector's recall for that dog

### Q7 · MCQ · Medium
What does **language-guided query selection** do in Grounding DINO?

- **A.** It selects which words of the prompt are nouns and discards the rest
- **B.** It chooses the backbone size (Swin-T or Swin-L) according to the prompt length
- **C.** It removes duplicate boxes after decoding, like NMS
- **D.** It selects the image tokens most similar to the text features to initialise the decoder queries

### Q8 · MCQ · Medium
A factory needs **60 FPS** detection of **5 fixed** part types on an edge device. It has 2,000 unlabelled images from its cameras and no annotations. What is the most appropriate approach?

- **A.** Run Grounding DINO + SAM on every frame on the edge device
- **B.** Auto-label the images with an open-vocabulary detector (e.g. Grounding DINO), review a sample by hand, and train a small closed-set real-time detector
- **C.** Use CLIP zero-shot classification on whole frames
- **D.** Use SAM automatic mask generation, which also outputs the part type of each mask

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
Stage times on a server GPU (given): Grounding DINO (Swin-T) **120 ms**; CLIP verification **5 ms per box**; SAM ViT-H image encoder **450 ms**; SAM prompt encoder + mask decoder **10 ms per box**; Depth Anything (ViT-L) **80 ms**. The image contains **3** dogs (3 boxes after thresholding).

(a) Total latency and FPS if all stages run **sequentially**. [1.5]
(b) The SAM image encoder and Depth Anything do not depend on the boxes, so on three GPUs they run **in parallel** with Grounding DINO + CLIP. The SAM decoder must wait for both the boxes and the SAM image embedding; depth-per-mask post-processing is negligible. Compute the critical-path latency and FPS. [1.5]
(c) SAM ViT-H is replaced by a light SAM variant whose encoder takes **40 ms** (decoder unchanged), still in the parallel setting of (b). New latency and FPS? Which branch is now the bottleneck? [1]
(d) After distillation, a YOLO-seg model (**12 ms**) and Depth-Anything-Small (**20 ms**) run on an edge GPU. FPS if run sequentially, and does it meet a 30 FPS requirement? FPS if they run in parallel? [1]

### Q10 · Numerical · Medium · [5 marks]
Grounding DINO is run with the prompt `"dog . person ."`. Six query predictions have the following sigmoid similarities to the tokens *dog* and *person* (separator tokens ignored). A box is kept if its **maximum** token score is **strictly greater** than `box_threshold`; its phrase is the set of tokens whose score is **strictly greater** than `text_threshold`.

| Box | dog | person |
|---|---|---|
| B1 | 0.62 | 0.08 |
| B2 | 0.31 | 0.12 |
| B3 | 0.15 | 0.48 |
| B4 | 0.40 | 0.28 |
| B5 | 0.36 | 0.05 |
| B6 | 0.22 | 0.33 |

(a) With `box_threshold = 0.35`, `text_threshold = 0.25`: which boxes are kept, with what phrase and score? [2]
(b) What changes with `text_threshold = 0.30` (box threshold 0.35)? [1]
(c) What changes with `box_threshold = 0.30`, `text_threshold = 0.30`? [0.5]
(d) Ground truth: 3 dogs and 1 person. B1, B5 and B4 each match a different GT dog (B4 is a real dog partly hidden behind the person), B3 matches the person, B2 is a duplicate of B1 (IoU 0.8 with B1, lower score), and B6 is a shop mannequin (no GT). A box counts as a *dog* prediction only if its phrase is exactly "dog"; each GT can be matched once. Compute precision and recall for the class *dog* under settings (a), (b) and (c). [1.5]

### Q11 · Numerical · Medium · [5 marks]
Box convention for IoU: $(x_1,y_1,x_2,y_2)$ in pixels, continuous coordinates, area $=(x_2-x_1)(y_2-y_1)$ (no +1).

(a) Grounding DINO outputs the normalised box $(c_x,c_y,w,h)=(0.50,\ 0.40,\ 0.30,\ 0.40)$ for an image of width $W=1000$ and height $H=800$ pixels. Convert it to $(x_1,y_1,x_2,y_2)$ pixels. [1]
(b) The ground-truth box is $(360,\ 180,\ 680,\ 500)$. Compute the IoU. Is it a true positive at IoU ≥ 0.5? At IoU ≥ 0.75? [1.5]
(c) On a test set of 200 images with **400** GT dogs, the zero-shot detector produces **340** detections of which **255** are TP at `box_threshold = 0.35`, and **220** detections of which **190** are TP at `box_threshold = 0.45`. Compute P, R and F1 for both. An application must **count** dogs: which threshold gives the smaller counting error? [1.5]
(d) A detector fine-tuned on labelled dogs produces 330 detections with 300 TP on the same set. Compute P, R, F1 and comment. [1]

### Q12 · Numerical · Hard · [5 marks]
A camera has focal length $f=1000$ px and principal point $c_x=500$ px. Depth Anything gives a relative inverse-depth map $r$ (larger = closer); model $1/Z=s\,r+t$.
A **door** (real height 2.0 m) appears 400 px tall; the median of $r$ inside the door's mask is 0.50. A **dog** mask has median $r=0.80$, pixel height 160 px and centre column $u=700$.

(a) Compute the door's depth from the pinhole model. Assuming $t=0$ (scale only), find $s$ and the dog's depth $Z$. [1.5]
(b) Estimate the dog's real height. [0.5]
(c) A laser rangefinder adds a second reference: a wall point at 10.0 m with $r=0.20$. Solve for $s$ and $t$ using both references and recompute the dog's depth and height. [1.5]
(d) Suppose the mean of $r$ over the dog's **bounding box** is used instead of the mask median, and 40% of the box pixels are background wall ($r=0.20$) while the rest have $r=0.80$. Using the calibration from (c), compute the resulting depth and the percentage error. [1]
(e) Using (c), compute the dog's lateral offset $X$ and its straight-line range (assume the dog's centre lies on the principal row, $Y\approx0$). [0.5]

### Q13 · Numerical · Medium · [5 marks]
A team wants instance masks for **50,000** images (≈3 objects per image).
- Grounded-SAM takes **0.8 s/image** on one GPU costing **$2.00/hour**.
- **20%** of images are flagged for human review at **6 s per image**; annotators cost **$15/hour**.
- Fully manual polygon annotation takes **60 s per object**.

(a) GPU time (hours) and cost of the auto-labelling run; wall-clock time on 4 GPUs. [1.5]
(b) Human review time and cost; total auto-labelling cost. [1.5]
(c) Cost of fully manual annotation and the cost ratio manual / auto. [1]
(d) The resulting YOLO-seg student runs at **8 ms/image**, Grounded-SAM at **800 ms/image**. Give the speed-up and the FPS of each. [1]

## Section C — Short answer

### Q14 · Short · Easy · [4 marks]
Distinguish closed-set, open-vocabulary and zero-shot detection. Explain why text embeddings make open-vocabulary recognition possible, and give the main trade-offs of zero-shot use versus few-shot adaptation and full fine-tuning.

### Q15 · Short · Medium · [4 marks]
Describe the architecture of Grounding DINO with a labelled diagram: backbones, feature enhancer, language-guided query selection and cross-modality decoder. State its inputs/outputs, the role of `box_threshold` and `text_threshold`, and what "DINO" stands for in its name.

### Q16 · Short · Medium · [4 marks]
Compare OWL-ViT and Grounding DINO: how image and text are fused, the kinds of queries supported, scalability to large vocabularies, accuracy and speed. When would you prefer each?

### Q17 · Short · Hard · [4 marks]
List the main failure modes of a text-prompted detection + segmentation pipeline (Grounding DINO + SAM) and give a concrete mitigation for each. Explain what "error propagation" means here and estimate the end-to-end success rate if detection recall is 0.90, the fraction of good masks is 0.95 and the fraction of acceptable depth estimates is 0.95 (assume independence).

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
**Design a system to find and segment all dogs in an image using text prompts and estimate their distance.** Include a pipeline diagram, the role and inputs/outputs of each model, how thresholds and prompts are chosen, how relative depth is converted to metres, how you would evaluate the system, its latency, and its main failure modes and mitigations. Briefly explain how you would extend it to video.

### Q19 · Long · Hard · [10 marks]
A logistics company wants real-time (≥ 25 FPS) detection of **forklifts, pallets and people without helmets** on 40 edge cameras, but has **no labelled data**. Design an approach that uses foundation models to bootstrap a deployable system: auto-labelling with Grounding DINO (and SAM if masks help), prompt and threshold design (including the hard "person without helmet" class), human quality control, distillation into a small detector, evaluation, and continuous improvement. Discuss costs, risks and why the foundation models are not deployed directly on the cameras.
