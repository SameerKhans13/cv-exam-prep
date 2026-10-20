# Computer Vision: Exam-Prep Study Pack

**Live site:** [imajij.github.io/cv-exam-prep](https://imajij.github.io/cv-exam-prep/) · **Quiz:** [quiz.html](https://imajij.github.io/cv-exam-prep/quiz.html) · **Golden Traps MCQs:** [mcq_golden_traps.html](mcq_golden_traps.html) · **Numericals Lab:** [numericals_formula.html](numericals_formula.html)

Detailed notes, practice questions with full solutions, two mock exams, a numericals drill, an interactive quiz and runnable code for the 16-session Computer Vision course. Built for **exam answers**: every formula has its symbols defined, every numerical shows full working, and every MCQ explains why each wrong option is wrong.

| What | Amount |
|---|---|
| Notes | 15 files (Sessions 1–15, with 9 and 16 merged into a revision file), ~9,300 lines, 74 Mermaid diagrams |
| Per-session practice | **284 questions**: 119 MCQ · 75 numerical · 60 short · 30 long/design, all with solutions |
| Mock exams | 2 full 3-hour papers (100 marks each) with marking schemes |
| Numericals drill | **67** calculation problems with full working + answer key |
| Quiz | [`quiz.html`](quiz.html): **119 MCQs**, offline, no dependencies |
| Golden Traps MCQs | [`mcq_golden_traps.html`](mcq_golden_traps.html): 10 high-yield Section A questions with explanations and next-step links |
| Numericals Lab | [`numericals_formula.html`](numericals_formula.html): visual formula cards, interactive checkers and fast Python snippets for the five high-yield Section B calculations |
| Code | 7 short runnable scripts ([`code/`](code/README.md)) |

> **Exam format assumed.** The format and date weren't filled in, so the mock exams and timings assume
> **3 hours, 100 marks: A 20 MCQs × 1 · B 6 numericals × 5 · C short answers (attempt 5 of 6) × 4 · D long/design (attempt 3 of 4) × 10.**
> If your real paper differs, the per-session question sets still apply. Just re-weight your time to match where the marks are.

---

## 1. What's where

```
cv-exam-prep/
├── README.md                  ← you are here: how to use, revision plans, coverage checklist
├── 00_formula_sheet.md        ← every formula, one table per session (last-minute sheet)
├── 00_comparison_tables.md    ← every "X vs Y" (94 tables, each with an exam verdict)
├── notes/                     ← 01 … 15, one per session (09 = Sessions 9 + 16 revision)
├── practice/
│   ├── NN_questions.md / NN_solutions.md   ← separate files so you can self-test
│   ├── numericals_drill.md                 ← all calculation types in one place
│   └── mock_exam_{1,2}.md (+ _solutions)
├── code/                      ← runnable demos of the practicals + a calculator for checking answers
└── quiz.html                  ← open in any browser
```

| # | Session | Notes | Practice |
|---|---|---|---|
| 1 | Image Processing & Features | [notes](notes/01_image_processing_features.md) | [Q](practice/01_questions.md) · [A](practice/01_solutions.md) |
| 2 | Feature Matching & Classical CV | [notes](notes/02_feature_matching_classical.md) | [Q](practice/02_questions.md) · [A](practice/02_solutions.md) |
| 3 | CNNs Under the Hood | [notes](notes/03_cnns_under_the_hood.md) | [Q](practice/03_questions.md) · [A](practice/03_solutions.md) |
| 4 | Training CNNs & Transfer Learning | [notes](notes/04_training_transfer_learning.md) | [Q](practice/04_questions.md) · [A](practice/04_solutions.md) |
| 5 | Embeddings & Image Similarity | [notes](notes/05_embeddings_similarity.md) | [Q](practice/05_questions.md) · [A](practice/05_solutions.md) |
| 6 | Object Detection I: Two-Stage | [notes](notes/06_detection_two_stage.md) | [Q](practice/06_questions.md) · [A](practice/06_solutions.md) |
| 7 | Object Detection II: One-Stage | [notes](notes/07_detection_one_stage.md) | [Q](practice/07_questions.md) · [A](practice/07_solutions.md) |
| 8 | Object Detection III: Transformers | [notes](notes/08_detection_transformers.md) | [Q](practice/08_questions.md) · [A](practice/08_solutions.md) |
| 9 + 16 | Practicals, Quiz/Project lessons, Revision & Compare Approaches | [notes](notes/09_revision_practicals.md) | [Q](practice/09_questions.md) · [A](practice/09_solutions.md) |
| 10 | Image Segmentation | [notes](notes/10_image_segmentation.md) | [Q](practice/10_questions.md) · [A](practice/10_solutions.md) |
| 11 | Camera Geometry | [notes](notes/11_camera_geometry.md) | [Q](practice/11_questions.md) · [A](practice/11_solutions.md) |
| 12 | 3D Vision & Motion | [notes](notes/12_3d_vision_motion.md) | [Q](practice/12_questions.md) · [A](practice/12_solutions.md) |
| 13 | Vision Foundation Models | [notes](notes/13_vision_foundation_models.md) | [Q](practice/13_questions.md) · [A](practice/13_solutions.md) |
| 14 | Vision++ with Foundation Models | [notes](notes/14_vision_plus_plus.md) | [Q](practice/14_questions.md) · [A](practice/14_solutions.md) |
| 15 | Generative Computer Vision | [notes](notes/15_generative_cv.md) | [Q](practice/15_questions.md) · [A](practice/15_solutions.md) |
| ★ | Mock exams | [Paper 1](practice/mock_exam_1.md) · [Paper 2](practice/mock_exam_2.md) | [Sol 1](practice/mock_exam_1_solutions.md) · [Sol 2](practice/mock_exam_2_solutions.md) |

Every notes file has the same 8 sections, in order: **1** Big picture · **2** Core concepts (one subsection per syllabus bullet) · **3** Key formulas with worked examples · **4** Algorithms step by step · **5** Diagrams · **6** Comparison tables · **7** Common exam traps · **8** 10-line cheat summary.

## 2. How to use the pack

**For each session (about 2–3 hours):**
1. Read **§1 Big picture** and **§8 cheat summary** first (5 min) so you know where you're going.
2. Read **§2–§4** properly. Rewrite each formula in §3 from memory once.
3. Do the **questions file closed-book**. MCQs take about 1 min each, numericals about 6 min, short answers about 8 min. Write long answers as bullet plans only (about 5 min).
4. Mark with the **solutions file**. Copy every miss into a mistakes log (one line: *what I thought → what's true*).
5. Read **§7 traps**. Most MCQ distractors come from there.
6. Open [`quiz.html`](quiz.html), filter to the session, and answer. Later, use **Show → My mistakes** to re-drill only what you got wrong. Progress is saved in your browser.

**Across sessions:**
- [`00_formula_sheet.md`](00_formula_sheet.md): write it out from memory every few days, then diff.
- [`00_comparison_tables.md`](00_comparison_tables.md): "Compare X and Y" questions are often 10 marks. Learn each table's **exam verdict** line.
- [`practice/numericals_drill.md`](practice/numericals_drill.md): Section B (30 marks) is the most reliable place to gain marks. Do the drill twice.
- [`code/`](code/README.md): run the scripts once to make the practicals concrete. `conv_output_and_receptive_field_calc.py` and `iou_nms_ap.py` double as **answer checkers** for your numericals.
- **Mock exams:** do them **timed (3 h) and closed-book**, then mark strictly with the marking schemes. Their MCQs are deliberately *not* in the quiz, so the papers stay unseen.

**Where the marks are.** Numerical types that come up again and again:
- conv output size, params and receptive field
- RANSAC N
- homography mapping
- IoU, NMS and AP from a ranked list
- Dice/IoU and mIoU
- projection x = K[R|t]X, with and without distortion
- stereo Z = fB/d and ΔZ
- Lucas–Kanade 2×2 solve
- Harris R
- CLIP softmax, InfoNCE and triplet loss
- DDPM ᾱ and CFG

Detection (Sessions 6–8) spans three sessions, so expect it to carry the most weight.

## 3. Revision plan

The exam date wasn't given, so pick the plan that matches the **days you have left** and count back from your exam date. Each "day" is about 5–6 focused hours.

### 14 days or more (full plan)

| Day | Focus | Output |
|---|---|---|
| 1 | S01 + S02 (classical features, matching, RANSAC, homography) | Q&A done, mistakes logged |
| 2 | S03 + S04 (CNN arithmetic, architectures, training, transfer) | conv calculator practice |
| 3 | S05 + S13 (embeddings and contrastive learning → CLIP, DINO, SAM, Depth Anything, OWL-ViT) | |
| 4 | S06 + S07 (two-stage → one-stage, IoU/NMS/AP) | AP-from-ranked-list practised twice |
| 5 | S08 + S14 (DETR → Grounding DINO, pipelines) | the 3-way detector comparison written out |
| 6 | S10 + S15 (segmentation; generative) | |
| 7 | S11 + S12 (camera geometry; stereo, flow, SfM, SLAM) | projection and stereo numericals fluent |
| 8 | **Mock exam 1**, timed; mark it | score + list of weak sessions |
| 9 | Re-study the weakest 2–3 sessions; numericals drill (first half) | |
| 10 | Numericals drill (second half); formula sheet from memory | |
| 11 | S09/16 revision notes + comparison tables; write 3 long answers under time | |
| 12 | **Mock exam 2**, timed; mark it | |
| 13 | Fix weak areas; quiz → *My mistakes*; redo every wrong numerical | |
| 14 | Light day: all 15 cheat summaries, formula sheet, traps sections; sleep early | |

### 7 days

| Day | Focus |
|---|---|
| 1 | S01, S02, S03 (notes + numericals + MCQs; skip long answers) |
| 2 | S04, S05, S06, S07 |
| 3 | S08, S10, S11, S12 |
| 4 | S13, S14, S15 + S09 revision notes |
| 5 | **Mock 1** timed + marking; numericals drill (odd-numbered problems) |
| 6 | **Mock 2** timed + marking; fix weak sessions; quiz → *My mistakes* |
| 7 | Cheat summaries + formula sheet + comparison-table verdicts; light quiz; sleep |

### 3 days (triage)

| Day | Focus |
|---|---|
| 1 | Numericals first: notes §3 of S03, S07, S10, S11, S12, then the [drill](practice/numericals_drill.md). Learn [`00_formula_sheet.md`](00_formula_sheet.md). |
| 2 | Concepts: all 15 **cheat summaries** + **traps** sections; quiz on all sessions; the comparison tables for detection, segmentation, foundation models and generative models |
| 3 | One **mock exam**, timed, then mark it. Re-read the model answers to every long question in that mock. Evening: formula sheet once more, then rest. |

### Night before and exam morning
- Read the formula sheet once, all 15 §8 cheat summaries, and your mistakes log. Nothing new.
- Know the conventions cold:
  - boxes $(x_1,y_1,x_2,y_2)$ with no +1
  - RANSAC N rounds **up**
  - DOF: H = 8, F = 7, E = 5, P = 11, K = 5
  - Dice ≥ IoU
  - a higher NMS threshold keeps **more** boxes
- In the exam, do Section B early while you're fresh. For every numerical write: formula → symbols → substitution → units → a sanity check.

## 4. Notation and conventions used throughout

Full list at the top of [`00_formula_sheet.md`](00_formula_sheet.md). The essentials:

- **Conv output** $\lfloor (W-K+2P)/S \rfloor+1$. **Params** $(K^2C_{in}+1)C_{out}$. **Receptive field** $r_l=r_{l-1}+(K_l-1)j_{l-1}$, $j_l=j_{l-1}S_l$.
- **Projection** $\lambda\tilde{\mathbf x}=K[R\mid\mathbf t]\tilde{\mathbf X}$ with $X_c=RX_w+\mathbf t$. **Stereo** $Z=fB/d$.
- **Boxes** are $(x_1,y_1,x_2,y_2)$, continuous, area $=(x_2-x_1)(y_2-y_1)$. **AP** defaults to all-point interpolation, and each question says which method it uses.
- **Confusion matrices:** rows are ground truth, columns are predictions.

A few results depend on the convention used, and the notes flag this where it matters:
- **Plane-induced homography:** $R-\mathbf t\mathbf n^\top/d$ for the plane $\mathbf n^\top\mathbf X+d=0$, but $R+\mathbf t\mathbf n^\top/d$ for $\mathbf n^\top\mathbf X=d$.
- **CFG weight:** $w=1$ means plain conditional. Some papers write $1+w$ instead.
- **11-point AP vs all-point AP.**
- **Retrieval AP vs detection AP.**

## 5. How accuracy was checked

- **Every numerical was checked twice.** The author ran it in Python. Then a separate auditor re-derived it from scratch in fresh code, including rounding. This covers all 75 per-session numericals, both mocks' Section B and the drill. The audit found and fixed a handful of small slips: a rounding error, a tensor shape in a notes example, an API call shape, and some wording.
- **Every MCQ was checked for consistency.** The key, the "why wrong" explanations, and the agreement between the questions file, solutions file and quiz were all checked.
- **All files were validated mechanically:**
  - all LaTeX parses (KaTeX)
  - all 111 Mermaid diagrams parse and are laid out to stay readable at page width
  - every relative link and `#anchor` resolves
  - `quiz.html` passes an automated browser test (answering, filters, shuffle, saved progress, mobile layout, light and dark)
- **All 7 scripts run** on OpenCV 5.0 / Python 3.9 and 3.11, and their outputs match hand calculations and `torchvision`.
- **Facts stated with care.** Hardware-dependent numbers (FPS, latencies) are given as orders of magnitude or marked "given" in the questions.

If you find an error, the matching `practice/NN_solutions.md` has the full working, so you can trace it quickly.

## 6. Syllabus coverage checklist

Every syllabus bullet and every must-cover item from the brief, each linked to where it's taught. Generated by a script that checks for the actual content (formulas, terms), not just headings.

**S01 Image Processing & Features**

- [x] [Image representation](notes/01_image_processing_features.md#image-representation)
- [x] [Filtering & convolution](notes/01_image_processing_features.md#filtering-and-convolution)
- [x] [Edge detection / Canny](notes/01_image_processing_features.md#edge-detection-and-canny)
- [x] [Harris corners](notes/01_image_processing_features.md#harris-corners)
- [x] [Keypoints and descriptors](notes/01_image_processing_features.md#keypoints-and-descriptors)
- [x] [SIFT, SURF, ORB](notes/01_image_processing_features.md#sift)
- [x] [Harris structure tensor + response + eigenvalue cases](notes/01_image_processing_features.md#harris-corners) *(must-cover detail)*
- [x] [SIFT: DoG scale space, orientation assignment, 128-d](notes/01_image_processing_features.md#sift) *(must-cover detail)*
- [x] [SURF: integral images + box filters](notes/01_image_processing_features.md#surf) *(must-cover detail)*
- [x] [ORB: FAST + rBRIEF, Hamming](notes/01_image_processing_features.md#orb) *(must-cover detail)*

**S02 Feature Matching & Classical CV**

- [x] [Feature matching](notes/02_feature_matching_classical.md#feature-matching)
- [x] [Distance metrics](notes/02_feature_matching_classical.md#distance-metrics)
- [x] [RANSAC](notes/02_feature_matching_classical.md#ransac)
- [x] [Homography](notes/02_feature_matching_classical.md#homography)
- [x] [Panorama stitching](notes/02_feature_matching_classical.md#panorama-stitching)
- [x] [Where classical CV works / fails](notes/02_feature_matching_classical.md#where-classical-cv-works-and-fails)
- [x] [Practical: SIFT/ORB matching + panorama](notes/02_feature_matching_classical.md#practical-sift-and-orb-matching-with-panorama-in-opencv)
- [x] [Lowe's ratio test](notes/02_feature_matching_classical.md#feature-matching) *(must-cover detail)*
- [x] [RANSAC N = log(1-p)/log(1-(1-e)^s)](notes/02_feature_matching_classical.md#ransac) *(must-cover detail)*
- [x] [Homography 8 DOF, DLT, >= 4 points, planar / pure rotation](notes/02_feature_matching_classical.md#homography) *(must-cover detail)*

**S03 CNNs Under the Hood**

- [x] [Why CNNs?](notes/03_cnns_under_the_hood.md#why-cnns)
- [x] [Convolution & feature maps](notes/03_cnns_under_the_hood.md#convolution-and-feature-maps)
- [x] [Receptive field & pooling](notes/03_cnns_under_the_hood.md#receptive-field-and-pooling)
- [x] [Hierarchical features](notes/03_cnns_under_the_hood.md#hierarchical-features)
- [x] [CNN architecture evolution](notes/03_cnns_under_the_hood.md#cnn-architecture-evolution)
- [x] [ResNet intuition](notes/03_cnns_under_the_hood.md#resnet-intuition)
- [x] [Conv output size, parameter count, RF recurrence](notes/03_cnns_under_the_hood.md#receptive-field-and-pooling) *(must-cover detail)*

**S04 Training CNNs & Transfer Learning**

- [x] [Overfitting & regularization](notes/04_training_transfer_learning.md#overfitting-and-regularisation)
- [x] [Data augmentation](notes/04_training_transfer_learning.md#data-augmentation)
- [x] [Pretrained models](notes/04_training_transfer_learning.md#pretrained-models)
- [x] [Feature extraction vs fine-tuning](notes/04_training_transfer_learning.md#feature-extraction-vs-fine-tuning)
- [x] [Practical image classification](notes/04_training_transfer_learning.md#practical-image-classification-pipeline)

**S05 Embeddings & Image Similarity**

- [x] [What is an embedding?](notes/05_embeddings_similarity.md#what-is-an-embedding)
- [x] [Learned representations](notes/05_embeddings_similarity.md#learned-representations)
- [x] [Similarity metrics & image retrieval](notes/05_embeddings_similarity.md#similarity-metrics)
- [x] [Siamese networks](notes/05_embeddings_similarity.md#siamese-networks)
- [x] [Contrastive learning](notes/05_embeddings_similarity.md#contrastive-learning)
- [x] [Contrastive, triplet and InfoNCE losses](notes/05_embeddings_similarity.md#contrastive-learning) *(must-cover detail)*

**S06 Object Detection I: Two-Stage**

- [x] [Classification vs localization vs detection](notes/06_detection_two_stage.md#21-classification-vs-localisation-vs-detection)
- [x] [Sliding windows & HOG + classifier](notes/06_detection_two_stage.md#22-sliding-windows-and-hog--classifier)
- [x] [Region proposals & R-CNN](notes/06_detection_two_stage.md#23-region-proposals-and-r-cnn)
- [x] [Fast R-CNN & Faster R-CNN](notes/06_detection_two_stage.md#24-fast-r-cnn-and-faster-r-cnn)
- [x] [Two-stage detector philosophy](notes/06_detection_two_stage.md#25-two-stage-detector-philosophy)

**S07 Object Detection II: One-Stage**

- [x] [YOLO](notes/07_detection_one_stage.md#21-yolo)
- [x] [SSD](notes/07_detection_one_stage.md#22-ssd)
- [x] [Anchor boxes & anchor-free detection](notes/07_detection_one_stage.md#23-anchor-boxes-and-anchor-free-detection)
- [x] [NMS & IoU](notes/07_detection_one_stage.md#25-iou-and-nms)
- [x] [Precision / Recall & AP / mAP](notes/07_detection_one_stage.md#26-precision-recall-ap-and-map)
- [x] [Focal loss; AP interpolation; mAP@0.5 vs COCO mAP@[.5:.95]](notes/07_detection_one_stage.md#26-precision-recall-ap-and-map) *(must-cover detail)*

**S08 Object Detection III: Transformers**

- [x] [DETR & object queries](notes/08_detection_transformers.md#detr-architecture-and-object-queries)
- [x] [Set prediction](notes/08_detection_transformers.md#set-prediction-hungarian-matching-and-the-set-loss)
- [x] [Two-stage vs one-stage vs DETR](notes/08_detection_transformers.md#two-stage-vs-one-stage-vs-detr)
- [x] [Modern detection approaches](notes/08_detection_transformers.md#modern-detection-approaches)
- [x] [Practical detector analysis](notes/08_detection_transformers.md#practical-detector-analysis)
- [x] [Why no NMS; drawbacks; Deformable DETR](notes/08_detection_transformers.md#why-detr-needs-no-nms-its-drawbacks-and-the-deformable-detr-fix) *(must-cover detail)*

**S09 Quiz/Project/Practicals + Exam Preparation (Sessions 9 & 16)**

- [x] [Practical classification / embedding exercise](notes/09_revision_practicals.md#practical-classification-exercise)
- [x] [Comprehensive revision & compare approaches](notes/09_revision_practicals.md#compare-approaches)

**S10 Image Segmentation**

- [x] [Semantic vs instance vs panoptic](notes/10_image_segmentation.md#semantic-vs-instance-vs-panoptic-segmentation)
- [x] [FCN, U-Net, Mask R-CNN](notes/10_image_segmentation.md#fcn-fully-convolutional-networks)
- [x] [IoU / Dice metrics](notes/10_image_segmentation.md#iou-and-dice-metrics)
- [x] [Modern segmentation overview](notes/10_image_segmentation.md#modern-segmentation-overview)
- [x] [Lab: run pretrained segmentation model](notes/10_image_segmentation.md#lab-running-a-pretrained-segmentation-model)
- [x] [U-Net concat skips; RoIAlign vs RoI Pool; Dice = 2IoU/(1+IoU)](notes/10_image_segmentation.md#mask-r-cnn) *(must-cover detail)*

**S11 Camera Geometry**

- [x] [Pinhole camera & perspective projection](notes/11_camera_geometry.md#pinhole-camera-and-perspective-projection)
- [x] [Intrinsic & extrinsic parameters](notes/11_camera_geometry.md#intrinsic-parameters)
- [x] [Camera calibration & distortion](notes/11_camera_geometry.md#camera-calibration)
- [x] [Homography revisited](notes/11_camera_geometry.md#homography-revisited)
- [x] [x = K[R|t]X](notes/11_camera_geometry.md#pinhole-camera-and-perspective-projection) *(must-cover detail)*

**S12 3D Vision & Motion**

- [x] [Stereo vision & depth from disparity](notes/12_3d_vision_motion.md#stereo-vision-and-depth-from-disparity)
- [x] [Optical flow & tracking](notes/12_3d_vision_motion.md#optical-flow)
- [x] [Structure from Motion](notes/12_3d_vision_motion.md#structure-from-motion-sfm)
- [x] [SLAM conceptual overview](notes/12_3d_vision_motion.md#slam-conceptual-overview)
- [x] [Epipolar geometry; E (5 DOF) vs F (7 DOF); rectification](notes/12_3d_vision_motion.md#epipolar-geometry) *(must-cover detail)*

**S13 Vision Foundation Models**

- [x] [What is a foundation model?](notes/13_vision_foundation_models.md#what-is-a-foundation-model)
- [x] [CLIP](notes/13_vision_foundation_models.md#clip)
- [x] [DINO / DINOv2](notes/13_vision_foundation_models.md#dino-and-dinov2)
- [x] [Segment Anything](notes/13_vision_foundation_models.md#segment-anything-sam)
- [x] [Depth Anything](notes/13_vision_foundation_models.md#depth-anything)
- [x] [OWL-ViT](notes/13_vision_foundation_models.md#owl-vit)

**S14 Vision++ with Foundation Models**

- [x] [Zero-shot vision](notes/14_vision_plus_plus.md#zero-shot-vision)
- [x] [Grounding DINO](notes/14_vision_plus_plus.md#grounding-dino)
- [x] [Detection + segmentation + depth pipelines](notes/14_vision_plus_plus.md#detection--segmentation--depth-pipelines)

**S15 Generative Computer Vision**

- [x] [Autoencoders & GANs (and VAEs)](notes/15_generative_cv.md#autoencoders)
- [x] [Diffusion models](notes/15_generative_cv.md#diffusion-models-ddpm)
- [x] [Image generation / editing](notes/15_generative_cv.md#image-generation-and-editing-controlnet-inpainting-img2img-personalisation)
- [x] [Brief modern applications](notes/15_generative_cv.md#brief-modern-applications-and-ethics)
- [x] [Latent diffusion, CFG, ControlNet, inpainting](notes/15_generative_cv.md#latent-diffusion-and-stable-diffusion) *(must-cover detail)*

**Also verified across all 15 notes files:**
- Sections 1–8 are present and in order, and every cheat summary has exactly 10 lines.
- Step-by-step algorithms are present for:
  - Canny, Harris, the SIFT pipeline (S01)
  - RANSAC and panorama stitching (S02)
  - NMS and AP computation (S07)
  - Hungarian matching in DETR (S08)
  - Lucas–Kanade and SfM (S12)
  - the diffusion forward and reverse processes (S15)
- The required diagrams are present: Faster R-CNN (S06), DETR (S08), U-Net (S10), CLIP (S13), and the Grounding DINO + SAM (+ depth) pipeline (S14).
- The practice mix holds in every session: 6–8 MCQ, 3–5 numerical, 3–4 short and 1–2 long. Every question is tagged with its type and difficulty.
