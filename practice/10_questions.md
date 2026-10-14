# Session 10 — Image Segmentation · Practice Questions

> Notes: [../notes/10_image_segmentation.md](../notes/10_image_segmentation.md) · Solutions: [10_solutions.md](10_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 70 marks** · Suggested time: 125 min

Conventions for this paper: IoU $=TP/(TP+FP+FN)$, Dice $=2TP/(2TP+FP+FN)$ on pixel counts. Transposed-conv output $=(W-1)S-2P+K$. Confusion matrices have **rows = ground truth, columns = prediction**. Feature value $F(x,y)$ is stored at integer column $x$, row $y$ (ignore half-pixel offsets).

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
A street-scene model must give **every** pixel a class label (road, sky, car, person) **and** give each individual car and person its own ID. Which task is this?

- **A.** Semantic segmentation
- **B.** Instance segmentation
- **C.** Panoptic segmentation
- **D.** Object detection followed by NMS

### Q2 · MCQ · Easy
How does U-Net combine encoder features with decoder features at the same resolution level?

- **A.** It concatenates the (cropped) encoder feature maps with the up-sampled decoder maps along the channel dimension.
- **B.** It adds class-score maps element-wise, as in FCN-8s.
- **C.** It passes the max-pooling indices to the decoder, which unpools with them.
- **D.** It multiplies the decoder features by an attention map computed from the encoder.

### Q3 · MCQ · Medium
RoIAlign improves on RoI Pool because it:

- **A.** rounds RoI coordinates to the nearest feature cell before pooling, which removes sub-pixel noise.
- **B.** keeps floating-point RoI and bin coordinates and computes features at regularly sampled points by bilinear interpolation.
- **C.** aligns every RoI with its nearest anchor box before pooling.
- **D.** resizes each RoI crop of the input image to 224×224 before running the backbone.

### Q4 · MCQ · Medium
Mask R-CNN's mask branch outputs $K$ masks of size 28×28 for each RoI ($K$ = number of classes). Which loss trains it?

- **A.** A per-pixel softmax cross-entropy across the $K$ masks.
- **B.** A Dice loss summed over all $K$ masks.
- **C.** An L2 regression of all $K$ masks to the ground-truth mask.
- **D.** A per-pixel sigmoid + binary cross-entropy on the mask of the RoI's ground-truth class only.

### Q5 · MCQ · Easy
In a medical dataset, 95% of pixels are background and 5% are tumour. A model predicts **every** pixel as background. Which statement is correct?

- **A.** Pixel accuracy is 0.95 and tumour IoU is also 0.95.
- **B.** Pixel accuracy is 0.95, tumour IoU is 0, and the two-class mIoU is 0.475.
- **C.** Pixel accuracy is 0.05, because the tumour is never detected.
- **D.** Pixel accuracy and mIoU are both 0.95, so the model is acceptable.

### Q6 · MCQ · Easy
A predicted mask has Dice = 0.80 with its ground truth. Its IoU is:

- **A.** 0.667
- **B.** 0.800
- **C.** 0.889
- **D.** 0.640

### Q7 · MCQ · Medium
DeepLab uses a 3×3 atrous (dilated) convolution with dilation rate 4. Its effective kernel size and its number of weights per input–output channel pair are:

- **A.** 12×12 and 144
- **B.** 3×3 and 9, applied with stride 4
- **C.** 9×9 and 9
- **D.** 9×9 and 81

### Q8 · MCQ · Medium
FCN-32s upsamples a 7×7 score map with a transposed convolution with kernel $K=64$, stride $S=32$, padding $P=16$. The output size is:

- **A.** 192×192
- **B.** 256×256
- **C.** 240×240
- **D.** 224×224

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
A tumour's ground-truth mask has 1,200 pixels. A model's predicted mask has 1,000 pixels, of which 900 lie inside the ground truth.

(a) Find TP, FP and FN. [1]
(b) Compute IoU and Dice. [1]
(c) Check that Dice $=2\,\text{IoU}/(1+\text{IoU})$. Model B on the same image reports Dice $=0.85$: convert it to IoU and say which model is better. [1]
(d) On a 2-image test set, model P has per-image IoU (0.50, 0.50) and model Q has (0.90, 0.15). Compute the mean IoU and mean Dice (averaged over images) of each. Do the two metrics rank P and Q the same way? Explain. [2]

### Q10 · Numerical · Medium · [5 marks]
A 3-class semantic-segmentation model (background, car, person) is evaluated on 10,000 labelled pixels. Confusion matrix (rows = GT, columns = prediction):

| GT \ Pred | background | car | person |
|---|---|---|---|
| **background** | 8850 | 100 | 50 |
| **car** | 120 | 660 | 20 |
| **person** | 90 | 10 | 100 |

(a) Compute the pixel accuracy. [1]
(b) Compute the IoU of each class and the mIoU. [2]
(c) Compute the mean class accuracy (mean per-class recall). [1]
(d) A trivial model predicts "background" for all pixels. Compute its pixel accuracy and mIoU, and comment. [1]

### Q11 · Numerical · Medium · [5 marks]
The original U-Net uses **unpadded** 3×3 convolutions (stride 1), 2×2 max-pooling (stride 2) and 2×2 up-convolutions (transposed conv, $K=2$, $S=2$, $P=0$). The input tile is $572\times572$; there are 4 pooling steps and 4 up-convolutions, with two 3×3 convs at every level including the bottleneck.

(a) Give the spatial size after every convolution and pooling of the contracting path down to the bottleneck output. [2]
(b) Using the transposed-conv formula, give the size after each up-convolution, the crop applied to each skip connection, and the size after each pair of decoder convs. [2]
(c) What is the final output size, and what fraction of the input tile area receives a prediction? [1]

### Q12 · Numerical · Hard · [5 marks]
A Faster/Mask R-CNN head works on a stride-16 feature map. A proposal in image pixels is $(x_1,y_1,x_2,y_2)=(75,42,300,250)$.

(a) Convert the RoI to feature-map coordinates. RoI Pool **rounds each coordinate to the nearest integer**: give the rounded RoI and the misalignment of each side in image pixels. [2]
(b) RoIAlign keeps the exact coordinates and divides the RoI into a $7\times7$ grid. What is the size of one bin (in feature cells)? [1]
(c) One RoIAlign sampling point falls at $(x,y)=(5.25,3.75)$. The neighbouring feature values are $F(5,3)=10$, $F(6,3)=20$, $F(5,4)=30$, $F(6,4)=60$. Compute the bilinearly interpolated value, and compare with nearest-neighbour sampling. [1]
(d) The mask head predicts $K\times28\times28$ logits for COCO ($K=80$). How many values does it output per RoI, and how many enter the mask loss? [1]

### Q13 · Numerical · Medium · [5 marks]
Panoptic evaluation of one image. Class **car** (thing): 5 GT segments and 5 predicted segments; the best same-class IoUs of the candidate pairs are 0.92, 0.81, 0.66, 0.55 and 0.45 (each prediction overlaps at most one GT). Class **road** (stuff): 1 GT segment and 1 predicted segment with IoU 0.90.

(a) Find TP, FP and FN for car. [1]
(b) Compute SQ, RQ and PQ for car. [2]
(c) Compute PQ for road and the overall PQ (mean over the two classes). [1]
(d) Why is the matching at IoU > 0.5 guaranteed to be unique? [1]

## Section C — Short answer

### Q14 · Short · Easy · [4 marks]
Distinguish **semantic**, **instance** and **panoptic** segmentation, and define **things** vs **stuff**. Use a street scene with two touching pedestrians, a car, the road and the sky as your example.

### Q15 · Short · Medium · [4 marks]
Explain how FCN turns an image classifier into a segmentation network: convolutionalisation, upsampling, and the FCN-32s/16s/8s skip connections. How does U-Net's skip connection differ?

### Q16 · Short · Medium · [4 marks]
Explain the two quantisation steps in RoI Pool and how RoIAlign removes them. Why does this matter much more for masks than for bounding boxes?

### Q17 · Short · Medium · [5 marks]
**Lab.** Write the key PyTorch/torchvision steps to run (i) a pretrained `deeplabv3_resnet50` and (ii) a pretrained `maskrcnn_resnet50_fpn` on one RGB image. Give the output structure and tensor shapes of each, how you turn them into a label map / binary masks, and two common bugs to avoid.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
Describe **Mask R-CNN** in detail: overall architecture (backbone, FPN, RPN, heads), the mask head and its output, the loss function (and why masks are class-decoupled), RoIAlign, and the inference procedure. Contrast it with a semantic-segmentation network such as U-Net.

### Q19 · Long · Hard · [10 marks]
**Design question.** A hospital gives you 300 annotated 2D MRI slices (from 40 patients) for brain-tumour segmentation; tumour pixels are about 2% of each slice. Design a segmentation system: architecture, loss, data augmentation, training strategy, evaluation metrics and protocol, and post-processing. Justify each choice and say which metrics you would **not** rely on.
