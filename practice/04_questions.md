# Session 04 — Training CNNs & Transfer Learning · Practice Questions

> Notes: [../notes/04_training_transfer_learning.md](../notes/04_training_transfer_learning.md) · Solutions: [04_solutions.md](04_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min
>
> Conventions (as in the notes): dropout probability $p$ = probability of **dropping** a unit (PyTorch); confusion matrices have **rows = true class, columns = predicted class**; logs are natural logs; ImageNet mean $[0.485, 0.456, 0.406]$, std $[0.229, 0.224, 0.225]$ applied after scaling pixels to $[0,1]$.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
A network contains `nn.Dropout(p=0.5)` (inverted dropout). After training you call `model.eval()` and run inference. What does the dropout layer do to its input?

- **A.** Randomly zeroes 50% of the activations and multiplies the rest by 2.
- **B.** Multiplies every activation by 0.5.
- **C.** Passes the activations through unchanged.
- **D.** Zeroes the same 50% of units that were dropped in the last training batch.

### Q2 · MCQ · Easy
Training loss decreases steadily for 40 epochs. Validation loss reaches its minimum at epoch 12 and then rises while the train–val gap keeps widening. What is the best diagnosis and first action?

- **A.** Underfitting — increase model capacity and train longer.
- **B.** Overfitting — keep the epoch-12 checkpoint (early stopping) and add regularisation/augmentation.
- **C.** Learning rate too low — increase it by 10×.
- **D.** The validation set must be corrupted — discard it and monitor training loss instead.

### Q3 · MCQ · Easy
In a standard image-classification pipeline, where should random augmentations such as `RandomResizedCrop`, flips and colour jitter be applied?

- **A.** To the training, validation and test sets, so all splits look alike.
- **B.** To the validation set only, to make model selection more robust.
- **C.** To the test set only, to estimate worst-case performance.
- **D.** To the training set only; validation and test use a fixed resize, centre crop and normalisation.

### Q4 · MCQ · Medium
You have 2,000 labelled photos of 5 dog breeds (similar to ImageNet content) and an ImageNet-pretrained ResNet-50. Which **first** strategy is most appropriate?

- **A.** Freeze the backbone (BN in eval mode), replace the head with a 5-way linear layer and train only the head.
- **B.** Train ResNet-50 from random initialisation, since the classes differ from ImageNet's.
- **C.** Fine-tune all layers with the original from-scratch learning rate of 0.1.
- **D.** Keep the 1000-way ImageNet head and map its predictions to breeds by name.

### Q5 · MCQ · Medium
You set `requires_grad=False` on every backbone parameter of a pretrained ResNet, train only the new head, and call `model.train()` each epoch. What happens to the backbone's BatchNorm layers during training?

- **A.** Nothing changes — the backbone is completely frozen.
- **B.** $\gamma$ and $\beta$ are updated, but the running statistics stay fixed.
- **C.** They normalise with batch statistics and keep updating their running mean/variance, so the "frozen" features drift; the fix is to put them in eval mode.
- **D.** PyTorch raises an error because BN cannot run in train mode with frozen parameters.

### Q6 · MCQ · Medium
What distinguishes **AdamW** from Adam with an L2 penalty added to the loss?

- **A.** AdamW replaces the L2 penalty by an L1 penalty to obtain sparse weights.
- **B.** Nothing — the two are mathematically identical for every optimiser.
- **C.** AdamW removes the momentum (first-moment) term to make decay stable.
- **D.** AdamW applies weight decay directly to the weights, decoupled from the adaptive gradient step, so the decay is not rescaled by $1/\sqrt{\hat v}$.

### Q7 · MCQ · Medium
Mixup with $\lambda = 0.3$ combines image A (label "cat") and image B (label "dog") as $\tilde x = 0.3\,x_A + 0.7\,x_B$. With label order [cat, dog], what is the training target?

- **A.** $[1, 0]$ — the label of the first image.
- **B.** $[0.3, 0.7]$
- **C.** $[0, 1]$ — the label of the dominant image.
- **D.** $[0.5, 0.5]$ — mixup always uses equal weights for labels.

### Q8 · MCQ · Medium
Which augmentation is most likely to **change the true label** of the training example?

- **A.** 180° rotation for handwritten digit classification.
- **B.** Horizontal flip for cat-vs-dog classification.
- **C.** ±10% brightness jitter for vehicle-type classification.
- **D.** A random crop covering 90% of the image for scene classification.

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
An ImageNet-pretrained ResNet-50 has **25,557,032** parameters in total, including its final FC layer (2048 → 1000, with bias). Its last residual stage `layer4` contains **14,964,736** parameters.
(a) How many parameters are in the original FC layer and in the rest of the network (the backbone)? **[1]**
(b) The FC layer is replaced by a new 2048 → 10 layer (with bias) and the backbone is frozen. Give the trainable parameters, the new total, and the trainable percentage. **[1.5]**
(c) `layer4` is now unfrozen as well. Give the trainable parameters and percentage. **[1]**
(d) Normalise the RGB pixel $(255, 0, 128)$ for this model (scale to $[0,1]$, then ImageNet mean/std). **[1.5]**

### Q10 · Numerical · Medium · [5 marks]
A 3-class classifier (classes: cat, dog, fox) outputs logits $z = [2.0,\ 1.0,\ 0.1]$ for an image whose true class is **cat**.
(a) Compute the softmax probabilities. **[1.5]**
(b) Compute the cross-entropy loss and the gradient $\partial\mathcal L/\partial z$. **[1]**
(c) With label smoothing $\varepsilon = 0.1$, give the smoothed target and the loss. **[1.5]**
(d) The image was in fact a mixup sample: $\lambda = 0.7$ of this cat image and $0.3$ of a fox image. Give the mixed target and the loss for the same prediction. **[1]**

### Q11 · Numerical · Medium · [5 marks]
One channel of a BatchNorm layer receives the mini-batch values $x = [2,\ 4,\ 6,\ 8]$ in **train mode**. Parameters $\gamma = 2$, $\beta = 1$, $\epsilon = 10^{-5}$ (negligible). Running statistics start at mean 0 and variance 1, BN momentum $\rho = 0.1$; as in PyTorch, the running variance is updated with the **unbiased** batch variance.
(a) Compute $\mu_B$, $\sigma_B^2$ (biased), $\hat x$ and the outputs $y$. **[2.5]**
(b) Compute the updated running mean and running variance. **[1]**
(c) Immediately afterwards the model is switched to eval mode and receives $x = 6$. What is the output? Comment on the difference from train mode. **[0.5]**
(d) The train-mode outputs $y$ go through inverted dropout with $p = 0.2$ and sampled keep-mask $[1, 0, 1, 1]$. Give the outputs, and the output at test time. **[1]**

### Q12 · Numerical · Medium · [5 marks]
A chest X-ray classifier is evaluated on 120 test images (rows = true, columns = predicted):

| True \ Pred | Normal | Pneumonia | COVID |
|---|---|---|---|
| **Normal** | 50 | 8 | 2 |
| **Pneumonia** | 5 | 30 | 5 |
| **COVID** | 1 | 4 | 15 |

(a) Overall accuracy. **[0.5]**
(b) Precision, recall and F1 for each class. **[3]**
(c) Macro-F1 and balanced accuracy. **[1]**
(d) Which single number would you report to a clinician worried about missed COVID cases, and why? **[0.5]**

### Q13 · Numerical · Medium · [5 marks]
(a) `StepLR`: initial LR $\eta_0 = 0.1$, step size 30 epochs, $\gamma = 0.1$, epochs counted from 0. Give the LR at epochs 29, 30 and 65. **[1]**
(b) Cosine annealing: $\eta_{max}=0.1$, $\eta_{min}=0$, $T = 100$ epochs, $\eta_t = \eta_{min}+\tfrac12(\eta_{max}-\eta_{min})(1+\cos(\pi t/T))$. Give the LR at $t = 30, 50, 75$. **[1.5]**
(c) Linear warm-up to 0.1 over 5 epochs with $\eta_t = 0.1\cdot t/5$ for $t = 1..5$. LR at epoch 2? **[0.5]**
(d) SGD with momentum $\mu = 0.9$, LR $\eta = 0.1$ and weight decay $\lambda_{wd} = 0.01$ (PyTorch convention: $g \leftarrow \nabla\mathcal L + \lambda_{wd}\theta$, $v \leftarrow \mu v + g$, $\theta \leftarrow \theta - \eta v$, with $v_0 = 0$). A single weight starts at $\theta_0 = 1.0$; the data-loss gradients at the first two steps are $0.5$ and $0.3$. Compute $\theta_1$ and $\theta_2$. **[2]**

## Section C — Short answer

### Q14 · Short · Medium · [4 marks]
Explain Batch Normalisation: give its equations, describe how it behaves differently in training and evaluation mode, list two benefits, and two practical pitfalls.

### Q15 · Short · Medium · [4 marks]
Compare **feature extraction** and **fine-tuning** of a pretrained CNN. Present the dataset-size × domain-similarity decision matrix and explain two techniques that make fine-tuning safer.

### Q16 · Short · Easy · [4 marks]
Describe four families of data augmentation with an example each, explain mixup/CutMix, give three situations in which a common augmentation would **not** be label-preserving, and state which data splits are augmented.

### Q17 · Short · Medium · [4 marks]
Diagnose each training run and propose one fix (10-class problem):
(i) train accuracy 99%, val accuracy 72%, val loss rising since epoch 10;
(ii) after 50 epochs train accuracy 55% and val accuracy 54%, both flat;
(iii) loss starts at 2.30, oscillates, then becomes NaN after ~300 iterations (LR = 1.0, SGD);
(iv) loss stays at exactly 2.303 for every iteration, even on a single batch.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
You have **800 labelled chest X-ray images** in 3 classes — Normal (500), Pneumonia (200), COVID (100) — from about 300 patients (some patients have several images). Design the full training strategy for a classifier: data splitting, preprocessing, augmentation (what is safe and what is not), model and transfer-learning strategy, handling class imbalance (give the class weights $w_c = N/(Kn_c)$), regularisation, optimiser and LR schedule, evaluation, and sanity checks.

### Q19 · Long · Hard · [10 marks]
A ResNet-18 trained **from scratch** on CIFAR-10 (50,000 training images, 32×32) reaches 99.5% training accuracy but only 76% validation accuracy. Propose and justify a **prioritised** plan of interventions. For each, explain *why* it helps (the mechanism), how its behaviour differs between training and inference where relevant, and how you would verify its effect. Conclude with whether transfer learning from ImageNet would help here and what you would need to change.
