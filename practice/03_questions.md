# Session 03 — CNNs Under the Hood · Practice Questions

> Notes: [../notes/03_cnns_under_the_hood.md](../notes/03_cnns_under_the_hood.md) · Solutions: [03_solutions.md](03_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min
>
> Conventions (as in the notes): output size $W_{out}=\lfloor (W-K+2P)/S\rfloor+1$; conv params $(K_hK_wC_{in}+1)C_{out}$ with bias; MACs $=H_{out}W_{out}K_hK_wC_{in}C_{out}$; receptive field $r_0=1, j_0=1$, $r_l=r_{l-1}+(K_l-1)j_{l-1}$, $j_l=j_{l-1}S_l$ (pooling counts as a layer).

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
A stride-1 convolution layer (ignore border effects) is applied to an image, giving feature map $F_1$. The image is shifted 5 pixels to the right and the same layer is applied again, giving $F_2$. Which statement is correct?

- **A.** $F_2 = F_1$, because convolution is translation invariant.
- **B.** $F_2$ equals $F_1$ shifted 5 pixels to the right, because convolution is translation equivariant.
- **C.** $F_2$ equals $F_1$ shifted 5 pixels to the left, because the kernel is flipped.
- **D.** $F_2$ is unrelated to $F_1$, because each position has its own weights.

### Q2 · MCQ · Easy
A $2\times2$ max-pooling layer with stride 2 is applied to a $56\times56\times64$ feature map. What are its number of learnable parameters and its output shape?

- **A.** 256 parameters; $28\times28\times64$
- **B.** 64 parameters; $28\times28\times64$
- **C.** 0 parameters; $28\times28\times64$
- **D.** 0 parameters; $28\times28\times16$

### Q3 · MCQ · Medium
A convolution layer uses $3\times3$ kernels, receives an input with 64 channels and has 128 filters (with bias). How many learnable parameters does it have?

- **A.** 1,280
- **B.** 73,728
- **C.** 147,584
- **D.** 73,856

### Q4 · MCQ · Medium
A network applies, in order: conv $3\times3$ (stride 1) → conv $3\times3$ (stride 1) → max-pool $2\times2$ (stride 2) → conv $3\times3$ (stride 1). What is the (theoretical) receptive field of one unit in the final layer, in input pixels?

- **A.** $10\times10$
- **B.** $8\times8$
- **C.** $7\times7$
- **D.** $14\times14$

### Q5 · MCQ · Medium
In the ResNet paper, a 56-layer plain CNN (with BatchNorm) had a **higher training error** than a 20-layer plain CNN on CIFAR-10. Which explanation is correct?

- **A.** Overfitting: the deeper network has more parameters, so it memorises the training set.
- **B.** Too little data: stronger data augmentation would remove the gap.
- **C.** Pooling layers in deeper networks destroy too much spatial information.
- **D.** An optimisation difficulty (degradation): the solver fails to find even the solution "shallow net + identity layers", which would be no worse.

### Q6 · MCQ · Medium
One $5\times5$ convolution ($C$ channels in, $C$ out) is replaced by two stacked $3\times3$ convolutions ($C\to C\to C$, ReLU in between). Ignoring biases, which statement is correct?

- **A.** The receptive field grows to $6\times6$ and the weights drop to $18C^2$.
- **B.** The receptive field stays $5\times5$, but weights increase because there are two layers.
- **C.** The receptive field stays $5\times5$, weights drop from $25C^2$ to $18C^2$, and an extra non-linearity is added.
- **D.** The receptive field shrinks to $3\times3$, because each kernel is only $3\times3$.

### Q7 · MCQ · Medium
A $1\times1$ convolution with 64 filters (stride 1, with bias) is applied to a $28\times28\times256$ feature map. Which statement is correct?

- **A.** Output $28\times28\times256$; it is an identity operation with 64 parameters.
- **B.** Output $28\times28\times64$; it has $(256+1)\cdot64 = 16{,}448$ parameters and mixes channels at every pixel.
- **C.** Output $28\times28\times64$; it has $64\cdot1\cdot1+64 = 128$ parameters.
- **D.** Output $1\times1\times64$; it acts as global average pooling.

### Q8 · MCQ · Medium
Which architecture–key-idea pairing is correct?

- **A.** EfficientNet — compound scaling of depth, width and input resolution together.
- **B.** VGG-16 — Inception modules with parallel $1\times1$, $3\times3$, $5\times5$ branches.
- **C.** AlexNet — identity shortcut connections that fix the degradation problem.
- **D.** GoogLeNet — about 138M parameters, most of them in three large FC layers.

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
A $128\times128\times3$ RGB image passes through:

| Layer | Operation |
|---|---|
| L1 | conv $5\times5$, 32 filters, stride 1, padding 2 (+ReLU) |
| L2 | max-pool $2\times2$, stride 2 |
| L3 | conv $3\times3$, 64 filters, stride 2, padding 1 (+ReLU) |
| L4 | conv $3\times3$, 64 filters, stride 1, padding 0 (+ReLU) |

All convs have biases.
(a) Give the output shape ($H\times W\times C$) after each layer. **[2]**
(b) Give the parameter count of each layer and the total for L1–L4. **[1.5]**
(c) Two classifier heads for 10 classes are considered: (i) flatten → FC(10); (ii) global average pooling → FC(10). Give the head parameters and the whole-network total for each, and comment. **[1.5]**

### Q10 · Numerical · Medium · [5 marks]
AlexNet's first layer takes a $227\times227\times3$ image and applies 96 filters of size $11\times11$ with stride 4 and no padding (with bias).
(a) Output feature-map size. **[1]**
(b) Number of learnable parameters. **[1]**
(c) Number of MACs (multiply–accumulates) for one image. **[1]**
(d) How many weights (ignore biases) would a fully connected layer need to map the same input to the same number of output values? By roughly what factor is that larger than the conv layer's parameter count? **[1]**
(e) The same layer is applied to a $454\times454\times3$ image. Give the new output size, parameter count and MACs. **[1]**

### Q11 · Numerical · Medium · [5 marks]
Compute the theoretical receptive field of the network below (all convs stride 1; pools stride 2):

conv $3\times3$ → conv $3\times3$ → max-pool $2\times2$ → conv $3\times3$ → conv $3\times3$ → max-pool $2\times2$ → conv $3\times3$ with dilation $D=2$.

(a) Tabulate $r_l$ and the jump $j_l$ after every layer. **[3]**
(b) What would the final receptive field be if the last layer were an ordinary (undilated) $3\times3$ conv? How many extra parameters does the dilation cost? **[1]**
(c) Does the final unit weight all pixels of its receptive field equally? Explain briefly. **[1]**

### Q12 · Numerical · Medium · [5 marks]
All layers have $C=256$ input and output channels unless stated; ignore biases (each conv is followed by BatchNorm).
(a) Compare the weights of one $5\times5$ conv with two stacked $3\times3$ convs; give the % saving and the receptive field of each option. **[1.5]**
(b) Repeat for one $7\times7$ conv vs three stacked $3\times3$ convs. **[1.5]**
(c) A ResNet-50 bottleneck block maps $256\to64$ ($1\times1$) $\to64$ ($3\times3$, padding 1) $\to256$ ($1\times1$). Compare its weights and its MACs on a $56\times56$ feature map with a basic block of two $3\times3$ convs at 256 channels (padding 1). **[2]**

### Q13 · Numerical · Hard · [5 marks]
(a) An Inception branch receives a $28\times28\times192$ input and applies a "same"-padded $5\times5$ conv with 32 filters (stride 1, with bias). Compute its MACs and parameters. **[1.5]**
(b) A $1\times1$ conv with 16 filters (with bias) is inserted before the $5\times5$ conv (which now has 16 input channels). Compute the new total MACs and parameters and the reduction factor in MACs. **[2]**
(c) A MobileNet-style layer replaces a standard $3\times3$ conv ($64\to128$, stride 1, "same", no bias) on a $56\times56$ map by a depthwise $3\times3$ conv followed by a pointwise $1\times1$ conv (no biases). Compute params and MACs of both and verify the ratio formula $\frac{1}{C_{out}}+\frac{1}{K^2}$. **[1.5]**

## Section C — Short answer

### Q14 · Short · Easy · [4 marks]
Why are convolutional layers preferred over fully connected layers for images? Support your answer with a parameter calculation for a $224\times224\times3$ input and name the inductive biases a CNN encodes.

### Q15 · Short · Medium · [4 marks]
Define translation **equivariance** and translation **invariance** mathematically. Which CNN components provide each? Give two reasons why a trained CNN classifier is only *approximately* translation invariant, and state how CNNs usually cope with rotations.

### Q16 · Short · Medium · [4 marks]
Distinguish the **theoretical** and **effective** receptive field. List four ways to increase the receptive field with one trade-off each, and explain why receptive field matters for detecting or segmenting large objects.

### Q17 · Short · Medium · [4 marks]
Describe the feature hierarchy that Zeiler & Fergus observed in a trained CNN (layer 1 to layer 5). Give three reasons why depth helps, and one reason why simply stacking more plain layers stopped helping before 2015.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
"Explain why ResNet enabled networks with 100+ layers." Your answer should cover: the degradation problem and the evidence for it (and why it is not overfitting); the residual formulation $F(x)=H(x)-x$ and why identity is easy to learn; a derivation of the gradient through a residual block; the bottleneck block and projection shortcuts (with a diagram and a parameter comparison); and the role of BatchNorm.

### Q19 · Long · Hard · [10 marks]
(a) Trace the evolution of CNN architectures from LeNet-5 to ViT: for LeNet-5, AlexNet, VGG-16, GoogLeNet, ResNet-50, EfficientNet-B0 and ViT-B/16 give the year, the key idea and the approximate parameter count. **[3]**
(b) For ViT-B/16 on a $224\times224\times3$ image, compute the number of tokens fed to the encoder and the parameters of the patch-embedding layer (embedding dimension 768, with bias). Why does ViT need more training data than a ResNet? **[2]**
(c) Design a CNN for $64\times64\times3$ images and 10 classes with **at most 100K parameters**. Give each layer's output shape and parameter count, the total, and the receptive field before the classifier head, and justify your main design choices. **[5]**
