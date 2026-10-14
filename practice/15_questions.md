# Session 15 — Generative Computer Vision · Practice Questions

> Notes: [../notes/15_generative_cv.md](../notes/15_generative_cv.md) · Solutions: [15_solutions.md](15_solutions.md)
> **Mix:** 8 MCQ · 5 Numerical · 4 Short · 2 Long · **Total: 69 marks** · Suggested time: 125 min

Conventions used throughout: natural logarithms; DDPM forward process $q(x_t\mid x_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_tI)$, $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{s=1}^t\alpha_s$; CFG $\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+w\,(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing))$.

## Section A — Multiple choice (1 mark each)

### Q1 · MCQ · Easy
In a DDPM, which statement about the forward (noising) process $q(x_t\mid x_{t-1})$ is correct?

- **A.** It is a learned encoder whose parameters are trained jointly with the denoiser, like a VAE encoder.
- **B.** It is a fixed Gaussian Markov chain defined by a pre-set variance schedule $\beta_1,\dots,\beta_T$; it has no learnable parameters.
- **C.** It is deterministic: $x_t$ is obtained from $x_{t-1}$ by a fixed blur with no randomness.
- **D.** It must be simulated step by step from $x_0$ to $x_t$ during training because no closed form for $q(x_t\mid x_0)$ exists.

### Q2 · MCQ · Medium
Why does a VAE write $z=\mu+\sigma\odot\epsilon$ with $\epsilon\sim\mathcal N(0,I)$ instead of sampling $z\sim\mathcal N(\mu,\operatorname{diag}\sigma^2)$ directly?

- **A.** It makes the encoder deterministic, so no sampling is needed during training.
- **B.** It guarantees that the KL term becomes zero.
- **C.** It removes noise from $z$, which makes the decoder's outputs sharper.
- **D.** It moves the randomness into $\epsilon$, which does not depend on the encoder's parameters, so $z$ becomes a differentiable function of $\mu$ and $\sigma$ and gradients can be back-propagated through the sampling step.

### Q3 · MCQ · Medium
Early in GAN training the discriminator easily rejects fakes, so $D(G(z))\approx0$. Which generator objective is used in practice to avoid vanishing gradients?

- **A.** Maximise $\log D(G(z))$ (equivalently, minimise $-\log D(G(z))$).
- **B.** Minimise $\log D(G(z))$.
- **C.** Minimise $\log(1-D(G(z)))$, because its gradient is largest when $D(G(z))\approx0$.
- **D.** Maximise $\log(1-D(G(z)))$.

### Q4 · MCQ · Medium
Which statement about classifier-free guidance (CFG) is correct?

- **A.** It requires training a separate noise-aware image classifier whose gradient steers sampling.
- **B.** Guidance scale $w=0$ gives the purely conditional prediction, and increasing $w$ increases sample diversity.
- **C.** A single network is trained both with and without the condition (the condition is dropped, e.g. about 10% of the time), and at sampling time $\hat\epsilon=\epsilon_\theta(x_t,\varnothing)+w\,(\epsilon_\theta(x_t,c)-\epsilon_\theta(x_t,\varnothing))$.
- **D.** It needs only one U-Net evaluation per denoising step, so it is cheaper than unguided conditional sampling.

### Q5 · MCQ · Easy
When Stable Diffusion (v1) generates a $512\times512$ RGB image, what does its U-Net actually denoise?

- **A.** The $512\times512\times3$ pixel image directly.
- **B.** A $64\times64\times4$ latent produced by the VAE (downsampling factor $f=8$).
- **C.** The $77\times768$ CLIP text embedding of the prompt.
- **D.** A single 4-dimensional latent vector per image.

### Q6 · MCQ · Medium
A VAE with a Gaussian decoder (pixel-wise MSE reconstruction loss) produces blurry samples. What is the **main** reason?

- **A.** Under a factorised Gaussian likelihood (MSE), the best decoder output for an uncertain $z$ is the average of all plausible images, which smears fine details.
- **B.** The KL term is computed with the wrong sign, which pushes latent codes away from the prior.
- **C.** The reparameterisation trick adds Gaussian noise directly to the decoder's output pixels.
- **D.** The decoder uses transposed convolutions, which cannot represent sharp edges.

### Q7 · MCQ · Easy
You want to translate summer photos into winter photos, but you have no pixel-aligned (summer, winter) pairs of the same scenes. Which method fits best, and why?

- **A.** pix2pix, because its L1 loss can be computed between unpaired images.
- **B.** DCGAN, because transposed convolutions translate one image style into another.
- **C.** A plain autoencoder trained on summer photos, because its decoder outputs winter images.
- **D.** CycleGAN, because the cycle-consistency losses ($F(G(x))\approx x$ and $G(F(y))\approx y$) constrain the mapping without paired examples.

### Q8 · MCQ · Hard
ControlNet attaches a trainable copy of the Stable Diffusion U-Net encoder to the frozen ("locked") U-Net through **zero convolutions** (1×1 convolutions whose weights and biases are initialised to 0). What is their main purpose?

- **A.** They keep the gradient with respect to their own weights permanently zero, so the locked model can never change.
- **B.** They downsample the $512\times512$ condition image to the $64\times64$ latent resolution.
- **C.** At initialisation they output exactly zero, so the combined model reproduces the original Stable Diffusion exactly and no random noise disturbs the pretrained features; their weights then grow from zero during training.
- **D.** They replace the cross-attention layers so that a text prompt is no longer needed.

## Section B — Numericals

### Q9 · Numerical · Medium · [5 marks]
A toy DDPM uses $T=3$ steps with $\beta_1=0.1$, $\beta_2=0.2$, $\beta_3=0.3$. A 2-pixel "image" is $x_0=(1.0,\,-0.5)$ and the noise draw is $\epsilon=(0.4,\,-1.2)$. Round final answers to 4 decimal places.

(a) Compute $\alpha_t$ and $\bar\alpha_t$ for $t=1,2,3$. [1.5]
(b) Compute $x_3$ in one shot using the closed form $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$. [1.5]
(c) Compute $\mathrm{SNR}(3)=\bar\alpha_3/(1-\bar\alpha_3)$ and interpret it. [0.5]
(d) Assume the trained network predicts the noise perfectly, $\epsilon_\theta(x_3,3)=\epsilon$. Compute the reverse-step mean $\mu_\theta(x_3,3)=\frac{1}{\sqrt{\alpha_3}}\big(x_3-\frac{\beta_3}{\sqrt{1-\bar\alpha_3}}\epsilon_\theta\big)$ and the implied clean-image estimate $\hat x_0=(x_3-\sqrt{1-\bar\alpha_3}\,\epsilon_\theta)/\sqrt{\bar\alpha_3}$. Then draw $x_2=\mu_\theta+\sigma_3z$ with $\sigma_3^2=\beta_3$ and $z=(0.5,\,-1.0)$. [1.5]

### Q10 · Numerical · Medium · [5 marks]
A VAE encoder backbone outputs a 512-d feature vector. Two linear heads produce $\mu\in\mathbb R^{32}$ and $\log\sigma^2\in\mathbb R^{32}$; the decoder starts with a fully-connected layer from $z\in\mathbb R^{32}$ to 512 units. Prior $p(z)=\mathcal N(0,I)$. All layers have biases.

(a) Count the parameters of the two encoder heads together and compare with a plain autoencoder that has a single 512 → 32 bottleneck layer. Also count the decoder's first layer. [1.5]
(b) For one image consider only the first two latent dimensions: $\mu=(0.5,\,-1.0)$, $\log\sigma^2=(0,\ \ln0.25)$. Compute $\sigma$ and the KL contribution of these two dimensions, $\mathrm{KL}(q_\phi(z\mid x)\,\Vert\,p(z))$, in nats. [2]
(c) With $\epsilon=(1.2,\,-0.4)$, compute the reparameterised sample $z$ for these two dimensions. [1]
(d) What is the KL if the encoder outputs $\mu=0,\ \sigma=1$ in every dimension for every input, and why is that bad? [0.5]

### Q11 · Numerical · Medium · [5 marks]
A toy data space has three outcomes $\{a,b,c\}$. The data distribution is $p_{data}=(0.5,\,0.3,\,0.2)$ and the current generator produces $p_g=(0.2,\,0.3,\,0.5)$. Use natural logs and 4 decimal places.

(a) Compute the optimal discriminator $D^*(x)$ for each outcome. [1]
(b) Compute $V(D^*,G)=\sum_x p_{data}(x)\log D^*(x)+\sum_x p_g(x)\log(1-D^*(x))$. [2]
(c) Using $V(D^*,G)=2\,\mathrm{JSD}(p_{data}\Vert p_g)-\log4$, deduce the JSD. What would $D^*$ and $V$ be if $p_g=p_{data}$? [1]
(d) For one fake sample $D(G(z))=0.02$, where $D=\sigma(a)$ and $a$ is the discriminator's logit. Compare the magnitude of the gradient with respect to $a$ of the saturating generator loss $\log(1-D(G(z)))$ and of the non-saturating loss $-\log D(G(z))$. [1]

### Q12 · Numerical · Medium · [5 marks]
A Stable-Diffusion-v1-style latent diffusion model uses a VAE with downsampling factor $f=8$ and 4 latent channels, a CLIP text encoder producing $77\times768$ token embeddings, and classifier-free guidance.

(a) For a $512\times512$ RGB output give the latent shape, and how many times fewer values the U-Net processes compared with pixel-space diffusion. [1]
(b) At one step, for three latent entries, $\epsilon_\theta(z_t,\varnothing)=(0.2,\,-0.4,\,0.1)$ and $\epsilon_\theta(z_t,c)=(0.5,\,-0.1,\,-0.3)$. Compute the guided $\hat\epsilon$ for $w=7.5$. What do $w=0$ and $w=1$ give? [2]
(c) How many U-Net evaluations does text-to-image generation with a 30-step sampler and CFG require? [1]
(d) An img2img (SDEdit) run uses the same 30-step sampler with strength $s=0.6$. Roughly what noise level does it start from, how many denoising steps are run, and how many U-Net evaluations are needed with CFG? [1]

### Q13 · Numerical · Medium · [5 marks]
FID compares Gaussian fits of Inception features: $\mathrm{FID}=\Vert\mu_r-\mu_g\Vert_2^2+\operatorname{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)$.

(a) Toy 1-D feature. Real images: $\mathcal N(\mu=0.5,\ \sigma=1.0)$. Model A: $\mathcal N(0.8,\ 1.0)$. Model B: $\mathcal N(0.5,\ 0.4)$. Compute both FIDs, state which model is better, and name the failure that Model B shows. [2.5]
(b) 2-D features with diagonal covariances: $\mu_r=(1,\,0)$, $\Sigma_r=\operatorname{diag}(4,\,1)$; $\mu_g=(0.5,\,0.5)$, $\Sigma_g=\operatorname{diag}(1,\,1)$. Compute the FID. [2]
(c) Why is FID normally computed with at least 10k (often 50k) samples, and only compared at equal sample size? [0.5]

## Section C — Short answer

### Q14 · Short · Easy · [4 marks]
Explain how an autoencoder works (encoder, bottleneck, reconstruction loss). Give two practical uses, and explain why a plain autoencoder is a poor image generator while a VAE is not.

### Q15 · Short · Medium · [4 marks]
Derive the VAE evidence lower bound (ELBO) using either Jensen's inequality or the KL decomposition, name and interpret its two terms, and explain the reparameterisation trick.

### Q16 · Short · Medium · [4 marks]
What is mode collapse in GANs? Explain why it happens, why it is not the same as overfitting, and give three remedies — explaining in particular how WGAN helps.

### Q17 · Short · Medium · [4 marks]
Contrast classifier guidance with classifier-free guidance: training requirements, sampling formula, cost per step, and the effect of the guidance scale on fidelity and diversity.

## Section D — Long answer / Design

### Q18 · Long · Hard · [10 marks]
A hospital has 2,000 labelled chest X-rays showing a rare pathology and 50,000 normal chest X-rays. It wants to use a generative model to synthesise extra pathology images to improve a classifier. Compare VAEs, GANs and diffusion models for this project (sample quality, diversity / mode coverage, training stability, data requirements, compute, controllability). Recommend a design, explain how you would evaluate whether the synthetic data actually helps, and discuss the ethical and privacy risks.

### Q19 · Long · Hard · [10 marks]
Explain step by step how a text prompt becomes an image in Stable Diffusion: tokenizer, CLIP text encoder, latent noise, U-Net with cross-attention and timestep embedding, scheduler, VAE decoder. Include how classifier-free guidance is trained and applied, and how ControlNet adds spatial control (for example from a pose skeleton or a depth map) without damaging the pretrained model. Draw a diagram of the pipeline.
