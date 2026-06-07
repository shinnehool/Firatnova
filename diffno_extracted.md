<!-- Page 1 -->

DiffFNO: Diffusion Fourier Neural Operator
Xiaoyi Liu$1,∗$
Hao Tang$2,\dagger$
1Washington University in St. Louis
2Peking University
jasonl@wustl.edu
haotang@pku.edu.cn
arXiv:2411.09911v2  [cs.CV]  5 Apr 2025
31
SRNO
DiffFNO (Ours)

<!-- [Image: 978x919 at (471,254)] -->

HiNOTE [35]

<!-- [Image: 978x919 at (332,265)] -->


<!-- [Image: 307x132 at (441,270)] -->

SRNO [54]
LMI [10]

<!-- [Image: 102x134 at (398,280)] -->


<!-- [Image: 125x102 at (407,280)] -->


<!-- [Image: 105x105 at (418,280)] -->

30
PSNR (dB)
LIT [5]
LIIF [7]

<!-- [Image: 978x919 at (472,304)] -->

DiﬀFNO

<!-- [Image: 978x919 at (332,316)] -->

LTE [24]
29
Forward Diﬀusion Process
Meta-SR [17]

<!-- [Image: 122x109 at (377,343)] -->

WFNO
28

<!-- [Image: 307x132 at (452,363)] -->

Gated 
140
160
180
200
220
240
260
AT-ODE 
Fusion
AttnNO
Solver
Inference Time (ms)
(b) DiffFNO vs SRNO [54]
(a) PSNR and inference time for$ \times$4 super-resolution
Figure 1. (a) All models use the EDSR-baseline [31] encoder, except HiNOTE [35] which has its own. (b) Compared to SRNO [54],
DiffFNO is strengthened by the fusion of spectral and spatial features and efficient refinement by a diffusion process.
Abstract
PSNR, including those beyond the training distribution. It
also achieves this at lower inference time (Fig. 1 (a)). Our
approach sets a new standard in super-resolution, deliver-
We introduce DiffFNO, a novel diffusion framework
ing both superior accuracy and computational efficiency.
for arbitrary-scale super-resolution strengthened by a
Weighted Fourier Neural Operator (WFNO). Mode Re-
balancing in WFNO effectively captures critical frequency
1. Introduction
components, significantly improving the reconstruction of
high-frequency image details that are crucial for super-
Image super-resolution (SR) reconstructs high-resolution
resolution tasks. Gated Fusion Mechanism (GFM) adap-
(HR) images from low-resolution (LR) inputs, recovering
tively complements WFNO’s spectral features with spa-
lost fine details to enhance visual quality.
SR is crucial
tial features from an Attention-based Neural Operator (At-
for applications like medical imaging [12], satellite imagery
tnNO). This enhances the network’s capability to capture
[23, 52], and video games [38]. The challenge in SR lies in
both global structures and local details. Adaptive Time-
its ill-posed nature: multiple HR images can correspond to
Step (ATS) ODE solver, a deterministic sampling strategy,
the same LR input due to information loss during downsam-
accelerates inference without sacrificing output quality by
pling. This ambiguity requires sophisticated algorithms ca-
dynamically adjusting integration step sizes ATS. Exten-
pable of inferring plausible and perceptually accurate high-
sive experiments demonstrate that DiffFNO achieves state-
frequency content from limited data.
of-the-art (SOTA) results, outperforming existing methods
Deep learning, particularly Convolutional Neural Net-
across various scaling factors by a margin of 2–4 dB in
works (CNNs) [58], has significantly advanced SR. Dong et
al. introduced SRCNN [9], demonstrating the effectiveness
$\dagger$Corresponding author.
$∗$Work Done during the visit at Peking University.
of end-to-end learning for SR. Subsequent models achieved
1

<!-- Page 2 -->

remarkable performance using deeper architectures and at-
mains robust across various upscaling factors—even those
tention mechanisms [6, 30, 31, 34, 57].
unseen during training.
Diffusion models have emerged as powerful generative
2. Related Work
frameworks modeling complex data distributions via itera-
tive denoising processes [11, 14, 45]. Their ability to gener-
Neural Operators and Fourier Methods. Neural Oper-
ate high-fidelity images is well-suited for inferring missing
ators (NO) [22] have emerged as a powerful framework
fine details. In SR, diffusion models progressively refine an
for learning mappings between infinite-dimensional func-
LR image by modeling the conditional distribution of HR
tion spaces, providing resolution-invariant models that gen-
images given the LR input [15, 25, 41, 51]. This iterative
eralize across different input resolutions. Unlike traditional
process reconstructs intricate textures and high-frequency
neural networks that map finite-dimensional vectors to other
components, producing realistic outputs.
vectors, neural operators learn mappings from functions to
However, diffusion models are computationally inten-
functions [27], making them well-suited for tasks involving
sive due to the iterative reverse diffusion process [46]. To
continuous data or data at varying resolutions.
address this, recent research explores efficient sampling
Multi-Layer Perceptrons (MLPs) often exhibit a spec-
strategies to accelerate reverse diffusion. One approach is
tral bias, favoring low-frequency functions [39]. This lim-
approximating the diffusion process through deterministic
its their ability to capture fine textures and sharp edges.
Ordinary Differential Equation (ODE), which can be solved
To overcome these limitations, techniques like positional
in fewer steps [33]. This accelerates inference and provides
encodings and Fourier feature mappings capture high-
consistent, reproducible results.
frequency details by embedding input coordinates into a
Arbitrary-scale SR models [7, 17, 24], which can up-
higher-dimensional sinusoidal space, allowing the network
sample images at user-defined scales beyond those seen
to represent complex patterns [44, 47].
in training, have gained attention in recent years. Meth-
Fourier Neural Operator (FNO) [26] is a variant of NO
ods involving attention mechanisms [5] and representing
that uses spectral convolution to efficiently capture global
images as continuous functions [10] have been explored.
data patterns, modeling long-range dependencies with
Operator-learning methods such as Super-Resolution Neu-
lower computational complexity than traditional CNNs. In
ral Operators (SRNO) [54] and HiNOTE [35] have further
physics and climate settings [19, 29, 55], FNOs can han-
advanced this field. However, the inherent differences be-
dle arbitrary input resolutions without retraining. Although
tween physics simulations and real-world images introduce
successful, FNOs may still lose high-frequency informa-
challenges from computational demands to the difficulty in
tion due to mode truncation (discarding higher-frequency
restoring high-frequency details.
Fourier modes). This loss impairs tasks such as SR that rely
To address these limitations, our contributions are:
on detailed reconstruction [13, 48, 49].
The Mode Rebalancing mechanism in WFNO over-
(1) We propose Weighted Fourier Neural Operator (WFNO)
comes these limitations.
Instead of being truncated, all
strengthened by iterative refinement from a diffusion frame-
Fourier modes are preserved, with additional learnable
work for high-frequency reconstruction, detailed in Fig. 2.
weights to modulate their impact on reconstruction. Fine-
Through Mode Rebalancing (MR), WFNO learns to empha-
grained feature representation is further enhanced by At-
size the most critical frequency components. This greatly
tnNO, which captures local details by processing data di-
enhances high-frequency image detail reconstruction, over-
rectly in the spatial domain
coming the limitations of standard FNOs and MLPs, which
Diffusion-Based SR and Efficient Sampling. Diffusion
underrepresent such details due to mode truncation and
models have gained prominence as powerful generative
spectral bias, respectively. (2) We develop Gated Fusion
models capable of producing high-quality images through
Mechanism (GFM) to dynamically adjust the influence of
iterative denoising techniques [14, 42]. In the context of SR,
Fourier space features from WFNO and complementary
diffusion models have been employed to model the condi-
spatial domain features from an Attention-based Neural Op-
tional distribution of HR images given LR inputs, achieving
erator (AttnNO). AttnNO is lightweight, sharing an encoder
higher resolutions after progressive enhancement [40, 41].
with and running in parallel to WFNO. (3) Additionally, we
Despite their effectiveness, diffusion models are compu-
present Adaptive Time-Step (ATS) ODE solver, which flex-
tationally intensive due to the large number of time steps re-
ibly adjusts integration step sizes based on data characteris-
tics by assessing the complexity of image regions, thereby
quired in the reverse diffusion process. Such computational
demands pose significant challenges for practical applica-
reducing computational overhead without compromising
tions [20], especially in real-time or resource-constrained
quality.
(4) DiffFNO achieves state-of-the-art results on
settings. Current solutions include: (i) Deterministic Sam-
multiple SR benchmarks, outperforming existing methods
pling via ODE Solvers: By reformulating the stochastic
by 2–4 dB in PSNR in reconstruction quality. It also offers
reverse diffusion as a deterministic ODE, advanced ODE
competitive inference time as Fig. 1 (a) shows. DiffFNO re-
2

<!-- Page 3 -->

Weighted Fourier Neural Operator
Gated Fusion Mechanism
Adaptive Time-Step ODE Solver

<!-- [Image: 420x91 at (243,87)] -->

Frequency Rebalancing
Learned Function

<!-- [Image: 78x65 at (223,95)] -->


<!-- [Image: 726x150 at (132,105)] -->


<!-- [Image: 1449x411 at (411,105)] -->


<!-- [Image: 105x108 at (313,112)] -->


<!-- [Image: 78x65 at (339,113)] -->

Runge-Kutta 4th-order

<!-- [Image: 106x108 at (194,117)] -->


<!-- [Image: 164x99 at (343,122)] -->


<!-- [Image: 71x135 at (119,123)] -->


<!-- [Image: 101x103 at (142,124)] -->


<!-- [Image: 459x91 at (244,128)] -->


<!-- [Image: 780x303 at (432,143)] -->


<!-- [Image: 248x132 at (187,146)] -->


<!-- [Image: 122x109 at (117,146)] -->


<!-- [Image: 39x100 at (407,151)] -->


<!-- [Image: 82x117 at (488,154)] -->


<!-- [Image: 322x90 at (321,156)] -->


<!-- [Image: 213x127 at (488,168)] -->


<!-- [Image: 307x132 at (278,172)] -->

Adaptive Time Step
Linear Transformation
Attention-based Neural Operator

<!-- [Image: 83x67 at (72,192)] -->

Galerkin 

<!-- [Image: 106x108 at (188,205)] -->


<!-- [Image: 78x65 at (217,212)] -->

Attention
RGB Result
Encoder (result concatenated 

<!-- [Image: 1175x1175 at (491,236)] -->

ODE
Noise Schedule
with Time Embedding)

<!-- [Image: 1175x1175 at (65,259)] -->

Element-Wise 
Element-Wise 
Concatenate
Dot Multiplication
Convolution

<!-- [Image: 305x99 at (424,267)] -->


<!-- [Image: 105x108 at (411,267)] -->

Multiplication
Addition
Kernel Integral 
Frequency Domain 

<!-- [Image: 106x108 at (208,286)] -->

Nonlinearity
Fourier Coeﬃcients

<!-- [Image: 71x135 at (113,288)] -->


<!-- [Image: 101x103 at (307,288)] -->


<!-- [Image: 78x65 at (412,290)] -->

Operator
Variables

<!-- [Image: 419x150 at (503,292)] -->


<!-- [Image: 405x150 at (67,292)] -->

Figure 2. The proposed Diffusion Fourier Neural Opeartor (DiffFNO) architecture for arbitrary-scale super-resolution begins by lifting a
low-resolution input image$ x$LR$(r)$ into a feature space using a convolutional encoder. Features extracted by the Weighted Fourier Neural
Operator (WFNO) and an Attention-based Neural Operator (AttnNO) are combined using a Gated Fusion Mechanism (GFM). The fused
features are then projected into RGB space, where Adaptive Time-Step (ATS) ODE solver efficiently completes the reverse diffusion
process with both accuracy and speed. This pipeline generates$ x$HR$(r)$, a high-resolution version of the input image.
tures from LR images. Unlike the simple linear transfor-
solvers can be employed to reduce the number of sam-
mations in standard FNO setups for physics simulations,
pling steps [21]. Methods like Denoising Diffusion Implicit
Models (DDIM) [45] and DPM-Solver [2, 33] have demon-
our encoder is tailored for SR, extracting complex patterns
and textures needed for high-quality reconstructions. We
strated the ability to generate high-quality images with sig-
nificantly fewer steps. (ii) Operator Learning for Fast Sam-
use the EDSR-baseline [31] and RDN models [58] in our
pling: Neural operators accelerates sampling by learning
experiments. (ii) WFNO and GFM: WFNO captures both
global and local details alongside the AttnNO. GFM com-
the solution operator of the reverse diffusion process [8, 59].
(iii) Progressive Distillation: Training a distilled model to
bines these into a unified HR feature map, which is then
approximate the behavior of the full diffusion model allows
projected into RGB. (iii) ATS ODE solver accelerates in-
ference speed by taking fewer, larger, and dynamically ad-
faster sampling with fewer steps [37, 43]. Although effec-
justed steps toward the reconstructed HR image. Fig. 2
tive, this method may require extensive retraining and po-
illustrates these components in detail.
tentially compromise image quality for increased speed.
The network minimizes the difference between the pre-
Applying these acceleration methods to diffusion-based
dicted image and the true image. The loss function is:
SR enables faster inference while maintaining high image
quality. With efficient sampling methods, diffusion models

$$
t
$$
$$
{L}(\th et a  ) = \ mathbb {E}_
$$
$$
\m a thcal
$$
$$
, \mathbf {x}_0} \left [ \left \| s_\theta (\mathbf {x}_t, t) - \nabla _{\mathbf {x}_t} \log p_t(\mathbf {x}_t | \mathbf {x}_0) \right \|_2^2 \right ], \label {eq:loss_function}
$$

(1)
become more practical for SR tasks, balancing performance
{
and computational efficiency [32].
where$ xt$ (i.e.$ x$LR) is obtained by adding noise to$ x0$ (i.e.
DiffFNO adopts (i) for its simplicity and the robustness
$x$HR).$ s\theta(xt, t)$ is the neural network approximating the true
of established numerical methods. We also strength it with
score function.$ \nablaxt log pt(xt|x0)$ is the true score function.
the ATS strategy, which adjusts integration step sizes adap-
tively to balance speed and quality.
3.2. Weighted Fourier Neural Operator
The Fourier Neural Operator (FNO) [26] is an efficient
3. The Proposed DiffFNO
NO variant designed to learn mappings between function
spaces. It operates directly on inputs of arbitrary resolu-
3.1. Network Architecture and Novel Components
tions, performing upscaling by mapping low-resolution in-
An overview of the proposed DiffFNO is shown in Fig. 1
puts to high-resolution outputs. It first transforms the input
(b). It has three parts: (i) A CNN encoder extracts fea-
data into the frequency domain, applies the learned filters,
3

<!-- Page 4 -->

and then transforms the data back to the spatial domain.
local dependencies. Working in tandem, they learn map-
Spectral convolution and mode truncation greatly enhance
pings from the low-resolution input function to the high-
computational efficiency.
resolution output function. Gated Fusion Mechanism opti-
Let$ x$LR$(r)$ denote the low-resolution input function
mally combines the complementary features from both op-
(e.g., an image), where$ r \inR2$ represents spatial coordi-
erators, adaptively balancing the contributions of each to a
nates. The goal is to learn an operator$ G$ such that:
fused feature map, which is then fed to a projection layer.
Efficient implementation of the kernel integral using the
$ $ \$mat h bf ${x$}_{\text {HR}}(\mathbf {r}) = \mathcal {G}[\mathbf {x}_{\text {LR}}(\mathbf {r})], \label {eq:operator_goal} $
(2)
Galerkin-type attention mechanism [4] has been explored
in the neural operator applied to SR tasks [35, 54]. Our
where$ x$HR$(r)$ is the output function (e.g., the super-
AttnNO is composed of bicubic interpolation, Galerkin at-
resolved image). The FNO models$ G$ by stacking:
tention, and nonlinearity, sharing an encoder with WFNO.
AttnNO models local interactions in the spatial domain, fo-

$$
\math b f  {v}_{l+ 1 }(\mathb f {r}) = \sigma \left ( \mathcal {W}_l \mathbf {v}_l(\mathbf {r}) + \mathcal {K}_l \mathbf {v}_l(\mathbf {r}) \right ), \label {eq:fno_layer}
$$

(3)
cusing on the most relevant spatial regions during the con-
volution process. Given the complementary role of AttnNO
where$ vl(r)$ is the feature representation at layer$ l$ evalu-
to WFNO, we simplify its structure to improve runtime.
ated at spatial location$ r$, and$ vl+1(r)$ is its updated repre-
While previous works have applied gating mechanisms
sentation in the following layer;$ \sigma$ is a nonlinear activation
in different contexts, our approach differs significantly.
function;$ Wl$ is a linear transformation.$ Kl$, the integral op-
Zheng et al. [60] use gating within recurrent CRF networks
erator at layer$ l$, transforms the features into the Fourier do-
primarily for semantic segmentation, controlling the infor-
main. Fourier modes are then truncated for computational
mation flow for boundary refinement rather than fusing fea-
efficiency. Global convolution is performed with a point-
ture maps with distinct representations. Hu et al. [16] intro-
wise multiplication between the transformed features and
duced channel-wise gating in Squeeze-and-Excitation (SE)
the learned Fourier coefficients.
blocks, focusing on adaptively recalibrating feature chan-

$$
\math c al {K}_l \ mathbf {v }_l(\mathbf {r}) = \mathcal {F}^{-1} \left ( \mathbf {P}_l(\boldsymbol {\xi }) \cdot \mathcal {F}[\mathbf {v}_l](\boldsymbol {\xi }) \right ), \label {eq:integral_operator}
$$

nels within a single network stream. In contrast, our Gated
(4)
Fusion Mechanism applies spatial gating to integrate global
where$ F$ and$ F−1$ denote the Fourier and inverse Fourier
dependencies captured by WFNO and local information
transforms, respectively.$ \xi$ is the frequency domain vari-
from AttnNO. This mechanism adaptively combines both
ables.$ F[vl](\xi)$ is the Fourier transform of$ vl$, evaluated at
operators’ feature maps, enhancing high-resolution image
frequency$ \xi$.$ Pl(\xi)$ is a complex-valued tensor of learnable
reconstruction by balancing global and local contributions
parameters representing the Fourier domain filters.
at each spatial location.
Let$  $\mat$ h bf {v} _{\text {\NewFNOShort }} \in \mathbb {R}^{B \times H \times W \times C} $and$  $\mathb$ f  {v}_{ \text {\SpatialNOShort }} \in \mathbb {R}^{B \times H \times W \times C} $
However,
mode
truncation
underrepresents
high-
frequency components that are critical to SR of real-world
denote the feature maps obtained from WFNO and AttnNO,
images.
To address this limitation, we introduce Mode
respectively, where$  B $is the batch size,$  H $and$  W $are the
Rebalancing. A learned weighting function$ wl(\xi)$ is ap-
height and width of the feature maps, and$  C $is the number
plied to the Fourier modes to amplify or attenuate specific
of channels. We first concatenate the feature maps along the
frequency components. It is defined at layer$ l$ as:
channel dimension and pass them through a convolutional
layer followed by a sigmoid activation to produce a gating

$$
\ma t h b f { w}_l(\boldsymbol {\xi }) = 1 + \gamma _l \cdot \| \boldsymbol {\xi } \|^\alpha , \label {eq:mode_weighting}
$$

(5)
map$  \ mathbf  {G} \in \mathbb {R}^{B \times H \times W \times 1} $:
where$ \gammal$ is a learnable scalar parameter at layer$ l$ that con-
$  \ m$athb$f { G} $= \s$i g$ma \le$ft ( \text {Conv}_{1 \times 1}\left ( \left [ \mathbf {v}_{\text {\NewFNOShort }}, \mathbf {v}_{\text {\SpatialNOShort }} \right ] \right ) \right ), \label {eq:gating_map} $
(7)
trols the strength of the weighting;$ \alpha$ is a hyperparameter
($0.7$ in our experiments or optionally a learnable parameter)
where$  \l eft [ \cdot , \cdot \right ] $denotes concatenation along the channel dimen-
that determines how the weight scales with the frequency
sion;  \te$xt {Conv}_{1 \times 1} $is a$  1 \times 1 $convolutional layer that reduces the
magnitude$ ∥\xi∥$.$ w(\xi)$ assigns higher weights to higher fre-
concatenated features to a single-channel gating map;$  \sigma (\cdot ) $
quencies when$ \alpha > 0$, thus emphasizing high-frequency
is the sigmoid activation function applied element-wise.
components. This yields an updated$ Kl$:
The fused feature map$  $\math$ b f {v}_ {\text {fused}} \in \mathbb {R}^{B \times H \times W \times C} $is the
element-wise weighted sum of the two feature maps:

$$
\math c al {K}_l \ mathb f  {v}_l(\m athbf {r}) = \mathcal {F}^{-1} \left ( \mathbf {w}_l(\boldsymbol {\xi }) \cdot \mathbf {P}_l(\boldsymbol {\xi }) \cdot \mathcal {F}[\mathbf {v}_l](\boldsymbol {\xi }) \right ). \label {eq:adaptive_integral_operator}
$$

(6)
$ $ \mat$ h b f  ${v}_$ { \t e xt  {$fused}$} = \mathbf {G} \odot \mathbf {v}_{\text {\NewFNOShort }} + \left ( 1 - \mathbf {G} \right ) \odot \mathbf {v}_{\text {\SpatialNOShort }}, \label {eq:fused_features} $
(8)
3.3. Gated Fusion Mechanism
where$  \odot $denotes element-wise multiplication, and subtrac-
While WFNO excels at capturing global dependencies
The gating map$  \mathbf {G} $is
through spectral convolutions, it may not fully exploit local
tion is performed element-wise.
interactions critical for detailed image reconstruction. We
broadcast across the channel dimension to match the dimen-
incorporate AttnNO to complement WFNO by capturing
sions of the feature maps.
4

<!-- Page 5 -->

drift toward$  \mathbf {D}\mathbf {x}_t $, the downscaled version of the image. The
Gated Fusion Mechanism brings two advantages com-
pared to a naive concatenation strategy: (i) Captures com-
added Gaussian noise further simulates the information loss
inherent in downscaling. At$  t  = T $, the image$  \mathbf {x}_T $approxi-
plementary Information: WFNO models global dependen-
mates the observed low-resolution image$  $\mathbf {x}_{\text {LR}} . The reverse
cies through spectral convolutions, effectively modeling
diffusion process then aims to recover the high-resolution
long-range interactions and overall structure. In contrast,
image$  \mathbf {x}_0 $(i.e.$ $\mathbf {x}_\text {HR} ) from$  \mathbf {x}_T $by reversing the degradation.
AttnNO excels at capturing local dependencies and fine-
grained details via attention mechanisms. (ii) Balances con-
Choice of$  \beta (t) $. The linear noise schedule is chosen for its
tributions dynamically: Gated Fusion Mechanism elicits the
simplicity and effectiveness. It provides a straightforward
importance of each feature map at each spatial location, dy-
way to control the rate of degradation over time. Parame-
namically balancing global and local information.
ters$  $\beta _{\text {min}} and$  $\beta _{\text {max}} are selected to balance the trade-off be-
tween sufficient degradation (to simulate downscaling) and
3.4. Forward Diffusion Process and Noise Schedule
numerical stability of the diffusion process.
Downsampling Operator$  \mathbf {D} $. The operator$  \mathbf {D} $is defined to
Motivation. NOs are well-suited for SR tasks due to their
reduce the spatial dimensions of the image by the desired
inherent resolution invariance and their ability to model
scaling factor. We use bicubic downsampling.
global dependencies efficiently. Diffusion models can it-
eratively refine a low-resolution image to a high-resolution
3.5. Adaptive Time-Step
one, capturing the complex conditional distribution of high-
resolution images given low-resolution inputs.
The standard reverse diffusion process is stochastic and re-
DiffFNO leverages the strengths of both frameworks.
quires a large number of sampling steps, making it compu-
WFNO is a powerful mechanism for handling arbitrary res-
tationally expensive. To accelerate inference, we reformu-
olutions and capturing high-frequency details, while the dif-
late the reverse diffusion as a deterministic Ordinary Differ-
fusion process iteratively improves reconstruction output.
ential Equation (ODE), allowing us to use advanced ODE
In our framework, the forward diffusion process mod-
solvers for faster sampling.
The ODE solver integrates
els the degradation of HR images to LR images, which in
the reverse diffusion process, and its output is the super-
our case is primarily due to downscaling. To incorporate
resolved image. The reverse diffusion process can be de-
this degradation into the diffusion model framework, we
scribed by a Stochastic Differential Equation (SDE) [46]:
define a forward process that simulates the downscaling ef-
fect over continuous time$  t  \i n [0, T] $. At, the image$ xT$
closely resemble the observed LR image$ x$LR after signifi-

$$
math bf  {x} = \ lef t [ f
$$
$$
\
$$
$$
(
$$
$$
\m a thbf x}, t) - g(t)^2 \nabla _{\mathbf {x}} \log p_t(\mathbf {x}) \right ] dt + g(t) d\bar {\mathbf {w}} \label {eq:reverse_diffusion_sde} {
$$
$$
d
$$

(11)
cant degradation. We adopt a modified variance-preserving
(VP) stochastic differential equation (SDE):
where$ x$ is the data;$ t$ is the time variable;$ f(x, t)$ and$ g(t)$
are drift and diffusion coefficients;$ \nablax log pt(x)$ is the score

$$
d \ ma
$$
$$
r
$$
$$
thbf {x} _ t = -\ f
$$

function;$ w¯$ is the reverse-time Wiener process. By remov-

$$
ac {1}{2} \beta (t) \left ( \mathbf {x}_t - \mathbf {D}\mathbf {x}_t \right ) dt + \sqrt {\beta (t)} d\mathbf {w}, \label {eq:forward_sde_modified}
$$

(9)
ing the stochastic term, we obtain the probability flow ODE,
which deterministically transports the data from the noise
where$  \beta (t) $is the noise schedule;$  \mathbf {D} $is the downsampling
distribution to the data distribution.
operator that reduces the resolution of the image;$  d\mathbf {w} $is the
standard Wiener process.
Our ATS ODE solver comprises three key components:
In this formulation, the term$  \mathbf {x} - \mathbf {D}\mathbf {x} $quantifies the high-
1. Adaptive Time Step Selection Using a Learned Func-
frequency details lost during downscaling. The drift term
tion.
Optimizing the allocation of time steps based on

$$
-
$$

$\frac  { 1 }{2 } \beta (t) \left ( \mathbf {x} - \mathbf {D}\mathbf {x} \right ) dt $models the gradual removal of these
data characteristics has been explored in previous works
We discretize the time interval$ [0, T]$ into$ N$
[28, 53].
$\sqrt {\beta (t)} d\mathbf {w} $adds Gaussian
details, while the diffusion term
non-uniform time steps$ {ti}N$
$i=0$, where$ t0 = 0$ and$ tN =$
noise to simulate further degradation.
$T$. We introduce a learned function$ \varphi\psi(t)$ as a weighted
Noise Schedule$  \beta (t) $. We define the noise schedule$  \beta (t) $
sum of polynomial basis functions, where the weight is
as a simple and effective linear function increasing over the
parameterized by a set of learnable coefficients$ \psi$
=
time interval$  [0 , T] $:
${\psi1, \psi2, . . . , \psiK}$, which adaptively determines the distri-
$  \b e t$a ($ t ) $= \$ b e$ta $_ { \$
bution of time steps based on the data characteristics.

$$
t ext {min}} + (\beta _{\text {max}} - \beta _{\text {min}}) \cdot \frac {t}{T}, \label {eq:beta_schedule_linear}
$$

(10)
Parameterization of$ \varphi\psi(t)$. We define$ \varphi\psi(t)$ as a normal-
ized weighted sum of$ K$ predefined monotonically increas-
where$  $\be$ta _{\text {min}} {=} 0.1 $and$  $\be$ta _{\text {max}} {=} 20 $. This linear schedule ensures
ing basis functions$  \{\phi $
$_k(t)\}_{k=1}^K $:
a gradual increase in the degradation strength from minimal
degradation at$  t {=} 0 $to maximum degradation at$  t {=} T $.

$$
_
$$
$$
\ps i (t) =
$$

Relation to Image Degradation. At each time$  t $, the im-

$$
_{ k =1}^K \psi _k \phi _k(t)}{\sum _{k=1}^K \psi _k \phi _k(T)}, \quad \psi _k = \exp (\omega _k), \label {eq:phi_parameterization}
$$
$$
\ph i
$$

(12)

$$
\
$$

age$  \mathbf {x}_t$ progressively loses high-frequency details due to the

$$
fra c {\sum
$$

5

<!-- Page 6 -->

where each basis function$  \phi  _k(t) = t^k $for$  k  =  1 ,  2 ,  \dots , K $is
balances computational cost and accuracy, requiring fewer
polynomial.$  \omega _k $are unconstrained learnable parameters that
steps than lower-order methods while retaining precision.
ensure$  \ p si _k \geq 0 $through the exponential mapping. We set
The benefits of ATS are threefold: (i) Deterministic Sam-
$K = 3$ to balance model flexibility with computational ef-
pling: It consistently produces the same results for identi-
ficiency. This setup allows$ \varphi\psi(t)$ to capture nonlinear time-
cal inputs, improving reproducibility. (ii) Reduced Compu-
step distributions without excessive complexity.
tation: Fewer sampling steps significantly decrease infer-
Selection of Time Steps. Using the learned function$ \varphi\psi(t)$,
ence time. (iii) High-Quality Reconstruction: It maintains

$$
i
$$

we map uniformly spaced normalized values$ si =$
$N$ to
high-quality reconstruction by efficiently allocating compu-
non-uniform time steps$ ti$:
tational resources.

$$
s i
$$
$$
^
$$

4. Experiments

$$
t _i
$$
$$
=  \ph i  _\
$$
$$
{
$$
$$
- 1 }\ le f t  (  s_i \right ) = \phi _\psi ^{-1}\left ( \frac {i}{N} \right ), \quad i = 0, 1, \dots , N \label {eq:adaptive_time_steps_learned}
$$

(13)

$$
p
$$

Datasets and Evaluation Metrics. We use the DIV2K [1]
Since$ \varphi\psi(t)$ is monotonically increasing, its inverse
dataset for training. For evaluation, we use the DIV2K val-
function$ \varphi−1$
$\psi (s)$ exists and can be efficiently computed.
idation set and four standard datasets: Set5 [3], Set14 [56],
2.
Neural Operator Score Network.
The score func-
BSD100 [36], and Urban100 [18].
tion, representing the gradient of the log probability density
We evaluate our model on upscaling factors of$ \times$2,$ \times$3,
$\qo pname \relax o{log}p_t(\mathbf {x})$, is approximated using a neural network$ s_\th eta (\mathbf {x}, t)$
$\times$4,$ \times$6,$ \times$8, and$ \times$12. Notably, scales$ \times$6,$ \times$8, and$ \times$12
parameterized by$ \theta $:
are outside the training distribution, as the training scales
are uniformly sampled from$ \times$1 to$ \times$4. This setup assesses

$$
\na bla _ { \math bf {x}} \log p_t(\mathbf {x}) \approx s_\theta (\mathbf {x}, t). \label {eq:score_function_approx}
$$

(14)
our model’s ability to generalize to arbitrary scales. We use
Peak Signal-to-Noise Ratio (PSNR) and Structural Similar-
In our architecture,$ s_\theta $consists of: (i) An encoder that ex-
ity Index Measure (SSIM) as our evaluation metrics.
tracts features from$ \$protect \mathbf  {x}_{\text {LR}}); (ii) WFNO for capturing global
Quantitative Results. Building on the quantitative gains of
dependencies and high-frequency details; (iii) AttnNO for
our model, we also present qualitative results to illustrate
modeling local dependencies and fine-grained structures;
the visual improvements achieved. We compare our pro-
(iv) Gated Fusion Mechanism to dynamically combine fea-
posed DiffFNO model with several SOTA arbitrary-scale
tures; (v) Time embedding$ e(t)$ incorporating the time vari-
SR methods, including Meta-SR [17], LIIF [7], LTE [24],
able$  t $into our neural network$  s_\t heta (\mathbf {x}, t) $using sinusoidal po-
SRNO [54], LIT [5], LMI [10], and HiNOTE [35]. All
sitional embeddings [14, 50], concatenating it and encoded
models are trained on the DIV2K dataset with identical set-
features along the channel dimension.
tings to ensure a fair comparison in Tables 1 and 2.
3. Efficient Solver. We solve the reverse-time stochastic
Among the compared methods, Meta-SR performs ad-
differential equation (SDE) of the diffusion process, trans-
equately at lower scales but struggles at higher scaling
formed into an ODE:
factors due to its generalized approach that lacks special-

$$
\f r ac { d\ m a
$$

ized mechanisms for fine detail capture.
LIIF and LTE

$$
thbf {x} }{d t} = f(\mathbf {x}, t) - \frac {1}{2} g(t)^2 \nabla _{\mathbf {x}} \log p_t(\mathbf {x}), \label {eq:reverse_sde}
$$

(15)
improve upon Meta-SR by using local implicit functions
and frequency-based estimations, respectively, which en-
where$ \ p rotect \mathbf  {x} \in \mathbb {R}^d$ is the image estimate at time$ t$ in the reverse
hance high-frequency texture representation.
However,
diffusion process;$ f(\m athbf {x}, t)$ and$ g(t)$ are coefficients derived
they still face limitations in capturing non-periodic textures
from the forward diffusion process.
and high-frequency details, resulting in blurred textures at
For the Variance Preserving (VP) SDE commonly used
larger scales. LIT and LMI further advance performance
in diffusion models, the coefficients are defined as:
by integrating attention mechanisms and MLP-mixer ar-
chitectures, effectively preserving high-frequency textures

$$
f( \m a th
$$
$$
bf {x},
$$
$$
t) =
$$
$$
-\frac {1}{2} \beta (t) \mathbf {x}, \quad g(t) = \sqrt {\beta (t)}, \label {eq:vp_sde_coefficients}
$$

(16)
and handling diverse scales, but they may not generalize
well across datasets with varying distributions. SRNO and
where$ \beta (t)$ is a predefined noise schedule specific to the dif-
HiNOTE employ neural operator frameworks with attention
fusion process and is consistent with our DiffFNO. By sub-
mechanisms and frequency-aware loss priors to better cap-
stituting the score function approximation from Eq. (14),
ture global spatial properties and enhance high-frequency
we define the approximate drift function:
detail reconstruction. We observed mixed results between
SRNO and HiNOTE: at certain scaling factors, one outper-

$$
f_\ th e ta ( \m a t
$$
$$
hbf {x}, t)  = f(\mathbf {x}, t) - \frac {1}{2} g(t)^2 s_\theta (\mathbf {x}, t). \label {eq:approximate_drift}
$$

(17)
forms the other, indicating their varying strengths at differ-
ent resolutions. Overall, their neural operator foundation
The adaptive time steps$ \ifmm$
$ode \lbrace \else \textbraceleft \fi  t_i \}_{i=0}^N$ discretize the ODE.
improves the handling of arbitrary scaling but may increase
We apply the Runge-Kutta 4th-order (RK4) method, as it
computational demands.
6

<!-- Page 7 -->


<!-- [Image: 481x321 at (66,72)] -->


<!-- [Image: 200x200 at (319,73)] -->


<!-- [Image: 200x200 at (376,73)] -->


<!-- [Image: 200x200 at (435,73)] -->


<!-- [Image: 200x200 at (494,73)] -->


<!-- [Image: 200x200 at (262,73)] -->

Bicubic
Meta-SR [17]
LTE [24]
LIIF [7]
LIT [5]

<!-- [Image: 200x200 at (262,139)] -->


<!-- [Image: 200x200 at (319,139)] -->


<!-- [Image: 200x200 at (376,139)] -->


<!-- [Image: 200x200 at (435,139)] -->


<!-- [Image: 200x200 at (494,139)] -->

LMI [10]
SRNO [54]
HiNOTE [35] DiffFNO (ours)
GT
BSD100 [36],$ \times12$

<!-- [Image: 1024x676 at (66,208)] -->


<!-- [Image: 200x200 at (320,208)] -->


<!-- [Image: 200x200 at (377,208)] -->


<!-- [Image: 200x200 at (436,208)] -->


<!-- [Image: 200x200 at (496,208)] -->


<!-- [Image: 200x200 at (263,208)] -->

Bicubic
Meta-SR [17]
LTE [24]
LIIF [7]
LIT [5]

<!-- [Image: 200x200 at (263,275)] -->


<!-- [Image: 200x200 at (320,275)] -->


<!-- [Image: 200x200 at (377,275)] -->


<!-- [Image: 200x200 at (436,275)] -->


<!-- [Image: 200x200 at (496,275)] -->

LMI [10]
SRNO [54]
HiNOTE [35] DiffFNO (ours)
GT
Urban100 [18],$ \times7.6$
Figure 3. Qualitative comparison on integer and continuous super-resolution scales. The models use RDN [58] as their encoder (except
HiNOTE [35], has its own). In the HR image, the cropped patch is outlined in green.
×2
×3
×4
×6
×8
×12
Model
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
EDSR-MetaSR [17]
33.32
0.913
30.10
0.800
28.23
0.830
26.10
0.792
24.77
0.742
23.95
0.720
EDSR-LTE [24]
33.83
0.921
30.50
0.880
28.79
0.852
26.55
0.800
25.05
0.760
24.20
0.736
EDSR-LIIF [7]
34.36
0.925
30.94
0.885
29.31
0.855
27.02
0.814
25.44
0.771
24.32
0.743
EDSR-LIT [5]
34.81
0.928
31.39
0.890
29.70
0.860
27.44
0.815
25.78
0.775
24.69
0.745
EDSR-LMI [10]
35.40
0.930
31.88
0.895
30.40
0.865
27.95
0.820
26.16
0.780
25.56
0.750
EDSR-SRNO [54]
34.85
0.928
31.45
0.890
30.05
0.863
27.36
0.810
26.00
0.772
25.91
0.760
EDSR-DiffFNO (Ours)
35.72
0.932
32.50
0.905
30.88
0.870
28.29
0.830
26.78
0.790
26.48
0.775
HiNOTE$\dagger$ [35]
35.29
0.931
31.90
0.895
30.46
0.842
27.83
0.799
26.41
0.772
26.23
0.732
RDN-MetaSR [17]
33.50
0.920
30.32
0.893
28.41
0.861
26.29
0.810
24.90
0.780
24.01
0.790
RDN-LTE [24]
33.98
0.922
30.65
0.882
28.94
0.852
26.70
0.802
25.20
0.762
24.35
0.732
RDN-LIIF [7]
34.51
0.927
31.09
0.887
29.46
0.857
27.17
0.812
25.59
0.772
24.47
0.742
RDN-LIT [5]
34.96
0.930
31.54
0.892
29.85
0.862
27.59
0.817
25.93
0.777
24.84
0.747
RDN-LMI [10]
35.55
0.932
32.03
0.897
30.55
0.867
28.10
0.822
26.31
0.782
25.71
0.752
RDN-SRNO [54]
35.00
0.930
31.60
0.892
30.20
0.862
27.51
0.812
26.15
0.772
26.06
0.762
RDN-DiffFNO (Ours)
35.87
0.934
32.65
0.902
31.03
0.872
28.44
0.832
26.93
0.792
26.63
0.777
Table 1. PSNR/SSIM comparison on the DIV2K [1] validation set using EDSR [31] and RDN [58] encoders. HiNOTE [35] uses its own.
Our DiffFNO model consistently achieves the high-
Mechanism, which adaptively balances global and local fea-
est PSNR and SSIM scores across all scaling factors and
tures, DiffFNO synthesizes global and local dependencies.
datasets. The performance gap widens at larger scaling fac-
The ATS ODE solver efficiently refines high-resolution im-
tors ($\times$8 and$ \times$12), demonstrating a superior generaliza-
ages, further enhancing quality. This combination addresses
tion to the out-of-distribution scales.
The improvements
the limitations of prior models, such as spectral bias and in-
are more pronounced on complex datasets like Urban100,
sufficient high-frequency detail capture.
which contain intricate textures and structures. By combin-
Qualitative Results. Fig. 3 compares arbitrary-scale SR
ing WFNO and AttnNO features through the Gated Fusion
methods on a BSD100 image (scaling factor of$ \times$12) with
7

<!-- Page 8 -->

Set5
Set14
BSD100
Urban100
Model
$\times$2
$\times$3
$\times$4
$\times$6
$\times$8
$\times$2
$\times$3
$\times$4
$\times$6
$\times$8
$\times$2
$\times$3
$\times$4
$\times$6
$\times$8
$\times$2
$\times$3
$\times$4
$\times$6
$\times$8
MetaSR [17]
37.50 34.05 31.52 28.23 26.02 33.51 30.03 28.02 25.53 24.02 31.02 28.05 26.52 24.82 23.52 32.02 28.03 25.82 23.52 22.03
LIIF [7]
38.02 34.42 32.04 28.57 26.25 34.03 30.43 28.43 25.84 24.33 31.52 28.55 27.03 25.03 23.83 32.52 28.53 26.03 23.83 22.33
LTE [24]
38.21 34.63 32.25 28.76 26.44 34.22 30.65 28.64 26.05 24.52 31.71 28.73 27.23 25.23 24.03 32.72 28.75 26.23 24.03 22.53
SRNO [54]
38.32 34.84 32.69 29.38 27.28 34.27 30.71 28.97 26.76 25.26 32.43 29.37 27.83 26.04 24.99 33.33 29.14 26.98 24.43 23.02
LIT [5]
38.53 35.02 32.82 29.51 27.42 34.44 30.83 29.03 26.82 25.33 32.52 29.51 27.92 26.12 25.01 33.42 29.22 27.02 24.52 23.12
LMI [10]
38.72 35.14 32.95 29.63 27.55 34.63 31.02 29.24 27.05 25.55 32.72 29.74 28.04 26.25 25.14 33.62 29.44 27.24 24.63 23.23
HiNOTE [35]
39.01 35.22 33.08 29.85 27.74 35.02 31.25 29.55 27.35 25.85 33.02 30.05 28.15 26.35 25.25 34.03 29.83 27.55 24.73 23.34
DiffFNO (Ours) 39.72 35.30 33.16 30.23 27.93 36.01 31.54 30.22 27.58 26.02 33.56 30.24 28.21 26.45 25.30 34.19 29.99 27.74 24.80 23.35
Table 2. PSNR comparison on four benchmark datasets: Set5 [3], Set14 [56], BSD100 [36], and Urban100 [18]. All models use RDN [58]
as their encoder, besides HiNOTE [35] which has its own.
$\times$2
$\times$3
$\times$4
$\times$6
$\times$8
$\times$12
Model
Inference
Steps
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
PSNR
SSIM
SRNO [54]
33.81
0.920
30.53
0.880
28.74
0.850
26.59
0.800
25.10
0.760
24.18
0.730
147
-
85
FNO [26]
34.36
0.925
30.94
0.885
29.31
0.855
27.02
0.810
25.44
0.770
24.32
0.740
-
WFNO
34.81
0.928
31.39
0.888
29.70
0.858
27.44
0.815
25.78
0.775
24.69
0.745
97
-
WFNO-AttnNO
35.40
0.930
31.88
0.892
30.40
0.862
27.95
0.820
26.16
0.780
25.56
0.750
139
1000
DiffFNO(-w, -a, -s)
34.85
0.928
31.45
0.890
30.05
0.860
27.36
0.815
26.00
0.775
25.91
0.760
204
1000
DiffFNO(-a, -s)
35.29
0.930
31.90
0.893
30.46
0.863
27.83
0.820
26.41
0.780
26.23
0.765
231
1000
0.932
DiffFNO(-s)
35.70
32.48
0.896
30.85
0.866
28.26
0.825
26.75
0.785
26.45
0.770
266
1000
DiffFNO
35.72
0.932
32.50
0.900
30.88
0.870
28.29
0.830
26.78
0.790
26.48
0.775
30
141
Table 3. Ablation study of variants of DiffFNO on the DIV2K [1] validation set. All use EDSR-baseline [31] backbone as their encoder.
Inference times are measured in milliseconds (ms). WFNO-AttnNO has Gated Fusion Mechanism.
fine-grained details like animal fur and rock textures and an
default FNO [26], at the cost of a slightly increased number
Urban100 (continuous scaling factor of$ \times$7.6) image featur-
of parameters and inference time.
ing large structures and fine local details such as reflections
Effect of Gated Fusion Mechanism and AttnNO. The
on the grass. SRNO and HiNOTE capture multiscale details
Gated Fusion Mechanism introduces minimal computa-
effectively, from the animal’s body to tiny gaps between
tional overhead. In addition, extra computational cost in-
glass panels. However, DiffFNO reconstructs crisper edges
curred by Attention-based Neural Operator is effectively
with fewer artifacts, enhancing texture in animal fur pat-
mitigated by running it in parallel with WFNO while em-
tern and reflections. WFNO captures large-scale patterns,
ploying a shared encoder.
while AttnNO and Gated Fusion Mechanism preserve intri-
Effect of ATS ODE Solver: ATS dramatically reduces the
number of inference steps from$ 1, 000$ to just$ 30$, which
cate textures. This multiscale approach followed by a dif-
fusion process enhanced by the ATS ’s ODE solver further
substantially improves the inference time while delivering
reduces visual artifacts.
competitive performance. As demonstrated in Tab. 3, this
acceleration not only preserves image quality but can even
Ablation Studies. Extensive ablation studies validate the
lead to slight improvements. In Fig. 1 (a), DiffFNO outper-
effectiveness and complementary nature of new compo-
forms existing methods in both PSNR and inference time.
nents in DiffFNO. Table 3 reports the PSNR results on the
DIV2K validation set for different model variants with scal-
5. Conclusion
ing factors from$ \times$2 to$ \times$12. -w denotes leaving out the
Mode Rebalancing (yielding the default FNO [26]). -a de-
We propose Diffusion Fourier Neural Opeartor (DiffFNO)
notes omitting AttnNO. -s denotes the removal of ATS ODE
for arbitrary-scale image super-resolution.
DiffFNO is
solver. We also measure inference time by averaging over
made of Weighted Fourier Neural Operator with a Mode
100 runs, and report inference steps. We establish a baseline
Rebalancing mechanism to emphasize high-frequency
details. It is complemented by a Attention-based Neural
with SRNO, whose architecture is the most similar to our
Operator through a Gated Fusion Mechanism that effec-
DiffFNO aamong the methods covered in our study. Over-
tively adjusts the influence of global and local features.
all, we observe notable improvements with the addition of
Image reconstruction is further refined by a diffusion pro-
model components. The complete DiffFNO achieves the
cess augmented with an Adaptive Time-Step ODE solver
highest PSNR and SSIM values across all upscaling factors.
that dynamically allocates time steps, drastically cutting
Effect of Mode Rebalancing. Incorporating Mode Rebal-
down inference time without compromising output quality.
Experiments demonstrate DiffFNO’s competitiveness in
ancing into WFNO boosted performance compared to the
8

<!-- Page 9 -->

both reconstruction quality and inference time across
[12] Hayit Greenspan. Super-Resolution in Medical Imaging. The
various benchmarks, establishing a new state-of-the-art.
Computer Journal, 52(1):43–63, 2008. 1
[13] Gaurav
Gupta,
Xiongye
Xiao,
and
Paul
Bogdan.
Multiwavelet-based
operator
learning
for
differential
References
equations. In Advances in Neural Information Processing
Systems, 2021. 2
[1] Eirikur Agustsson and Radu Timofte. Ntire 2017 challenge
[14] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising dif-
on single image super-resolution: Dataset and study. In The
fusion probabilistic models. Advances in neural information
IEEE Conference on Computer Vision and Pattern Recogni-
processing systems, 33:6840–6851, 2020. 2, 6
tion (CVPR) Workshops, 2017. 6, 7, 8
[15] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet,
[2] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-
Mohammad Norouzi, and Tim Salimans. Cascaded diffu-
DPM: an analytic estimate of the optimal reverse variance in
sion models for high fidelity image generation. Journal of
diffusion probabilistic models. In International Conference
Machine Learning Research, 23(47):1–33, 2022. 2
on Learning Representations, 2022. 3
[16] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation net-
[3] Marco Bevilacqua, Aline Roumy, Christine Guillemot, and
works. In Proceedings of the IEEE conference on computer
Marie-line Alberi-Morel.
Low-complexity single-image
vision and pattern recognition, pages 7132–7141, 2018. 4
super-resolution based on nonnegative neighbor embedding.
[17] Xuecai Hu, Haoyuan Mu, Xiangyu Zhang, Zilei Wang,
In Proceedings of the British Machine Vision Conference
Tieniu Tan, and Jian Sun.
Meta-sr:
A magnification-
(BMVC), pages 135.1–135.10, 2012. 6, 8
In Proceedings of
arbitrary network for super-resolution.
[4] Shuhao Cao. Choose a transformer: Fourier or galerkin. Ad-
the IEEE/CVF conference on computer vision and pattern
vances in neural information processing systems, 34:24924–
recognition, pages 1575–1584, 2019. 1, 2, 6, 7, 8
24940, 2021. 4
[18] Jia-Bin Huang, Abhishek Singh, and Narendra Ahuja. Sin-
[5] Hao-Wei Chen, Yu-Syuan Xu, Min-Fong Hong, Yi-Min
gle image super-resolution from transformed self-exemplars.
Tsai, Hsien-Kai Kuo, and Chun-Yi Lee.
Cascaded local
In Proceedings of the IEEE Conference on Computer Vision
implicit transformer for arbitrary-scale super-resolution. In
and Pattern Recognition (CVPR), pages 5197–5206, 2015.
Proceedings of the IEEE/CVF Conference on Computer Vi-
6, 7, 8
sion and Pattern Recognition, pages 18257–18267, 2023. 1,
[19] Peishi Jiang, Zhao Yang, Jiali Wang, Chenfu Huang, Pengfei
2, 6, 7, 8
Xue, T. C. Chakraborty, Xingyuan Chen, and Yun Qian. Ef-
[6] Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao,
ficient super-resolution of near-surface climate modeling us-
and Chao Dong.
Activating more pixels in image super-
ing the fourier neural operator. Journal of Advances in Mod-
resolution transformer. In Proceedings of the IEEE/CVF con-
eling Earth Systems, 15(7), 2023. 2
ference on computer vision and pattern recognition, pages
[20] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine.
22367–22377, 2023. 2
Elucidating the design space of diffusion-based generative
[7] Yinbo Chen, Sifei Liu, and Xiaolong Wang.
Learning
models. In Proc. NeurIPS, 2022. 2
continuous image representation with local implicit image
[21] Zhifeng Kong and Wei Ping. On fast sampling of diffusion
function. In Proceedings of the IEEE/CVF conference on
probabilistic models. In ICML Workshop on Invertible Neu-
computer vision and pattern recognition, pages 8628–8638,
ral Networks, Normalizing Flows, and Explicit Likelihood
2021. 1, 2, 6, 7, 8
Models, 2021. 3
[8] Tim Dockhorn, Arash Vahdat, and Karsten Kreis.
Score-
[22] Nikola B Kovachki, Zongyi Li, Burigede Liu, Kamyar Az-
based generative modeling with critically-damped langevin
izzadenesheli, Kaushik Bhattacharya, Andrew M Stuart, and
diffusion. In International Conference on Learning Repre-
Anima Anandkumar. Neural operator: Learning maps be-
sentations, 2022. 3
tween function spaces with applications to pdes. Journal of
Machine Learning Research, 24(146):1–63, 2023. 2
[9] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou
Tang. Image super-resolution using deep convolutional net-
[23] Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero,
works. IEEE Transactions on Pattern Analysis and Machine
Andrew P. Aitken, Alykhan Tejani, Johannes Totz, Zehan
Intelligence, 38:295–307, 2014. 1
Wang, and Wenzhe Shi. Photo-realistic single image super-
CoRR,
resolution using a generative adversarial network.
[10] Huiyuan Fu, Fei Peng, Xianwei Li, Yejun Li, Xin Wang,
abs/1609.04802, 2016. 1
and Huadong Ma. Continuous optical zooming: A bench-
mark for arbitrary-scale image super-resolution in real world.
[24] Jaewon Lee and Kyong Hwan Jin.
Local texture estima-
In Proceedings of the IEEE/CVF Conference on Computer
tor for implicit representation function. In Proceedings of
Vision and Pattern Recognition (CVPR), pages 3035–3044,
the IEEE/CVF conference on computer vision and pattern
recognition, pages 1929–1938, 2022. 1, 2, 6, 7, 8
2024. 1, 2, 6, 7, 8
[25] Haoying Li, Yifan Yang, Meng Chang, Shiqi Chen, Huajun
[11] Sicheng Gao, Xuhui Liu, Bohan Zeng, Sheng Xu, Yan-
Feng, Zhihai Xu, Qi Li, and Yueting Chen. Srdiff: Single
jing Li, Xiaoyan Luo, Jianzhuang Liu, Xiantong Zhen, and
image super-resolution with diffusion probabilistic models.
Baochang Zhang.
Implicit diffusion models for continu-
Neurocomputing, 479:47–59, 2022. 2
ous super-resolution. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition, pages
[26] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli,
Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and
10021–10030, 2023. 2
9

<!-- Page 10 -->

[38] NVIDIA Corporation. Deep learning super sampling (dlss),
Anima Anandkumar. Fourier neural operator for parametric
partial differential equations. In International Conference on
2024. 1
Learning Representations, 2021. 2, 3, 8
[39] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix
[27] Zongyi Li, Nikola Kovachki, and Anima Anandkumar.
Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and
Fourier neural operator with learned deformations for pdes
Aaron Courville. On the spectral bias of neural networks. In
on general geometries. In Proceedings of the 39th Interna-
International conference on machine learning, pages 5301–
tional Conference on Machine Learning, 2022. 2
5310. PMLR, 2019. 2
[28] Zhengyu Li, Kyungmin Kim, Jungwoo Lee, and Thomas
[40] Robin Rombach, Andreas Blattmann, Dominik Lorenz,
Huang. Autodiffusion: Training-free optimization of time
Patrick Esser, and Bj¨orn Ommer.
High-resolution image
steps and architectures for automated diffusion sampling. In
In Proceedings of
synthesis with latent diffusion models.
IEEE Conference on Computer Vision and Pattern Recogni-
the IEEE/CVF Conference on Computer Vision and Pattern
tion (CVPR), 2023. 5
Recognition, pages 10684–10695, 2022. 2
[29] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin,
[41] Chitwan Saharia, Jonathan Ho, William Chan, Tim Sali-
Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and
mans, David J Fleet, and Mohammad Norouzi. Image super-
Anima Anandkumar. Physics-informed neural operator for
IEEE transactions on
resolution via iterative refinement.
learning partial differential equations. ACM/JMS Journal of
pattern analysis and machine intelligence, 45(4):4713–4726,
Data Science, 1(3):1–27, 2024. 2
2022. 2
[30] Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc
[42] Chitwan Saharia, Jonathan Ho, William Chan, Tim Sali-
Van Gool, and Radu Timofte. Swinir: Image restoration us-
mans, David J Fleet, and Mohammad Norouzi. Sr3: Image
ing swin transformer. In Proceedings of the IEEE/CVF inter-
super-resolution via repeated refinement. In Proceedings of
national conference on computer vision, pages 1833–1844,
the IEEE/CVF Conference on Computer Vision and Pattern
2021. 2
Recognition (CVPR), pages 2736–2745, 2022. 2
[31] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and
[43] Tim Salimans and Jonathan Ho. Progressive distillation for
Kyoung Mu Lee. Enhanced deep residual networks for single
fast sampling of diffusion models. In International Confer-
image super-resolution. In The IEEE Conference on Com-
ence on Learning Representations, 2022. 3
puter Vision and Pattern Recognition (CVPR) Workshops,
[44] Vincent Sitzmann,
Julien N.P. Martel,
Alexander W.
2017. 1, 2, 3, 7, 8
Bergman, David B. Lindell, and Gordon Wetzstein. Implicit
[32] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and
neural representations with periodic activation functions. In
Joshua B Tenenbaum. Compositional visual generation with
Advances in Neural Information Processing Systems, 2020.
composable diffusion models. In Computer Vision–ECCV
2
2022: 17th European Conference, Tel Aviv, Israel, Octo-
ber 23–27, 2022, Proceedings, Part XVII, pages 423–439.
[45] Jiaming
Song,
Chenlin
Meng,
and
Stefano
Ermon.
Springer, 2022. 3
arXiv preprint
Denoising diffusion implicit models.
[33] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan
arXiv:2010.02502, 2020. 2, 3
Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion
[46] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Ab-
probabilistic model sampling in around 10 steps. Advances
hishek Kumar, Stefano Ermon, and Ben Poole. Score-based
in Neural Information Processing Systems, 35:5775–5787,
generative modeling through stochastic differential equa-
2022. 2, 3
tions. arXiv preprint arXiv:2011.13456, 2020. 2, 5
[34] Zhisheng Lu, Juncheng Li, Hong Liu, Chaoyan Huang, Lin-
[47] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara
lin Zhang, and Tieyong Zeng. Transformer for single im-
Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ra-
age super-resolution. In Proceedings of the IEEE/CVF con-
mamoorthi, Jonathan Barron, and Ren Ng. Fourier features
ference on computer vision and pattern recognition, pages
let networks learn high frequency functions in low dimen-
457–466, 2022. 2
sional domains. Advances in neural information processing
[35] Xihaier Luo, Xiaoning Qian, and Byung-Jun Yoon. Hierar-
systems, 33:7537–7547, 2020. 2
chical neural operator transformer with learnable frequency-
[48] Huy Tran, Levon Nurbekyan, and Houman Owhadi. Fac-
aware loss prior for arbitrary-scale super-resolution. arXiv
torized fourier neural operators. In Proceedings of the 39th
preprint arXiv:2405.12202, 2024. 1, 2, 4, 6, 7, 8
International Conference on Machine Learning, 2022. 2
[36] David Martin, Charless Fowlkes, Doron Tal, and Jitendra
[49] Tapas Tripura and Souvik Chakraborty. Wavelet neural op-
Malik.
A database of human segmented natural images
erator for solving parametric partial differential equations in
and its application to evaluating segmentation algorithms and
computational mechanics problems. Computer Methods in
measuring ecological statistics. In Proceedings of the IEEE
Applied Mechanics and Engineering, 404:115783, 2023. 2
International Conference on Computer Vision (ICCV), pages
[50] A Vaswani. Attention is all you need. Advances in Neural
416–423, 2001. 6, 7, 8
Information Processing Systems, 2017. 6
[37] Taehong Moon, Moonseok Choi, EungGu Yun, Jongmin
[51] Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin CK
Yoon, Gayoung Lee, and Juho Lee. Early exiting for acceler-
ated inference in diffusion models. In ICML 2023 Workshop
Chan, and Chen Change Loy. Exploiting diffusion prior for
real-world image super-resolution. International Journal of
on Structured Probabilistic Inference & Generative Model-
Computer Vision, pages 1–21, 2024. 2
ing, 2023. 3
10

<!-- Page 11 -->

[52] Peijuan Wang, Bulent Bayram, and Elif Sertel. A compre-
hensive review on deep learning based remote sensing im-
age super-resolution methods. Earth-Science Reviews, 232:
104110, 2022. 1
[53] Rachel Watson, Anirudh Mehta, Hyojin Choi, and Ajay
Singh. Align your steps: Optimizing sampling schedules in
diffusion models. arXiv preprint arXiv:2404.14507, 2024. 5
[54] Min Wei and Xuesong Zhang. Super-resolution neural oper-
ator. In Proceedings of the IEEE/CVF Conference on Com-
puter Vision and Pattern Recognition (CVPR), pages 18247–
18256, 2023. 1, 2, 4, 6, 7, 8
[55] Qidong Yang, Paula Harder, Venkatesh Ramesh, Alex
Hernandez-Garcia, Daniela Szwarcman, Prasanna Sattigeri,
Campbell D Watson, and David Rolnick. Fourier neural op-
erators for arbitrary resolution climate data downscaling. In
ICLR 2023 Workshop on Tackling Climate Change with Ma-
chine Learning, 2023. 2
[56] Roman Zeyde, Michael Elad, and Matan Protter. On sin-
gle image scale-up using sparse-representations. In Interna-
tional Conference on Curves and Surfaces, pages 711–730,
2010. 6, 8
[57] Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng
Zhong, and Yun Fu.
Image super-resolution using very
deep residual channel attention networks. In Proceedings of
the European conference on computer vision (ECCV), pages
286–301, 2018. 2
[58] Yulun Zhang, Yapeng Tian, Yu Kong, Bineng Zhong, and
Yun Fu. Residual dense network for image super-resolution.
In CVPR, 2018. 1, 3, 7, 8
[59] Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Aziz-
zadenesheli, and Anima Anandkumar. Fast sampling of dif-
fusion models via operator learning. In Proceedings of the
40th International Conference on Machine Learning, pages
42390–42402. PMLR, 2023. 3
[60] Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-
Paredes, Vibhav Vineet, Zhile Su, Dalong Du, Chang Huang,
and Philip HS Torr. Conditional random fields as recurrent
neural networks. In Proceedings of the IEEE international
conference on computer vision, pages 1529–1537, 2015. 4
11