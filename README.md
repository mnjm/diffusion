# Diffusion

A minimal, hackable repro of [Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239) with parts of [Improved DDPM](https://arxiv.org/pdf/2102.09672) (cosine beta schedule) and [Classifier Free Guidance](https://arxiv.org/pdf/2207.12598). Model is roughly same UNet used in the paper with Self Attention. Also support classifier free guidance for conditional generation, cosine and linear beta schedules, EMA etc. Experimented with include CIFAR-10, MNIST and Landscapes datasets. Training script uses Hydra for config and (optional) WandB logging. Still work in progress, but good enough to mess around with generative diffusion models.

## Generated samples

### CIFAR10 with CFG (32x32)
![CIFAR10 with CFG](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/cifar10-cfg-cosine.jpg)

### MNIST with CFG (28x28)
![MNIST with CFG](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/mnist-cfg.jpg)

### Tiny ImageNet with CFG (64x6x) (10 epochs)
![Tiny ImageNet CFG 10 Epochs](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/tiny-imagenet-10epochs-cfr.jpg)

### Landscapes (32x32)
![Generated Landscapes](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/landscapes.png)

### CIFAR10 (32x32)
![Generated CIFAR10](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/cifar10.png)


## Usage

### Install dependencies

```sh
pip install -r requirements.txt
```

### CIFAR10 with linear and without CFG

```sh
uv run train.py
```

### Landscapes with linear and without CFG

```sh
uv run train.py +run=landscapes
```

### MNIST with cosine and CFG

```sh
uv run train.py +run=mnist_cfg
```
* uses lite version of the model since mnist is 24x24x1

### CIFAR10 with cosine and CFG

```sh
uv run train.py +run=cifar10_cfg
```

### Tiny ImageNet with cosine and CFG

```sh
uv run train.py +run=tiny_imagenet_cfg
```

### Sampling

```sh
uv run sample.py <model checkpoint.pt> -n <number of samples> <--save>
```
