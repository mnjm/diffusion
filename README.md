# Diffusion

A minimal, hackable repro of [Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2006.11239) with parts of [Improved DDPM](https://arxiv.org/pdf/2102.09672) (cosine beta schedule) and [Classifier Free Guidance](https://arxiv.org/pdf/2207.12598). Model is roughly same UNet used in the paper with Self Attention. Also support classifier free guidance for conditional generation, cosine and linear beta schedules, EMA etc. Datasets experimented with include MNIST. CIFAR-1o and Landscapes. Training script uses Hydra for config and (optional) WandB logging. Still work in progress, but good enough to mess around with generation diffusion models.

## Generated samples

### Landscapes (32x32)
![Generated Landscapes](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/landscapes.png)

### CIFAR10 (32x32)
![Generated CIFAR10](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/cifar10.png)

### MNIST with CFG (28x28)
![MNIST with CFG](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/mnist-cfg.jpg)

### CIFAR10 with CFG (32x32)

**Note**: The whitening and mode collapse below is because I picked the wrong hyperparameters during training. I naively set p_drop to 0.1 without thinking much, which meant the model didn’t learn the unconditional part well. So when generating samples, I noticed it was noisy and bad. (I also sadly did not observe intermediate samples generated during training) To fix it during sampling, I had to use a high CFG scale(6). This helped get strong class samples but also made the images collapse toward class mean hence whitening and mode collapse.

![CIFAR10 with CFG](https://raw.githubusercontent.com/mnjm/diffusion/refs/heads/assets/cifar10-cfg-bad.png)

## Usage

### Install dependencies

```sh
pip install -r requirements.txt
```

### CIFAR10 with linear and without CFG

```sh
python train.py
```

### Landscapes with linear and without CFG

```sh
python train.py +run=landscapes
```

### MNIST with cosine and CFG

```sh
python train.py +run=mnist_cfg
```
* uses lite version of the model since mnist is 24x24x1

### MNIST with cosine and CFG

```sh
python train.py +run=cifar10_cfg
```

### Sampling

```sh
python sample.py <model checkpoint.pt> -n <number of samples> <--save>
```