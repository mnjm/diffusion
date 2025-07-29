from datetime import datetime
from pathlib import Path
from copy import deepcopy

import pytz
import torch
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import math

supported_datasets = ['mnist', 'cifar10', 'tiny-imagenet', 'landscapes']

def torch_get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
            device = xm.xla_device()
        except ImportError:
            device = torch.device("cpu")
    return device

def torch_set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_compile_ckpt_fix(state_dict):
    # when torch.compiled a model, state_dict is updated with a prefix '_orig_mod.', renaming this
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict

def get_ist_time():
    ist = pytz.timezone('Asia/Kolkata')
    now_ist = datetime.now(ist)
    return now_ist.strftime('%d-%m-%Y-%H%M')

class HFDatasetWrapper(Dataset):

    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Ensure image has 3 channels and is in (H, W, C) format before transforming
        image = item['image'].convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

def get_dataset(dataset_str):
    assert dataset_str in supported_datasets
    
    print(f"Loading {dataset_str} dataset")
    cache_dir = Path("./dataset-cache") / dataset_str
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    img_size = (32, 32) 
    img_chls = 3
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    
    if dataset_str == "tiny-imagenet":
        train_ds = load_dataset('Maysee/tiny-imagenet', split='train', cache_dir=cache_dir)
        val_ds = load_dataset('Maysee/tiny-imagenet', split='valid', cache_dir=cache_dir)
        combined_ds = concatenate_datasets([train_ds, val_ds])
        torch_ds = HFDatasetWrapper(combined_ds, transform)
    elif dataset_str == "cifar10":
        train_ds = datasets.CIFAR10(root=cache_dir, train=True, download=True, transform=transform)
        val_ds = datasets.CIFAR10(root=cache_dir, train=False, download=True, transform=transform)
        torch_ds = ConcatDataset([train_ds, val_ds])
    elif dataset_str == "landscapes":
        assert any(cache_dir.iterdir()), "Download dataset pls. run ./download-landscape.sh"
        torch_ds = datasets.ImageFolder(cache_dir, transform=transform)
    else: # MNIST
        img_size = (24, 24)
        img_chls = 1
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        train_ds = datasets.MNIST(root=cache_dir, train=True, download=True, transform=transform)
        val_ds = datasets.MNIST(root=cache_dir, train=False, download=True, transform=transform)
        torch_ds = ConcatDataset([train_ds, val_ds])
    return img_size, img_chls, torch_ds

def save_imgs(imgs, img_path, nrow=0):
    n = imgs.shape[0]
    if nrow == 0:
        nrow = math.ceil(math.sqrt(n))
    grid = make_grid(imgs.float(), nrow=nrow, padding=2, normalize=True)
    save_image(grid, img_path)

class EMAModelWrapper:
    
    def __init__(self, model, beta=0.995, warmup_steps=100):
        self.ema_model = deepcopy(model).eval().requires_grad_(False)
        self.beta = beta
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def is_active(self):
        return self.step > self.warmup_steps
    
    def state_dict(self):
        return {
            'beta': self.beta,
            'warmup_steps': self.warmup_steps,
            'step': self.step,
            'ema_model': self.ema_model.state_dict(),
        }
    
    def load(self, state_dict):
        self.beta = state_dict['beta']
        self.warmup_steps = state_dict['warmup_steps']
        self.step = state_dict['step']
        self.ema_model.load_state_dict(state_dict['ema_model'])
    
    def update(self, base_model):
        self.step += 1
        if self.step == self.warmup_steps:
            self.ema_model.load_state_dict(base_model.state_dict())
        elif self.is_active():
            for base_param, ema_param in zip(base_model.parameters(), self.ema_model.parameters()):
                old, new = ema_param.data, base_param.data
                ema_param.data = old * self.beta + (1-self.beta) * new
        return
    
    def __call__(self, x, t, lbl=None):
        assert self.is_active(), "EMA model is not active yet"
        return self.ema_model(x, t, lbl)