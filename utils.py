import torch
from torch.utils.data import Dataset

def get_torch_device():
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

class HFDataset(Dataset):
    
    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def get_index_from_list(vals, t, x_shape):
    """ 
    from: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=qWw50ui9IZ5q
    Returns a specific index t of a passed list of values vals while considering the batch dimension for broadcasting
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)