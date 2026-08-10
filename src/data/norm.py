import torch
from torchvision import transforms
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODE = "imagenet"

class PerImageStandardise:
    """Z-score each image individually, then repeat to 3 channels."""
    def __call__(self, t):
        g = t.mean(dim=0, keepdim=True)
        return ((g - g.mean()) / g.std().clamp_min(1e-6)).repeat(3, 1, 1)

def make_norm():
    if MODE == "perimage":
        return PerImageStandardise()
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def make_resize():
    if MODE == "perimage":
        return transforms.Compose([transforms.Resize(IMG_SIZE, max_size=None), transforms.CenterCrop(IMG_SIZE)])
    return transforms.Resize((IMG_SIZE, IMG_SIZE))
