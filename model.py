import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(369, 369, 1, bias=False),
            nn.LeakyReLU(),

            nn.Conv2d(369, 184, 1, bias=False),
            nn.LeakyReLU(),

            nn.Conv2d(184, 92, 1, bias=False),
            nn.LeakyReLU(),

            nn.Conv2d(92, 46, 1, bias=False),
            nn.LeakyReLU(),

            nn.Flatten(), 
            nn.Linear(46, 2)  
        )

    def forward(self, x): 
        return self.model(x)