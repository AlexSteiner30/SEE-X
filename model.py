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
            nn.Conv2d(369, 369, (1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(369, 369//2, (1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(369//2, 369//4, (1,1)),
            nn.LeakyReLU(),

            nn.Conv2d(369//4, 369//8, (1,1)),
            nn.LeakyReLU(),

            nn.Flatten(), 

            nn.Linear(369//8, 2),
        )

    def forward(self, x):
        return self.model(x)