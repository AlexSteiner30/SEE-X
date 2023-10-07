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
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 128, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(128, 256, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(33362176, 2)  
        )

    def forward(self, x):
        return self.model(x)