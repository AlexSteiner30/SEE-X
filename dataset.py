import torch.utils.data as data
import json
import numpy as np
import torch
from PIL import Image

class LoadDataset(data.Dataset):
    def __init__(self):
        self.images = []
        self.x = []

        f = open("description.json")
        self.data = json.load(f)

        for i, itImg in enumerate(self.data):
            img_path = "dataset/" + str(itImg['index']) + ".png"

            self.x.append([itImg['x']])
     
            img = Image.open(img_path)
            self.images.append(np.array(img, dtype=np.float32))

        f.close()
  
    def __getitem__(self, idx):
        images = torch.tensor(self.images[idx])
        x = torch.tensor(self.x[idx])

        return images, x  
    
    def __len__(self):
        return len(self.images)