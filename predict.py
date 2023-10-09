from model import *
from dataset import *
from flask import request

dataset = LoadDataset()

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

with open('static/model_state.pt', 'rb') as f: 
    clf.load_state_dict(load(f))  

def predict(x):
    img = Image.open(x)  
    img_tensor = ToTensor()(img).reshape(1,1,369,369).to('cuda')

    return torch.argmax(clf(img_tensor)).item()