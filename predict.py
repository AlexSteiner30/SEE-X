from model import *
from dataset import *

dataset = LoadDataset()

clf = ImageClassifier()#.to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

with open('static/model_state.pt', 'wb') as f: 
    save(clf.state_dict(), f) 

with open('static/model_state.pt', 'rb') as f: 
    clf.load_state_dict(load(f))  

def predict(x):
    img = Image.open(x) 
    img_tensor = ToTensor()(img).unsqueeze(0).reshape(1,1, 369,369)#.to('cuda')

    return torch.argmax(clf(img_tensor)).item()

x = input()
print(predict('dataset/' + str(x) + '.png'))