from model import *
from dataset import *

dataset = LoadDataset()

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

if __name__ == "__main__": 
    
    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    x = input()
    img = Image.open('dataset/' + x + '.png') 
    img_tensor = ToTensor()(img).unsqueeze(0).reshape(1,1, 369,369).to('cuda')

    print(torch.argmax(clf(img_tensor)))