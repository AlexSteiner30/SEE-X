from model import *
from dataset import *

dataset = LoadDataset()

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

if __name__ == "__main__": 
    for epoch in range(330): 
        for batch in dataset: 
            x,y = batch 

            x = x.reshape(1,1,369,369)

            x, y = x.to('cuda'), y.to('cuda') 

            yhat = clf(x)

            loss = loss_fn(yhat, y) 

            opt.zero_grad()
            loss.backward() 
            opt.step() 

        print(f"Epoch:{epoch} loss is {loss.item()}")
    
    with open('model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open('dataset/1.png') 
    img_tensor = ToTensor()(img).unsqueeze(0).reshape(1,1,369,369).to('cuda')

    print(torch.argmax(clf(img_tensor)))