from model import *
from dataset import *
import matplotlib.pyplot as plt

dataset = LoadDataset()

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

losses = []

if __name__ == "__main__": 
    for epoch in range(10): 
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
        losses.append(loss.item())

    with open('static/model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    with open('static/model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    plt.plot(losses)
    plt.savefig("losses.png", bbox_inches='tight',transparent=True, pad_inches=0)