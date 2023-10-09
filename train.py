from model import *
from dataset import *
import matplotlib.pyplot as plt

dataset = LoadDataset()

clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

losses = []

if __name__ == "__main__": 
    for epoch in range(30): 
        running_loss = 0.0

        for batch in dataset: 
            x,y = batch 
            x = x.reshape(1,1,369,369)

            x, y = x.to('cuda'), y.to('cuda') 

            opt.zero_grad()
            outputs = clf(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / 1}')


    with open('static/model_state.pt', 'wb') as f: 
        save(clf.state_dict(), f) 

    with open('static/model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open('dataset/0.png') 
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))

    img = Image.open('dataset/2.png') 
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))
