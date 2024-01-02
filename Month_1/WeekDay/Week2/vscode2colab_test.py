import os
import torch 
import torch.nn as nn 
from torch.optim import Adam 
from torchvision import datasets 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 100 
num_classes = 10 
hidden_size = 500
lr = 0.001
epochs = 5 

if not os.path.exists('../../data') : 
    os.mkdir('../../data')

train_dataset = datasets.MNIST(root='../../data/mnist', transform=ToTensor(), train=True, download=True)
test_dataset = datasets.MNIST(root='../../data/mnist', transform=ToTensor(), train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

class myNetwork(nn.Module): 
    def __init__(self, hidden_size, num_classes): 
        super().__init__()
        self.hidden_layer1 = nn.Linear(28*28, hidden_size)  
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_layer3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape(-1, 28*28)
        x = self.hidden_layer1(x) 
        x = self.hidden_layer2(x) 
        x = self.hidden_layer3(x) 
        return x 


model = myNetwork(hidden_size, num_classes).to(device)
criteria = nn.CrossEntropyLoss() 
optim = Adam(model.parameters(), lr=lr)

for epoch in range(epochs) : 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)
        label = label.to(device) 

        output = model(image)
        loss = criteria(output, label)
        optim.zero_grad()
        loss.backward() 
        optim.step() 
        
        if idx % 100 == 0: 
            print(f'{epoch}/{epochs} , {idx} step | Loss : {loss.item():.4f}')