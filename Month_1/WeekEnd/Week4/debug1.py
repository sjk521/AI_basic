import torch
import torch.nn as nn 
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

image_size = 28 
batch_size = 100 
hidden_size = 500 
num_class = 10 
lr = 0.001
epoch = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'


train_dataset = MNIST(root='../../data', train=True, download=True, transform=ToTensor())
test_dataset = MNIST(root='../../data', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class myMLP(nn.Module):
    def __init__(self, image_size, hidden_size, num_class): 
        super().__init__()
        self.image_size = image_size 
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_class)

    def forward(self, x): 
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, self.image_size * self.image_size))
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x) 
        return x 

model = myMLP(image_size=image_size, 
              hidden_size=hidden_size, 
              num_class=num_class).to(device) 
criteria = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

step = 0 
for ep in range(epoch): 
    for idx, (image, label) in enumerate(train_loader):
        step += 1 
        if step == 1157 : 
            print('dddd')
        image = image.to(device) 
        label = label.to(device) 

        output = model(image)
        loss = criteria(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if idx % 100 == 0 :
            print(f'Epoch : {ep}/{epoch}, step : {idx}, Loss : {loss.item():.3f}')