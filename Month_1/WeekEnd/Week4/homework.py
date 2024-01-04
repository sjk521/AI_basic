# Debug mode를 이용해 상해있는 이 코드를 정상적으로 작동시켜보세요. 
# 총 11가지의 강제 error를 만들었습니다. 
# 에러를 고치면 에러가 발생한 line 뒤쪽에 주석으로 error의 원인을 적어주세요. 
# Debug mode의 call stack, debug console, variable 등의 과정을 충분히 활용해보세요 ^^ 
# 제출은 고친 파일(error의 원인이 적혀있는)과 
# debug mode를 활용해 디버깅하는 과정의 스크린샷을 찍어서 보내주세요!

import torch 
import torch.nn as nn 
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

lr = 0.001 # lr이 숫자여야 함.
image_size = 28 
num_classes = 10 
batch_size = 100
hidden_size = 500 
total_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 반대로

class MLP(nn.Module): 
    def __init__(self, image_size, hidden_size, num_classes) : 
        super().__init__() #self.mlp1이라고 불러낼꺼면 우선 parent class에서 정의된 것부터 가져왔어야 함.
        self.image_size = image_size
        self.mlp1 = nn.Linear(in_features=image_size*image_size, out_features=hidden_size)
        self.mlp2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.mlp3 = nn.Linear(in_features=hidden_size, out_features=hidden_size) # out_features=hidden_size
        self.mlp4 = nn.Linear(in_features=hidden_size, out_features=num_classes)
    
    def forward(self, x) : 
        batch_size = x.shape[0]
        #x = torch.reshape(x, (batch_size, -1, self.image_size * self.image_size)) # cross entropy구할 때 size에러남. RuntimeError: Expected target size [100, 10], got [100]
        x = torch.reshape(x, (batch_size, self.image_size * self.image_size))
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x

myMLP = MLP(image_size, hidden_size, num_classes).to(device)

train_mnist = MNIST(root='../../data/mnist', train=True, transform=ToTensor(), download=True)
test_mnist = MNIST(root='../../data/mnist', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True) #batch_size는 변수여서 따옴표 제거
test_loader = DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=True)

loss_fn = nn.CrossEntropyLoss() ## function이어서 괄호 필요


optim = Adam(params=myMLP.parameters(), lr=lr) # parameters()는 메서드여서 괄호 필요.

for epoch in range(total_epochs): 
    for idx, (image, label) in enumerate(train_loader) : 
        image = image.to(device)
        label = label.to(device)

        output = myMLP(image)

        loss = loss_fn(output, label)

        loss.backward()
        optim.step() #AttributeError: 'Tensor' object has no attribute 'step'
        optim.zero_grad()

        if idx % 100 == 0 : # 100으로 나누는데 나머지가 100일 리가. 0으로 변경
            print(f'Epoch : {epoch}/{total_epochs}, step : {idx}, Loss : {loss.item():.3f}') # 이렇게 보여주면 loss를 볼 수가 없지.
