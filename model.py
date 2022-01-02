import torch

class Model(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.fc1 = torch.nn.Linear(784, 128)
    self.fc2 = torch.nn.Linear(128, 10)

  def forward(self , x):
    x = x.reshape((x.shape[0], 784))
    x = self.fc1(x)
    x = torch.relu(x)
    x = torch.dropout(x, 0.2 , train=True)
    x = self.fc2(x)
    x = torch.softmax(x, dim=1)
    return x

    