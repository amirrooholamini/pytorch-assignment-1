import argparse
import torch
import torchvision
from model import Model
# import wandb
# wandb.init(project="assignment-1")

parser = argparse.ArgumentParser(description='train model params')
parser.add_argument('--device', type=str, default='cpu', help='device you want to train on it: cpu or cuda')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train')
parser.add_argument('--batch_size', type=int, default=10, help='number of batch size in each iteration')
parser.add_argument('--lr', type=float, default=10, help='learning rate')
args = parser.parse_args()


def accuracyCalc(preds, lables):
  _, preds_max = torch.max(preds,1)
  acc = torch.sum(preds_max == lables , dtype=torch.float64)/len(preds)
  return acc

# config = wandb.config
# config.batch_size = 64
# config.epochs = 10
# config.lr = 0.005
batch_size = args.batch_size
epochs = args.epochs
lr = args.lr


cpuDevice = torch.device(args.device)
model = Model()
model.to(cpuDevice)
model.train(True)

transforms = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),
       torchvision.transforms.Normalize((0),(1))                      
])

dataset = torchvision.datasets.FashionMNIST("./dataset",download=False, transform=transforms, train=True)
trainData = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lossFunction = torch.nn.CrossEntropyLoss()


for epoch in range(epochs):
  trainLoss = 0
  trainAccuracy = 0
  for images, labels in trainData:
    images = images.to(cpuDevice)
    labels = labels.to(cpuDevice)
    optimizer.zero_grad()

    predictions = model(images)
    loss = lossFunction(predictions,labels)
    loss.backward()
    optimizer.step()

    trainLoss += loss
    trainAccuracy += accuracyCalc(predictions, labels)

  totalLoss = trainLoss/len(trainData)
  totalAccuracy = trainAccuracy/len(trainData)
  print(f"Epoch: {epoch +1}, Loss: {totalLoss}, Accuracy: {totalAccuracy}")
  # wandb.log({"loss": totalLoss, "acc":totalAccuracy})

torch.save(model.state_dict(), "weights.pth")
