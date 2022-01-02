import argparse
import torch
import torchvision
from model import Model

parser = argparse.ArgumentParser(description='test model params')
parser.add_argument('--device', type=str, default='cpu', help='device you want to train on it: cpu or cuda')
parser.add_argument('--weight_file', type=str, default='weights.pth', help='model weight file path')
args = parser.parse_args()

cpuDevice = torch.device(args.device)
model = Model()

model.load_state_dict(torch.load(args.weight_file))
model.train(False)
model.eval()

def accuracyCalc(preds, lables):
  _, preds_max = torch.max(preds,1)
  acc = torch.sum(preds_max == lables)/len(preds)
  return acc

transforms = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),  
       torchvision.transforms.Normalize((0),(1))                 
])

dataset = torchvision.datasets.FashionMNIST("./dataset_test",download=False, transform=transforms, train=False)
testData = torch.utils.data.DataLoader(dataset, batch_size=32)

trainAccuracy = 0
for images, labels in testData:
  images = images.to(cpuDevice)
  labels = labels.to(cpuDevice)
  predictions = model(images)
  trainAccuracy += accuracyCalc(predictions, labels)

totalAccuracy = trainAccuracy/len(testData)
print(totalAccuracy)