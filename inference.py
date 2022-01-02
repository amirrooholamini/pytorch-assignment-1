import argparse

import time
start = time.time()
import torch
import torchvision
import cv2 as cv
import numpy as np
from model import Model

parser = argparse.ArgumentParser(description='inference model params')
parser.add_argument('--device', type=str, default='cpu', help='device you want to train on it: cpu or cuda')
parser.add_argument('--image', type=str, default='sample.png', help='image path')
args = parser.parse_args()

string_labels = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','	Ankle boot']

cpuDevice = torch.device(args.device)
model = Model()

model.load_state_dict(torch.load('weights.pth'))
model.train(False)
model.eval()

img = cv.imread(args.image)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.resize(img, (28,28))

transforms = torchvision.transforms.Compose([
       torchvision.transforms.ToTensor(),                 
])
tensor = transforms(img).unsqueeze(0)
tensor = tensor.to(cpuDevice)
prediction = model(tensor).cpu().detach().numpy()

end = time.time()
print(string_labels[np.argmax(prediction)])
print(f'time: {end-start} seconds')