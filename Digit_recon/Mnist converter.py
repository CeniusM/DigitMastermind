#convert mnist data set to strings
## digit recognition
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import math
import numpy
import matplotlib.pyplot as plt
from random import randint

# load data
train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)


loaders = {
    "train": DataLoader(train_data, shuffle=False),
    "test": DataLoader(test_data, shuffle=True)
}





def zerocombiner(linenums):
    #combine zeros so 0, 0, 0, 0, 0 becomes !5
    newlinenums = []
    count = 0
    for i in range(len(linenums)):
        if linenums[i] == "0":
            count += 1
        else:
            if count != 0:
                newlinenums.append("!" + str(count))
                count = 0
            newlinenums.append(linenums[i])

def dataconverter(data, target):
    lines = []
    
    for i in range(28):
        #convert numbers to strings if with 4 decimal places
        linenums = [str(round(data[i][j], 4)) for j in range(28)] #convert objects to strings
        linenums = ["0" if num == "0.0" else num for num in linenums] #replace 0.0 with 0
            
        linenums = zerocombiner(linenums) #combine zeros
        
        #join strings with ;
        if len(linenums) != 1:
            line = "|".join(linenums)
        lines.append(line)

    #join lines with |
    data = ";".join(lines)    
    target = target.item()

    data = str(target) + "?" + data

    print(data)
    return data


for batch_idx, (data, target) in enumerate(loaders["train"]):
    data = data.squeeze(0).squeeze(0).cpu().numpy()
    data = dataconverter(data, target)

    break


print("done")