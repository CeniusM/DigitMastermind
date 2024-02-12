import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


# define model
class CNN(nn.Module):

    # define layers
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1 = input channels, 10 = output channels, 5 = kernel size
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10 = input channels, 20 = output channels, 5 = kernel size from conv1
        self.fc1 = nn.Linear(320, 50) # 320 = 20 * 4 * 4 (output of conv2) fc = fully connected
        self.fc2 = nn.Linear(50, 10) # 10 = number of classes

    # define forward pass
    def forward(self, x):

        # conv1 -> relu -> maxpool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        # flatten -> relu-> fully connected
        x = x.view(-1, 320) # flatten
        x = F.relu(self.fc1(x)) # relu
        x = self.fc2(x) # fully connected

        # softmax
        return F.softmax(x, dim=1)

modelpath = input("model path: ")
if modelpath == "":
    modelpath = "model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load(modelpath))

def img_test(data, model):

    model.eval()

    data = data.unsqueeze(0).to(device)
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True).item()


    #image = data.squeeze(0).squeeze(0).cpu().numpy()
    return pred

def dataexctracter(inputdata):
    lines = []
    inputdata=inputdata.split(";")

    for line in inputdata:
        line = line.split("|")
        nums = []
        for num in line:
            nums.append(float(num))
        lines.append(nums)

    #create tensor
    tensor = torch.tensor(lines)
    return tensor, lines  


import os
n=0

while True:

    #open the n'th file in the img folder
    folder= "All code\Digit_recon\imgs"
    path, dirs, files = next(os.walk(folder))


    filepath = os.path.join(folder, files[n])

    #get input from img.txt
    with open(filepath, "r") as file:
        inputdata = file.read()
    n+=1

    # inputdata = input()    

    if inputdata == "exit":
        break

    tensor, lines = dataexctracter(inputdata)
    pred = img_test(tensor, model)
    print(pred)

    print("predection is: ", pred)

    plt.imshow(lines,cmap="gray")
    plt.title("The number is: "+str(pred))
    plt.show()


input("press enter to exit")