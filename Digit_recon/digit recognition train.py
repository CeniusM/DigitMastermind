# digit recognition
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import math
import numpy
import matplotlib.pyplot as plt
from random import randint
import os



learn_rate = 0.001
epoch = 4
btcs = 1024
traintesting = False
modelname = "model16_97.pth"
create_new_model = True


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
    "train": DataLoader(train_data, batch_size=btcs, shuffle=True),
    "traintest": DataLoader(train_data, batch_size=btcs, shuffle=True),
    "test": DataLoader(test_data, batch_size=btcs, shuffle=True)
}

# define model
class CNN(nn.Module):

    # define layers
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # 1 = input channels, 10 = output channels, 5 = kernel size
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # 10 = input channels, 20 = output channels, 5 = kernel size from conv1
        self.conv2_drop = nn.Dropout2d() # dropout layer
        self.fc1 = nn.Linear(320, 50) # 320 = 20 * 4 * 4 (output of conv2) fc = fully connected
        self.fc2 = nn.Linear(50, 10) # 10 = number of classes

    # define forward pass
    def forward(self, x):

        # conv1 -> relu -> maxpool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # flatten -> relu -> dropout -> fully connected
        x = x.view(-1, 320) # flatten
        x = F.relu(self.fc1(x)) # relu
        x = F.dropout(x, training=self.training) # dropout
        x = self.fc2(x) # fully connected

        # softmax
        return F.softmax(x, dim=1)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
modelpath = f"Digit_recon\Models\{modelname}"
model.load_state_dict(torch.load(modelpath))


optimizer = optim.Adam(model.parameters(), lr=learn_rate)

# define loss function
loss_fn  = nn.CrossEntropyLoss()

# define training function
def train(epoch, model):
    model.train()
    batch_percentage = loaders['train'].batch_size/len(loaders["train"])
    for batch_idx, (data, target) in enumerate(loaders["train"]):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()


        if batch_idx % batch_percentage == 0:
            # Calculate the values
            processed_samples = batch_idx * len(data)
            total_samples = len(loaders["train"].dataset)
            percentage = 100. * batch_idx / len(loaders["train"])
            loss_value = loss.item()
            # Print the values
            print(f"Train Epoch: {epoch} [{processed_samples}/{total_samples} \t({percentage:.0f}%)]\t{loss_value:.6f}")    



def test():
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loaders["test"]:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loaders["test"].dataset)
    accuracy = 100. * correct/len(loaders["test"].dataset)
    print(f"\nTest average loss: {test_loss:.4f}, accuracy: {accuracy:.2f}%\n")

    return accuracy

if traintesting:
    def traindata_test():
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in loaders["traintest"]:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loaders["traintest"].dataset)
        accuracy = 100. * correct/len(loaders["traintest"].dataset)
        print(f"\nTraintest average loss: {test_loss:.4f}, accuracy: {accuracy:.2f}%\n")

        return accuracy


accuracylist = []
traindata_accuracylist = []

for epoch in range(1, epoch+1):
    train(epoch, model=model)

    accuracy = test()
    accuracylist.append(accuracy)

    if traintesting:
        traindata_accuracy = traindata_test()
        traindata_accuracylist.append(traindata_accuracy)

if epoch>0:
    def savepath(savefolder, accuracylist):
        path, dirs, files = next(os.walk(savefolder))
        lastaccuracy = int(accuracylist[-1])
        n = len(files)+1
        return f"{savefolder}\\model{n}_{lastaccuracy}.pth"


    savefolder = "Digit_recon\Models"
    if modelname == "" or create_new_model:
        savename = savepath(savefolder, accuracylist)
    else:
        savename = modelpath

    torch.save(model.state_dict(), savename)


def img_test(model):

    model.eval()
    data, target = test_data[randint(0, len(test_data))]

    data = data.unsqueeze(0).to(device)

    output = model(data)
    prob = F.softmax(output, dim=1)[0].cpu().detach().numpy()
    pred = output.argmax(dim=1, keepdim=True).item()
    image = data.squeeze(0).squeeze(0).cpu().numpy()

    return image, pred, target, prob


def img_plot(model, iters=10):

    # set window size
    window_length = 6

    # set table size
    columns = math.ceil((math.sqrt(iters)))
    rows = math.ceil(iters/columns)

    window_high = window_length/columns*rows
    
    # set image size to be able to plot all images
    imgsize = window_length/rows*1.2
    textsize = min(math.sqrt(imgsize),imgsize)*6
    space_in_between = imgsize/4

    # set window size
    plt.figure(figsize=(window_length, window_high))
    plt.subplots_adjust(wspace=space_in_between, hspace=space_in_between)

    # plot images
    for i in range(iters):
        plt.subplot(rows, columns, i+1)
        image, pred, target, prob = img_test(model)   

        #print prob
        #     

        plt.axis('off')
        if pred == target:
            plt.title(f'Predicted: {pred}', color='green', fontsize=textsize)
        else:
            plt.title(f'Predicted: {pred}', color='red', fontsize=textsize)
        plt.imshow(image, cmap='gray')
    plt.show()

if traintesting:
    print(accuracylist, traindata_accuracylist)
    for acc in range(len(accuracylist)):
        print(f"{accuracylist[acc]:.1f}%\t{traindata_accuracylist[acc]:.1f}%")
else:
    print(accuracylist)
    for acc in accuracylist:
        print(f"{acc:.1f}%")
img_plot(model, iters=25)






#plot image with probability distribution

#create a draw function window for the user

#create a function that takes the drawn image and converts it to a well formatted tensor

#create a function that takes any image and converts it to a well formatted tensor
