import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torchvision
import torch.nn.functional as F
from fashion import FashionMNIST
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import numpy as np

# This line is needed for ipython magic stuff
#%matplotlib inline  


############ IMPORT DATA ############
# Fetch data
train_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

# Get second copy of data
valid_data = FashionMNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))


# Randomly select indexes that will be used as training examples
train_idx = np.random.choice(train_data.train_data.shape[0], 54000, replace=False)



# Remove non-train examples from 'train_data'
train_data.train_data = train_data.train_data[train_idx, :]
train_data.train_labels = train_data.train_labels[torch.from_numpy(train_idx).type(torch.LongTensor)]

# Create a mask for all non-train data
mask = np.ones(60000)
mask[train_idx] = 0


# Remove train examples from valid_data
valid_data.train_data = valid_data.train_data[torch.from_numpy(np.argwhere(mask)), :].squeeze()
valid_data.train_labels = valid_data.train_labels[torch.from_numpy(mask).type(torch.ByteTensor)]


# Initialize batch size and test bacth size
batch_size = 100
test_batch_size = 100


# Create dataloaders for both the train_data and the valid_data
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)


# Create dataloader for test data
test_loader = torch.utils.data.DataLoader(
    FashionMNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


# Show image
#plt.imshow(train_loader.dataset.train_data[1].numpy())
#plt.show()

#show second image
#plt.imshow(train_loader.dataset.train_data[10].numpy())
#plt.show()


####### MODELS #########
class FcNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, image):
        batch_size = image.size()[0]
        x = image.view(batch_size, -1)
        x = torch.sigmoid(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


class TwoConv3Full(nn.Module):
    def __init__(self):
        super(TwoConv3Full, self).__init__()
        self.n_filters_1 = 6
        self.kernel_size_1 = 5
        self.pool_downsize_1 = 2
        self.n_filters_2 = 16
        self.kernel_size_2 = 5
        self.pool_downsize_2 = 2
        self.fc1_outFactors = 120
        self.fc2_outFactors = 100
        self.fc3_outFactors = 10
        self.conv1 = nn.Conv2d(1 , self.n_filters_1, self.kernel_size_1, padding=2)
        self.pool1 = nn.MaxPool2d(self.pool_downsize_1, stride = self.pool_downsize_1)
        self.conv2 = nn.Conv2d(self.n_filters_1, self.n_filters_2, self.kernel_size_2, padding=2)
        self.pool2 = nn.MaxPool2d(self.pool_downsize_2, stride = self.pool_downsize_2)
        self.fc1 = nn.Linear(7 * 7 * self.n_filters_2, self.fc1_outFactors)
        self.fc2 = nn.Linear(self.fc1_outFactors, self.fc2_outFactors)
        self.fc3 = nn.Linear(self.fc2_outFactors, self.fc3_outFactors)


    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * self.n_filters_2)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return F.softmax(x, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
        
class TwoConvs(nn.Module):
    def __init__(self):
        super(TwoConvs, self).__init__()
        self.conv1 = nn.Conv2d( 1, 20, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(20, 10, 5, 1, padding=2)

        self.fc1 = nn.Linear(28*28*10, 10)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(-1, 28 * 28 * 10)

        x = F.log_softmax(self.fc1(x), dim=1)

        return x


######### FUNCTIONS ##########
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return model


def valid(model, valid_loader):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
            #data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            #valid_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            #valid_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            valid_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    valid_loss /= len(valid_loader.dataset)
    print("valid_loss =========> %f" % valid_loss)
    print('\n' + "valid" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))
    print("correct ==========> %d" % correct)
    print("correct / len(valid_loader.dataset) =======> %f" % (float(correct) / float(len(valid_loader.dataset))))
    return float(correct) / float(len(valid_loader.dataset))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data, target = Variable(data, volatile=True).cuda(), Variable(target).cuda() # if you have access to a gpu
            #data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            #test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n' + "test" + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    
def experiment(model, epochs=10, lr=0.001):
    best_precision = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        model = train(model, train_loader, optimizer)
        precision = valid(model, valid_loader)

        print("precision: %f" % precision)
        print("best_precision: %f" % best_precision)
    
        if precision > best_precision:
            best_precision = precision
            best_model = model

    return best_model, best_precision

best_precision = 0
for model in [TwoConvs()]:  # add your models in the list
    # model.cuda()  # if you have access to a gpu
    model, precision = experiment(model)

    print(precision)
    #if not best_precision:
    #    best_precision = precision

    if precision > best_precision:
        best_precision = precision
        best_model = model

test(best_model, test_loader)



