"""
Learning rate finder in pytorch
Reference:
[1]:https://nbviewer.jupyter.org/github/sgugger/Deep-Learning/blob/master/Learning%20rate%20finder.ipynb

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from utils import progress_bar
import os

PATH = '.'

print(torch.cuda.is_available())
transfms = transforms.ToTensor()
train_set = datasets.MNIST(PATH, train=True, download=True, transform=transfms)
test_set = datasets.MNIST(PATH, train=False, download=True, transform=transfms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

# mb_example = next(iter(train_loader))
# print(mb_example[0].size(), mb_example[1].size())

# fig = plt.figure()
# for i in range(4):
#     sub_plot = fig.add_subplot(1, 4, i + 1)
#     sub_plot.axis('Off')
#     plt.imshow(mb_example[0][i, 0].numpy(), cmap='Greys')
#     sub_plot.set_title(mb_example[1][i])
# plt.show()

mean = torch.mean(train_set.train_data.type(torch.FloatTensor)) / 255.
std = torch.std(train_set.train_data.type(torch.FloatTensor)) / 255.

transfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])
train_set = datasets.MNIST(PATH, train=True, download=True, transform=transfms)
test_set = datasets.MNIST(PATH, train=False, download=True, transform=transfms)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)


class SimpleNet(nn.Module):
    """docstring for SimpleNet"""

    def __init__(self, in_features, hidden_features, out_features):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))

        return F.log_softmax(self.linear2(x), dim=-1)


net = SimpleNet(28 * 28, 100, 10)
optimizer = optim.SGD(net.parameters(), lr=1e-2)
criterion = F.nll_loss
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)


# def train(epochs):
#     # net.train()
#     for epoch in range(epochs):
#         running_loss = 0
#         corrects = 0
#         print(f"Epoch {epoch + 1}:")

#         for data in train_loader:
#             inputs, labels = data
#             # inputs, labels = Variable(inputs), Variable(labels)
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()

#             outputs = net(inputs)
#             _, preds = outputs.max(1)
#             # Compute
#             loss = F.nll_loss(outputs, labels)
#             running_loss += loss.item()
#             corrects += preds.eq(labels).sum().item()

#             # Backpropagate the computation of the gradients
#             loss.backward()
#             # Do the step of the sgd
#             optimizer.step()

#         print(f'Loss: {running_loss/len(train_set)}  Accuracy: {100.*corrects/len(train_set)}')

def train(epoch):
    print("\n Epoch: %d" % epoch)
    net.train()
    running_loss = 0
    corrects = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)

        # Compute
        loss = F.nll_loss(outputs, targets)
        # Backpropagate the computation of the gradients
        loss.backward()
        # Do the step of the sgd
        optimizer.step()

        _, preds = outputs.max(1)
        running_loss += loss.item()
        total += targets.size(0)
        corrects += preds.eq(targets).sum().item()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (running_loss / (batch_idx + 1), 100. * corrects / total, corrects, total))


best_acc = 0


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    corrects = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = F.nll_loss(outputs, targets)

            test_loss += loss.item()
            _, pred = outputs.max(1)
            total += targets.size(0)
            corrects += pred.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                         % (test_loss / (batch_idx + 1), 100. * corrects / total, corrects, total))

    # save checkpoint
    acc = 100 * corrects / total
    if acc > best_acc:
        print("Saving...")
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


# find the best lr
def find_lr(init_value=1e-8, final_value=10., beta=0.98):
    num = len(train_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []

    total = 0
    correct = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_num += 1

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        # compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        loss.backward()
        optimizer.step()

        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

        _, pred = outputs.max(1)
        total += targets.size(0)
        correct += pred.eq(targets).sum().item()
        progress_bar(batch_idx, len(train_loader), "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                     % (avg_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return log_lrs, losses



# find best lr
logs, losses = find_lr()
print(logs)
print(losses)
plt.plot(logs, losses)
plt.show()
# optimizer = optim.SGD(net.parameters(), lr=1e-1)
# for epoch in range(1, 11):
#     train(epoch)
#     test(epoch)
# train(10)
