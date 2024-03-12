import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from LeNet_model_MNIST_disHier import LeNet_client_side

LeNetf = LeNet_client_side()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(LeNetf.parameters())


def accuracy(net, test_loder):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs in test_loader:
            data, label = inputs
            outputs = net(data)
            outputs = outputs
            _, pred = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
    acc = 100.0 * correct / total
    return acc


for epoch in range(10):
    running_loss = 0.0
    for i, d in enumerate(train_loader, 0):
        inputs, labels = d
        optimizer.zero_grad()
        outputs = LeNetf(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('\r[epoch %02d][batch %3.d] loss:%.3f' % (epoch + 1, i + 1, loss), end='')
    print('')
acc = accuracy(LeNetf, test_loader)
print('Done! acc = %.3f%%' % acc)
if acc >= 90:
    torch.save(LeNetf.state_dict(), '../LeNet.pth')
