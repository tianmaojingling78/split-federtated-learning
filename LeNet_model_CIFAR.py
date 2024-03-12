from torch import nn
import torch.nn.functional as F
import math


class LeNet_client_side(nn.Module):
    def __init__(self):
        super(LeNet_client_side, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)  # cifar10
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # mnist
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        return x


class LeNet_server_side(nn.Module):
    def __init__(self):
        super(LeNet_server_side, self).__init__()
        self.fc1 = nn.Linear(500, 50)  # cifar10
        # self.fc1 = nn.Linear(320, 50)  # mnist
        self.fc2 = nn.Linear(50, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x