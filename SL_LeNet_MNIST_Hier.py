import torch
import torchvision.datasets
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import random
import numpy as np
import copy
from get_data import get_mnist, get_cifar10, dataset_iid, DatasetSplit

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(torch.cuda.get_device_name(0))

# ===================================================================
program = "SL LeNet on MNIST"
print(f"---------{program}----------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# To print in color -------test/train of the client side
def prRed(skk): print("\033[91m {}\033[00m".format(skk))


def prGreen(skk): print("\033[92m {}\033[00m".format(skk))


num_clients = 4
num_edges = 2
epochs = 50
frac = 1  # participation of clients; if 1 then 100% clients participate in SL
lr = 0.001


# =====================================================================================================
#                           Client-side Model definition
# =====================================================================================================
# Model at client side
class LeNet_client_side(nn.Module):
    def __init__(self):
        super(LeNet_client_side, self).__init__()
        # self.conv1 = nn.Conv2d(3, 10, kernel_size=5)  # cifar10
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # mnist
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


net_glob_client = LeNet_client_side()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(net_glob_client)  # to use the multiple GPUs

net_glob_client.to(device)
print(net_glob_client)


# =====================================================================================================
#                           Edge-side Model definition
# =====================================================================================================
# Model at client side
class LeNet_edge_side(nn.Module):
    def __init__(self):
        super(LeNet_edge_side, self).__init__()
        # self.fc1 = nn.Linear(500, 50)  # cifar10
        self.fc1 = nn.Linear(320, 50)  # mnist

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
        return x


net_glob_edge = LeNet_edge_side()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_edge = nn.DataParallel(net_glob_edge)  # to use the multiple GPUs

net_glob_edge.to(device)
print(net_glob_edge)


# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
class LeNet_server_side(nn.Module):
    def __init__(self):
        super(LeNet_server_side, self).__init__()
        self.fc2 = nn.Linear(50, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc2(x)
        return x


net_glob_server = LeNet_server_side()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_server = nn.DataParallel(net_glob_server)  # to use the multiple GPUs

net_glob_server.to(device)
print(net_glob_server)


# ====================================================================================================
#                                  data processing
# ====================================================================================================
dataset_train, dataset_test = get_mnist()

dict_users = dataset_iid(dataset_train, num_clients)
dict_users_test = dataset_iid(dataset_test, num_clients)

# ====================================================================================================
#                                  Server Side Program
# ====================================================================================================
# For Server Side Loss and Accuracy
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []
batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

criterion = nn.CrossEntropyLoss()
count1 = 0
count2 = 0


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# to print train - test together in each round-- these are made global
acc_avg_all_user_train = 0
loss_avg_all_user_train = 0
loss_train_collect_user = []
acc_train_collect_user = []
loss_test_collect_user = []
acc_test_collect_user = []

# client idx collector
idx_collect = []
fed_check = False  # every client has trained one epoch


# Server-side function associated with Training
def train_server(fx_client, labels, idx, len_batch):
    global net_glob_server, criterion, device, batch_acc_train, batch_loss_train, fed_check
    global loss_train_collect, acc_train_collect, count1, acc_avg_all_user_train, loss_avg_all_user_train, idx_collect
    global loss_train_collect_user, acc_train_collect_user

    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client.to(device)
    labels = labels.to(device)

    # ---------forward prop-------------
    fx_server = net_glob_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, labels)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, labels)

    # --------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train =>\tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_train,
                                                                     loss_avg_train))

        # print("accuracy = ", acc_avg_train)
        acc_avg_train_all = acc_avg_train
        loss_avg_train_all = loss_avg_train

        # accumulate accuracy and loss for each new user
        loss_train_collect_user.append(loss_avg_train_all)
        acc_train_collect_user.append(acc_avg_train_all)

        # collect the id of each new user
        if idx not in idx_collect:
            idx_collect.append(idx)
            # print(idx_collect)

        # This is to check if all users are served for one round --------------------
        if len(idx_collect) == num_clients:
            fed_check = True  # for evaluate_server function  - to check fed check has hitted
            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client


# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_clients, acc_avg_train_all, loss_avg_train_all, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train, loss_avg_all_user_train

    net_glob_server.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        # ---------forward prop-------------
        fx_server = net_glob_server(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            prGreen('Client{} Test =>\tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                          loss_avg_test))

            # Store the last accuracy and loss
            acc_avg_test_all = acc_avg_test
            loss_avg_test_all = loss_avg_test

            loss_test_collect_user.append(loss_avg_test_all)
            acc_test_collect_user.append(acc_avg_test_all)

            # if all users are served for one round ----------                    
            if fed_check:
                fed_check = False

                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    return


# ==============================================================================================================
#                                       Edges Side Program
# ==============================================================================================================
# Edge-side functions associated with Training and Testing
class Edge(object):
    def __init__(self, net_edge_model, idx, lr, device, cids):
        self.net = net_edge_model
        self.idx = idx
        self.device = device
        self.lr = lr
        self.selected_clients = cids

    def train(self, fx_client, labels, idx, len_batch):
        self.net.train()
        optimizer_edge = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # train and update
        optimizer_edge.zero_grad()

        fx_client = fx_client.to(device)
        labels = labels.to(device)

        # ---------forward prop-------------
        fx = self.net(fx_client)
        edge_fx = fx.clone().detach().requires_grad_(True)

        # Sending activations to server and receiving gradients from server
        dfx = train_server(edge_fx, labels, idx, len_batch)

        # --------backward prop--------------
        fx.backward(dfx)
        dfx_client = fx_client.grad.clone().detach()
        optimizer_edge.step()

        return dfx_client

    def evaluate(self, fx_client, labels, idx, len_batch, ell):
        self.net.eval()

        with torch.no_grad():
            fx_client = fx_client.to(device)
            labels = labels.to(device)
            # ---------forward prop-------------
            fx_edge = self.net(fx_client)
            evaluate_server(fx_edge, labels, self.idx, len_batch, ell)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))

        return


# ==============================================================================================================
#                                       Clients Side Program
# ==============================================================================================================
# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                 idxs_test=None):
        self.net = net_client_model
        self.idx = idx
        self.device = device
        self.lr = lr
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256 * 4, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256 * 4, shuffle=True)

    def train(self, net, edge):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

        len_batch = len(self.ldr_train)
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer_client.zero_grad()
            # ---------forward prop-------------
            fx = net(images)
            client_fx = fx.clone().detach().requires_grad_(True)
            dfx = edge.train(client_fx, labels, self.idx, len_batch)

            # --------backward prop -------------
            fx.backward(dfx)
            optimizer_client.step()

        return net.state_dict()

    def evaluate(self, net, ell, edge):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)
                edge.evaluate(fx, labels, self.idx, len_batch, ell)

        return


m = max(int(frac * num_clients), 1)
cids = np.arange(num_clients)
selected_cids = []
for i in range(num_edges):
    temp = np.random.choice(cids, int(m / num_edges), replace=False)
    selected_cids.append(temp)
    cids = list(set(cids) - set(temp))

# this epoch is global epoch, also known as rounds
for ep in range(epochs):
    for i in range(num_edges):
        edge = Edge(net_edge_model=net_glob_edge, idx=ep, lr=lr, device=device, cids=selected_cids[i])
        for idx in selected_cids[i]:
            local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                           idxs=dict_users[idx], idxs_test=dict_users_test[idx])
            # Training
            w_client = local.train(net=copy.deepcopy(net_glob_client).to(device), edge=edge)

            # Testing
            local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=ep, edge=edge)

            # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
            net_glob_client.load_state_dict(w_client)
print("Training and Evaluation completed!")
