import torch
from torch.utils.data import DataLoader, Dataset

import random
import numpy as np
import copy
from get_data import get_mnist, get_cifar10, dataset_iid, DatasetSplit
from LeNet_model_MNIST_disHier import *

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


num_clients = 5
epochs = 50
frac = 1  # participation of clients; if 1 then 100% clients participate in SL
lr = 0.001


# =====================================================================================================
#                           Client-side Model definition
# =====================================================================================================
# Model at client side
net_glob_client = LeNet_client_side()
if torch.cuda.device_count() > 1:
    print("We use", torch.cuda.device_count(), "GPUs")
    net_glob_client = nn.DataParallel(
        net_glob_client)  # to use the multiple GPUs

net_glob_client.to(device)
print(net_glob_client)

# ===================================================================================
# For Loss and Accuracy
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


# ====================================================================================================
#                                  Server Side Program
# ====================================================================================================
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
fed_check = False


# server-side function associated with Training
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

    # server-side model net_glob_server is global so it is updated automatically in each pass to this function
    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train =>\tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_train,
                                                                                      loss_avg_train))

        # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
        # this is because we work on the last trained model and its accuracy (not earlier cases)

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
            # all users served for one round ------------------------- output print and update is done in evaluate_server()
            # for nicer display

            idx_collect = []

            acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)

            loss_train_collect.append(loss_avg_all_user_train)
            acc_train_collect.append(acc_avg_all_user_train)

            acc_train_collect_user = []
            loss_train_collect_user = []

    # send gradients to the client
    return dfx_client


# server-side functions associated with Testing
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

                print("====================== EDGE V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user_train,
                                                                                          loss_avg_all_user_train))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    return


# ==============================================================================================================
#                                       Clients Side Program
# ==============================================================================================================
# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, device, dataset_train=None, dataset_test=None, idxs=None,
                 idxs_test=None):
        self.idx = idx
        self.device = device

        self.lr = lr
        # self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=256 * 4, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=256 * 4, shuffle=True)

    def train(self, net):
        net.train()
        optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

        len_batch = len(self.ldr_train)
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer_client.zero_grad()
            # ---------forward prop-------------
            fx = net(images)
            client_fx = fx.clone().detach().requires_grad_(True)

            # Sending activations to server and receiving gradients from server
            dfx = train_server(client_fx, labels, self.idx, len_batch)

            # --------backward prop -------------
            fx.backward(dfx)
            optimizer_client.step()

        # prRed('Client{} Train => Epoch: {}'.format(self.idx, ell))

        return net.state_dict()

    def evaluate(self, net, ell):
        net.eval()

        with torch.no_grad():
            len_batch = len(self.ldr_test)
            for batch_idx, (images, labels) in enumerate(self.ldr_test):
                images, labels = images.to(self.device), labels.to(self.device)
                # ---------forward prop-------------
                fx = net(images)

                # Sending activations to server
                evaluate_server(fx, labels, self.idx, len_batch, ell)

            # prRed('Client{} Test => Epoch: {}'.format(self.idx, ell))

        return


dataset_train, dataset_test = get_mnist()

dict_users = dataset_iid(dataset_train, num_clients)
dict_users_test = dataset_iid(dataset_test, num_clients)

# this epoch is global epoch, also known as rounds
for iter in range(epochs):
    m = max(int(frac * num_clients), 1)
    idxs_users = np.random.choice(range(num_clients), m, replace=False)

    # Sequential training/testing among clients
    for idx in idxs_users:
        local = Client(net_glob_client, idx, lr, device, dataset_train=dataset_train, dataset_test=dataset_test,
                       idxs=dict_users[idx], idxs_test=dict_users_test[idx])
        # Training
        w_client = local.train(net=copy.deepcopy(net_glob_client).to(device))

        # Testing
        local.evaluate(net=copy.deepcopy(net_glob_client).to(device), ell=iter)

        # copy weight to net_glob_client -- use to update the client-side model of the next client to be trained
        net_glob_client.load_state_dict(w_client)

print("Training and Evaluation completed!")
