# %%

from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
import time
import os
import adamw.cyclic_scheduler as CyclicLRWithRestarts
import adamw.adamw as AdamW
import Caps_basics.CapsNet_Layers_multiFC as CapsNet_Layers_MFC
import Caps_basics.CapsNet_Layers as CapsNet_Layers
import Caps_basics.ResNetCaps_E as ResNetCaps_E
import Fixed_weight_loss

# torch.autograd.set_detect_anomaly(True)

CUDA, USE_CUDA, model_name, db_used = "cuda:0", True, "CapsNet_MR", "cifar10"
n_epochs, ADAM_LR, ADAM_WD = 100, 0.001, 0.0
lossAN, lossML = True, True
D_simplex, U_simplex = False, False  # <------------------False False = POLYGON
onlyML, ML_REC, onlyREC = False, True, False
batch_size = 128
alpha = 0.3

folder_results = "ALPHA_Resumed_Fixed_weight_loss_train_moreFC_" + db_used
implementation_name = str(alpha) + db_used
imageFolder = 'missClassifiedImages'
#######################
if not os.path.exists(folder_results):
    os.mkdir(folder_results)
if not os.path.exists(os.path.join(folder_results, str(alpha) + "model_log")) and os.path.exists(folder_results):
    os.mkdir(os.path.join(folder_results, str(alpha) + "model_log"))
implementation_folder_name = os.path.join(folder_results, implementation_name)

with open(implementation_folder_name + ".txt", "w") as text_file:
    text_file.write(
        "ADAM_LR {} ADAM_WD {} USE CUDA {} ON {} model {} dataset {} loss AN {} ML {}".format(ADAM_LR, ADAM_WD,
                                                                                              USE_CUDA, CUDA,
                                                                                              model_name, db_used,
                                                                                              lossAN, lossML))
    text_file.write("loss type:onlyML {} ML_REC {} onlyREC {}".format(onlyML, ML_REC, onlyREC))
with open(implementation_folder_name + "TEST.txt", "w") as text_file:
    text_file.write("ALPHA {}, database {}\n".format(str(alpha), db_used))


def lr_decrease(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        init_lr = param_group['lr']
        param_group['lr'] = init_lr * lr_decay


def isnan(x):
    return x != x


def compute_loss(LossAN, LossML, loss_AN, loss_ML, alpha):
    if LossAN and not LossML:
        loss = loss_AN
    elif LossML and not LossAN:
        loss = loss_ML
    elif LossML and LossAN:
        loss = alpha * loss_AN + (1 - alpha) * loss_ML
    return loss


def resume_model(name_file, model, optimizer, map_location):
    if os.path.isfile(name_file):
        print("=> loading checkpoint '{}'".format(name_file))
        checkpoint = torch.load(name_file, map_location=map_location)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(name_file, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(name_file))

    return start_epoch, model, optimizer


###########################DATASET
resize_dim = (32, 32)

dataset_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

NUM_CLASSES = 10
print("MNIST")
image_datasets = {'train': datasets.MNIST('../data', train=True, download=True, transform=dataset_transform),
                  'val': datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)}
print("Initializing Datasets and Dataloaders...")
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
               'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True)}
print("Initializing Datasets and Dataloaders...")
MNIST_bo = True

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

############################################ MODEL
USE_CUDA = True
device = torch.device(CUDA if torch.cuda.is_available() else "cpu")

model = CapsNet_Layers_MFC.CapsNet_MR(NUM_CLASSES, D_simplex, U_simplex, MNIST=MNIST_bo, smallNORB=False)

optimizer = AdamW.AdamW(model.parameters(), lr=ADAM_LR, weight_decay=ADAM_WD)
scheduler = CyclicLRWithRestarts.CyclicLRWithRestarts(optimizer, batch_size, 60000, restart_period=5, t_mult=1.2,
                                                      policy="cosine")
criterion = nn.CrossEntropyLoss().to(device)
model_path = 'model/cifar10/checkpoint__CapsNet_MR.pth.tar'

start_epoch, model, optimizer = resume_model(model_path, model, optimizer, map_location=CUDA)
for param in model.parameters():
    param.requires_grad = False
if USE_CUDA:
    model = model.to(device)  # cuda()
    print('cuda')

FC_layers = CapsNet_Layers_MFC.FC_layer_single(NUM_CLASSES)
for param in FC_layers.parameters():
    param.requires_grad = True
FC_layers.to(device)

model_FC = nn.ModuleList()
model_FC.append(model)
model_FC.append(FC_layers)
model_FC.to(device)

criterion = nn.CrossEntropyLoss().to(device)
######NEWLOSS
criterionNew = Fixed_weight_loss.Fixed_weight_loss(device, False, False, in_feature=NUM_CLASSES,
                                                   out_feature=NUM_CLASSES)
criterionNew = criterionNew.to(device)
#############


# %%
model.eval()
test_loss, test_accuracy = 0, 0

start = time.time()

for batch_id, (data, target) in enumerate(dataloaders['val']):
    target = torch.eye(NUM_CLASSES).index_select(dim=0, index=target)
    data, target = Variable(data), Variable(target)
    data, target = data.to(device), target.to(device)  # .cuda()

    target_m = []
    for i in range(len(target)):
        n_loc = (target[i, :] == 1).nonzero()
        m = torch.zeros(NUM_CLASSES, NUM_CLASSES)
        m[n_loc, n_loc] = 1
        target_m.append(m)
    target_m = torch.stack(target_m).to(device)
    output_digit, reconstruction, masked, output_fc = model(data)
    output = output_digit
    #########NEWLOSS########
    L_angle = criterionNew.arc_loss(output.squeeze(), target_m, val=1)

    b = []
    for i in range(len(L_angle)):
        b.append(torch.diag(L_angle[i]))
    b = torch.stack(b)

    _, label = torch.max(target, 1)
    loss_AN = criterion(b, label.long())
    if lossML:
        if onlyML:
            loss_ML = criterionNew.margin_loss(output_digit, target)
        elif ML_REC:
            loss_ML = criterionNew.loss(data, output_digit, target, reconstruction)
        elif onlyREC:
            loss_ML = criterionNew.reconstruction_loss(data, reconstruction)
    else:
        loss_ML = 0.0
    loss = compute_loss(loss_AN, loss_ML)
    test_loss += float(loss.data)
    test_accuracy += float(
        sum(np.argmax(b.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))) / float(
        data.size(0))

    if batch_id % 100 == 0:
        print("test accuracy:".format(batch_id, sum(np.argmax(b.data.cpu().numpy(), 1) ==
                                                              np.argmax(target.data.cpu().numpy(), 1)) / float(
            data.size(0))))
        print("loss {} margin loss {}".format(batch_id, loss.data, loss_ML))
end = time.time()
print("TEST: Validation time execution {}".format(end - start))
print("TEST: Loss value for test phase: {}".format(test_loss / len(dataloaders['val'])))
print("TEST: Accuracy value for test phase: {} %".format((test_accuracy / len(dataloaders['val'])))*100)
torch.cuda.empty_cache()



