import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch.nn.functional as F

import csv
from skimage import io

from PIL import Image
import pandas as pd

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import time
import os
import copy
import adamw.cyclic_scheduler as CyclicLRWithRestarts
import adamw.adamw as AdamW
import Caps_basics.CapsNet_Layers_multiFC as CapsNet_Layers_MFC
import Caps_basics.CapsNet_Layers as CapsNet_Layers
import Caps_basics.ResNetCaps_E as ResNetCaps_E
import Fixed_weight_loss
import smallNorb as small_norb


# torch.autograd.set_detect_anomaly(True)


class Fixed_weight_loss_MFC(nn.Module):

    def __init__(self, dataset="mnist", batch_size=128, n_epochs=100, bool_alpha=True, alpha=0.0, save_files=False):
        super(Fixed_weight_loss_MFC, self).__init__()
        ############SET VAR###########
        self.CUDA, self.USE_CUDA = "cuda:0", True  # <-------------------------cuda

        self.model_name, self.FC, self.CV, self.db_used = "CapsNet_MR", False, False, dataset  # <----model

        self.n_epochs, self.batch_size, self.ADAM_LR, self.ADAM_WD = n_epochs, batch_size, 0.001, 1e-7  # <---optimizer

        self.lossAN, self.lossML = True, True  # <-----------------------loss required
        self.onlyML, self.ML_REC, self.onlyREC = True, False, False  # <-----------------loss computed
        self.D_simplex, self.U_simplex = False, True  # <----------simplex distribution

        self.alpha, self.bool_alpha = alpha, bool_alpha  # <-------------------------alpha

        #########################################LOG if required########################
        self.save_files = save_files
        if self.save_files:
            print(
                " saved in {}: train {}, test {} ".format(self.folder_results, self.implementation_folder_name + ".txt",
                                                          self.implementation_folder_name + "TEST.txt"))
            self.folder_results = "ALPHA_U_NoRec_Fixed_weight_loss_train_moreFC_" + self.db_used
            self.implementation_name = str(self.alpha) + self.db_used
            if not os.path.exists(self.folder_results):
                os.mkdir(self.folder_results)
            if not os.path.exists(os.path.join(self.folder_results, str(alpha) + "model_log")):
                os.mkdir(os.path.join(self.folder_results, str(alpha) + "model_log"))
            self.implementation_folder_name = os.path.join(self.folder_results, self.implementation_name)
            with open(self.implementation_folder_name + ".txt", "w") as text_file:
                text_file.write(
                    "self.ADAM_LR {} self.ADAM_WD {} USE CUDA {} ON {} model {} dataset {} loss AN {} ML {}\n".format(
                        self.ADAM_LR, self.ADAM_WD, self.USE_CUDA, self.CUDA, self.model_name, self.db_used,
                        self.lossAN, self.lossML))
                text_file.write(
                    "loss type:self.onlyML {} self.ML_REC {} self.onlyREC {}\n".format(self.onlyML, self.ML_REC,
                                                                                       self.onlyREC))
                text_file.write("self.D_simplex {} self.U_simplex {}\n".format(self.D_simplex, self.U_simplex))

            with open(self.implementation_folder_name + "TEST.txt", "w") as text_file:
                text_file.write("ALPHA {}, database {}\n".format(self.alpha, self.db_used))
        ################################################################
        print(
            "self.ADAM_LR {} self.ADAM_WD {} USE CUDA {} ON {} model {}; \n self.D_simplex {} self.U_simplex {}\n".format(
                self.ADAM_LR, self.ADAM_WD, self.USE_CUDA, self.CUDA, self.model_name, self.D_simplex, self.U_simplex))
        print("Loss Type: ", self.lossAN, self.lossML)
        print("loss type:self.onlyML {} self.ML_REC {} self.onlyREC {}".format(self.onlyML, self.ML_REC, self.onlyREC))

        ################################setting dataset#####################################
        resize_dim = (32, 32)
        self.MNIST_bo, self.smallNORB_bo = False, False
        if self.db_used == 'cifar10':
            dataset_transform = transforms.Compose([
                transforms.Resize(resize_dim),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

            self.NUM_CLASSES = 10
            print("CIFAR10")
            self.image_datasets = {
                'train': datasets.CIFAR10('../data', train=True, download=True, transform=dataset_transform),
                'val': datasets.CIFAR10('../data', train=False, download=True, transform=dataset_transform)}
            print("Initializing Datasets and Dataloaders...")
            self.dataloaders = {
                'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=self.batch_size,
                                                     shuffle=True),
                'val': torch.utils.data.DataLoader(self.image_datasets['val'], batch_size=self.batch_size,
                                                   shuffle=True)}
            print("Initializing Datasets and Dataloaders...")
        elif self.db_used == 'mnist':
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.NUM_CLASSES = 10
            print("MNIST")
            self.image_datasets = {
                'train': datasets.MNIST('../data', train=True, download=True, transform=dataset_transform),
                'val': datasets.MNIST('../data', train=False, download=True, transform=dataset_transform)}
            print("Initializing Datasets and Dataloaders...")
            self.dataloaders = {
                'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=self.batch_size,
                                                     shuffle=True),
                'val': torch.utils.data.DataLoader(self.image_datasets['val'], batch_size=self.batch_size,
                                                   shuffle=True)}
            print("Initializing Datasets and Dataloaders...")
            self.MNIST_bo = True
        elif self.db_used == 'smallnorb':
            train_transform = transforms.Compose([
                transforms.Resize(48),
                transforms.RandomCrop(32),
                transforms.ColorJitter(brightness=32. / 255, contrast=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.0,), (0.3081,))
            ])
            test_transform = transforms.Compose([
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.,), (0.3081,))
            ])
            self.NUM_CLASSES = 5
            print("SMALLNORB")
            path = os.path.join("/home/rita/JupyterProjects/EYE-SEA/DataSets", "small_norb_root")
            self.image_datasets = {
                'train': small_norb.SmallNORB(path, train=True, download=True, transform=train_transform),
                'val': small_norb.SmallNORB(path, train=False, transform=test_transform)}
            print("Initializing Datasets and Dataloaders...")
            self.dataloaders = {
                'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=self.batch_size,
                                                     shuffle=True),
                'val': torch.utils.data.DataLoader(self.image_datasets['val'], batch_size=self.batch_size,
                                                   shuffle=True)}
            print("Initializing Datasets and Dataloaders...")
            self.smallNORB_bo = True
        else:
            print('Unknown dataset')

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}

    def lr_decrease(self, optimizer, lr_decay):
        for param_group in optimizer.param_groups:
            init_lr = param_group['lr']
            param_group['lr'] = init_lr * lr_decay

    def isnan(self, x):
        return x != x

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def compute_loss(self, loss_AN, loss_ML):
        if self.lossAN and not self.lossML:
            loss = loss_AN
        elif self.lossAN and not self.lossML:
            loss = loss_ML
        elif self.lossML and self.lossAN and not self.bool_alpha:
            loss = loss_AN + loss_ML
        elif self.lossML and self.lossAN and self.bool_alpha:
            loss = self.alpha * loss_AN + (1 - self.alpha) * loss_ML
        return loss

    # In[ ]:
    def procedure_train(self):
        print("=> using model CapsuleNET with the new loss")
        device = torch.device(self.CUDA if torch.cuda.is_available() else "cpu")
        model = CapsNet_Layers_MFC.CapsNet_MR(self.NUM_CLASSES, self.FC, self.CV, MNIST=False,
                                              smallNORB=self.smallNORB_bo)

        if self.USE_CUDA:
            model = model.to(device)
            print('load model on cuda')

        optimizer = AdamW.AdamW(model.parameters(), lr=self.ADAM_LR, weight_decay=self.ADAM_WD)
        scheduler = CyclicLRWithRestarts.CyclicLRWithRestarts(optimizer, self.batch_size,
                                                              len(self.image_datasets['train']), restart_period=5,
                                                              t_mult=1.2, policy="cosine")
        criterion = nn.CrossEntropyLoss().to(device)
        ######NEWLOSS
        criterionNew = Fixed_weight_loss.Fixed_weight_loss(device, self.D_simplex, self.U_simplex,
                                                           in_feature=self.NUM_CLASSES, out_feature=self.NUM_CLASSES)
        criterionNew = criterionNew.to(device)
        #############

        accuracy_train, loss_train, loss_train_AN, loss_train_ML = [], [], [], []

        start = time.time()
        for epoch in range(self.n_epochs):
            model.train()
            scheduler.step()  # <----------------------------------------------adamwr
            print('TRAINING: epoch {}:{}'.format(epoch + 1, self.n_epochs))
            train_loss, train_loss_angle, train_loss_margin, train_accuracy = 0, 0, 0, 0
            losted = 0
            batch_accuracy = []

            for batch_id, (data, target) in enumerate(self.dataloaders['train']):
                if self.save_files and batch_id == 0:
                    fig, ax = plt.subplots()
                    A = data[1, 0, :, :].numpy()
                    im = ax.imshow(A)
                    plt.savefig(str(batch_id) + "image" + str(target[1].item) + str(epoch) + ".jpg")

                target = torch.eye(self.NUM_CLASSES).index_select(dim=0, index=target)
                data, target = Variable(data), Variable(target)
                data, target = data.to(device), target.to(device)  # .cuda()

                target_m = []
                for i in range(len(target)):
                    n_loc = (target[i, :] == 1).nonzero()
                    m = torch.zeros(self.NUM_CLASSES, self.NUM_CLASSES)
                    m[n_loc, n_loc] = 1
                    target_m.append(m)
                del m
                target_m = torch.stack(target_m).to(device)

                optimizer.zero_grad()

                output_digit, recostruction, masked, output_fc = model(data)
                if self.FC or self.CV:
                    output = output_fc.view(output_fc.size(0), self.NUM_CLASSES, -1)
                else:
                    output = output_digit

                #########NEWLOSS########
                L_angle = criterionNew.arc_loss(output.squeeze(), target_m, val=0)
                del target_m
                #############################################only diagonal#####################################################
                b = []
                for i in range(len(L_angle)):
                    b.append(torch.diag(L_angle[i]))
                b = torch.stack(b)
                _, label = torch.max(target, 1)

                loss_AN = criterion(b, label.long())

                ################################################################################################################
                ############################################### marginal loss ##################################################
                if self.lossML:
                    if self.onlyML:
                        loss_ML = criterionNew.margin_loss(output_digit, target)
                    elif self.ML_REC:
                        loss_ML = criterionNew.loss(data, output_digit, target, recostruction)
                    elif self.onlyREC:
                        loss_ML = criterionNew.reconstruction_loss(data, recostruction)
                else:
                    loss_ML = loss_AN
                loss = self.compute_loss(loss_AN, loss_ML)
                ################################################################################################################

                if self.isnan(loss):
                    print("loss lost batch_id", batch_id)
                    losted += 1
                else:

                    loss.backward()
                    optimizer.step()
                    # print( batch_id)
                    scheduler.batch_step()  # <--------------------------------------------adamwr
                    train_loss += float(loss.data)
                    train_accuracy += float(
                        sum(np.argmax(b.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1)) / float(
                            data.size(0)))
                    train_loss_angle += float(loss_AN.data)
                    train_loss_margin += float(loss_ML.data)

                    if batch_id % 1000 == 0:
                        print("train diag accuracy: {}%".format(sum(np.argmax(b.data.cpu().numpy(), 1) ==
                                                                             np.argmax(target.data.cpu().numpy(),
                                                                                       1)) / float(data.size(0))*100))
                        print("angle loss {} margin loss {}".format(batch_id, loss_AN.data, loss_ML))

                        batch_accuracy.append(float(sum(np.argmax(b.data.cpu().numpy(), 1) ==
                                                        np.argmax(target.data.cpu().numpy(), 1)) / float(data.size(0))))
                del data, target
            accuracy_train.append(np.mean(batch_accuracy))
            loss_train.append(train_loss / (len(self.dataloaders['train']) - losted))
            loss_train_AN.append(train_loss_angle / (len(self.dataloaders['train']) - losted))
            loss_train_ML.append(train_loss_margin / (len(self.dataloaders['train']) - losted))

            del b, L_angle, loss_AN, loss_ML, output_digit, output_fc, masked, batch_accuracy

            if epoch % 20 == 0 and not epoch == 0:
                test_loss, test_accuracy = 0, 0

                start_test = time.time()

                for batch_id, (data, target) in enumerate(self.dataloaders['val']):

                    target = torch.eye(self.NUM_CLASSES).index_select(dim=0, index=target)
                    data, target = Variable(data), Variable(target)
                    data, target = data.to(device), target.to(device)  # .cuda()

                    target_m = []
                    for i in range(len(target)):
                        n_loc = (target[i, :] == 1).nonzero()
                        m = torch.zeros(self.NUM_CLASSES, self.NUM_CLASSES)
                        m[n_loc, n_loc] = 1
                        target_m.append(m)
                    del m
                    target_m = torch.stack(target_m).to(device)
                    output_digit, _, masked, output_fc = model(data)
                    if self.FC:
                        output = output_fc.view(output_fc.size(0), self.NUM_CLASSES, -1)
                    else:
                        output = output_digit
                    #########NEWLOSS########
                    L_angle = criterionNew.arc_loss(output.squeeze(), target_m, epoch, batch_id, "heatmap/", val=1)
                    del target_m
                    b = []
                    for i in range(len(L_angle)):
                        b.append(torch.diag(L_angle[i]))
                    b = torch.stack(b)

                    _, label = torch.max(target, 1)
                    loss_AN = criterion(b, label.long())
                    if self.lossML:
                        if self.onlyML:
                            loss_ML = criterionNew.margin_loss(output_digit, target)
                        elif self.ML_REC:
                            loss_ML = criterionNew.loss(data, output_digit, target, recostruction)
                        elif self.onlyREC:
                            loss_ML = criterionNew.reconstruction_loss(data, recostruction)
                    else:
                        loss_ML = 0.0
                    loss = self.compute_loss(loss_AN, loss_ML)

                    test_loss += float(loss.data)
                    test_accuracy += float(
                        sum(np.argmax(b.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1)) / float(
                            data.size(0)))

                    if batch_id % 100 == 0:
                        print("test accuracy: {}%".format(sum(np.argmax(b.data.cpu().numpy(), 1) ==
                                                                                np.argmax(target.data.cpu().numpy(),
                                                                                          1)) / float(data.size(0))*100))
                        print("loss {} margin loss {}".format(loss_AN.data, loss_ML))
                    del data, target, b
                end_test = time.time()

                print(
                    "VALIDATION{}:Validation time execution {}".format(epoch, end_test - len(self.dataloaders['val'])))
                print("VALIDATION{}:Loss value for test phase: {}".format(epoch,
                                                                          test_loss / len(self.dataloaders['val'])))
                print("VALIDATION{}:Accuracy value for test phase: {}".format(epoch, test_accuracy / len(
                    self.dataloaders['val'])))
                if self.save_files:
                    with open(self.implementation_folder_name + "TEST.txt", "a") as text_file:
                        text_file.write("Validation time execution {}\n".format(end_test - start_test))
                        text_file.write(
                            "Loss value for test phase: {}\n".format(test_loss / len(self.dataloaders['val'])))
                        text_file.write(
                            "Accuracy value for test phase: {}\n".format(test_accuracy / len(self.dataloaders['val'])))
                del test_loss, test_accuracy, loss, loss_AN, loss_ML, output_digit, output_fc, masked, output
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'loss_type': self.implementation_name,
                    'arch': 'CapsNet',
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, self.folder_results + "/" + str(
                    self.alpha) + "model_log/checkpoint_" + "_" + self.model_name + "_" + str(epoch) + ".pth.tar")

        end = time.time()
        print("TRAIN: Training time execution {}".format(end - start))
        print("TRAIN: Loss value for training phase: {}".format(train_loss / len(self.dataloaders['train'])))
        print("TRAIN: Accuracy value for training phase: {}".format(train_accuracy / len(self.dataloaders['train'])))
        if self.save_files:
            with open(self.implementation_folder_name + ".txt", "a") as text_file:
                text_file.write("Training time execution {}\n".format(end - start))
                text_file.write(
                    "Loss value for training phase: {}\n".format(train_loss / len(self.dataloaders['train'])))
                text_file.write(
                    "Accuracy value for training phase: {}\n".format(train_accuracy / len(self.dataloaders['train'])))
            self.save_checkpoint({
                'epoch': epoch + 1,
                'loss_type': self.implementation_name,
                'arch': 'CapsNet',
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, self.folder_results + "/" + str(
                self.alpha) + "model_log/checkpoint_" + "_" + self.model_name + "_" + str(epoch) + ".pth.tar")

            epochs = np.arange(1, self.n_epochs + 1)
            plt.plot(epochs, loss_train, color='g')
            plt.plot(epochs, loss_train_AN, color='b')
            plt.plot(epochs, loss_train_ML, color='c')
            plt.plot(epochs, accuracy_train, color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy - Loss')
            plt.title('Training phase')
            plt.savefig(self.folder_results + "/" + self.implementation_name + ".png")

        model.eval()
        test_loss, test_accuracy = 0, 0

        start = time.time()

        for batch_id, (data, target) in enumerate(self.dataloaders['val']):
            target = torch.eye(self.NUM_CLASSES).index_select(dim=0, index=target)
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)  # .cuda()

            target_m = []
            for i in range(len(target)):
                n_loc = (target[i, :] == 1).nonzero()
                m = torch.zeros(self.NUM_CLASSES, self.NUM_CLASSES)
                m[n_loc, n_loc] = 1
                target_m.append(m)
            target_m = torch.stack(target_m).to(device)
            output_digit, _, masked, output_fc = model(data)
            if self.FC or self.CV:
                output = output_fc.view(output_fc.size(0), self.NUM_CLASSES, -1)
            else:
                output = output_digit
            #########NEWLOSS########
            L_angle = criterionNew.arc_loss(output.squeeze(), target_m, val=1)

            b = []
            for i in range(len(L_angle)):
                b.append(torch.diag(L_angle[i]))
            b = torch.stack(b)

            _, label = torch.max(target, 1)
            loss_AN = criterion(b, label.long())
            if self.lossML:
                if self.onlyML:
                    loss_ML = criterionNew.margin_loss(output_digit, target)
                elif self.ML_REC:
                    loss_ML = criterionNew.loss(data, output_digit, target, recostruction)
                elif self.onlyREC:
                    loss_ML = criterionNew.reconstruction_loss(data, recostruction)
            else:
                loss_ML = 0.0
            loss = self.compute_loss(loss_AN, loss_ML)
            test_loss += float(loss.data)
            test_accuracy += float(
                sum(np.argmax(b.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))) / float(
                data.size(0))

            if batch_id % 100 == 0:
                print("BATCH_{}: test accuracy:".format(batch_id, sum(np.argmax(b.data.cpu().numpy(), 1) ==
                                                                      np.argmax(target.data.cpu().numpy(), 1)) / float(
                    data.size(0))))
                print("BATCH_{}:loss {} margin loss {}".format(batch_id, loss.data, loss_ML))
        end = time.time()
        print("TEST: Validation time execution {}".format(end - start))
        print("TEST: Loss value for test phase: {}".format(test_loss / len(self.dataloaders['val'])))
        print("TEST: Accuracy value for test phase: {}".format(test_accuracy / len(self.dataloaders['val'])))
        if self.save_files:
            with open(self.implementation_folder_name + "TEST.txt", "a") as text_file:
                text_file.write("Validation time execution {}\n".format(end - start))
                text_file.write("Loss value for test phase: {}\n".format(test_loss / len(self.dataloaders['val'])))
                text_file.write(
                    "Accuracy value for test phase: {}\n".format(test_accuracy / len(self.dataloaders['val'])))

        torch.cuda.empty_cache()






