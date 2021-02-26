
import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import Polytope_basics.simplex_coordinates2 as simplex_coordinates2
import numpy as np
import matplotlib.pyplot as plt
from sklearn  import preprocessing

verbose = False


# %%

def polygon(sides, radius=1, rotation=0, translation=None):
    one_segment = math.pi * 2 / sides
    points = [
        (math.sin(one_segment * i + rotation) * radius,
         math.cos(one_segment * i + rotation) * radius)
        for i in range(sides)]
    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]
    return points


# %%

class Fixed_weight_loss(nn.Module):
    def __init__(self ,device, D_simplex ,U_simplex, s=2.0, in_feature=10 ,out_feature=10):
        super(Fixed_weight_loss, self).__init__()

        self.device = device
        if D_simplex: self.d = out_feature - 1
        elif U_simplex: self.d = out_feature
        else: self.d = out_feature
        m = 0  # math.acos(-1/self.d)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.alpha = ( 1 -math.sqrt(self.d +1)) / self.d
        self.s = s
        if D_simplex or U_simplex:
            vertex_simplex = torch.Tensor(
                simplex_coordinates2.simplex_coordinates2(self.d, D=D_simplex, U=U_simplex)).permute(1, 0)
        elif not D_simplex and not U_simplex:
            self.polygon_vertex = polygon(out_feature)
            vertex_simplex = torch.Tensor(self.polygon_vertex)
            self.weight = Parameter(vertex_simplex)  # .permute(1,0))
        if D_simplex:
            lvalue = torch.stack([self.alpha * sum(vertex_simplex[i, :]) for i in range(vertex_simplex.size()[0])])
            self.weight = Parameter(torch.cat((vertex_simplex, lvalue.unsqueeze(1)), 1))
        elif U_simplex:
            self.weight = Parameter(vertex_simplex)
        self.weight.requires_grad = False

        # nn.init.xavier_uniform_(self.weight)
        if verbose: print("Norm_Arc_loss _ weight matrix {}".format(self.weight.size()))

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.mse_loss = nn.MSELoss()
        print("margin {}  cos_m {}  sin_m {}".format(m, self.sin_m, self.cos_m))

    def saveFigure(self, data, epoch, batch_id, folder_name, name_var):
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

        fig, ax = plt.subplots()
        A = data[1, :, :].cpu().detach().numpy()
        im = ax.imshow(A)
        cbar = ax.figure.colorbar(im)
        ax.set_xticks(np.arange(10), (classes))
        ax.set_yticks(np.arange(10), (classes))
        plt.savefig(folder_name + "/" + str(batch_id) + name_var + str(epoch) + ".jpg")

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            self.device)
        order_index.required_grad = True
        return torch.index_select(a, dim, order_index)

    def isnan(self, x):
        return x != x

    def dotproduct(self, v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    def length(self, v):
        return math.sqrt(self.dotproduct(v, v))

    def angle(self, v1, v2):
        return math.acos(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2)))

    def arc_loss(self, x, label, val=0):
        if verbose: print("ARC LOSS")
        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))
        m_c = v_c.view(x.size(0), 1, -1).repeat(1, len(self.weight), 1)
        cosine = []
        for i in range(len(m_c)):
            x_i = m_c[i, :]
            if (x_i.size()) != self.weight.size():
                print("x dimension and weight dimension do not match")
                break
            cosine_i = F.linear(F.normalize(x_i), F.normalize(self.weight))
            cosine.append(cosine_i)
        self.cosine = torch.stack(cosine)

        self.sine = torch.sqrt(1.0 - torch.pow(self.cosine, 2))
        self.phi = self.cosine * self.cos_m - self.sine * self.sin_m

        if val == 0:
            output = (label * self.phi) + ((1.0 - label) * self.cosine)
        else:
            output = self.cosine
        output = output * self.s

        return output

    def arc_loss2D(self, x, label, val=0):
        if verbose: print("ARC LOSS")

        # v_c = torch.sqrt((x**2).sum(dim=2,keepdim=True))
        m_c = x.view(x.size(0), 1, -1).repeat(1, len(self.weight), 1)
        cosine = []
        for i in range(len(m_c)):
            x_i = m_c[i, :, :]
            if not len(x_i) == len(self.weight):  # (x_i.size()) != self.weight.size():
                print("x dimension and weight dimension do not match")
                break
            cosine_i = F.linear(F.normalize(x_i), F.normalize(self.weight))
            cosine.append(cosine_i)
        self.cosine = torch.stack(cosine)
        self.sine = torch.sqrt((1.0 - torch.pow(self.cosine, 2)) + 1e-8)
        self.phi = self.cosine * self.cos_m - self.sine * self.sin_m

        if val == 0:
            output = (label * self.phi) + ((1.0 - label) * self.cosine)
        else:
            output = self.cosine
        output = output * self.s

        return output

    def digit_angle_loss(self, x, labels):
        margin = np.repeat(0.5, x[0, :].size()[1] - 1)
        loss_cosine = []
        for i in range(len(x)):
            ind = (labels[i, :] == 1).nonzero()
            x_class = (x[i, ind, :].squeeze()).unsqueeze(1).view(1, -1)
            if i == 0: print(ind)
            cosine = []
            for b in range(len(x[i, :])):
                if not ind == b:
                    x_b = x[i, b, :].view(1, -1)
                    cosine_b = F.linear(F.normalize(x_b), F.normalize(x_class))
                    cosine.append(math.acos(cosine_b))
            loss_cosine.append(cosine - margin)
        loss_cosine = torch.tensor(loss_cosine)
        loss_cosine.requires_grad = True
        loss = loss_cosine.sum(dim=1).mean()

        return loss

    def loss(self, data, x, target, reconstructions):  # <--------------------------------------ML+REC
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    # margin loss di capsule
    def margin_loss(self, x, labels, size_average=True):  # <------------------------------------ML
        if verbose: print("x {}".format(x.size()))
        if verbose: print("labels {}".format(labels.size()))
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))  # <-L2
        if verbose: print("v_c {}".format(v_c.size()))
        left = F.relu(0.9 - v_c).view(batch_size, -1)  # **2
        right = F.relu(v_c - 0.1).view(batch_size, -1)  # **2

        loss = labels * left + 0.5 * (1.0 - labels) * right

        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):  # <-------------------------------------REC
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005

    def forward(self, x, labels, L_angle):
        L_margin = self.margin_loss(x, labels)

        loss = L_angle + L_margin

# %%


