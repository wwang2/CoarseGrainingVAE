import torch
from torch import nn 
import numpy as np
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from functools import partial

class shifted_softplus(torch.nn.Module):

    def __init__(self):
        super(shifted_softplus, self).__init__()

    def forward(self, input):
        return F.softplus(input) - np.log(2.0)

class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(x)

zeros_initializer = partial(constant_, val=0.0)

def preprocess_r(r_ij):

    dist = ((r_ij ** 2 + 1e-9).sum(-1)) ** 0.5
    unit = r_ij / dist.reshape(-1, 1)

    return dist, unit

layer_types = {
    "linear": torch.nn.Linear,
    "Tanh": torch.nn.Tanh,
    "ReLU": torch.nn.ReLU,
    "shifted_softplus": shifted_softplus,
    "sigmoid": torch.nn.Sigmoid,
    "Dropout": torch.nn.Dropout,
    "LeakyReLU": torch.nn.LeakyReLU,
    "ELU":  torch.nn.ELU,
    "swish": Swish
}


class CosineEnvelope(nn.Module):
    # Behler, J. Chem. Phys. 134, 074106 (2011)
    def __init__(self, cutoff):
        super().__init__()

        self.cutoff = cutoff

    def forward(self, d):

        output = 0.5 * (torch.cos((np.pi * d / self.cutoff)) + 1)
        exclude = d >= self.cutoff
        output[exclude] = 0

        return output

def gaussian_smearing(distances, offset, widths, centered=False):

    if not centered:
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances - offset

    else:
        coeff = -0.5 / torch.pow(offset, 2)
        diff = distances

    gauss = torch.exp(coeff * torch.pow(diff, 2))

    return gauss


class Dense(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        dropout_rate=0.0,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
    ):

        self.weight_init = weight_init
        self.bias_init = bias_init

        super().__init__(in_features, out_features, bias)

        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def reset_parameters(self):
        """
            Reinitialize model parameters.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        
        y = super().forward(inputs)

        # kept for compatibility with earlier versions of nff
        if hasattr(self, "dropout"):
            y = self.dropout(y)

        if self.activation:
            y = self.activation(y)

        return y

class GaussianSmearing(nn.Module):

    def __init__(self,
                 start,
                 stop,
                 n_gaussians,
                 centered=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        widths = torch.FloatTensor(
            (offset[1] - offset[0]) * torch.ones_like(offset))
        self.register_buffer("width", widths)
        self.register_buffer("offsets", offset)
        self.centered = centered

    def forward(self, distances):
        result = gaussian_smearing(
            distances, self.offsets, self.width, centered=self.centered
        )

        return result


class PainnRadialBasis(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff):
        super().__init__()

        self.n = torch.arange(1, n_rbf + 1).float()
        self.cutoff = cutoff

    def forward(self, dist):
        """
        Args:
            d (torch.Tensor): tensor of distances
        """

        shape_d = dist.unsqueeze(-1)
        n = self.n.to(dist.device)
        coef = n * np.pi / self.cutoff
        device = shape_d.device

        # replace divide by 0 with limit of sinc function

        denom = torch.where(shape_d == 0,
                            torch.tensor(1.0, device=device),
                            shape_d)
        num = torch.where(shape_d == 0,
                          coef,
                          torch.sin(coef * shape_d))

        output = torch.where(shape_d >= self.cutoff,
                             torch.tensor(0.0, device=device),
                             num / denom)

        return output


class DistanceEmbed(nn.Module):
    def __init__(self,
                 n_rbf,
                 cutoff,
                 feat_dim,
                 dropout):

        super().__init__()
        rbf = PainnRadialBasis(n_rbf=n_rbf,
                               cutoff=cutoff)
        dense = Dense(in_features=n_rbf,
                      out_features=feat_dim,
                      bias=True,
                      dropout_rate=dropout)
        self.block = nn.Sequential(rbf, dense)
        self.f_cut = CosineEnvelope(cutoff=cutoff)

    def forward(self, dist):
        rbf_feats = self.block(dist)
        envelope = self.f_cut(dist).reshape(-1, 1)
        output = rbf_feats * envelope

        return output


class SchNetEdgeFilter(nn.Module):
    def __init__(self,
                 cutoff,
                 n_gaussians,
                 n_filters,
                 activation='shifted_softplus'):

        super(SchNetEdgeFilter, self).__init__()

        self.filter = nn.Sequential(
            GaussianSmearing(
                start=0.0,
                stop=cutoff,
                n_gaussians=n_gaussians,
            ),
            Dense(
                in_features=n_gaussians,
                out_features=n_gaussians,
            ),
            layer_types[activation](),
            Dense(
                in_features=n_gaussians,
                out_features=n_filters,
            ))

    def forward(self, e):
        return self.filter(e)