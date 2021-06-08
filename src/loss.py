import torch

def KL(mu, std):
     return -0.5 * torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2), dim=-1).mean()