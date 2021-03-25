import torch
import torch.nn as nn

#normalize the input image data
class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:,i] = (x[:,i] - self.mean[i])/self.std[i]

        return x

#swap input dimensions for preprocessing pur·pose.
class Permute(nn.Module):

    def __init__(self, permutation = [2,1,0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        
        return input[:, self.permutation]
