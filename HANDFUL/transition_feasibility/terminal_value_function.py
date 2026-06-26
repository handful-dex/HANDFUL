import numpy as np
import os
import torch
import random

from torch import nn

import pickle

class RetriGraspTValue(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RetriGraspTValue, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.output_layer = nn.Linear(128, output_dim)

        self.activate_func = nn.ELU()

    def forward(self, inputs):
        x = self.activate_func(self.linear1(inputs))
        x = self.activate_func(self.linear2(x))
        x = self.activate_func(self.linear3(x))
        outputs = self.output_layer(x)

        return outputs
    
class GraspInsertTValue(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraspInsertTValue, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 8)
        self.output_layer = nn.Linear(8, output_dim)

        self.activate_func = nn.ELU()
        self.dropout = nn.Dropout(p=0.3)

    
    def forward(self, inputs):
        x = self.dropout(self.activate_func(self.linear1(inputs)))
        x = self.dropout(self.activate_func(self.linear2(x)))
        x = self.dropout(self.activate_func(self.linear3(x)))
        outputs = self.output_layer(x)

        return outputs
    