import psi4
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
import os
import time

# Creating NN object
    
class MLP_t1_test(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()  
        num_nodes = 196
        
        # T1 block
        self.t1_L1 = nn.Linear(14, 196) 
        self.t1_AF1 = nn.ReLU()
        
        self.t1_L2 = nn.Linear(196, 196)
        self.t1_AF2 = nn.ReLU()
        
        self.t1_L3 = nn.Linear(196, 196)
        self.t1_AF3 = nn.ReLU()
        
        self.t1_L4 = nn.Linear(196, 196)
        self.t1_AF4 = nn.ReLU()
        
        self.t1_L5 = nn.Linear(196, 196)
        self.t1_AF5 = nn.ReLU()
        
        self.t1_L6 = nn.Linear(196, 196)
        self.t1_AF6 = nn.ReLU()
        
        self.t1_L7 = nn.Linear(196, 196)
        self.t1_AF7 = nn.ReLU()
        
        self.t1_L8 = nn.Linear(196, 1)
        
    def forward(self, x_t1):
        
        # Passing input through T1 layer
        
        x_t1 = self.t1_L1(x_t1)
        x_t1 = self.t1_AF1(x_t1)
        
        x_t1 = self.t1_L2(x_t1)
        x_t1 = self.t1_AF2(x_t1)
        
        x_t1 = self.t1_L3(x_t1)
        x_t1 = self.t1_AF3(x_t1)
        
        x_t1 = self.t1_L4(x_t1)
        x_t1 = self.t1_AF4(x_t1)
        
        x_t1 = self.t1_L5(x_t1)
        x_t1 = self.t1_AF5(x_t1)
        
        x_t1 = self.t1_L6(x_t1)
        x_t1 = self.t1_AF6(x_t1)
        
        x_t1 = self.t1_L7(x_t1)
        x_t1 = self.t1_AF7(x_t1)
        
        x_t1 = self.t1_L8(x_t1)
        
        return x_t1  #Output gives t1 amplitudes

class MLP_t2_test(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self):
        super().__init__()  
        num_nodes = 128
        
        # T1 block
        self.t2_L1 = nn.Linear(30, 196) # Six more features added to the T1 block
        self.t2_AF1 = nn.ReLU()
        
        self.t2_L2 = nn.Linear(196, 196)
        self.t2_AF2 = nn.ReLU()
        
        self.t2_L3 = nn.Linear(196, 196)
        self.t2_AF3 = nn.ReLU()
        
        self.t2_L4 = nn.Linear(196, 196)
        self.t2_AF4 = nn.ReLU()
        
        self.t2_L5 = nn.Linear(196, 196)
        self.t2_AF5 = nn.ReLU()
        
        self.t2_L6 = nn.Linear(196, 196)
        self.t2_AF6 = nn.ReLU()
        
        self.t2_L7 = nn.Linear(196, 196)
        self.t2_AF7 = nn.ReLU()
        
        self.t2_L8 = nn.Linear(196, 1)
        
    def forward(self, x_t2):
        
        # Passing input through T1 layer
        
        x_t2 = self.t2_L1(x_t2)
        x_t2 = self.t2_AF1(x_t2)
        
        x_t2 = self.t2_L2(x_t2)
        x_t2 = self.t2_AF2(x_t2)
        
        x_t2 = self.t2_L3(x_t2)
        x_t2 = self.t2_AF3(x_t2)
        ########Best R2 (Not stable Learning curve)##########
        x_t2 = self.t2_L4(x_t2)
        x_t2 = self.t2_AF4(x_t2)
        
        x_t2 = self.t2_L5(x_t2)
        x_t2 = self.t2_AF5(x_t2)
       ### Stable ###############
    
        x_t2 = self.t2_L6(x_t2)
        x_t2 = self.t2_AF6(x_t2)
        
        x_t2 = self.t2_L7(x_t2)
        x_t2 = self.t2_AF7(x_t2)
        
        x_t2 = self.t2_L8(x_t2)
        
        return x_t2  #Output gives t2 amplitudes