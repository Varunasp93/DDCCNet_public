import psi4
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
import os
import time

# Creating NN object
    
class Linear_Block(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, num_in, num_nodes=196):
        super(Linear_Block, self).__init__()  
        self.num_in = num_in
        self.num_nodes = num_nodes
        
        self.L1 = nn.Linear(self.num_in, self.num_nodes) 
        self.AF1 = nn.ReLU()
        
        self.L2 = nn.Linear(self.num_nodes, self.num_nodes)
        self.AF2 = nn.ReLU()
        
        self.L3 = nn.Linear(self.num_nodes, self.num_nodes)
        self.AF3 = nn.ReLU()
        
        self.L4 = nn.Linear(self.num_nodes, self.num_nodes)
        self.AF4 = nn.ReLU()
        
        self.L5 = nn.Linear(self.num_nodes, self.num_nodes)
        self.AF5 = nn.ReLU()
        
        self.L6 = nn.Linear(self.num_nodes, self.num_nodes)
        self.AF6 = nn.ReLU()
        
        self.L7 = nn.Linear(self.num_nodes, self.num_nodes)
        self.AF7 = nn.ReLU()
        
        self.L_skip = nn.Linear(self.num_in, self.num_nodes) 
        
        
    def forward(self, x):
        
        x_out = self.L1(x)
        x_out = self.AF1(x_out)
        
        x_out = self.L2(x_out)
        x_out = self.AF2(x_out)
        
        x_out = self.L3(x_out)
        x_out = self.AF3(x_out)
        
        x_out = self.L4(x_out)
        x_skip = self.L_skip(x)
        x_out += x_skip
        x_out = self.AF4(x_out)
        
        x_out = self.L5(x_out)
        x_out = self.AF5(x_out)
        
        x_out = self.L6(x_out)
        x_out = self.AF6(x_out)
        
        x_out = self.L7(x_out)
        x_out = self.AF7(x_out)
        
        return x_out

class MLP_Amp(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, indMO_in, MOint_in, MOvec_in, Amp_in, Block_nodes=196):
        super(MLP_Amp, self).__init__()  
        self.indMO_in = indMO_in
        self.MOint_in = MOint_in
        self.MOvec_in = MOvec_in
        self.Amp_in = Amp_in
        self.Block_nodes = Block_nodes
        
        # Individual MOs
        self.indMO_block = Linear_Block(self.indMO_in, self.Block_nodes)
        
        # MO Interactions
        self.MOint_block = Linear_Block(self.MOint_in, self.Block_nodes)
        
        # MO vector
        self.MOvec_block = Linear_Block(self.MOvec_in, self.Block_nodes)
        
        # Amp_block
        self.Amp_block = Linear_Block(self.Amp_in, self.Block_nodes)
        
        # Combined_block
        self.Combined_block = Linear_Block(self.Block_nodes, self.Block_nodes)
        self.AF_combined = nn.ReLU()
        
        # Linear layer
        self.L1 = nn.Linear(self.Block_nodes, self.Block_nodes)
        self.AF1 = nn.ReLU()
        
        # Output layer
        self.L_out = nn.Linear(self.Block_nodes,1)
        
    def forward(self, x_indMO, x_MOint, x_MOvec, x_Amp):
        
        x_out_indMO = self.indMO_block(x_indMO)
        
        x_out_MOint = self.MOint_block(x_MOint)
        
        x_out_MOvec = self.MOvec_block(x_MOvec)
        
        x_out_Amp = self.Amp_block(x_Amp)
        
        x_combined = x_out_indMO+x_out_MOint+x_out_MOvec+x_out_Amp
        
        x = self.Combined_block(x_combined)
        x = self.AF_combined(x)
        
        x = self.L1(x)
        x += x_out_Amp
        x = self.AF1(x)
        
        x = self.L_out(x)
        
        return x  #Output gives amplitudes