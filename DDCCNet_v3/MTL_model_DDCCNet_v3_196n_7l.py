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
        
        self.L8 = nn.Linear(self.num_nodes, 1)
        
        self.L_skip = nn.Linear(self.num_in, self.num_nodes) 
        
        
    def forward(self, x):
        
        x_out = self.L1(x)
        x_out = self.AF1(x_out)
        
        x_out = self.L2(x_out)
        x_out = self.AF2(x_out)
        
        x_out = self.L3(x_out)
        x_out = self.AF3(x_out)
        
        x_out = self.L4(x_out)
        x_out = self.AF4(x_out)
        
        x_out = self.L5(x_out)
        x_skip = self.L_skip(x)
        x_out += x_skip
        x_out = self.AF5(x_out)
        
        x_out = self.L6(x_out)
        x_out = self.AF6(x_out)
        
        x_out = self.L7(x_out)
        x_out = self.AF7(x_out)
        
        x_out = self.L8(x_out)
        
        return x_out

class MLP_t2(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, t2block_in, tau_in, Wmnij_in, Wmbej_in, Wmbje_in, Zmbij_in,
                 Block_nodes=196):
        super(MLP_t2, self).__init__()  
        self.t2block = t2block_in
        self.tau_in = tau_in
        self.Wmnij_in = Wmnij_in
        self.Wmbej_in = Wmbej_in
        self.Wmbje_in = Wmbje_in
        self.Zmbij_in = Zmbij_in
        self.Block_nodes = Block_nodes
        
        # t2_block
        self.t2_block = Linear_Block(self.t2block, self.Block_nodes)
        
        # tau block
        self.tau_block = Linear_Block(self.tau_in, self.Block_nodes)
        
        # Wmnij
        self.Wmnij_block = Linear_Block(self.Wmnij_in, self.Block_nodes)
        
        # Wmbej
        self.Wmbej_block = Linear_Block(self.Wmbej_in, self.Block_nodes)
        
        # Wmbje
        self.Wmbje_block = Linear_Block(self.Wmbje_in, self.Block_nodes)
        
        # Zmbij
        self.Zmbij_block = Linear_Block(self.Zmbij_in, self.Block_nodes)
        
        self.L1 = nn.Linear(10, 16)
        self.AF1 = nn.ReLU()
        self.L2 = nn.Linear(16, 32)
        self.AF2 = nn.ReLU()
        self.L3 = nn.Linear(32, 16)
        self.AF3 = nn.ReLU()
        self.L4 = nn.Linear(16, 8)
        self.AF4 = nn.ReLU()
        self.L5 = nn.Linear(8, 1)
        
        # Combined_block
        #self.Combined_block = Linear_Block(self.Block_nodes, self.Block_nodes)
        #self.AF_combined = nn.ReLU()
        
        # Linear layer
        #self.L1 = nn.Linear(self.Block_nodes, self.Block_nodes)
        #self.AF1 = nn.ReLU()
        
        # Output layer
        #self.L_out = nn.Linear(self.Block_nodes,1)
        
    def forward(self, x_t2block, x_tau, x_Wmnij, x_Wmbej, x_Wmbje, x_Zmbij, x_t2):
        
        x_out_t2block = self.t2_block(x_t2block)
        
        x_tau = torch.cat((x_tau,x_out_t2block),1)
        
        x_Wmnij = torch.cat((x_Wmnij,x_out_t2block),1)
        
        x_Wmbej = torch.cat((x_Wmbej,x_out_t2block),1)
        
        x_Wmbje = torch.cat((x_Wmbje,x_out_t2block),1)
        
        x_Zmbij = torch.cat((x_Zmbij,x_out_t2block),1)
        
        x_out_tau = self.tau_block(x_tau)
        
        x_out_Wmnij = self.Wmnij_block(x_Wmnij)
        
        x_out_Wmbej = self.Wmbej_block(x_Wmbej)
        
        x_out_Wmbje = self.Wmbje_block(x_Wmbje)
        
        x_out_Zmbij = self.Zmbij_block(x_Zmbij)
        
        x_combined = torch.cat((x_out_tau,x_out_Wmnij,x_out_Wmbej,x_out_Wmbje,x_out_Zmbij),1)
        
        x_combined = torch.cat((x_combined,x_t2), 1)
        
        x = self.L1(x_combined)
        x = self.AF1(x)
        x = self.L2(x)
        x = self.AF2(x)
        x = self.L3(x)
        x = self.AF3(x)
        x = self.L4(x)
        x = self.AF4(x)
        x = self.L5(x)
        
        return x, x_out_t2block, x_out_tau, x_out_Wmnij, x_out_Wmbej, x_out_Wmbje, x_out_Zmbij  #Output gives t2 amplitudes
    
    
    
class MLP_t1(nn.Module):
    '''
    Multilayer Perceptron for regression.
    '''
    def __init__(self, t1block_in, Fae_in, Fmi_in, Fme_in, Block_nodes=196):
        super(MLP_t1, self).__init__()  
        self.t1block_in = t1block_in
        self.Fae_in = Fae_in
        self.Fmi_in = Fmi_in
        self.Fme_in = Fme_in
        self.Block_nodes = Block_nodes
        
        # t1_block
        self.t1_block = Linear_Block(self.t1block_in, self.Block_nodes)
        
        # Fae block
        self.Fae_block = Linear_Block(self.Fae_in, self.Block_nodes)
        
        # Fmi block
        self.Fmi_block = Linear_Block(self.Fmi_in, self.Block_nodes)
        
        # Fme block
        self.Fme_block = Linear_Block(self.Fme_in, self.Block_nodes)
        
        self.L1 = nn.Linear(6, 16)
        self.AF1 = nn.ReLU()
        self.L2 = nn.Linear(16, 32)
        self.AF2 = nn.ReLU()
        self.L3 = nn.Linear(32, 16)
        self.AF3 = nn.ReLU()
        self.L4 = nn.Linear(16, 8)
        self.AF4 = nn.ReLU()
        self.L5 = nn.Linear(8, 1)
        
        # Combined_block
        #self.Combined_block = Linear_Block(self.Block_nodes, self.Block_nodes)
        #self.AF_combined = nn.ReLU()
        
        # Linear layer
        #self.L1 = nn.Linear(self.Block_nodes, self.Block_nodes)
        #self.AF1 = nn.ReLU()
        
        # Output layer
        #self.L_out = nn.Linear(self.Block_nodes,1)
        
    def forward(self, x_t1block, x_Fae, x_Fmi, x_Fme, x_t1):
        
        x_out_t1block = self.t1_block(x_t1block)
        
        x_Fae = torch.cat((x_Fae,x_out_t1block),1)
        
        x_Fmi = torch.cat((x_Fmi,x_out_t1block),1)
        
        x_Fme = torch.cat((x_Fme,x_out_t1block),1)
        
        x_out_Fae = self.Fae_block(x_Fae)
        
        x_out_Fmi = self.Fmi_block(x_Fmi)
        
        x_out_Fme = self.Fme_block(x_Fme)
        
        x_combined = torch.cat((x_out_Fae,x_out_Fmi,x_out_Fme),1)
        
        x_combined = torch.cat((x_combined,x_t1), 1)
        
        x = self.L1(x_combined)
        #x = self.AF1(x)
        x = self.L2(x)
        #x = self.AF2(x)
        x = self.L3(x)
        #x = self.AF3(x)
        x = self.L4(x)
        #x = self.AF4(x)
        x = self.L5(x)
        
        return x, x_out_t1block, x_out_Fae, x_out_Fmi, x_out_Fme #Output gives t1 amplitudes 