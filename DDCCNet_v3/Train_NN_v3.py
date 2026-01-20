#!/usr/bin/env python
# coding: utf-8

# # Importing packages
# 
# ##### Necessary packages
# 1. psi4
# 2. numpy
# 3. scikit-learn
# 4. pytorch
# 5. os
# 6. sys
# 
# ##### Packages for saving and data and plotting
# 1. matplotlib
# 2. pandas
# 
# ##### helper files
# 1. helper_CC_ML_pert_T_fc_t1: for PSI4 calculations
# 2. MTL_model_t1_t2_Combined_separate_2: the machine model functions

# In[1]:


# Import packages

import psi4
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
import os
import time
import pickle
from helper_CC_ML_pert_T_fc_t1_lo import *
from MTL_model_DDCCNet_v3_196n_7l import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import sys


# # Progress bar
# 
# To show the progress during the training 

# In[2]:


def progressBar(count_value, total, time):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    if percentage > 0:
        total_time = round(time/percentage*100.0, 1)
    else:
        total_time = 0.0
    bar = '|' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('Training[%s] %s%s %s%s%s\r' %(bar, percentage, '%', round(time,1), '/',total_time))
    sys.stdout.flush()

#This code is Contributed by PL VISHNUPPRIYAN


# # Selecting the device to train and test models
# 
# First look if CUDA available\
# If not MPS (for apple)\
# If not cpu

# In[3]:


# CUDA device

if torch.cuda.is_available():
    device = "cuda:0"
    print("Device is CUDA")

elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Device is MPS")

else:
    device = "cpu"
    print("Device is CPU")

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")


# # Defining features
# 
# Features are defined. SHAP was used to select 18 fearues for t2 and 14 features were used for t1.
# 
# ##### t2 features
# * Eoccn: Energy of occupied orbital n
# * Evirn: Energy of virtual orbital n
# * Joccn: Coulomb contribution to the energy of occupied orbital n
# * Jvirn: Coulomb contribution to the energy of virtual orbital n
# * Eoccn: Exchange contribution to the energy of occupied orbital n
# * Evirn: Exchange contribution to the energy of virtual orbital n
# * Hoccn: One-electron contribution to the energy of occupied orbital n
# * Hvirn: One-electron contribution to the energy of virtual orbital n
# * Jian: Coulomb integral between occupied orbital n and virtual orbital n
# * Kian: Exchange integral between occupied orbital n and virtual orbital n
# * diag: Are the two excited electrons go to the same virtual orbital (categorical)
# * orbdiff: Difference in orbital energies
# * doublecheck: Two-electron integral (<ij||ab>)
# * t2start: MP2 t2 amplitude
# * t2mag: log10 value of magnitude of MP2 amplitude
# * t2sign: Sign of MP2 t2 amplitude
# * Jianmag: log10 of magnitude of Jian
# * Kianmag: log10 of magnitude of Kian
# 
# 
# ##### t1 features
# * E_i: energy of occupied orbital
# * E_a: energy of virtual orbital
# * H_i: One-electron contribution to the energy of occupied orbital
# * H_a: One-electron contribution to the energy of virtual orbital
# * J_i: Coulomb contribution to the energy of occupied orbital
# * J_a: Coulomb contribution to the energy of virtual orbital
# * K_i: Exchange contribution to the energy of occupied orbital
# * K_a: Exchange contribution to the energy of virtual orbital
# * Jia: Coulomb integral between occupied and virtual orbitals
# * Kia: Exchange integral between occupied and virtual orbitals
# * Jiamag_1: log10 of magnitude of Jia
# * Kiamag_1: log10 of magnitude of Kia
# * orbdiff_1: Difference between orbital energies
# * t1_mp2: t1 value of MP2 amplitudes (performing 1 CC iteration with MP2 t2 amplitudes)

# In[4]:


# Getting features

#features_t2 = [ 'Eocc1','Jocc1', 'Kocc1', 'Hocc1', 'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Evir1', 'Hvir1', 
#               'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2','Jia1', 'Jia2', 
#               'Kia1', 'Kia2', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 
#               'Jia1mag', 'Jia2mag', 'Kia1mag', 'Kia2mag']

#features_t2 = [ 'Eocc1', 'Hocc1', 'Eocc2', 'Hocc2', 'Evir1', 'Hvir1', 'Jvir1', 'Hvir2', 'Jvir2', 'Kvir2',
#               'Jia1', 'Jia2', 'orbdiff', 'doublecheck', 't2start', 't2mag', 'Jia1mag', 'Jia2mag']
#
#features_t1 = ['E_i', 'E_a', 'H_i', 'H_a', 'J_i', 'J_a', 'K_i', 'K_a', 'Jia', 'Kia', 'Jiamag_1', 'Kiamag_1', 
#               'orbdiff_1', 't1_mp2']

features_t2 = ['orbdiff', 'doublecheck', 't2start']
features_t1 = ['orbdiff_1','t1_mp2']

intermediates_t2 = ['tmp_tau', 'Wmnij', 'Wmbej', 'Wmbje', 'Zmbij']
intermediates_t1 = ['Fae', 'Fmi', 'Fme']


# # Loading data for training
# 
# Here we load the training data which was saved as pickle file
# * Molecule_data: list containing t1 and t2 amplitudes, features, MO and F matrices
# * scaler_t2: scaler for t2 amplitudes
# * ff_t2: Weights for t2 amplitudes
# * scaler_t1: scaler for t1 amplitudes
# * ff_t1: Weights for t1 amplitudes

# In[5]:


# Loading data

with open('Train_CO2_mono_di_1tri_5t2_3t1_Allamps_LA4_fc.sav', 'rb') as handle:
    Train_data = pickle.load(handle)

Molecule_data = Train_data["Train data big"]
scaler_t2 = Train_data['Scaler_t2']
ff_t2 = Train_data['Weights_t2']
scaler_t1 = Train_data['Scaler_t1']
ff_t1 = Train_data['Weights_t1']


# # Combining data from the molecule list
# 
# This is done to calculate the mean absolute error and r2 for training molecules
# 
# ##### variables
# * Bigfeats_combined: Combinded array for t2 features
# * Bigamps_combined: Combinded t2 amplitude array
# * Bigfeats_t1_combined: Combinded array for t1 features
# * Bigamps_t1_combined: Combinded t1 amplitude array
# * Int_array: Array containing <ij||ab> integers
# * Bigints: Combinded array for integers

# In[6]:


# Combined data for MAE calculation
i=0
for m_no in range(len(Molecule_data)):
    Int_array = Molecule_data[m_no]['Integrals'].reshape(-1,1)
    if i==0:
        Bigfeats_combined = (Molecule_data[m_no]['Feats_t2'])
        Bigamps_combined = (Molecule_data[m_no]['Amps_t2'])
        Bigfeats_t1_combined = (Molecule_data[m_no]['Feats_t1'])
        Bigamps_t1_combined = (Molecule_data[m_no]['Amps_t1'])
        Bigints = Int_array
        i=1
    else:
        Bigfeats_combined = torch.cat((Bigfeats_combined,Molecule_data[m_no]['Feats_t2']), axis=0)
        Bigamps_combined = torch.cat((Bigamps_combined,Molecule_data[m_no]['Amps_t2']), axis=0)
        Bigfeats_t1_combined = torch.cat((Bigfeats_t1_combined,Molecule_data[m_no]['Feats_t1']), axis=0)
        Bigamps_t1_combined = torch.cat((Bigamps_t1_combined,Molecule_data[m_no]['Amps_t1']), axis=0)
        Bigints = torch.cat((Bigints,Int_array), axis=0)

print(Bigfeats_combined.shape)
print(Bigamps_combined.shape)
print(Bigfeats_t1_combined.shape)
print(Bigamps_t1_combined.shape)


# # Loading test data
# 
# Here we load the pickle file containing data of the test molecule.\
# This is done to calculate MAE and r2 test
# 
# ##### variables:
# * Test_feats: Test features for t2 amplitudes
# * Test_amps: Test t2 amplitudes
# * Test_t1_feats: Test features for t1 amplitudes
# * Test_t1_amps: Test t1 amplitudes
# 

# In[7]:


# Loading test data

with open('Validate_CO2_1tri_5t2_3t1_Allamps_LA4_fc.sav', 'rb') as handle:
    Test_data = pickle.load(handle)

Test_molecule_data = Test_data["Train data"]
Test_feats = Test_molecule_data[0]["Feats_t2"]
Test_amps = Test_molecule_data[0]["Amps_t2"]

Test_feats_unscaled = Test_feats.copy()

Test_t1_feats = Test_molecule_data[0]["Feats_t1"]
Test_t1_amps = Test_molecule_data[0]["Amps_t1"]

Test_feats = scaler_t2.transform(Test_feats)
#Test_feats = Test_feats*ff_t2
#
Test_t1_feats = scaler_t1.transform(Test_t1_feats)
#Test_t1_feats = Test_t1_feats*ff_t1

Test_feats = torch.from_numpy(Test_feats)
Test_amps = torch.from_numpy(Test_amps)
Test_t1_feats = torch.from_numpy(Test_t1_feats)
Test_t1_amps = torch.from_numpy(Test_t1_amps)

print(Test_feats.shape)
print(Test_amps.shape)
print(Test_t1_feats.shape)
print(Test_t1_amps.shape)


# In[8]:


def MAPE(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))


# # Training function
# 
# Here we define the training function
# 
# ##### Parameters: 
# 
# * Molecule_array: molecule data with amplitudes and features (list)
# * epoch_size: number of epochs (int, Default=100)
# * plot: plot parity plots? (Boolean, Default=False)
# 
# 
# ##### Returns:
# 
# * Model_dict: Python dictionary containig t1 and t2 models
# * Average_xx_loss_array: Arrays with each errors for each loss function at each epoch
# 
# 
# ##### Important variables:
# 
# * T2_NN: NN for t2
# * loss_function_T2: Loss function for t2
# * optimizer_T2: optimizer for t2
# * T1_NN: NN for t1
# * loss_function_T1: Loss function for t1
# * optimizer_T1: optimizer for t1
# * running_xx_loss: cumulative total of each loss. sets to zero at each epoch
# * inputs_t2: featurs for t2
# * targets_t2: amplitudes for t2
# * inputs_t1: featurs for t1
# * targets_t1: amplitudes for t1
# * t2_pred: predicted t2
# * t1_pred: predicted t1
# * tmp_t1, tmp_t2: t1 and t2 tensors used for calculation of loss for t2
# * tmp2_t1, tmp2_t2: t1 and t2 tensors used for calculation of loss for t1
# * corre_loss, corre_loss2: Correlation loss for t2 and t1 NNs respectively
# * total_rms, total_rms2: rms for t2 and t1 NNs respectively
# * t2_loss: Loss for t2 using loss_function_T2
# * t1_loss: Loss for t1 using loss_function_T1
# * Average_xx_error: Average error calculated for each loss. Calculated at after each epoch using running_xx_loss and appended to Avg_xx_loss_array
# * pred_nn_t2_train: predicted t2 for training molecules
# * pred_nn_t1_train: predicted t1 for training molecules
# * pred_nn_t2_test: predicted t2 for test molecules
# * pred_nn_t1_test: predicted t1 for test molecules
# * Exact_corre_train: Exact correlation for train t2 amplitudes
# * Pred_corre_train: Exact correlation for train t2 amplitudes
# * Exact_corre_test: Exact correlation for test t2 amplitudes
# * Pred_corre_test: Exact correlation for test t2 amplitudes

# In[9]:


# Training function

def train(Molecule_array, epoch_size=100, plot=False):


    ########################## Setting parameters for T ########################## 

    # Define the model
    T2_NN = MLP_t2(t2block_in=5, tau_in=6, Wmnij_in=6, Wmbej_in=6, Wmbje_in=6, Zmbij_in=6)

    # Initializing the weights
    #def init_weights_uniform(m):
    #    if type(m) == nn.Linear:
    #        torch.nn.init.kaiming_uniform_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
    #T_NN.apply(init_weights_uniform)

    # Passing the model to GPU
    T2_NN = T2_NN.to(device)

    # Define common loss function for t2 and t1 amplitudes
    loss_function_T2 = nn.MSELoss()

    # Defining the optimizer
    optimizer_T2 = torch.optim.Adam(T2_NN.parameters(), lr=1e-6)


    # Define the model
    T1_NN = MLP_t1(t1block_in=3, Fae_in=4, Fmi_in=4, Fme_in=4)

    # Initializing the weights
    #def init_weights_uniform(m):
    #    if type(m) == nn.Linear:
    #        torch.nn.init.kaiming_uniform_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
    #T_NN.apply(init_weights_uniform)

    # Passing the model to GPU
    T1_NN = T1_NN.to(device)

    # Define common loss function for t2 and t1 amplitudes
    loss_function_T1 = nn.MSELoss()

    # Defining the optimizer
    optimizer_T1 = torch.optim.Adam(T1_NN.parameters(), lr=1e-6)

    ########################## End setting parameters ########################## 


    ########################## Start training ##################################
    start_time = time.time()

    epoch_array = []
    Avg_corre_loss_array = []
    Avg_rms_loss_array = []
    Avg_t2_loss_array = []
    Avg_t1_loss_array = []
    Avg_t2_loss0_array = []
    Avg_t1_loss0_array = []
    Avg_tau_loss_array = []
    Avg_Wmnij_loss_array = []
    Avg_Wmbej_loss_array = []
    Avg_Wmbje_loss_array = []
    Avg_Zmbij_loss_array = []
    Avg_Fae_loss_array = []
    Avg_Fmi_loss_array = []
    Avg_Fme_loss_array = []
    Corre_min = 100.0

    # Start learning
    print('Start Training')
    for epoch in range(epoch_size): # 5 epochs at maximum
        # Print epoch
        #print(f'Starting epoch {epoch+1}')

        running_corre_loss = 0
        running_rms_loss = 0
        running_t2_loss = 0
        running_t1_loss = 0
        running_t2_loss0 = 0
        running_t1_loss0 = 0
        running_tau_loss = 0
        running_Wmnij_loss = 0
        running_Wmbej_loss = 0
        running_Wmbje_loss = 0
        running_Zmbij_loss = 0
        running_Fae_loss = 0
        running_Fmi_loss = 0
        running_Fme_loss = 0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(Molecule_data):

            inputs_t2 = data['Feats_t2']
            targets_t2 = data['Amps_t2']
            targets_tmptau = (data['targets_tmptau']).float().to(device)
            targets_Wmnij = (data['targets_Wmnij']).float().to(device)
            targets_Wmbej = (data['targets_Wmbej']).float().to(device)
            targets_Wmbje = (data['targets_Wmbje']).float().to(device)
            targets_Zmbij = (data['targets_Zmbij']).float().to(device)

            inputs_t1 = data['Feats_t1']
            targets_t1 = data['Amps_t1']
            targets_Fae = (data['targets_Fae']).float().to(device)
            targets_Fmi = (data['targets_Fmi']).float().to(device)
            targets_Fme = (data['targets_Fme']).float().to(device) 

            Integral = data['Integrals']
            Integral2 = data['Integrals2']
            Multi_array = data['Multi_array']
            target_corre = data['Corr_e']
            F = data['F_ov']
            MO = data['MO_oovv']
            nmo = ((data['Orbs']).float()).to(device)
            nocc = int(nmo[0])
            nvirt = int(nmo[1])

            inputs_t2 = inputs_t2.float()
            targets_t2 = targets_t2.float()
            inputs_t1 = inputs_t1.float()
            targets_t1 = targets_t1.float()
            target_corre = target_corre.float()
            Integral= Integral.float()
            Integral2= Integral2.float()
            Multi_array = Multi_array.float()
            F = F.float()
            MO = MO.float()

            inputs_t2 = inputs_t2.to(device)
            targets_t2 = targets_t2.to(device)
            inputs_t1 = inputs_t1.to(device)
            targets_t1 = targets_t1.to(device)
            target_corre = target_corre.to(device)
            Integral = Integral.to(device)
            Integral2 = Integral2.to(device)
            Multi_array = Multi_array.to(device)
            F = F.to(device)
            MO = MO.to(device)
            exact_corre = target_corre[0]
            SA_corre = target_corre[1]

            # Zero the gradients
            optimizer_T2.zero_grad()
            optimizer_T1.zero_grad()

            # Perform forward pass
            t2_pred, t2_pred0, tau_pred, Wmnij_pred, Wmbej_pred, Wmbje_pred, Zmbij_pred = T2_NN(inputs_t2,inputs_t2,
                                                                                      inputs_t2,inputs_t2,
                                                                                      inputs_t2,inputs_t2,
                                                                                      inputs_t2)
            t1_pred, t1_pred0, Fae_pred, Fmi_pred, Fme_pred = T1_NN(inputs_t1,inputs_t1,
                                                                    inputs_t1,inputs_t1,
                                                                    inputs_t1)

            tmp_t1 = t1_pred.detach().clone()
            tmp_t1 = tmp_t1.reshape(nocc,nvirt)
            #tmp_t2 = t2_pred.reshape(nocc,nocc,nvirt,nvirt)

            tmp2_t1 = t1_pred.reshape(nocc,nvirt)
            tmp2_t2 = t2_pred.detach().clone()

            Int_Multi = Multi_array*Integral
            Int_Multi2 = Multi_array*Integral2

            # Calculating the correlation energy loss
            # For t2
            CCSDcorr_E = 2.0 * torch.einsum('ia,ia->', F, tmp_t1)
            tmp_tau = torch.einsum('ia,jb->ijab', tmp_t1, tmp_t1)
            CCSDcorr_E += 2.0 * torch.einsum('ijab,ijab->', tmp_tau, MO)
            CCSDcorr_E -= 1.0 * torch.einsum('ijab,ijba->', tmp_tau,MO)
            CCSDcorr_E += SA_corre
            CCSDcorr_E += 2.0 * torch.sum(t2_pred*Int_Multi)
            CCSDcorr_E -= 1.0 * torch.sum((t2_pred*Int_Multi2))

            corre_loss = torch.abs(exact_corre-CCSDcorr_E)


            # For t1
            MO2 = MO.detach().clone()
            F2 = F.detach().clone()
            CCSDcorr_E2 = 2.0 * torch.einsum('ia,ia->', F2, tmp2_t1)
            tmp_tau2 = torch.einsum('ia,jb->ijab', tmp2_t1, tmp2_t1)
            CCSDcorr_E2 += 2.0 * torch.einsum('ijab,ijab->', tmp_tau2, MO2)
            CCSDcorr_E2 -= 1.0 * torch.einsum('ijab,ijba->', tmp_tau2,MO2)
            CCSDcorr_E2 += SA_corre
            CCSDcorr_E2 += 2.0 * torch.sum((tmp2_t2*Multi_array*Integral))
            CCSDcorr_E2 -= 1.0 * torch.sum((tmp2_t2*Multi_array*Integral2))

            corre_loss2 = torch.abs(exact_corre.detach().clone()-CCSDcorr_E2)

            # Calculate RMS
            # For t2
            rms_t1 = targets_t1.detach().clone() - t1_pred.detach().clone()
            rms_t2 = targets_t2 - t2_pred
            total_rms = torch.sqrt(torch.sum(rms_t1*rms_t1) + torch.sum(rms_t2*rms_t2))

            # For t1
            rms2_t1 = targets_t1 - t1_pred
            rms2_t2 = targets_t2.detach().clone() - t2_pred.detach().clone()
            total_rms2 = torch.sqrt(torch.sum(rms2_t1*rms2_t1) + torch.sum(rms2_t2*rms2_t2))

            # Compute loss
            t2_loss = loss_function_T2(t2_pred, targets_t2)
            t1_loss = loss_function_T1(t1_pred, targets_t1)

            t2_loss0 = loss_function_T2(t2_pred0, targets_t2)
            t1_loss0 = loss_function_T1(t1_pred0, targets_t1)

            tau_loss = loss_function_T2(tau_pred, targets_tmptau)
            Wmnij_loss = loss_function_T2(Wmnij_pred, targets_Wmnij)
            Wmbej_loss = loss_function_T2(Wmbej_pred, targets_Wmbej)
            Wmbje_loss = loss_function_T2(Wmbje_pred, targets_Wmbje)
            Zmbij_loss = loss_function_T2(Zmbij_pred, targets_Zmbij)

            Fae_loss = loss_function_T1(Fae_pred, targets_Fae)
            Fme_loss = loss_function_T1(Fme_pred, targets_Fme)
            Fmi_loss = loss_function_T1(Fmi_pred, targets_Fmi)

            Total_loss_t2 = corre_loss+t2_loss+total_rms+tau_loss+Wmnij_loss+Wmbej_loss+Wmbje_loss+Zmbij_loss#+t2_loss0
            Total_loss_t1 = corre_loss2+t1_loss+total_rms2+Fae_loss+Fme_loss+Fmi_loss#+t1_loss0

            #Total_loss_t2 = total_rms+t2_loss+tau_loss+Wmnij_loss+Wmbej_loss+Wmbje_loss+Zmbij_loss
            #Total_loss_t1 = total_rms2+t1_loss+Fae_loss+Fme_loss+Fmi_loss

            current_corre_loss = corre_loss.detach().clone()
            current_rms_loss = total_rms.detach().clone()
            current_t2_loss = t2_loss.detach().clone()
            current_t1_loss = t1_loss.detach().clone()
            current_t2_loss0 = t2_loss0.detach().clone()
            current_t1_loss0 = t1_loss0.detach().clone()
            current_tau_loss = tau_loss.detach().clone()
            current_Wmnij_loss = Wmnij_loss.detach().clone()
            current_Wmbej_loss = Wmbej_loss.detach().clone()
            current_Wmbje_loss = Wmbje_loss.detach().clone()
            current_Zmbij_loss = Zmbij_loss.detach().clone()
            current_Fae_loss = Fae_loss.detach().clone()
            current_Fmi_loss = Fmi_loss.detach().clone()
            current_Fme_loss = Fme_loss.detach().clone()

            running_corre_loss += current_corre_loss
            running_rms_loss += current_rms_loss
            running_t2_loss += current_t2_loss
            running_t1_loss += current_t1_loss
            running_t2_loss0 += current_t2_loss0
            running_t1_loss0 += current_t1_loss0
            running_tau_loss += current_tau_loss
            running_Wmnij_loss += current_Wmnij_loss
            running_Wmbej_loss += current_Wmbej_loss
            running_Wmbje_loss += current_Wmbje_loss
            running_Zmbij_loss += current_Zmbij_loss
            running_Fae_loss += current_Fae_loss
            running_Fmi_loss += current_Fmi_loss
            running_Fme_loss += current_Fme_loss

            # Perform backward pass
            Total_loss_t2.backward()
            Total_loss_t1.backward()

            # Perform optimization
            optimizer_T2.step()
            optimizer_T1.step()

        progressBar(epoch+1, epoch_size, (time.time()-start_time))

        Average_corre_error = running_corre_loss/len(Molecule_data)
        Average_rms_error = running_rms_loss/len(Molecule_data)
        Average_t2_error = running_t2_loss/len(Molecule_data)
        Average_t1_error = running_t1_loss/len(Molecule_data)
        Average_t2_error0 = running_t2_loss0/len(Molecule_data)
        Average_t1_error0 = running_t1_loss0/len(Molecule_data)
        Average_tau_error = running_tau_loss/len(Molecule_data)
        Average_Wmnij_error = running_Wmnij_loss/len(Molecule_data)
        Average_Wmbej_error = running_Wmbej_loss/len(Molecule_data)
        Average_Wmbje_error = running_Wmbje_loss/len(Molecule_data)
        Average_Zmbij_error = running_Zmbij_loss/len(Molecule_data)
        Average_Fae_error = running_Fae_loss/len(Molecule_data)
        Average_Fmi_error = running_Fmi_loss/len(Molecule_data)
        Average_Fme_error = running_Fme_loss/len(Molecule_data)

        epoch_array.append(epoch+1)
        Avg_corre_loss_array.append(Average_corre_error.cpu())
        Avg_rms_loss_array.append(Average_rms_error.cpu())
        Avg_t2_loss_array.append(Average_t2_error.cpu())
        Avg_t1_loss_array.append(Average_t1_error.cpu())
        Avg_t2_loss0_array.append(Average_t2_error0.cpu())
        Avg_t1_loss0_array.append(Average_t1_error0.cpu())
        Avg_tau_loss_array.append(Average_tau_error.cpu())
        Avg_Wmnij_loss_array.append(Average_Wmnij_error.cpu())
        Avg_Wmbej_loss_array.append(Average_Wmbej_error.cpu())
        Avg_Wmbje_loss_array.append(Average_Wmbje_error.cpu())
        Avg_Zmbij_loss_array.append(Average_Zmbij_error.cpu())
        Avg_Fae_loss_array.append(Average_Fae_error.cpu())
        Avg_Fmi_loss_array.append(Average_Fmi_error.cpu())
        Avg_Fme_loss_array.append(Average_Fme_error.cpu())

        #epoch_tensor = torch.from_numpy(np.asarray(epoch_array))
        #Avg_t2_loss_tensor = torch.from_numpy(np.asarray(Avg_t2_loss_array))
        #Avg_t1_loss_tensor = torch.from_numpy(np.asarray(Avg_t1_loss_array))

        if corre_loss < Corre_min:
            Corre_min = corre_loss
            t2_min_corre_model = MLP_t2(t2block_in=5, tau_in=6, Wmnij_in=6, Wmbej_in=6, Wmbje_in=6, Zmbij_in=6)
            t1_min_corre_model = MLP_t1(t1block_in=3, Fae_in=4, Fmi_in=4, Fme_in=4)

            t2_min_corre_model.load_state_dict(T2_NN.state_dict())
            t1_min_corre_model.load_state_dict(T1_NN.state_dict())

            Corre_Model_dict = {
            'T1_Model': t1_min_corre_model,
            'T2_Model': t2_min_corre_model
            }

        if (epoch)%5000 == 0:

            t2_min_corre_model = MLP_t2(t2block_in=5, tau_in=6, Wmnij_in=6, Wmbej_in=6, Wmbje_in=6, Zmbij_in=6)
            t1_min_corre_model = MLP_t1(t1block_in=3, Fae_in=4, Fmi_in=4, Fme_in=4)

            t2_min_corre_model.load_state_dict(T2_NN.state_dict())
            t1_min_corre_model.load_state_dict(T1_NN.state_dict())

            Corre_Model_dict = {
            'T1_Model': t1_min_corre_model,
            'T2_Model': t2_min_corre_model
            }

            model_name = 'Model_v6_mono_di_1tri_fc_'+str(epoch)+'.sav'
            with open(model_name, 'wb') as handle:
                pickle.dump(Corre_Model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ########################## End training ####################################        

    # Calculating the time spent
    print()
    print('End Training')
    duration = time.time()-start_time 
    hrs = int(duration/3600)
    mins = int((duration%3600)/60)
    sec = int(np.round(duration%60))

    print('Training process has finished.')
    print('Duration :', hrs, 'hrs', mins, 'mins', sec, 'sec')
    #print('')
    #print('epoch array')
    #print(epoch_array)
    #print('')
    #print('Avg t2 error')
    #print(Avg_t2_loss_array)
    #print('')
    #print('Avg t1 error')
    #print(Avg_t1_loss_array)
    #print('')

    T2_NN.eval()

    # T2 train error
    with torch.no_grad():
        pred_nn_t2_train,_1,_2,_3,_4,_5,_6 = T2_NN(Bigfeats_combined.float().to(device),
                                               Bigfeats_combined.float().to(device),
                                               Bigfeats_combined.float().to(device),
                                               Bigfeats_combined.float().to(device),
                                               Bigfeats_combined.float().to(device),
                                               Bigfeats_combined.float().to(device),
                                               Bigfeats_combined.float().to(device))

    pred_nn_t2_train = pred_nn_t2_train.to("cpu").detach().numpy()

    t2_MAE_train = np.average(np.abs(Bigamps_combined.reshape(-1,)-pred_nn_t2_train.reshape(-1,)))
    t2_r2_train = r2_score(Bigamps_combined.reshape(-1,),pred_nn_t2_train.reshape(-1,))

    T1_NN.eval()
    with torch.no_grad():
        pred_nn_t1_train,_1,_2,_3,_4 = T1_NN(Bigfeats_t1_combined.float().to(device),
                                         Bigfeats_t1_combined.float().to(device),
                                         Bigfeats_t1_combined.float().to(device),
                                         Bigfeats_t1_combined.float().to(device),
                                         Bigfeats_t1_combined.float().to(device))

    pred_nn_t1_train = pred_nn_t1_train.to("cpu").detach().numpy()

    t1_MAE_train = np.average(np.abs(Bigamps_t1_combined.reshape(-1,)-pred_nn_t1_train.reshape(-1,)))
    t1_r2_train = r2_score(Bigamps_t1_combined.reshape(-1,),pred_nn_t1_train.reshape(-1,))

    # T2 test error
    T2_NN.eval()
    with torch.no_grad():
        pred_nn_t2_test,_1,_2,_3,_4,_5,_6 = T2_NN(Test_feats.float().to(device),
                                             Test_feats.float().to(device),
                                             Test_feats.float().to(device),
                                             Test_feats.float().to(device),
                                             Test_feats.float().to(device),
                                             Test_feats.float().to(device),
                                             Test_feats.float().to(device))

    pred_nn_t2_test = pred_nn_t2_test.to("cpu").detach().numpy()
    t2_MAE_test = np.average(np.abs(Test_amps.reshape(-1,)-pred_nn_t2_test.reshape(-1,)))
    t2_r2_test = r2_score(Test_amps.reshape(-1,),pred_nn_t2_test.reshape(-1,))

    T1_NN.eval()
    with torch.no_grad():
        pred_nn_t1_test,_1,_2,_3,_4 = T1_NN(Test_t1_feats.float().to(device), Test_t1_feats.float().to(device),
                                        Test_t1_feats.float().to(device), Test_t1_feats.float().to(device),
                                        Test_t1_feats.float().to(device))

    pred_nn_t1_test = pred_nn_t1_test.to("cpu").detach().numpy()
    t1_MAE_test = np.average(np.abs(Test_t1_amps.reshape(-1,)-pred_nn_t1_test.reshape(-1,)))
    t1_r2_test = r2_score(Test_t1_amps.reshape(-1,),pred_nn_t1_test.reshape(-1,))

    Exact_corre_train = Bigamps_combined.reshape(-1,)*Bigints.reshape(-1,)
    Pred_corre_train = pred_nn_t2_train.reshape(-1,)*Bigints.reshape(-1,).detach().numpy()

    Exact_corre_test = Test_amps.reshape(-1,)*Test_feats_unscaled[:,1]
    Pred_corre_test = pred_nn_t2_test.reshape(-1,)*Test_feats_unscaled[:,1]

    if plot:
        current_filename = 'Plots/Train_t2_mol6_sym_v6_2.png'   
        plt.scatter(Bigamps_combined.reshape(-1,), pred_nn_t2_train, label='train')
        plt.scatter(Test_amps.reshape(-1,), pred_nn_t2_test, label='test')
        plt.plot([torch.min(Bigamps_combined),torch.max(Bigamps_combined)],[torch.min(Bigamps_combined),torch.max(Bigamps_combined)], c='black')
        plt.xlabel('Exact $\it{t_{2}}$ amplitudes')
        plt.ylabel('Predicted $\it{t_{2}}$ amplitudes')
        plt.legend()
        plt.savefig(current_filename,dpi=600)
        plt.close()

        current_filename = 'Plots/Train_t1_mol6_sym_v6_2.png'   
        plt.scatter(Bigamps_t1_combined.reshape(-1,), pred_nn_t1_train, label='train')
        plt.scatter(Test_t1_amps.reshape(-1,), pred_nn_t1_test, label='test')
        plt.plot([torch.min(Bigamps_t1_combined),torch.max(Bigamps_t1_combined)],[torch.min(Bigamps_t1_combined),torch.max(Bigamps_t1_combined)], c='black')
        plt.xlabel('Exact $\it{t_{1}}$ amplitudes')
        plt.ylabel('Predicted $\it{t_{1}}$ amplitudes')
        plt.legend()
        plt.savefig(current_filename,dpi=600)
        plt.close()

        #current_filename = 'Plots/Test_t2_mol6_sym_LA_6.png'   
        #plt.scatter(Test_amps.reshape(-1,), pred_nn_t2_test, label='test')
        #plt.plot([torch.min(Test_amps),torch.max(Test_amps)],[torch.min(Test_amps),torch.max(Test_amps)], c='black')
        #plt.xlabel('Exact $\it{t_{2}}$ amplitudes')
        #plt.ylabel('Predicted $\it{t_{2}}$ amplitudes')
        #plt.legend()
        #plt.savefig(current_filename,dpi=600)
        #plt.close()

        #current_filename = 'Plots/Test_t1_mol6_sym_LA_6.png'   
        #plt.scatter(Test_t1_amps.reshape(-1,), pred_nn_t1_test, label='test')
        #plt.plot([torch.min(Test_t1_amps),torch.max(Test_t1_amps)],[torch.min(Test_t1_amps),torch.max(Test_t1_amps)], c='black')
        #plt.xlabel('Exact $\it{t_{1}}$ amplitudes')
        #plt.ylabel('Predicted $\it{t_{1}}$ amplitudes')
        #plt.legend()
        #plt.savefig(current_filename,dpi=600)
        #plt.close()

        current_filename = 'Plots/Train_t2_corre_mol6_sym_v6_2.png'   
        plt.scatter(Exact_corre_train, Pred_corre_train, label='train correlation')
        plt.scatter(Exact_corre_test, Pred_corre_test, label='test correlation')
        plt.plot([torch.min(Exact_corre_train),torch.max(Exact_corre_train)],[torch.min(Exact_corre_train),torch.max(Exact_corre_train)], c='black')
        plt.xlabel('Exact correlation energy (Eh)')
        plt.ylabel('Predicted correlation energy (Eh)')
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.legend()
        plt.savefig(current_filename,dpi=600)
        plt.close()

        #current_filename = 'Plots/Test_t2_corre_mol6_sym_LA_6.png'   
        #plt.scatter(Exact_corre_test, Pred_corre_test, label='test correlation')
        #plt.plot([torch.min(Exact_corre_test),torch.max(Exact_corre_test)],[torch.min(Exact_corre_test),torch.max(Exact_corre_test)], c='black')
        #plt.xlabel('Exact correlation energy (Eh)')
        #plt.ylabel('Predicted correlation energy (Eh)')
        #plt.xticks(fontsize=5)
        #plt.yticks(fontsize=5)
        #plt.legend()
        #plt.savefig(current_filename,dpi=600)
        #plt.close()

    print('')
    print('Train t2 MAE = ', t2_MAE_train)
    print('Train t2 R2  = ', t2_r2_train)
    print('Train t1 MAE = ', t1_MAE_train)
    print('Train t1 R2  = ', t1_r2_train)
    print('')
    print('Test t2 MAE = ', t2_MAE_test)
    print('Test t2 R2  = ', t2_r2_test)
    print('Test t1 MAE = ', t1_MAE_test)
    print('Test t1 R2  = ', t1_r2_test)

    Model_dict = {
        'T1_Model': T1_NN,
        'T2_Model': T2_NN
    }

    return Model_dict,Corre_Model_dict,Avg_corre_loss_array,Avg_rms_loss_array,Avg_t2_loss_array,Avg_t1_loss_array,Avg_t2_loss0_array,Avg_t1_loss0_array,Avg_tau_loss_array, Avg_Wmnij_loss_array, Avg_Wmbej_loss_array, Avg_Wmbje_loss_array, Avg_Zmbij_loss_array, Avg_Fae_loss_array, Avg_Fmi_loss_array, Avg_Fme_loss_array


# # Execute train function

# In[10]:


n_epochs = 25000
Models, Corre_Models, corre_loss, rms_loss, t2_loss, t1_loss, t2_loss0, t1_loss0, tau_loss, Wmnij_loss, Wmbej_loss, Wmbje_loss, Zmbij_loss, Fae_loss, Fmi_loss, Fme_loss= train(Molecule_data, epoch_size=n_epochs, plot=True)


# In[11]:


# Save models


# In[12]:


# For KNN and RF
Model_dict = {
            'DDCCNet_25k_model': Models,
            'DDCCNet_25k_model_Best_corr': Corre_Models,
            }

with open('Model_v6_mono_di_1tri_fc_25k.sav', 'wb') as handle:
    pickle.dump(Model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # Plot learning curves
# 
# Plotting learning curves

# In[13]:


epoch_array = np.arange(n_epochs)

plt.plot(epoch_array[1000:], corre_loss[1000:], label='Corre')
plt.plot(epoch_array[1000:], rms_loss[1000:], label='RMS')
plt.plot(epoch_array[1000:], t2_loss[1000:], label='t2')
plt.plot(epoch_array[1000:], t1_loss[1000:], label='t1')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.savefig('L123_10k_1000.png', dpi=600)


# In[14]:


plt.plot(epoch_array[1000:], tau_loss[1000:], label='tau')
plt.plot(epoch_array[1000:], Wmnij_loss[1000:], label='Wmnij')
plt.plot(epoch_array[1000:], Wmbej_loss[1000:], label='Wmbej')
plt.plot(epoch_array[1000:], Wmbje_loss[1000:], label='Wmbje')
plt.plot(epoch_array[1000:], Zmbij_loss[1000:], label='Zmbij')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.savefig('T2_int_loss_10k_1000.png', dpi=600)


# In[15]:


plt.plot(epoch_array, Fae_loss, label='Fae')
plt.plot(epoch_array, Fmi_loss, label='Fmi')
plt.plot(epoch_array, Fme_loss, label='Fme')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')
plt.savefig('Plots/L1+L2+L3_separate2_mol6_sym_10k_3.png', dpi=600)


# In[16]:


corre_loss = np.asarray(corre_loss)
rms_loss = np.asarray(rms_loss)
t2_loss = np.asarray(t2_loss)
t1_loss = np.asarray(t1_loss)
tau_loss = np.asarray(tau_loss)
Wmnij_loss = np.asarray(Wmnij_loss)
Wmbej_loss = np.asarray(Wmbej_loss)
Wmbje_loss = np.asarray(Wmbje_loss)
Zmbij_loss = np.asarray(Zmbij_loss)
Fae_loss = np.asarray(Fae_loss)
Fmi_loss = np.asarray(Fmi_loss)
Fme_loss = np.asarray(Fme_loss)
t2_loss0 = np.asarray(t2_loss0)
t1_loss0 = np.asarray(t1_loss0)
Total_loss = corre_loss+rms_loss+t2_loss+t1_loss+tau_loss+Wmnij_loss+Wmbej_loss+Wmbje_loss+Zmbij_loss+Fae_loss+Fmi_loss+Fme_loss+t2_loss0+t1_loss0


# In[17]:


plt.plot(epoch_array[1000:], Total_loss[1000:], label='Total')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('Total_loss_10k_1000.png', dpi=600)


# In[18]:


plt.plot(epoch_array, t2_loss0, label='t2 loss0')
plt.plot(epoch_array, t1_loss0, label='t1 loss0')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yscale('log')


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


epoch_array = np.arange(n_epochs)

plt.plot(epoch_array[100:], corre_loss[100:], label='Corre')
#plt.plot(epoch_array, rms_loss, label='RMS')
#plt.plot(epoch_array, t2_loss, label='t2')
#plt.plot(epoch_array, t1_loss, label='t1')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Correlation Energy Loss (Eh)')
#plt.yscale('log')
plt.savefig('Correlation_Energy_loss_10k_1000.png', dpi=600)


# In[20]:


epoch_array = np.arange(n_epochs)

#plt.plot(epoch_array, corre_loss, label='Corre')
plt.plot(epoch_array, rms_loss, label='RMS')
#plt.plot(epoch_array, t2_loss, label='t2')
#plt.plot(epoch_array, t1_loss, label='t1')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.yscale('log')
#plt.savefig('Plots/L1+L2+L3_separate2_mol6_sym_LA_3.png', dpi=600)


# In[21]:


epoch_array = np.arange(n_epochs)

#plt.plot(epoch_array, corre_loss, label='Corre')
#plt.plot(epoch_array, rms_loss, label='RMS')
plt.plot(epoch_array, t2_loss, label='t2')
#plt.plot(epoch_array, t1_loss, label='t1')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.yscale('log')
#plt.savefig('Plots/L1+L2+L3_separate2_mol6_sym_LA_3.png', dpi=600)


# In[22]:


epoch_array = np.arange(n_epochs)

#plt.plot(epoch_array, corre_loss, label='Corre')
#plt.plot(epoch_array, rms_loss, label='RMS')
#plt.plot(epoch_array, t2_loss, label='t2')
plt.plot(epoch_array, t1_loss, label='t1')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.yscale('log')
#plt.savefig('Plots/L1+L2+L3_separate2_mol6_sym_LA_3.png', dpi=600)


# In[ ]:





# # Printing loss of each loss function

# In[23]:


print('RMS loss = ', rms_loss[-1])
print('Correlation energy loss = ', corre_loss[-1])
print('t2 loss = ', t2_loss[-1])
print('t1 loss = ', t1_loss[-1])


# # Test function
# 
# ##### Parameters:
# 
# * Foldername: Folder with test xyz files
# * model_dict: Python dictionary with models to predict t1 and t2 amplitudes
# * scaler_t2: scaler for t2 amplitudes
# * scaler_t1: scaler for t1 amplitudes
# * ff_t2: weights for t2 amplitudes
# * ff_t1: weights for t1 amplitudes
# 
# ##### Saves data from training at save_file_name
# 
# * Molecule: Molecule file name
# * Error (mEh): Error for molecule in mEh
# * CCSD steps: Number of steps for conventional CCSD calculation
# * DDCCSD steps: Number of steps for DDCCSD calculation

# In[24]:


def Test(Foldername, model_dict, scaler_t2, scaler_t1, ff_t2, ff_t1, 
        save_file_name='dat_file.csv'):
    steps=list()
    difference=list()
    filenames=list()
    CCSD_steps_array=list()
    DDCCSD_steps_array=list()
    CCSD_Energy_array=list()
    DDCCSD_Energy_array=list()

    LA_lim = 0.0001

    t2_model = model_dict['T2_Model']
    t1_model = model_dict['T1_Model']

    t2_model = t2_model.to(device)
    t1_model = t1_model.to(device)

    for filename in os.listdir(Foldername):
        if filename.endswith('.xyz'):
            psi4.core.clean()
            filenames.append(filename)
            print ("filename is "+filename)
            path1=str(Foldername+filename)
            text = open(path1, 'r').read()
            #print(text)
            mol = psi4.geometry(text)

            psi4.set_options({'basis':        'cc-pvdz',
                              'scf_type':     'pk',
                              'maxiter':      1000,
                              'reference':    'rhf',
                              'mp2_type':     'conv',
                              'e_convergence': 1e-8,
                              'Freeze_core':   False,
                              'd_convergence': 1e-8})

            MLt2=0  #  scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
            A=HelperCCEnergy(mol) # Pred t1 and pred t2
            #B=HelperCCEnergy(mol) # zero t1 and MP2 t2

            A.compute_t1_mp2()
            matrixsize=(A.nocc-A.nfzc)*(A.nocc-A.nfzc)*A.nvirt*A.nvirt
            Xnew=np.zeros([(A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt,len(features_t2)])
            for x in range(len(features_t2)):
                Xnew[:,:,:,:,x]=getattr(A, features_t2[x])

            Xnew = Xnew.swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt,len(features_t2))
            iu = np.triu_indices((A.nocc-A.nfzc)*A.nvirt)
            Xnew = Xnew[iu]
            one_over_orbdiff = 1/(Xnew[:,0].reshape(-1,))
            Xnew[:,0] = one_over_orbdiff

            tia_tjb = np.einsum('ia,jb->ijab', A.t1_mp2, A.t1_mp2, optimize=True)
            tau = tia_tjb + A.t2start.copy()

            tia_tjb = tia_tjb.swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt)
            tia_tjb = tia_tjb[iu]

            tau = tau.swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt)
            tau = tau[iu]

            Xnew = np.concatenate((Xnew, tia_tjb.reshape(-1,1)), axis=1)
            Xnew = np.concatenate((Xnew, tau.reshape(-1,1)), axis=1)

            Positions = np.arange(Xnew.shape[0])

            Abs_MP2 = np.abs(Xnew[:,2])
            Largefeatures = Xnew[Abs_MP2>=LA_lim]
            Largepositions = Positions[Abs_MP2>=LA_lim]

            X_new_scaled= scaler_t2.transform(Largefeatures)
            #X_newer_scaled= X_new_scaled * ff_t2

            matrixsize_t1=(A.nocc-A.nfzc)*A.nvirt
            Xnew_t1=np.zeros([1,matrixsize_t1,len(features_t1)])
            for x in range (0,len(features_t1)):
                Xnew_t1[0,:,x]=getattr(A, features_t1[x]).reshape(matrixsize_t1)

            Xnew_t1 = Xnew_t1.reshape(matrixsize_t1,len(features_t1))
            Positions_t1 = np.arange(matrixsize_t1)

            one_over_orbdiff_1 = 1/(A.orbdiff_1.reshape(-1,))
            Xnew_t1[:,0] = one_over_orbdiff_1
            F_ov = A.F_ov
            Xnew_t1 = np.concatenate((Xnew_t1, F_ov.reshape(-1,1)),axis=1)

            X_new_t1_scaled = scaler_t1.transform(Xnew_t1)
            #X_newer_t1_scaled = X_new_t1_scaled * ff_t1

            X_test_t2_scaled = torch.from_numpy(X_new_scaled)
            X_test_t1_scaled = torch.from_numpy(X_new_t1_scaled)

            t2_model.eval()
            with torch.no_grad():
                y_t2,_1,_2,_3,_4,_5,_6 = t2_model(X_test_t2_scaled.float().to(device),
                                               X_test_t2_scaled.float().to(device),
                                               X_test_t2_scaled.float().to(device),
                                               X_test_t2_scaled.float().to(device),
                                               X_test_t2_scaled.float().to(device),
                                               X_test_t2_scaled.float().to(device),
                                               X_test_t2_scaled.float().to(device))

            t1_model.eval()
            with torch.no_grad():
                y_t1,_1,_2,_3,_4 = t1_model(X_test_t1_scaled.float().to(device),
                                         X_test_t1_scaled.float().to(device),
                                         X_test_t1_scaled.float().to(device),
                                         X_test_t1_scaled.float().to(device),
                                         X_test_t1_scaled.float().to(device))

            y_t2 = y_t2.to("cpu").detach().numpy()
            y_t1 = y_t1.to("cpu").detach().numpy()

            newamps = (Xnew[:,2]).reshape(-1,1)
            newamps[Largepositions] = y_t2.reshape(-1,1)

            MLt2 = np.zeros(((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt))
            MLt2[iu] = newamps.reshape(-1,)
            il = np.tril_indices((A.nocc-A.nfzc)*A.nvirt,-1)
            MLt2[il] = (MLt2.copy()).T[il]
            MLt2=MLt2.reshape(A.nocc-A.nfzc,A.nvirt,A.nocc-A.nfzc,A.nvirt).swapaxes(1,2)
            A.t2=MLt2.copy()

            newamps_t1 = np.zeros((matrixsize_t1,1))
            newamps_t1[Positions_t1] = y_t1.reshape(-1,1)

            MLt1 = newamps_t1.reshape(A.nocc-A.nfzc, A.nvirt)
            A.t1=MLt1.copy()

            print()
            print('Start DDCCSD iterations')
            A.compute_energy()
            #rhfenergy.append(A.rhf_e)
            #startenergy.append(A.StartEnergy)

            if hasattr(A,"FinalEnergy"):
                DDCCSD_steps = A.steps
            else:
                DDCCSD_steps = 101

            print()
            print('Start CCSD iterations')
            #B.compute_energy()

            DDCCSD_error = abs(A.StartEnergy-A.FinalEnergy)*1000

            DDCCSD_Energy = A.StartEnergy

            CCSD_Energy = A.FinalEnergy

            difference.append(DDCCSD_error)
            #CCSD_steps_array.append(B.steps)
            DDCCSD_steps_array.append(DDCCSD_steps)

            CCSD_Energy_array.append(CCSD_Energy)
            DDCCSD_Energy_array.append(DDCCSD_Energy)

            #MAE_t2 = np.average(np.abs(B.t2.copy().reshape(-1,) - MLt2.copy().reshape(-1,)))
            #MAE_t1 = np.average(np.abs(B.t1.copy().reshape(-1,) - y_t1.copy().reshape(-1,)))
            #MAE_MP2 = np.average(np.abs(B.t2.copy().reshape(-1,) - B.t2start.copy().reshape(-1,)))

    #print ('Filenames')
    #print (filenames)

    #print ('Individual CCSD Differences')
    #print (difference)
    #
    #print ('Average CCSD Differences')

    #print ('CCSD iterations')
    #print (CCSD_steps)
    #print ('DDCCSD iterations')
    #print (DDCCSD_steps)
    #
    #print()
    #print('t2 MAE = ', MAE_t2)
    #print('t1 MAE = ', MAE_t1)
    #print('MP2 MAE = ', MAE_MP2)

    Data_dict = {
        'Molecule': filenames,
        'Difference (mEh)': difference,
        #'CCSD steps': CCSD_steps_array,
        'CCSD Energy': CCSD_Energy_array,
        'DDCCSD Energy': DDCCSD_Energy_array,
        'DDCCSD steps': DDCCSD_steps_array
    }

    Data_df = pd.DataFrame.from_dict(Data_dict)

    Data_df.to_csv(save_file_name)

    #Amp_dict = {
    #    'MP2_amps': A.t2start.reshape(-1,),
    #    'Exact_t2': B.t2.reshape(-1,),
    #    'Predicted': MLt2.reshape(-1,),
    #}

    #Amp_df = pd.DataFrame.from_dict(Amp_dict)
    #Amp_df.to_csv('Amp_file_51mol_50000epochs.csv')
    return MLt2


# # Execute Test function

# In[25]:


T2_dict = Test(Foldername='../DDCCNet_CO2_dimers/Testing/', scaler_t1=scaler_t1, scaler_t2=scaler_t2, ff_t1=ff_t1, ff_t2=ff_t2,
     model_dict=Models, save_file_name = 'Test_data_196n_7ly_25k.csv')


# In[ ]:


T2_dict = Test(Foldername='../DDCCNet_CO2_dimers/Training/', scaler_t1=scaler_t1, scaler_t2=scaler_t2, ff_t1=ff_t1, ff_t2=ff_t2,
     model_dict=Models, save_file_name = 'Train_data_196n_7ly_25k.csv')


# In[ ]:


T2_dict = Test(Foldername='../DDCCNet_CO2_dimers/Testing/', scaler_t1=scaler_t1, scaler_t2=scaler_t2, ff_t1=ff_t1, ff_t2=ff_t2,
     model_dict=Corre_Models, save_file_name = 'Test_data_bestcorr_196n_7ly_25k.csv')


# In[ ]:


-0.367902230663003-(-0.367035574940637)


# In[ ]:




