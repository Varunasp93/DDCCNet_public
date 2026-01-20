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

#import psi4
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
import os
import time
import pickle
#from helper_CC_ML_pert_T_fc_t1_lo import *
from MTL_model_t1_t2_Combined_separate_196n_7ly import *
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
    #print(f"Training[%s] %s%s %s%s%s\r" %(bar, percentage, '%', round(time,1), '/',total_time,end=""))
    print(f"Training[{bar}] {percentage}% {round(time,1)}/{total_time}\r", end="")

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

features_t2 = [ 'Eocc1','Jocc1', 'Kocc1', 'Hocc1', 'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Evir1', 'Hvir1', 
               'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2','Jia1', 'Jia2', 'Kia1', 'Kia2','Jia1mag',
               'Jia2mag', 'Kia1mag', 'Kia2mag', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 
               't2sign']


features_t1 = ['E_i', 'E_a', 'H_i', 'H_a', 'J_i', 'J_a', 'K_i', 'K_a', 'Jia', 'Kia', 'Jiamag_1', 'Kiamag_1', 
               'orbdiff_1', 't1_mp2']


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

with open('GDB5_scaler_weights.sav', 'rb') as handle:
    Train_data = pickle.load(handle)

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

# In[7]:


# Training function

def train(Molecule_array, model_name_prefix, start_epoch=0, epoch_size=100, plot=False, pretrained=False, pretrained_model=None):
    

    ########################## Setting parameters for T ########################## 
    
    learning_rate = 1e-6

    # Define the model
    T2_NN = MLP_Amp(indMO_in=16, MOint_in=8, MOvec_in=48, Amp_in=6)
    
    # Initializing the weights
    #def init_weights_uniform(m):
    #    if type(m) == nn.Linear:
    #        torch.nn.init.kaiming_uniform_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
    #T_NN.apply(init_weights_uniform)
    
    
    # Define the model
    T1_NN = MLP_Amp(indMO_in=8, MOint_in=4, MOvec_in=24, Amp_in=2)
    
    # Initializing the weights
    #def init_weights_uniform(m):
    #    if type(m) == nn.Linear
    #        torch.nn.init.kaiming_uniform_(m.weight, mode = 'fan_in', nonlinearity = 'relu')
    #T_NN.apply(init_weights_uniform)
    
    if pretrained:
        with open(pretrained_model, 'rb') as handle:
            pretrained_models = pickle.load(handle)
            
        t1_model = pretrained_models['T1_Model']
        t2_model = pretrained_models['T2_Model']

        T2_NN.load_state_dict(t2_model.state_dict())
        T1_NN.load_state_dict(t1_model.state_dict())

        # Passing the model to GPU
        T2_NN = T2_NN.to(device)

        # Passing the model to GPU
        T1_NN = T1_NN.to(device)

        if 'T1_optimizer_dict' in pretrained_models.keys():

            loss_function_T2 = nn.L1Loss()
            loss_function_T1 = nn.L1Loss()
            
            optimizer_T2 = torch.optim.Adam(T2_NN.parameters(), lr=learning_rate)
            optimizer_T1 = torch.optim.Adam(T1_NN.parameters(), lr=learning_rate)

            optimizer_T1_dict = pretrained_models['T1_optimizer_dict']
            optimizer_T2_dict = pretrained_models['T2_optimizer_dict']

            optimizer_T2.load_state_dict(optimizer_T2_dict)
            optimizer_T1.load_state_dict(optimizer_T1_dict)

        else:
            
            loss_function_T2 = nn.L1Loss()
            loss_function_T1 = nn.L1Loss()

            optimizer_T2 = torch.optim.Adam(T2_NN.parameters(), lr=learning_rate)
            optimizer_T1 = torch.optim.Adam(T1_NN.parameters(), lr=learning_rate)
        
    else:
    
        # Passing the model to GPU
        T2_NN = T2_NN.to(device)
    
        # Define common loss function for t2 and t1 amplitudes
        #loss_function_T2 = nn.MSELoss()
        loss_function_T2 = nn.L1Loss()
    
        # Defining the optimizer
        optimizer_T2 = torch.optim.Adam(T2_NN.parameters(), lr=learning_rate)
    
        # Passing the model to GPU
        T1_NN = T1_NN.to(device)
    
        # Define common loss function for t2 and t1 amplitudes
        #loss_function_T1 = nn.MSELoss()
        loss_function_T1 = nn.L1Loss()
    
        # Defining the optimizer
        optimizer_T1 = torch.optim.Adam(T1_NN.parameters(), lr=learning_rate)

    ########################## End setting parameters ########################## 
    
    
    ########################## Start training ##################################
    start_time = time.time()
    
    epoch_array = []
    Avg_corre_loss_array = []
    Avg_rms_loss_array = []
    Avg_t2_loss_array = []
    Avg_t1_loss_array = []
    Corre_min = 100.0
    # Start learning
    print('Start Training')

    print(f'    Epoch     Correlation error     RMS error    t1 error    t2 error    Molecules   time\n', flush=True)

    for epoch in range(start_epoch, start_epoch + epoch_size): 
        # Print epoch
        #print(f'Starting epoch {epoch+1}')
        tick = time.time() 
        running_corre_loss = 0
        running_rms_loss = 0
        running_t2_loss = 0
        running_t1_loss = 0
        n_molecules = 0
        
        for mol_data_file in Molecule_array:
            with open(mol_data_file, 'rb') as handle:
                Train_data = pickle.load(handle)

            Molecule_data = Train_data["Train data big"]

            # Iterate over the DataLoader for training data
            for i, data in enumerate(Molecule_data):
            
                inputs_t2 = data['Feats_t2']
                targets_t2 = data['Amps_t2']
                inputs_t1 = data['Feats_t1']
                targets_t1 = data['Amps_t1']
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
                t2_pred = T2_NN(inputs_t2[:,:16],inputs_t2[:,16:24],inputs_t2[:,30:], inputs_t2[:,24:30])
                t1_pred = T1_NN(inputs_t1[:,:8],inputs_t1[:,8:12],inputs_t1[:,14:], inputs_t1[:,12:14])
                
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
            
                corre_loss = (torch.abs(exact_corre-CCSDcorr_E))*1000
            
            
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
            
                corre_loss2 = (torch.abs(exact_corre.detach().clone()-CCSDcorr_E2))*1000
            
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
            
                Total_loss_t2 = corre_loss  + t2_loss + total_rms 
                Total_loss_t1 = corre_loss2  + t1_loss + total_rms2
            
                #Total_loss_t2 = total_rms  + t2_loss 
                #Total_loss_t1 = total_rms2 + t1_loss
            
                current_corre_loss = corre_loss.detach().clone()
                current_rms_loss = total_rms.detach().clone()
                current_t2_loss = t2_loss.detach().clone()
                current_t1_loss = t1_loss.detach().clone()
            
                running_corre_loss += current_corre_loss
                running_rms_loss += current_rms_loss
                running_t2_loss += current_t2_loss
                running_t1_loss += current_t1_loss
            
                # Perform backward pass
                Total_loss_t2.backward()
                Total_loss_t1.backward()

                # Perform optimization
                optimizer_T2.step()
                optimizer_T1.step()

            n_molecules += len(Molecule_data)
        
        tock = time.time()

        epoch_time = tock-tick


        #progressBar(epoch+1, epoch_size, (time.time()-start_time))
                 
        Average_corre_error = running_corre_loss/n_molecules
        Average_rms_error = running_rms_loss/n_molecules
        Average_t2_error = running_t2_loss/n_molecules
        Average_t1_error = running_t1_loss/n_molecules
 
        epoch_array.append(epoch+1)
        Avg_corre_loss_array.append(Average_corre_error.cpu())
        Avg_rms_loss_array.append(Average_rms_error.cpu())
        Avg_t2_loss_array.append(Average_t2_error.cpu())
        Avg_t1_loss_array.append(Average_t1_error.cpu()) 

        print(f'    {epoch+1:10d}    {Average_corre_error:10.4f}    {Average_rms_error:10.4f}    {Average_t1_error:10.4f}    {Average_t2_error:10.4f}    {n_molecules:10d}    {epoch_time:10.5f}', flush=True)
        
        if (epoch+1)%1000 == 0:
            
            t2_tmp_model = MLP_Amp(indMO_in=16, MOint_in=8, MOvec_in=48, Amp_in=6)
            t1_tmp_model = MLP_Amp(indMO_in=8, MOint_in=4, MOvec_in=24, Amp_in=2)
            
            t2_tmp_model.load_state_dict(T2_NN.state_dict())
            t1_tmp_model.load_state_dict(T1_NN.state_dict())
            
            Model_dict = {
            'T1_Model': t1_tmp_model,
            'T2_Model': t2_tmp_model,
            'T1_optimizer_dict': optimizer_T1.state_dict(),
            'T2_optimizer_dict': optimizer_T2.state_dict(),
            'current_epoch': epoch+1
            }
            
            model_name = model_name_prefix + '_' +str(epoch+1) + '.sav'
            
            with open(model_name, 'wb') as handle:
                pickle.dump(Model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #epoch_tensor = torch.from_numpy(np.asarray(epoch_array))
        #Avg_t2_loss_tensor = torch.from_numpy(np.asarray(Avg_t2_loss_array))
        #Avg_t1_loss_tensor = torch.from_numpy(np.asarray(Avg_t1_loss_array))

    ########################## End training ####################################        
    
            Train_data_dict = {
            'epoch' : epoch_array, 
            'Avg_corre_loss' : Avg_corre_loss_array,
            'Avg_rms_loss' : Avg_rms_loss_array,
            'Avg_t2_loss_array' : Avg_t2_loss_array,
            'Avg_t1_loss_array' : Avg_t1_loss_array
            }
    
            dict_name = model_name_prefix + '_data_dict.sav'
            with open(dict_name, 'wb') as f:
                pickle.dump(Train_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    # Calculating the time spent
    print()
    print('End Training')
    duration = time.time()-start_time 
    hrs = int(duration/3600)
    mins = int((duration%3600)/60)
    sec = int(np.round(duration%60))
    
    print('Training process has finished.')
    print('Duration :', hrs, 'hrs', mins, 'mins', sec, 'sec')


mol_data_file_array = ['GDB5_Training_1_Allfeats_LA4.sav', 'GDB5_Training_2_Allfeats_LA4_same-sw.sav',
                       'GDB5_Training_3_Allfeats_LA4_same-sw.sav', 'GDB5_Training_4_Allfeats_LA4_same-sw.sav', 
                       'GDB5_Training_5_Allfeats_LA4_same-sw.sav', 'GDB5_Training_6_Allfeats_LA4_same-sw.sav',
                       'GDB5_Training_7_Allfeats_LA4_same-sw.sav', 'GDB5_Training_8_Allfeats_LA4_same-sw.sav',
                       'GDB5_Training_9_Allfeats_LA4_same-sw.sav', 'GDB5_Training_10_Allfeats_LA4_same-sw.sav',
                       'GDB5_Testing_1_Allfeats_LA4_same-sw.sav', 'GDB5_Testing_17_Allfeats_LA4_same-sw.sav',
                       'GDB5_Testing_18_Allfeats_LA4_same-sw.sav', 'GDB5_Training_200_1_Allfeats_LA4_same-sw.sav',
                       'GDB5_Training_200_2_Allfeats_LA4_same-sw.sav', 'GDB5_Training_200_8_Allfeats_LA4_same-sw.sav',
                       'GDB5_Training_200_3_Allfeats_LA4_same-sw.sav', 'GDB5_Training_200_4_Allfeats_LA4_same-sw.sav',
                       'GDB5_Training_200_5_Allfeats_LA4_same-sw.sav', 'GDB5_Training_200_6_Allfeats_LA4_same-sw.sav',
                       'GDB5_Training_200_7_Allfeats_LA4_same-sw.sav',]

pretrained = False

with open('GDB5_scaler_weights.sav', 'rb') as handle:
    Train_data = pickle.load(handle)

scaler_t2 = Train_data['Scaler_t2']
ff_t2 = Train_data['Weights_t2']
scaler_t1 = Train_data['Scaler_t1']
ff_t1 = Train_data['Weights_t1']

Model_name_prefix = 'Model_v5_Training200_chain_combined_pretrained_75-test_medium_mEh'
train(mol_data_file_array,
      start_epoch=2000,
      epoch_size=25000, 
      plot=False, 
      model_name_prefix=Model_name_prefix, 
      pretrained=True, 
      pretrained_model='Model_v5_Training200_chain_combined_pretrained_75-test_medium_mEh_2000.sav')
