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
#from helper_CC_ML_pert_T_fc_t1 import *
from helper_CC_ML_pert_T_fc_t1_lo import *
from MTL_model_t1_t2_Combined_separate_2_Allfeats import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import sys


# # Progress bar
# 
# To show the progress during the training 

# In[2]:


def progressBar(count_value, total):
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '|' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('Training[%s] %s%s\r' %(bar, percentage, '%'))
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

features_t2 = [ 'Eocc1','Jocc1', 'Kocc1', 'Hocc1', 'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Evir1', 'Hvir1', 
               'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2','Jia1', 'Jia2', 
               'Kia1', 'Kia2', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 
               'Jia1mag', 'Jia2mag', 'Kia1mag', 'Kia2mag']

#features_t2 = [ 'Eocc1', 'Hocc1', 'Eocc2', 'Hocc2', 'Evir1', 'Hvir1', 'Jvir1', 'Hvir2', 'Jvir2', 'Kvir2',
#               'Jia1', 'Jia2', 'orbdiff', 'doublecheck', 't2start', 't2mag', 'Jia1mag', 'Jia2mag']

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

def loading_train(Train_file): 

    with open(Train_file, 'rb') as handle:
        Train_data = pickle.load(handle)

    # Molecule_data = Train_data["Train data big"]
    scaler_t2 = Train_data['Scaler_t2']
    ff_t2 = Train_data['Weights_t2']
    scaler_t1 = Train_data['Scaler_t1']
    ff_t1 = Train_data['Weights_t1']

    return scaler_t2, ff_t2, scaler_t1, ff_t1


def Test(Foldername, model_dict, scaler_t2, scaler_t1, ff_t2, ff_t1, lim):
    steps=list()
    difference=list()
    filenames=list()
    CCSD_steps_array=list()
    DDCCSD_steps_array=list()
    DDCCSD_Error_list=list()
    MAE_t2_list=list()
    MAE_t1_list=list()
    DDCCSD_array=list() 

    LA_lim = lim

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
            B=HelperCCEnergy(mol) # zero t1 and MP2 t2
            #C=HelperCCEnergy(mol) # Pred t1 and Exact t2
            #D=HelperCCEnergy(mol) # Exact t1 and Pred t2

            A.compute_t1_mp2()
            matrixsize=(A.nocc-A.nfzc)*(A.nocc-A.nfzc)*A.nvirt*A.nvirt
            Xnew=np.zeros([(A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt,len(features_t2)])
            for x in range(len(features_t2)):
                Xnew[:,:,:,:,x]=getattr(A, features_t2[x])

            Xnew = Xnew.swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt,len(features_t2))
            iu = np.triu_indices((A.nocc-A.nfzc)*A.nvirt)
            Xnew = Xnew[iu]
            Positions = np.arange(Xnew.shape[0])

            Abs_MP2 = np.abs(Xnew[:,23])
            Largefeatures = Xnew[Abs_MP2>=LA_lim]
            Largepositions = Positions[Abs_MP2>=LA_lim]

            X_new_scaled= scaler_t2.transform(Largefeatures)
            X_newer_scaled= X_new_scaled * ff_t2

            matrixsize_t1=(A.nocc-A.nfzc)*A.nvirt
            Xnew_t1=np.zeros([1,matrixsize_t1,len(features_t1)])
            for x in range (0,len(features_t1)):
                Xnew_t1[0,:,x]=getattr(A, features_t1[x]).reshape(matrixsize_t1)

            Xnew_t1 = Xnew_t1.reshape(matrixsize_t1,len(features_t1))
            Positions_t1 = np.arange(matrixsize_t1)

            X_new_t1_scaled = scaler_t1.transform(Xnew_t1)
            X_newer_t1_scaled = X_new_t1_scaled * ff_t1

            X_test_t2_scaled = torch.from_numpy(X_newer_scaled)
            X_test_t1_scaled = torch.from_numpy(X_newer_t1_scaled)

            t2_model.eval()
            with torch.no_grad():
                y_t2 = t2_model(X_test_t2_scaled.float().to(device))

            t1_model.eval()
            with torch.no_grad():
                y_t1 = t1_model(X_test_t1_scaled.float().to(device))

            y_t2 = y_t2.to("cpu").detach().numpy()
            y_t1 = y_t1.to("cpu").detach().numpy()

            newamps = (Xnew[:,23]).reshape(-1,1)
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
            print('Start DDCCSD iterations: Predicted t1 and Predicted t2')
            A.compute_energy(maxiter=1)
            #rhfenergy.append(A.rhf_e)
            #startenergy.append(A.StartEnergy)

            DDCCSD_array.append(A.StartEnergy)

            Data_dict = {
                         'File_names': filenames,
                         'DDCCSD_energy': DDCCSD_array
                        }
            save_file_name = 'DDCCNetv3_CO2_mono-di-10tri_3000_di-tri.csv'
            Data_df = pd.DataFrame.from_dict(Data_dict)
            Data_df.to_csv(save_file_name)

        save_file_name = 'DDCCNetv3_CO2_mono-di-10tri_3000_di-tri.csv'
        Data_df = pd.DataFrame.from_dict(Data_dict)
        Data_df.to_csv(save_file_name)
    return np.average(DDCCSD_Error_list), np.average(MAE_t2_list), np.average(MAE_t1_list)




scaler_t2, ff_t2, scaler_t1, ff_t1 = loading_train('Train_mono-di-10tri_LA.sav')

Model_dict_array = ['DDCCNetv3_CO2-mono-di-10tri_3000.sav']
                    #'DDCCNetv3_CO2-mono-di_5000.sav']

print_lim_array = []
print_epoch_array = []
print_error_array = []
print_t2_mae = []
print_t1_mae = []

for la_cut in [0.0001]:
    for model_dict_name in Model_dict_array:
    
        epoch = int((model_dict_name.split('.')[0]).split('_')[2])

        with open(model_dict_name, 'rb') as handle:
            model_dict = pickle.load(handle)

        ddccsd_error, t2_mae, t1_mae = Test(Foldername='../Testing_di_tri/', model_dict=model_dict, scaler_t1=scaler_t1, scaler_t2=scaler_t2, 
                                                ff_t1=ff_t1, ff_t2=ff_t2, lim=la_cut)

