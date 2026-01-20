#!/usr/bin/env python
# coding: utf-8

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
from MTL_model_t1_t2_Combined_separate_196n_7ly import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
import sys


# In[2]:


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


# In[3]:


# Features

features_t2 = [ 'Eocc1','Jocc1', 'Kocc1', 'Hocc1', 'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Evir1', 'Hvir1', 
               'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2','Jia1', 'Jia2', 'Kia1', 'Kia2','Jia1mag',
               'Jia2mag', 'Kia1mag', 'Kia2mag', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 
               't2sign']


features_t1 = ['E_i', 'E_a', 'H_i', 'H_a', 'J_i', 'J_a', 'K_i', 'K_a', 'Jia', 'Kia', 'Jiamag_1', 'Kiamag_1', 
               'orbdiff_1', 't1_mp2']


# In[4]:


def Test(feat_file, model_dict,scaler_t2, 
         scaler_t1, ff_t2, ff_t1, count = 1):
    
    t2_model = model_dict['T2_Model']
    t1_model = model_dict['T1_Model']
    
    t2_model = t2_model.to(device)
    t1_model = t1_model.to(device)
    
    dat_file = np.load(feat_file)
    Largefeatures = dat_file['t2_feats']
    Xnew_t1 = dat_file['t1_feats']
    MP2_amps = dat_file['MP2_amps']
    Largepositions = dat_file['t2_pos']
    Positions_t1 = dat_file['t1_pos']
    file_path = dat_file['mol_file']
    iu = dat_file['iu']
    print(MP2_amps.shape)
    f_path = file_path[0].replace('isaac/','isaac24/')

    text = open(f_path, 'r').read()
    mol = psi4.geometry(text)
    psi4.set_options({'basis':        'cc-pVDZ',
                      'scf_type':     'pk',
                      'maxiter':      1000,
                      'reference':    'rhf',
                      'mp2_type':     'conv',
                      'e_convergence': 1e-8,
                      'Freeze_core':   False,
                      'd_convergence': 1e-8})
                
    A=HelperCCEnergy(mol)
    
    X_new_scaled= scaler_t2.transform(Largefeatures)
    X_newer_scaled= X_new_scaled * ff_t2
            
    X_new_t1_scaled = scaler_t1.transform(Xnew_t1)
    X_newer_t1_scaled = X_new_t1_scaled * ff_t1
    
    X_test_t2_scaled = torch.from_numpy(X_newer_scaled)
    X_test_t1_scaled = torch.from_numpy(X_newer_t1_scaled)

    X_t2_tensor = X_test_t2_scaled.float().to(device)
    X_t1_tensor = X_test_t1_scaled.float().to(device)
            
    t2_model.eval()
    with torch.no_grad():
        y_t2 = t2_model(X_t2_tensor[:,:16],X_t2_tensor[:,16:24],X_t2_tensor[:,30:],X_t2_tensor[:,24:30])
                
    t1_model.eval()
    with torch.no_grad():
         y_t1 = t1_model(X_t1_tensor[:,:8],X_t1_tensor[:,8:12],X_t1_tensor[:,14:],X_t1_tensor[:,12:14])
            
    y_t2 = y_t2.to("cpu").detach().numpy()
    y_t1 = y_t1.to("cpu").detach().numpy()
            
    newamps = MP2_amps.reshape(-1,1)
    newamps[Largepositions] = y_t2.reshape(-1,1)
    
    iu = np.triu_indices((A.nocc-A.nfzc)*A.nvirt)
    MLt2 = np.zeros(((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt))
    MLt2[iu] = newamps.reshape(-1,)
    il = np.tril_indices((A.nocc-A.nfzc)*A.nvirt,-1)
    MLt2[il] = (MLt2.copy()).T[il]
    MLt2=MLt2.reshape(A.nocc-A.nfzc,A.nvirt,A.nocc-A.nfzc,A.nvirt).swapaxes(1,2)
    A.t2=MLt2.copy()
            
    newamps_t1 = np.zeros(((A.nocc-A.nfzc)*A.nvirt,1))
    newamps_t1[Positions_t1] = y_t1.reshape(-1,1)
            
    MLt1 = newamps_t1.reshape(A.nocc-A.nfzc, A.nvirt)
    A.t1=MLt1.copy()
            
    print()
    print('Start DDCCSD iterations: Predicted t1 and Predicted t2')
            
    if count == 1:
        A.compute_energy(maxiter=100)
        #rhfenergy.append(A.rhf_e)
        #startenergy.append(A.StartEnergy)
    
        if hasattr(A,"FinalEnergy"):
            DDCCSD_steps = A.steps
        else:
            DDCCSD_steps = 101
    
        print()
        print('Start CCSD iterations: Zero t1 and MP2 t2 (Conventional CCSD)')
        #B.compute_energy()
    
        DDCCSD_error = abs(A.StartEnergy-A.FinalEnergy)*1000
        
        DDCCSD_Energy = A.StartEnergy
        
        CCSD_Energy = A.FinalEnergy
        
        return feat_file, CCSD_Energy, DDCCSD_Energy, DDCCSD_error
                
    else:
        A.compute_energy(maxiter=1)
        
        DDCCSD_Energy = A.StartEnergy
        
        return feat_file, DDCCSD_Energy


# In[5]:


model_array = ['Model_v5_Training200_chain_combined_pretrained_75-test_medium_mEh_4000.sav']

main_folder = 'Testing_final_medium_75'

molecule_array = []
# for test_folder in os.listdir(main_folder):
folder_path = main_folder + '/'
#if (os.path.isdir(folder_path)) and (test_folder.startswith('Testing_')):
for mol in os.listdir(folder_path):
    if mol.endswith('.npz'):
        mol_path = folder_path+mol
        molecule_array.append(mol_path)


with open('GDB5_scaler_weights.sav', 'rb') as handle: 
    Train_data = pickle.load(handle)

scaler_t2 = Train_data['Scaler_t2'] 
ff_t2 = Train_data['Weights_t2'] 
scaler_t1 = Train_data['Scaler_t1'] 
ff_t1 = Train_data['Weights_t1'] 
count = 2

for model_name in model_array:

    with open(model_name, 'rb') as handle:
        Model_v5 = pickle.load(handle)

    dat_file = model_name.split('.')[0]+'_testing-example_combined_chain.csv'
    steps=list()
    difference=list()
    filenames=list()
    CCSD_steps_array=list()
    DDCCSD_steps_array=list()
    CCSD_Energy_array=list()
    DDCCSD_Energy_array=list() 
    
    if dat_file in os.listdir('./'):
        dat_df = pd.read_csv(dat_file)
        dat_dict = dat_df.to_dict()
        for x, fname in dat_dict['Molecule'].items():
            filenames.append(fname)
        for x,dif in dat_dict['Difference (mEh)'].items():
            difference.append(dif)
        for x,CCSD_Energy in dat_dict['CCSD Energy'].items():
            CCSD_Energy_array.append(CCSD_Energy)
        for x,DDCCSD_Energy in dat_dict['DDCCSD Energy'].items():
            DDCCSD_Energy_array.append(DDCCSD_Energy)
    
    if count == 1:
        for mol in molecule_array:
            if mol not in filenames:
                print('Molecular File name: ', mol)
                molecule_name, CCSD_Energy1, DDCCSD_Energy1, DDCCSD_error1 = Test(mol, model_dict=Model_v5,
                                                                              scaler_t1=scaler_t1, 
                                                               scaler_t2=scaler_t2, ff_t1=ff_t1, 
                                                               ff_t2=ff_t2,count = count)
        
                filenames.append(molecule_name)
                difference.append(DDCCSD_error1)
                CCSD_Energy_array.append(CCSD_Energy1)
                DDCCSD_Energy_array.append(DDCCSD_Energy1)
                   
                Data_dict = {
                'Molecule': filenames,
                'Difference (mEh)': difference,
                'CCSD Energy': CCSD_Energy_array,
                'DDCCSD Energy': DDCCSD_Energy_array,
                }
                
                Data_df = pd.DataFrame.from_dict(Data_dict)
            
                Data_df.to_csv(dat_file)
    
        count += 1
    
    else:
        for mol in molecule_array:
            if mol not in filenames:
                molecule_name, DDCCSD_Energy1 = Test(mol, model_dict=Model_v5,
                                                 scaler_t1=scaler_t1, 
                                                 scaler_t2=scaler_t2, ff_t1=ff_t1, 
                                                ff_t2=ff_t2,count = count)
        
                filenames.append(molecule_name)
                DDCCSD_Energy_array.append(DDCCSD_Energy1)
                   
                Data_dict = {
                'Molecule': filenames,
                'DDCCSD Energy': DDCCSD_Energy_array
                }
                
                Data_df = pd.DataFrame.from_dict(Data_dict)
            
                Data_df.to_csv(dat_file)
    
        count += 1    
    


