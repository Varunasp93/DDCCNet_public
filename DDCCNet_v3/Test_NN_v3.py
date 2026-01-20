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
from MTL_model_DDCCNet_v3_196n_7l import *
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

features_t2 = ['orbdiff', 'doublecheck', 't2start']
features_t1 = ['orbdiff_1','t1_mp2']

intermediates_t2 = ['tmp_tau', 'Wmnij', 'Wmbej', 'Wmbje', 'Zmbij']
intermediates_t1 = ['Fae', 'Fmi', 'Fme']


# In[4]:


# Test Models
def Test(Foldername, model_dict, scaler_t2, scaler_t1, ff_t2, ff_t1, 
        save_file_name='dat_file.csv', count=1):
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

            if count==1:
                print('Start DDCCSD iterations')
                A.compute_energy(maxiter=100)

                if hasattr(A,"FinalEnergy"):
                    DDCCSD_steps = A.steps
                else:
                    DDCCSD_steps = 101

                print()
                print('Start CCSD iterations')

                DDCCSD_error = abs(A.StartEnergy-A.FinalEnergy)*1000

                DDCCSD_Energy = A.StartEnergy

                CCSD_Energy = A.FinalEnergy

                difference.append(DDCCSD_error)
                CCSD_steps_array.append(A.steps)
                DDCCSD_steps_array.append(DDCCSD_steps)

                CCSD_Energy_array.append(CCSD_Energy)
                DDCCSD_Energy_array.append(DDCCSD_Energy)

                Data_dict = {
                    'Molecule': filenames,
                    'Difference (mEh)': difference,
                    'CCSD steps': CCSD_steps_array,
                    'CCSD Energy': CCSD_Energy_array,
                    'DDCCSD Energy': DDCCSD_Energy_array,
                    'DDCCSD steps': DDCCSD_steps_array
                }

                Data_df = pd.DataFrame.from_dict(Data_dict)

                Data_df.to_csv(save_file_name)

            else:
                print('Start DDCCSD iterations')
                A.compute_energy(maxiter=1)

                DDCCSD_Energy = A.StartEnergy

                DDCCSD_Energy_array.append(DDCCSD_Energy)

                Data_dict = {
                    'Molecule': filenames,
                    'DDCCSD Energy': DDCCSD_Energy_array,
                }

                Data_df = pd.DataFrame.from_dict(Data_dict)

                Data_df.to_csv(save_file_name)


# In[ ]:


#Testing



with open('Train_CO2_mono_5t2_3t1_Allamps_LA4_fc.sav', 'rb') as handle: Train_data = pickle.load(handle)

Molecule_data = Train_data["Train data big"] 
scaler_t2 = Train_data['Scaler_t2'] 
ff_t2 = Train_data['Weights_t2'] 
scaler_t1 = Train_data['Scaler_t1'] 
ff_t1 = Train_data['Weights_t1'] 
count = 1

with open('Model_v6_mono_fc_4999.sav', 'rb') as handle:
    Model_v6 = pickle.load(handle)

dat_file = model_name.split('.')[0]+'.csv'

T2_dict = Test(Foldername='../Testing_di_tri/', 
               scaler_t1=scaler_t1, scaler_t2=scaler_t2, 
               ff_t1=ff_t1, ff_t2=ff_t2, model_dict=Model_v6, 
               save_file_name = dat_file, count = count)

