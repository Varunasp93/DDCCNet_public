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
#from MTL_model_separate import *
from sklearn.neighbors import KNeighborsRegressor


# In[2]:


# Getting features

features_t2 = [ 'Eocc1','Jocc1', 'Kocc1', 'Hocc1', 'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Evir1', 'Hvir1', 
               'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2','Jia1', 'Jia2', 
               'Kia1', 'Kia2', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 't2sign', 
               'Jia1mag', 'Jia2mag', 'Kia1mag', 'Kia2mag']
#features_t2 = [ 'Eocc1', 'Hocc1', 'Eocc2', 'Hocc2', 'Evir1', 'Hvir1', 'Jvir1', 'Hvir2', 'Jvir2', 'Kvir2',
#               'Jia1', 'Jia2', 'orbdiff', 'doublecheck', 't2start', 't2mag', 'Jia1mag', 'Jia2mag']

#features_t1 = ['E_i', 'E_a', 'H_i', 'H_a', 'J_a', 'K_a', 'Jia', 'Kia', 'Kiamag_1', 't1_mp2']
features_t1 = ['E_i', 'E_a', 'H_i', 'H_a', 'J_i', 'J_a', 'K_i', 'K_a', 'Jia', 'Kia', 'Jiamag_1', 'Kiamag_1', 
               'orbdiff_1', 't1_mp2']

debug=2
#LA_lim=1.0e-2
#LA_lim = 0.0

def GetAmps_ddccnet(Foldername, lim, occ=False, vir=False):
    i=1
    LA_lim = 0.0
    Molecule_array = []
    for filename in os.listdir(str(Foldername)):
        Bigfeature = []
        Amp_ids = []
        Bigamp = []
        Bigamp_t1 = []
        psi4.core.clean()
        path1=str(str(Foldername)+filename)
        if path1.endswith('.xyz'):
            text = open(path1, 'r').read()
            mol = psi4.geometry(text)
            psi4.core.clean()


            psi4.set_options({'basis':        'cc-pVDZ',#'6-31g',
                              'scf_type':     'pk',
                              'reference':    'rhf',
                              'mp2_type':     'conv',
                              'e_convergence': 1e-8,
                              'Freeze_core':   False,
                              'd_convergence': 1e-8})
            A=HelperCCEnergy(mol)
            A.compute_t1_mp2()
            corr_e = A.compute_energy()
            matrixsize=(A.nocc-A.nfzc)*(A.nocc-A.nfzc)*A.nvirt*A.nvirt
            Bigmatrix=np.zeros([(A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt, len(features_t2)])
            for x in range(0,len(features_t2)):
                Bigmatrix[:,:,:,:, x]=getattr(A, features_t2[x])

            Bigmatrix_t1=np.zeros([(A.nocc-A.nfzc),A.nvirt, len(features_t1)])
            for x in range(0,len(features_t1)):
                Bigmatrix_t1[:,:, x]=getattr(A, features_t1[x])

            Bigfeature = Bigmatrix.swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt,len(features_t2))
            iu = np.triu_indices((A.nocc-A.nfzc)*A.nvirt)
            Bigfeature = Bigfeature[iu]

            Bigamp = (A.t2).swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt)
            Bigamp = Bigamp[iu].reshape(-1,1)

            Bigint = (A.doublecheck).swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt)
            Bigint = Bigint[iu].reshape(-1,1)

            Bigint2 = (A.doublecheck).swapaxes(2,3).swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt)
            Bigint2 = Bigint2[iu].reshape(-1,1)

            Multiplicative_array = np.zeros(((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt))
            iu2 = np.triu_indices((A.nocc-A.nfzc)*A.nvirt, 1)

            Multiplicative_array[iu] = 1.0
            Multiplicative_array[iu2] = 2.0
            Multiplicative_array = Multiplicative_array[iu].reshape(-1,1)

            Abs_mp2 = np.abs(Bigfeature[:,23])
            Largefeature = Bigfeature[Abs_mp2>=LA_lim]
            Largeamp = Bigamp[Abs_mp2>=LA_lim]
            Largeint = Bigint[Abs_mp2>=LA_lim]
            Largeint2 = Bigint2[Abs_mp2>=LA_lim]
            Multiplicative_array = Multiplicative_array[Abs_mp2>=LA_lim]

            Bigfeature_t1=Bigmatrix_t1.reshape(-1,len(features_t1))
            Bigamp_t1 = A.t1.reshape(-1,1)
            F_ov = A.F_ov

            LA_corre = 2.0*np.sum((Bigamp.copy()).reshape(-1,)*(Bigint.copy()).reshape(-1,))
            LA_corre -= 1.0*np.sum((Bigamp.copy()).reshape(-1,)*(Bigint2.copy()).reshape(-1,))

            Smallamp = Bigamp[Abs_mp2<LA_lim]
            Smallint = Bigint[Abs_mp2<LA_lim]
            Smallint2 = Bigint2[Abs_mp2<LA_lim]

            SA_corre = 2.0*np.sum((Smallamp.copy()).reshape(-1,)*(Smallint.copy()).reshape(-1,))
            SA_corre -= 1.0*np.sum((Smallamp.copy()).reshape(-1,)*(Smallint2.copy()).reshape(-1,))

            Data_dict = {
                'Feats_t2': Largefeature,
                'Feats_t1': Bigfeature_t1,
                'Multi_array': Multiplicative_array,
                'Amps_t2': Largeamp,
                'Amps_t1': Bigamp_t1,
                'Integrals': Largeint,
                'Integrals2': Largeint2,
                'Corr_e': np.asarray([corr_e,SA_corre]),
                'F_ov': F_ov,
                'MO_oovv': A.doublecheck,
                'Orbs': np.asarray([A.nocc,A.nvirt])
                }
            Molecule_array.append(Data_dict)
    return Molecule_array


# In[3]:


lim_list=[0.0001]
lim_name = ['LA']

for lim_in, lim in enumerate(lim_list):
    Molecule_data = GetAmps_ddccnet('../Training_mono_di_10tri/', lim)

    # Scaling and calculating weights for t2

    is_new = 0
    for molecule in range(len(Molecule_data)):
        if is_new == 0:
            Feats = Molecule_data[molecule]['Feats_t2']
            Amps = Molecule_data[molecule]['Amps_t2']
            is_new = 1
        else:
            Feats = np.concatenate((Feats, Molecule_data[molecule]['Feats_t2']), axis=0)
            Amps = np.concatenate((Amps, Molecule_data[molecule]['Amps_t2']), axis=0)
    print(Feats.shape)
    print(Amps.shape)

    scaler_t2 = MinMaxScaler().fit(Feats)

    Bigmat = np.concatenate([Feats, Amps], axis=1)
    corr_mat = np.corrcoef(Bigmat.T)
    finalfactor = np.nan_to_num(corr_mat[len(features_t2)][:len(features_t2)], nan = 0.0)


    # Scaling and calculating weights for t1
    is_new = 0
    for molecule in range(len(Molecule_data)):
        if is_new == 0:
            Feats_t1 = Molecule_data[molecule]['Feats_t1']
            Amps_t1 = Molecule_data[molecule]['Amps_t1']
            is_new = 1
        else:
            Feats_t1 = np.concatenate((Feats_t1, Molecule_data[molecule]['Feats_t1']), axis=0)
            Amps_t1 = np.concatenate((Amps_t1, Molecule_data[molecule]['Amps_t1']), axis=0)
    print(Feats_t1.shape)
    print(Amps_t1.shape)

    scaler_t1 = MinMaxScaler().fit(Feats_t1)

    Bigmat_t1 = np.concatenate([Feats_t1, Amps_t1], axis=1)
    corr_mat_t1 = np.corrcoef(Bigmat_t1.T)
    finalfactor_t1 = np.nan_to_num(corr_mat_t1[len(features_t1)][:len(features_t1)], nan = 0.0)  

    # Converting them into pytorch tensors

    for molecule in range(len(Molecule_data)):
        new_feats_t2 = scaler_t2.transform(Molecule_data[molecule]['Feats_t2'])
        new_feats_t2 = new_feats_t2*finalfactor
        new_feats_t1 = scaler_t1.transform(Molecule_data[molecule]['Feats_t1'])
        new_feats_t1 = new_feats_t1*finalfactor_t1
        Feats_tensor = torch.from_numpy(new_feats_t2)
        Feats_t1_tensor = torch.from_numpy(new_feats_t1)
        Amps_tensor = torch.from_numpy(Molecule_data[molecule]['Amps_t2'])
        Amps_t1_tensor = torch.from_numpy(Molecule_data[molecule]['Amps_t1'])
        Corre_tensor = torch.from_numpy(Molecule_data[molecule]['Corr_e'])
        F_ov_tensor = torch.from_numpy(Molecule_data[molecule]['F_ov'])
        MO_tensor = torch.from_numpy(Molecule_data[molecule]['MO_oovv'])
        Orbs_tensor = torch.from_numpy(Molecule_data[molecule]['Orbs'])
        Int_tensor = torch.from_numpy(Molecule_data[molecule]['Integrals'])
        Int_tensor2 = torch.from_numpy(Molecule_data[molecule]['Integrals2'])
        Multi_array_tensor = torch.from_numpy(Molecule_data[molecule]['Multi_array'])

        Molecule_data[molecule]['Feats_t2'] = Feats_tensor
        Molecule_data[molecule]['Feats_t1'] = Feats_t1_tensor
        Molecule_data[molecule]['Amps_t2'] = Amps_tensor
        Molecule_data[molecule]['Amps_t1'] = Amps_t1_tensor
        Molecule_data[molecule]['Corr_e'] = Corre_tensor
        Molecule_data[molecule]['F_ov'] = F_ov_tensor
        Molecule_data[molecule]['MO_oovv'] = MO_tensor
        Molecule_data[molecule]['Orbs'] = Orbs_tensor
        Molecule_data[molecule]['Integrals'] = Int_tensor
        Molecule_data[molecule]['Integrals2'] = Int_tensor2
        Molecule_data[molecule]['Multi_array'] = Multi_array_tensor   

    # Saving data

    # For pytorch
    Data_dict = {
                'Train data big': Molecule_data,
                'Scaler_t2': scaler_t2,
                'Weights_t2': finalfactor,
                'Scaler_t1': scaler_t1,
                'Weights_t1': finalfactor_t1,
                }
    Train_name = 'Train_mono-di-10tri_LA.sav'
    with open(Train_name, 'wb') as handle:
        pickle.dump(Data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
