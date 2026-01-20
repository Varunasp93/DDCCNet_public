
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

features_t2 = [ 'Eocc1','Jocc1', 'Kocc1', 'Hocc1', 'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Evir1', 'Hvir1', 
               'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2','Jia1', 'Jia2', 'Kia1', 'Kia2','Jia1mag',
               'Jia2mag', 'Kia1mag', 'Kia2mag', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 
               't2sign']


features_t1 = ['E_i', 'E_a', 'H_i', 'H_a', 'J_i', 'J_a', 'K_i', 'K_a', 'Jia', 'Kia', 'Jiamag_1', 'Kiamag_1', 
               'orbdiff_1', 't1_mp2']

def Save_feats(Foldername, save_folder_name):
    steps=list()
    difference=list()
    filenames=list()
    CCSD_steps_array=list()
    DDCCSD_steps_array=list()
    CCSD_Energy_array=list()
    DDCCSD_Energy_array=list()
    
    LA_lim = 0.0001

    for filename in os.listdir(Foldername):
        if filename.endswith('.xyz'):
            psi4.core.clean()
            filenames.append(filename)
            print (filename)
            path1=str(Foldername+filename)
            text = open(path1, 'r').read()
            #print(text)
            mol = psi4.geometry(text)

            psi4.set_options({'basis':        'cc-pVDZ',
                              'scf_type':     'pk',
                              'maxiter':      1000,
                              'reference':    'rhf',
                              'mp2_type':     'conv',
                              'e_convergence': 1e-8,
                              'Freeze_core':   False,
                              'd_convergence': 1e-8})
                
            MLt2=0
            A=HelperCCEnergy(mol) 
            
            A.compute_t1_mp2()
            matrixsize=(A.nocc-A.nfzc)*(A.nocc-A.nfzc)*A.nvirt*A.nvirt
            Xnew=np.zeros([(A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt,len(features_t2)])
            for x in range(len(features_t2)):
                Xnew[:,:,:,:,x]=getattr(A, features_t2[x])
            
            matrixsize_t1=(A.nocc-A.nfzc)*A.nvirt
            Xnew_t1=np.zeros([(A.nocc-A.nfzc),A.nvirt, len(features_t1)])
            for x in range(0,len(features_t1)):
                Xnew_t1[:,:, x]=getattr(A, features_t1[x])
            
            MO_coefficient_matrix = (A.npC).T
            Occ_MO_coefficient_matrix = MO_coefficient_matrix[:(A.nocc-A.nfzc)]
            Virt_MO_coefficient_matrix = MO_coefficient_matrix[(A.nocc-A.nfzc):A.nmo]
            
            print()
            print(MO_coefficient_matrix.shape)
            print(Occ_MO_coefficient_matrix.shape)
            print(Virt_MO_coefficient_matrix.shape)
            print()
            
            i_coef_mat_t2 = np.zeros(((A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt,12))
            j_coef_mat_t2 = np.zeros(((A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt,12))
            
            a_coef_mat_t2 = np.zeros(((A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt,12))
            b_coef_mat_t2 = np.zeros(((A.nocc-A.nfzc),(A.nocc-A.nfzc),A.nvirt,A.nvirt,12))
            
            i_coef_mat_t1 = np.zeros(((A.nocc-A.nfzc),A.nvirt,12))
            a_coef_mat_t1 = np.zeros(((A.nocc-A.nfzc),A.nvirt,12))
            
            in_size = A.nmo
            out_size = 12
            
            stride = in_size // out_size
            
            kernel = in_size - (out_size-1)*stride
            
            maxpool_layer = nn.MaxPool1d(kernel_size=kernel, stride=stride)
            
            # Slice the relevant parts
            occ_coef = Occ_MO_coefficient_matrix[:(A.nocc - A.nfzc)]  # shape: [n_occ, nmo]
            virt_coef = Virt_MO_coefficient_matrix[:A.nvirt]           # shape: [n_virt, nmo]

            # Convert to torch
            occ_tensor = torch.from_numpy(occ_coef).float()            # shape: [n_occ, nmo]
            virt_tensor = torch.from_numpy(virt_coef).float()          # shape: [n_virt, nmo]

            # Reshape for MaxPool1d (input shape: [batch, channel, width])
            occ_tensor_reshaped = occ_tensor.unsqueeze(1)              # [n_occ, 1, nmo]
            virt_tensor_reshaped = virt_tensor.unsqueeze(1)            # [n_virt, 1, nmo]

            # Apply maxpool
            maxpool_occ = maxpool_layer(occ_tensor_reshaped)           # [n_occ, 1, pooled_dim]
            maxpool_virt = maxpool_layer(virt_tensor_reshaped)         # [n_virt, 1, pooled_dim]

            # Normalize virtual orbitals
            norm_virt = 1.0 / torch.sqrt(torch.sum(virt_tensor ** 2, dim=1, keepdim=True))
            virt_tensor_normalized = virt_tensor * norm_virt
            virt_tensor_norm_reshaped = virt_tensor_normalized.unsqueeze(1)
            maxpool_virt_normalized = maxpool_layer(virt_tensor_norm_reshaped)

            # Fill T1 matrices
            #i_coef_mat_t1 = maxpool_occ.unsqueeze(2).repeat(1, 1, A.nvirt)  # [n_occ, 1, n_virt]
            i_coef_mat_t1 = maxpool_occ.expand(-1, A.nvirt, -1)
            #a_coef_mat_t1 = maxpool_virt_normalized.unsqueeze(0).repeat(A.nocc - A.nfzc, 1, 1)  # [n_occ, 1, n_virt]
            a_coef_mat_t1 = maxpool_virt_normalized.transpose(0, 1).expand(A.nocc - A.nfzc, -1, -1)  # [n_occ, n_virt, pooled_dim]

            # For T2 tensors, build the cross products
            occ1_exp = occ_tensor.unsqueeze(1).repeat(1, A.nocc - A.nfzc, 1)     # [n_occ, n_occ, nmo]
            occ2_exp = occ_tensor.unsqueeze(0).repeat(A.nocc - A.nfzc, 1, 1)     # [n_occ, n_occ, nmo]

            virt1_exp = virt_tensor_normalized.unsqueeze(1).repeat(1, A.nvirt, 1)  # [n_virt, n_virt, nmo]
            virt2_exp = virt_tensor_normalized.unsqueeze(0).repeat(A.nvirt, 1, 1)  # [n_virt, n_virt, nmo]

            # Reshape and apply pooling
            def pool_batch(x):  # x: [B1, B2, nmo] → [B1, B2, pooled]
                x = x.reshape(-1, 1, A.nmo)
                x_pooled = maxpool_layer(x)  # shape: [B1*B2, 1, pooled]
                return x_pooled.reshape(x.shape[0] // A.nvirt, A.nvirt, -1)

            # T2 coefficients
            i_coef_mat_t2 = pool_batch(occ1_exp.unsqueeze(2).repeat(1, 1, A.nvirt * A.nvirt, 1).reshape(-1, A.nmo))
            j_coef_mat_t2 = pool_batch(occ2_exp.unsqueeze(2).repeat(1, 1, A.nvirt * A.nvirt, 1).reshape(-1, A.nmo))
            a_coef_mat_t2 = pool_batch(virt1_exp.reshape(-1, A.nmo))
            b_coef_mat_t2 = pool_batch(virt2_exp.reshape(-1, A.nmo))

            # i and j from occ1 x occ2 x virt1 x virt2
            i_coef_mat_t2 = i_coef_mat_t2.reshape(A.nocc - A.nfzc, A.nocc - A.nfzc, A.nvirt, A.nvirt, -1)
            j_coef_mat_t2 = j_coef_mat_t2.reshape(A.nocc - A.nfzc, A.nocc - A.nfzc, A.nvirt, A.nvirt, -1)

            # a and b from virt1 x virt2 → broadcasted over occ1 x occ2
            a_coef_mat_t2 = a_coef_mat_t2.reshape(1, 1, A.nvirt, A.nvirt, -1).expand(A.nocc - A.nfzc, A.nocc - A.nfzc, -1, -1, -1)
            b_coef_mat_t2 = b_coef_mat_t2.reshape(1, 1, A.nvirt, A.nvirt, -1).expand(A.nocc - A.nfzc, A.nocc - A.nfzc, -1, -1, -1)

            Xnew = np.concatenate((Xnew,i_coef_mat_t2),axis=4)
            Xnew = np.concatenate((Xnew,j_coef_mat_t2),axis=4)
            Xnew = np.concatenate((Xnew,a_coef_mat_t2),axis=4)
            Xnew = np.concatenate((Xnew,b_coef_mat_t2),axis=4)
            
            Xnew = Xnew.swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt,len(features_t2)+4*12)
            iu = np.triu_indices((A.nocc-A.nfzc)*A.nvirt)
            Xnew = Xnew[iu]
            Positions = np.arange(Xnew.shape[0])
            
            Abs_MP2 = np.abs(Xnew[:,27])
            Largefeatures = Xnew[Abs_MP2>=LA_lim]
            Largepositions = Positions[Abs_MP2>=LA_lim]

            Xnew_t1 = np.concatenate((Xnew_t1,i_coef_mat_t1), axis=2)
            Xnew_t1 = np.concatenate((Xnew_t1,a_coef_mat_t1), axis=2)
            
            Xnew_t1 = Xnew_t1.reshape(matrixsize_t1,len(features_t1)+2*12)
            Positions_t1 = np.arange(matrixsize_t1)
            
            # Save the features
            feature_file_name = save_folder_name+filename.split('.')[0]+'_feats'
            np.savez_compressed(feature_file_name, 
                                t2_feats = Largefeatures,
                                t2_pos = Largepositions,
                                t1_pos = Positions_t1,
                                t1_feats = Xnew_t1,
                                MP2_amps = np.asarray(Xnew[:,27]),
                                mol_file = np.asarray([path1]),
                                iu = iu)
                

#Save_feats('/lustre/isaac24/proj/UTK0022/DDCCNet/QM9/Training_folders/Testing_example/', 'Testing_feats/Testing_example/')

for i in range(11,29):
    xyz_folder  = '/lustre/isaac24/proj/UTK0022/DDCCNet/GDB5/Diverse/Group_' + str(i) + '/'
    save_folder = 'Testing_feats/Group_' + str(i) + '/'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    Save_feats(xyz_folder, save_folder)
