
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

# Getting features

features_t2 = [ 'Eocc1','Jocc1', 'Kocc1', 'Hocc1', 'Eocc2', 'Jocc2', 'Kocc2', 'Hocc2', 'Evir1', 'Hvir1', 
               'Jvir1', 'Kvir1', 'Evir2', 'Hvir2', 'Jvir2', 'Kvir2','Jia1', 'Jia2', 'Kia1', 'Kia2','Jia1mag',
               'Jia2mag', 'Kia1mag', 'Kia2mag', 'diag', 'orbdiff', 'doublecheck', 't2start', 't2mag', 
               't2sign']

#features_t2 = ['Eocc1', 'Hocc1', 'Eocc2', 'Hocc2', 'Evir1', 'Hvir1', 'Jvir1', 'Hvir2', 'Jvir2', 'Kvir2',
#               'Jia1', 'Jia2', 'orbdiff', 'doublecheck', 't2start', 't2mag', 'Jia1mag', 'Jia2mag']

features_t1 = ['E_i', 'E_a', 'H_i', 'H_a', 'J_i', 'J_a', 'K_i', 'K_a', 'Jia', 'Kia', 'Jiamag_1', 'Kiamag_1', 
               'orbdiff_1', 't1_mp2']

debug=2
LA_lim=0.0001

def GetAmps_ddccnet(Foldername, occ=False, vir=False):
    i=1
    Molecule_array = []
    for filename in os.listdir(str(Foldername)):
        Bigfeature = []
        Amp_ids = []
        Bigamp = []
        Bigamp_t1 = []
        psi4.core.clean()
        path1=str(str(Foldername)+filename)
        print(filename)
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

            print()
            print(Bigmatrix.shape)
            print(i_coef_mat_t2.shape)
            print(a_coef_mat_t2.shape)
            print(i_coef_mat_t1.shape)
            print(a_coef_mat_t1.shape)
            
            Bigmatrix = np.concatenate((Bigmatrix,i_coef_mat_t2),axis=4)
            Bigmatrix = np.concatenate((Bigmatrix,j_coef_mat_t2),axis=4)
            Bigmatrix = np.concatenate((Bigmatrix,a_coef_mat_t2),axis=4)
            Bigmatrix = np.concatenate((Bigmatrix,b_coef_mat_t2),axis=4)
            
            print(Bigmatrix.shape)
            
            Bigfeature = Bigmatrix.swapaxes(1,2).reshape((A.nocc-A.nfzc)*A.nvirt,(A.nocc-A.nfzc)*A.nvirt,len(features_t2)+4*12)
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
            
            Abs_mp2 = np.abs(Bigfeature[:,27])
            Largefeature = Bigfeature[Abs_mp2>=LA_lim]
            Largeamp = Bigamp[Abs_mp2>=LA_lim]
            Largeint = Bigint[Abs_mp2>=LA_lim]
            Largeint2 = Bigint2[Abs_mp2>=LA_lim]
            Multiplicative_array = Multiplicative_array[Abs_mp2>=LA_lim]
            
            Bigmatrix_t1 = np.concatenate((Bigmatrix_t1,i_coef_mat_t1), axis=2)
            Bigmatrix_t1 = np.concatenate((Bigmatrix_t1,a_coef_mat_t1), axis=2)
            
            Bigfeature_t1=Bigmatrix_t1.reshape(-1,len(features_t1)+2*12)
            Bigamp_t1 = A.t1.reshape(-1,1)
            
            print(Bigmatrix_t1.shape)
            
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

Molecule_data = GetAmps_ddccnet('../Training_folders/Training_1/')

# Scaling and calculating weights for t2

is_new = 0
for molecule in range(len(Molecule_data)):
    if is_new == 0:
        Feats = Molecule_data[molecule]['Feats_t2'][:,:len(features_t2)+48]
        Amps = Molecule_data[molecule]['Amps_t2']
        is_new = 1
    else:
        Feats = np.concatenate((Feats, Molecule_data[molecule]['Feats_t2'][:,:len(features_t2)+48]), axis=0)
        Amps = np.concatenate((Amps, Molecule_data[molecule]['Amps_t2']), axis=0)
print(Feats.shape)
print(Amps.shape)

scaler_t2 = MinMaxScaler().fit(Feats)

Bigmat = np.concatenate([Feats, Amps], axis=1)
corr_mat = np.corrcoef(Bigmat.T)
finalfactor = np.nan_to_num(corr_mat[len(features_t2)+48][:len(features_t2)+48], nan = 0.0)


# Scaling and calculating weights for t1
is_new = 0
for molecule in range(len(Molecule_data)):
    if is_new == 0:
        Feats_t1 = Molecule_data[molecule]['Feats_t1'][:,:len(features_t1)+24]
        Amps_t1 = Molecule_data[molecule]['Amps_t1']
        is_new = 1
    else:
        Feats_t1 = np.concatenate((Feats_t1, Molecule_data[molecule]['Feats_t1'][:,:len(features_t1)+24]), axis=0)
        Amps_t1 = np.concatenate((Amps_t1, Molecule_data[molecule]['Amps_t1']), axis=0)
print(Feats_t1.shape)
print(Amps_t1.shape)

scaler_t1 = MinMaxScaler().fit(Feats_t1)

Bigmat_t1 = np.concatenate([Feats_t1, Amps_t1], axis=1)
corr_mat_t1 = np.corrcoef(Bigmat_t1.T)
finalfactor_t1 = np.nan_to_num(corr_mat_t1[len(features_t1)+24][:len(features_t1)+24], nan = 0.0)


# Converting them into pytorch tensors

for molecule in range(len(Molecule_data)):
    new_feats_t2 = scaler_t2.transform(Molecule_data[molecule]['Feats_t2'][:,:len(features_t2)+48])
    new_feats_t2 = new_feats_t2*finalfactor
    new_feats2_t2 = Molecule_data[molecule]['Feats_t2']
    new_feats2_t2[:,:len(features_t2)+48] = new_feats_t2
    
    new_feats_t1 = scaler_t1.transform(Molecule_data[molecule]['Feats_t1'][:,:len(features_t1)+24])
    new_feats_t1 = new_feats_t1*finalfactor_t1
    new_feats2_t1 = Molecule_data[molecule]['Feats_t1']
    new_feats2_t1[:,:len(features_t1)+24] = new_feats_t1
    
    Feats_tensor = torch.from_numpy(new_feats2_t2)
    Feats_t1_tensor = torch.from_numpy(new_feats2_t1)
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
    print(Feats_tensor.shape)
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

with open('QM9_Training_1_Allfeats_LA4.sav', 'wb') as handle:
    pickle.dump(Data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
scaler_weight_dict = {
            'Scaler_t2': scaler_t2,
            'Weights_t2': finalfactor,
            'Scaler_t1': scaler_t1,
            'Weights_t1': finalfactor_t1,
            }

with open('QM9_scaler_weights.sav', 'wb') as handle:
    pickle.dump(scaler_weight_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Molecule_data_test = GetAmps_ddccnet('../Validate/')

# # Saving data
# 
# # For KNN and RF
# Data_dict_test = {
#             'Train data': Molecule_data_test,
#             }
# 
# with open('Validate_CO2_tri_Allfeats_LA4.sav', 'wb') as handle:
#     pickle.dump(Data_dict_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

