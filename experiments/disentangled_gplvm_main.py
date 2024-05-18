#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Disentangled GPLVM with independent GPs modelling molecular properties With individual hyperparameters.

TODO: 
    
    Test with MAP inference / other latent variable types
    
"""
import torch
import gpytorch
import os 
import pickle as pkl
import numpy as np
import yaml
import gc
import pandas as pd
from tqdm import trange
from sklearn.decomposition import PCA
from gpytorch.mlls import VariationalELBO
from models.likelihood import GaussianLikelihood
from utils.load_data import load_full_dataset
from utils.visualisation import plot_disentangled_2d_latents, plot_y_label_comparison
from utils.metrics import rmse_missing, nll
from models.disentangled_gplvm import DisentangledGPLVM, predict_joint_latent
from utils.config import BASE_SEED

save_model = True

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setting torch and numpy seed for reproducibility
    
    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    
    # Model settings file
    
    if os.path.exists("utils/settings.yml"):
        settings = yaml.safe_load(open("utils/settings.yml", "r"))
    else:
        print("Expected a file settings.yml but didn't find it.")
    
    # Load data 
    
    file_path = 'data/zinc_50k.txt'
    df = pd.read_csv(file_path)
    df_prop = ['logP','qed','SAS']
    data = torch.Tensor(np.array(df[df_prop]))
    
    #train_data, test_data = load_full_dataset(5000, 512)
    #Y_train= train_data['y1']
    #Y_test = test_data['y1']    
    
    Y_train = data[0:40000]
    Y_test = data[40000:50000]
    
    Y_train = Y_train.to(device)
    Y_test = Y_test.to(device)
    
    ## Extract model settings
    
    latent_dim = settings['gplvm']['latent_dim']
    num_inducing = settings['gplvm']['num_inducing']
    
    N = len(Y_train)
    data_dim = Y_train.shape[1]
  
    # Shared Model
    
    disentangled_model = DisentangledGPLVM(N, latent_dim, num_inducing, latent_config='point').to(device)
    
    # Likelihood
    
    likelihood_logP = GaussianLikelihood(batch_shape = disentangled_model.model_logP.batch_shape).to(device)
    likelihood_qed = GaussianLikelihood(batch_shape = disentangled_model.model_qed.batch_shape).to(device)
    likelihood_sas = GaussianLikelihood(batch_shape = disentangled_model.model_sas.batch_shape).to(device)
    
    mll_qed = VariationalELBO(likelihood_logP, disentangled_model.model_qed, num_data=len(Y_train)).to(device)
    mll_sas = VariationalELBO(likelihood_sas, disentangled_model.model_sas, num_data=len(Y_train)).to(device)
    mll_logP = VariationalELBO(likelihood_qed, disentangled_model.model_logP, num_data=len(Y_train)).to(device)
    
    optimizer = torch.optim.Adam([
        dict(params=disentangled_model.parameters(), lr=0.01),
        dict(params=likelihood_qed.parameters(), lr=0.01),
        dict(params=likelihood_logP.parameters(), lr=0.01),
        dict(params=likelihood_sas.parameters(), lr=0.01)
    ])
           
    disentangled_model.get_trainable_param_names()
    
    ############## Training loop - optimises the objective wrt kernel hypers, ######
    ################  variational params and inducing inputs using the optimizer provided ########
    
    torch.use_deterministic_algorithms(False)
    
    loss_list = []
    iterator = trange(3000, leave=True)
    batch_size = 100
    
    for i in iterator: 
        
        joint_loss = 0.0
        batch_index = disentangled_model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = disentangled_model.Z.Z  # a full sample returns latent Z across all N
        sample_batch = sample[batch_index]
        
        ### Getting the output of the two groups of GPs
        
        output_logP = disentangled_model.model_logP(sample_batch)
        output_qed = disentangled_model.model_qed(sample_batch)
        output_sas = disentangled_model.model_sas(sample_batch)
        
        ### Adding together the ELBO losses 
        
        joint_loss += -mll_logP(output_logP, Y_train[batch_index].T[0]).sum()
        joint_loss += -mll_qed(output_qed, Y_train[batch_index].T[1]).sum()
        joint_loss += -mll_sas(output_sas, Y_train[batch_index].T[2]).sum()
                
        loss_list.append(joint_loss.item())
        
        iterator.set_description('Loss: ' + str(float(np.round(joint_loss.item(),2))) + ", iter no: " + str(i))
        joint_loss.backward()
        disentangled_model.inducing_inputs.grad = disentangled_model.inducing_inputs.grad.to(device)
        optimizer.step()
        
    ## clean-up CUDA memory
    
    torch.cuda.empty_cache()
    gc.collect()
        
    ########## Analysis ##########
    
    Z_train = disentangled_model.Z.Z
    
    y_ls_qed = torch.topk(disentangled_model.model_qed.covar_module.base_kernel.lengthscale.cpu().detach(), k=2, largest=False)[1][0]
    y_ls_sas = torch.topk(disentangled_model.model_sas.covar_module.base_kernel.lengthscale.cpu().detach(), k=2, largest=False)[1][0]
    y_ls_logP= torch.topk(disentangled_model.model_logP.covar_module.base_kernel.lengthscale.cpu().detach(), k=2, largest=False)[1][0]
        
    ########## Plot 2d informative latents ##########
    
    title = 'Continuous Molecular representation [GPLVM Latent space]' + '\n' + 'Shaded by property values for the ZINC dataset'
    plot_disentangled_2d_latents(Z_train, Y_train, y_ls_logP, y_ls_qed, y_ls_sas, title)
    
    ######### PCA Analysis and projection #########
    
    pca = PCA(n_components=2)
    Z_pca = torch.Tensor(pca.fit_transform(Z_train.cpu().detach()))
    cols = torch.Tensor([0,1]).to(int)
    
    title = 'Continuous Molecular representation [GPLVM Latent space]' + '\n' + 'Shaded by property values for the ZINC dataset'
    plot_disentangled_2d_latents(Z_pca, Y_train, cols, cols, cols, title)

    ##########  Training recon ##########
    
    truncate_idx = 1000
    
    train_post_loP = likelihood_logP(disentangled_model.model_logP(Z_train[0:truncate_idx]))
    train_post_qed = likelihood_qed(disentangled_model.model_qed(Z_train[0:truncate_idx]))
    train_post_sas = likelihood_sas(disentangled_model.model_sas(Z_train[0:truncate_idx]))

    Y_train_pred_logP = train_post_loP.loc.cpu().detach()
    Y_train_pred_qed = train_post_qed.loc.cpu().detach()
    Y_train_pred_sas = train_post_sas.loc.cpu().detach()
    
    Y_train_pred_loc = torch.hstack((Y_train_pred_logP.T, Y_train_pred_qed.T, Y_train_pred_sas.T)).T
    
    diags_Y_logP = [m.cpu().detach().diag().sqrt() for m in train_post_loP.covariance_matrix][0]
    diags_Y_qed = [m.cpu().detach().diag().sqrt() for m in train_post_qed.covariance_matrix][0] 
    diags_Y_sas = [m.cpu().detach().diag().sqrt() for m in train_post_sas.covariance_matrix][0] 

    diags_Y = torch.hstack((diags_Y_logP.T, diags_Y_qed.T, diags_Y_sas.T)).T
    
    plot_y_label_comparison(Y_train_pred_loc.T, Y_train[0:truncate_idx], diags_Y)
    
    ############# Save / Load trained model ################
    
    fname = 'trained_models/gplvm_disent ' + '_n_train_' + str(len(Y_train)) + '_M_' + str(num_inducing) + '_Q_' + str(latent_dim) + '.pkl' 
    if os.path.isfile(fname):
        with open(fname, 'rb') as file:
            model_sd, likl_sd_logP, likl_sd_qed, likl_sd_sas = pkl.load(file)
            disentangled_model.load_state_dict(model_sd)
            likelihood_logP.load_state_dict(likl_sd_logP)
            likelihood_qed.load_state_dict(likl_sd_qed)
            likelihood_sas.load_state_dict(likl_sd_sas)
            
    with open(fname, 'wb') as file:
          pkl.dump((disentangled_model.cpu().state_dict(), likelihood_logP.cpu().state_dict(), 
                    likelihood_qed.cpu().state_dict(), likelihood_sas.cpu().state_dict()), file)
          
    ########## Testing ##########
    
    # Initialise test model at training params and reset the latent variables.
   
    TEST = True

    if TEST:
        
        test_model = disentangled_model.initialise_model_test(len(Y_test), latent_dim).to(device)
        
        test_loss, test_model, Z_test = predict_joint_latent(test_model, Y_test, likelihood_qed, likelihood_sas, likelihood_logP, lr=0.01, prior_z = None, steps = 4000)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            
            likelihood_qed.eval()
            likelihood_logP.eval()
            likelihood_sas.eval()
            test_model.eval()
            
            truncate_idx = 1000
            
            test_post_logP = likelihood_logP(disentangled_model.model_logP(Z_test.Z[0:truncate_idx]))
            test_post_qed = likelihood_qed(disentangled_model.model_qed(Z_test.Z[0:truncate_idx]))
            test_post_sas = likelihood_sas(disentangled_model.model_sas(Z_test.Z[0:truncate_idx]))

            Y_test_pred_logP = test_post_logP.loc.cpu().detach()
            Y_test_pred_qed = test_post_qed.loc.cpu().detach()
            Y_test_pred_sas = test_post_sas.loc.cpu().detach()
            
            Y_test_pred_loc = torch.hstack((Y_test_pred_logP.T, Y_test_pred_qed.T, Y_test_pred_sas.T)).T
            
            diags_Y_logP = [m.cpu().detach().diag().sqrt() for m in test_post_logP.covariance_matrix][0]
            diags_Y_qed = [m.cpu().detach().diag().sqrt() for m in test_post_qed.covariance_matrix][0] 
            diags_Y_sas = [m.cpu().detach().diag().sqrt() for m in test_post_sas.covariance_matrix][0] 

            diags_Y_test = torch.hstack((diags_Y_logP.T, diags_Y_qed.T, diags_Y_sas.T)).T
        
            Y_test_pred_sigma = diags_Y_test
    
    ######### Plot test latents and property prediction ##########
    
    plot_y_label_comparison(Y_test_pred_loc.T, Y_test[0:truncate_idx], diags_Y_test)
    
    title = 'Continuous Molecular representation [GPLVM Latent space]' + '\n' + 'Shaded by property values for the ZINC dataset'
    plot_disentangled_2d_latents(Z_test.Z, Y_test, y_ls_logP, y_ls_qed, y_ls_sas, title)
    
    ########## Test Reconstruction error & NLL ##########

    rmse_test = rmse_missing(Y_test[0:truncate_idx].cpu(), Y_test_pred_loc.T)
    
    nll_test_logP = nll(test_post_logP, Y_test[:,0][0:truncate_idx])
    nll_test_qed = nll(test_post_qed, Y_test[:,1][0:truncate_idx])
    nll_test_sas = nll(test_post_sas, Y_test[:,2][0:truncate_idx])

    print('Test Reconstruction error  = ' + str(rmse_test))
    