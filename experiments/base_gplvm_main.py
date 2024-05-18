#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for Canonical GPLVM with independent GPs modelling molecular properties With shared hyperparameters.

TODO: 
    
    Ablation with MAP inference / Bayesian latent variables.
    
"""
import torch
import gpytorch
import os 
import gc
import yaml
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from celluloid import Camera
from gpytorch.mlls import VariationalELBO
from models.likelihood import GaussianLikelihood
from utils.load_data import load_full_dataset
from models.base_gplvm import BaseGPLVM, predict_joint_latent
from utils.visualisation import plot_2d_latents, plot_y_label_comparison
from utils.metrics import rmse_missing
from utils.config import BASE_SEED

save_model = True
missing = False

def get_Y_missing(Y, N_train, percent):
    
    idx = np.random.binomial(n=1, p=percent, size=(N_train, Y.shape[1])).astype(bool)
    train_idx = np.random.randint(0, Y.shape[0], N_train)
    Y_missing = Y.clone()[train_idx]
    Y_missing[idx] = np.nan
    return Y_missing, train_idx

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
    
    if missing:
        Y_train, train_idx = get_Y_missing(Y_train, 5000, .30)
            
    Y_train = Y_train.to(device)
    Y_test = Y_test.to(device)
    
    ## Extract model settings
    
    latent_dim = settings['gplvm']['latent_dim']
    num_inducing = settings['gplvm']['num_inducing']
    
    N = len(Y_train)
    data_dim = Y_train.shape[1]
      
    # Base Model
    
    base_model = BaseGPLVM(N, data_dim, latent_dim, num_inducing, latent_config='point').to(device)
    likelihood = GaussianLikelihood(batch_shape = base_model.batch_shape).to(device)
    mll = VariationalELBO(likelihood, base_model, num_data=len(Y_train)).to(device)
    
    optimizer = torch.optim.Adam([
        dict(params=base_model.parameters(), lr=0.01),
        dict(params=likelihood.parameters(), lr=0.01)
    ])

    base_model.get_trainable_param_names()
    
    ############## Training loop - optimises the objective wrt kernel hypers, ######
    ################  variational params and inducing inputs using the optimizer provided ########
    
    torch.use_deterministic_algorithms(False)
    
    loss_list = []
    iterator = trange(10000, leave=True)
    batch_size = 100
    
    Z_seq = []
    
    for i in iterator: 
        
        joint_loss = 0.0
        batch_index = base_model._get_batch_idx(batch_size)
        optimizer.zero_grad()
        sample = base_model.Z.Z  # a full sample returns latent Z across all N
        sample_batch = sample[batch_index]
        
        output = base_model(sample_batch)        
        joint_loss = -mll(output, Y_train[batch_index].T).sum()
     
        loss_list.append(joint_loss.item())

        iterator.set_description('Loss: ' + str(float(np.round(joint_loss.item(),2))) + ", iter no: " + str(i))
    
        joint_loss.backward()
        #base_model.inducing_inputs.grad = base_model.inducing_inputs.grad.to(device)
        optimizer.step()
        
        if i%50 == 0:
            Z_seq.append(base_model.Z.Z.detach().cpu().numpy())
   
            
    Z_train = base_model.Z.Z
    
    ## clean-up CUDA memory
    torch.cuda.empty_cache()
    gc.collect()

    ########## Plot 2d informative latents ##########
    
    y_ls = torch.topk(base_model.covar_module.base_kernel.lengthscale, k=2, largest=False)[1][0]

    col_id1 = y_ls[0].cpu().item()
    col_id2 = y_ls[1].cpu().item()
    
    title = 'Continuous Molecular representation [GPLVM Latent space]' + '\n' + 'Shaded by property values for the ZINC dataset'
    plot_2d_latents(Z_train, Y_train, col_id1, col_id2, title)
    
    ########## Plotting animation of Z_seq ###########
    
    Y_plot = Y_train.cpu().detach().numpy()
    
    camera = Camera(plt.figure(figsize=(14,14)))
    
    for i in np.arange():
        
         Z_plot = Z_seq[i]          
         plt.scatter(*Z_plot[:,(col_id1, col_id2)].T, s=10, c=Y_plot[:,1])
         camera.snap()
    plt.colorbar()
    anim = camera.animate(blit=True)
    plt.title('Evolution of latent space [Learning QED]')
    anim.save('QED.mp4')

    ########## Save / Load trained model ##########
    
    fname = 'trained_models/gplvm_base ' + '_n_train_' + str(len(Y_train)) + '_M_' + str(num_inducing) + '_Q_' + str(latent_dim) + '.pkl' 
    
    if os.path.isfile(fname):
          with open(fname, 'rb') as file:
              model_sd, likl_sd = pkl.load(file)
              base_model.load_state_dict(model_sd)
              likelihood.load_state_dict(likl_sd)

    with open(fname, 'wb') as file:
        pkl.dump((base_model.cpu().state_dict(), likelihood.cpu().state_dict()), file)
            
    ########## Testing ##########
    
    # Initialise test model at training params and reset the latent variables.
   
    TEST = True

    if TEST:
        
        test_model = base_model.initialise_model_test(len(Y_test), latent_dim).to(device)
        
        if missing:
        
            Y_test, test_idx = get_Y_missing(Y_test, 1000, .30)
            
        test_loss, test_model, Z_test = predict_joint_latent(test_model, Y_test, likelihood, lr=0.01, prior_z = None, steps = 4000)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            
            truncate_idx = 5000
            
            test_model.eval()
            likelihood.eval()
            
            pred_posterior = likelihood(base_model(Z_test.Z[0:truncate_idx]))
            
            Y_test_recon = pred_posterior.loc.cpu().detach()
            Y_test_pred_covar = pred_posterior.covariance_matrix.cpu().detach()[0:truncate_idx]
            diags_Y_test = torch.Tensor([np.array(m.diag().sqrt()) for m in Y_test_pred_covar])

    ######### Plot test latents and property prediction ##########
    
    plot_y_label_comparison(Y_test_recon.T, Y_test[0:truncate_idx], diags_Y_test)
    plot_2d_latents(Z_test.Z, Y_test, col_id1, col_id2, title)

    ########## Test Reconstruction error ##########
    
    #rmse_test = rmse_missing(test_data['y1'][test_idx][0:200], Y_test_recon.T)    
    rmse_test = rmse_missing(Y_test[0:truncate_idx].cpu(), Y_test_recon.T)
    print('Test Reconstruction error = ' + str(rmse_test))
    