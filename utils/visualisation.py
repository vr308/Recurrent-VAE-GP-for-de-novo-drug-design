#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation code for reconstruction plots, label reproduction and test predictions

"""

import matplotlib.pylab as plt
from sklearn.metrics import jaccard_score

    
def plot_ECFP_matrices(X_true, X_recon):
    
    plt.figure(figsize=(7,10))
    plt.subplot(211)
    plt.imshow(X_true.cpu().detach())
    plt.axis('off')
    plt.xlabel('bits')
    plt.ylabel('num molecules')
    plt.title('Ground truth')
    plt.subplot(212)
    plt.imshow(X_recon.cpu().detach())
    plt.axis('off')
    plt.xlabel('bits')
    plt.ylabel('num molecules')
    plt.title('Reconstruction')
    cbar = plt.colorbar()
    cbar.set_ticks(ticks=[0,1])
    jaccard = jaccard_score(X_true, X_recon, average='weighted')
    plt.title('ECFP Generative Reconstructions [J-score = ' + str(jaccard) + ' ]')

def plot_y_label_comparison(Y_test_recon, Y_test, Y_test_pred_sigma):
    
    plt.figure(figsize=(20,8))
    labels = ['logP', 'QED', 'SAS']
    Y_plot = Y_test.cpu().detach().numpy()
    
    for i in [0,1,2]:
        
        plt.subplot(1,3,i+1)
        plt.scatter(Y_test_recon[:,i], Y_plot[:,i], s=3,c=Y_plot[:,i],cmap='jet')
        plt.errorbar(Y_test_recon[:,i], Y_plot[:,i], yerr=2*Y_test_pred_sigma[i], fmt='None', color='k', alpha=0.1, errorevery=1)
        plt.colorbar()
        plt.xlabel(r'Predicted ')
        plt.ylabel(r'Measured ')
        plt.title(labels[i])
        xpoints = ypoints = plt.xlim()
        plt.plot(xpoints, ypoints, linestyle='--', color='k',alpha=0.7, scalex=False, scaley=False)
    plt.tight_layout()
    plt.suptitle('Predicting properties with the GPLVM [ZINC dataset]' + '\n' + '\n' + r'2$\sigma$ prediction intervals')
    

def plot_2d_latents(Z_train, Y_train, col_id1, col_id2, title):

    Z_plot = Z_train.cpu().detach().numpy()
    Y_plot = Y_train.cpu().detach().numpy()
       
    plt.figure(figsize=(20,8))
    plt.suptitle(title, fontsize='small')
    
    plt.subplot(131)
    plt.scatter(Z_plot[:,col_id1], Z_plot[:,col_id2],s=3, c=Y_plot[:,0])
    plt.colorbar()
    plt.title('logP')
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')

    plt.subplot(132)
    plt.scatter(Z_plot[:,col_id1], Z_plot[:,col_id2], s=3, c=Y_plot[:,1])
    plt.colorbar()
    plt.title('QED')
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')


    plt.subplot(133)
    plt.scatter(Z_plot[:,col_id1], Z_plot[:,col_id2], s=3, c=Y_plot[:,2])
    plt.colorbar()
    plt.title('SAS')
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')


def plot_disentangled_2d_latents(Z_train, Y_train, y_ls_logP, y_ls_qed, y_ls_sas, title):

    Z_plot = Z_train.cpu().detach().numpy()
    Y_plot = Y_train.cpu().detach().numpy()
       
    plt.figure(figsize=(20,8))
    plt.suptitle(title, fontsize='small')
    
    col_id1 = y_ls_logP[0].cpu().item()
    col_id2 = y_ls_logP[1].cpu().item()
    
    plt.subplot(131)
    plt.scatter(Z_plot[:,col_id1], Z_plot[:,col_id2],s=3, c=Y_plot[:,0])
    plt.colorbar()
    plt.title('logP')
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')
    
    col_id1 = y_ls_qed[0].cpu().item()
    col_id2 = y_ls_qed[1].cpu().item()
    
    plt.subplot(132)
    plt.scatter(Z_plot[:,col_id1], Z_plot[:,col_id2], s=3, c=Y_plot[:,1])
    plt.colorbar()
    plt.title('QED')
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')

    col_id1 = y_ls_sas[0].cpu().item()
    col_id2 = y_ls_sas[1].cpu().item()
 
    plt.subplot(133)
    plt.scatter(Z_plot[:,col_id1], Z_plot[:,col_id2], s=3, c=Y_plot[:,2])
    plt.colorbar()
    plt.title('SAS')
    plt.xlabel(r'$z_{1}$')
    plt.ylabel(r'$z_{2}$')
