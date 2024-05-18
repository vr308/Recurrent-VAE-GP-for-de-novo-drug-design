#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical GPLVM with a single GP decoder with shared hyperparameters

@author: vr308  

"""
import torch
import numpy as np
from tqdm import trange
from prettytable import PrettyTable
from gpytorch.models import ApproximateGP
from gpytorch.mlls import VariationalELBO
from gpytorch.means import ConstantMean
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from models.latent_variable import PointLatentVariable, MAPLatentVariable
from gpytorch.priors import NormalPrior
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal

class BaseGPLVM(ApproximateGP):
    
     def __init__(self, n: int, data_dim: int, latent_dim: int, n_inducing: int, latent_config='point'):
         
        """The GPLVM model class for unsupervised learning. The current class supports
        
        (a) Point estimates for latent Z 
        (b) MAP Inference for Z where prior_Z is necessary
        (c) Gaussian variational distribution q(Z) when prior_z is not None 

        Z  (LatentVariable) is initialised in the constructor: 
            An instance of a sub-class of the LatentVariable class.
                                    One of,
                                    PointLatentVariable / 
                                    MAPLatentVariable / 
                                    GaussianLatentVariable to
                                    facilitate inference with (a), (b) or (c) respectively.
       
        """
        self.n = n
        self.batch_shape = torch.Size([data_dim])
        self.inducing_inputs = torch.randn(n_inducing, latent_dim)

        Z_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
        Z_init = torch.nn.Parameter(torch.zeros(n, latent_dim))
          
        # Latent variable configuration
        
        if latent_config == 'map':
            
            prior_z = NormalPrior(Z_prior_mean, torch.ones_like(Z_prior_mean))
            Z = MAPLatentVariable(n, latent_dim, Z_init, prior_z)
        
        elif latent_config == 'point':
        
            Z = PointLatentVariable(Z_init)
            
        # Sparse Variational Formulation

        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape) 
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)
        
        super(BaseGPLVM, self).__init__(q_f)
        
        self.Z = Z
        
        # Kernel 

        self.mean_module = ConstantMean(batch_shape=self.batch_shape)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))
                   
     def forward(self, Z):
                 
        mean_z = self.mean_module(Z)
        covar_z = self.covar_module(Z)
        dist = MultivariateNormal(mean_z, covar_z)
        return dist
    
     def _get_batch_idx(self, batch_size: int):
           
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)
    
     def get_trainable_param_names(self):
       
       ''' Prints a list of parameters (model + variational) which will be 
       learnt in the process of optimising the objective '''
       
       table = PrettyTable(["Modules", "Parameters"])
       total_params = 0
       for name, parameter in self.named_parameters():
           if not parameter.requires_grad: continue
           param = parameter.numel()
           table.add_row([name, param])
           total_params+=param
       print(table)
       print(f"Total Trainable Params: {total_params}")
       
     def initialise_model_test(self, n_test: int, latent_dim: int):
    
       '''This method needs to be called before test inference (learning 
        Z_test corresponding X_test)'''
       
       # Initialise test models 
       self.n = n_test
       #self.model.n = n_test
       Z_init_test = torch.nn.Parameter(torch.randn(self.n, latent_dim))
       self.Z.reset(Z_init_test)
       
       return self

def predict_joint_latent(model, X_test, likelihood, lr=0.001, prior_z = None, steps = 2000, batch_size = 100):
   
    # Initialise a new test optimizer with just the test model latents
    
    test_optimizer = torch.optim.Adam(model.Z.parameters(), lr=lr)
    
    mll = VariationalELBO(likelihood, model, num_data=model.n)
    
    print('---------------Learning variational parameters for test ------------------')
    
    for name, param in model.Z.named_parameters():
        print(name)
        
    loss_list = []
    iterator = trange(steps, leave=True)
    batch_size = batch_size
    
    for i in iterator: 
        
           loss = 0.0
           batch_index = model._get_batch_idx(batch_size)
           test_optimizer.zero_grad()
           sample_batch = model.Z.Z[batch_index] # a full sample returns latent Z across all N
           
           output = model(sample_batch)
           loss = -mll(output, X_test[batch_index].T).sum() 
               
           loss_list.append(loss.item())
           iterator.set_description('Loss: ' + str(float(np.round(loss.item(),2))) + ", iter no: " + str(i))
           loss.backward()
           test_optimizer.step()
        
    return loss_list,model, model.Z