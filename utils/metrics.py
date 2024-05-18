#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Negative test log-likelihood and sq. root mean squared error

@author: vr308

"""

import torch
import numpy as np
from scipy.stats import stats 
  
def nll(Y_test_pred, Y_test):
    
      lpd = Y_test_pred.log_prob(Y_test) 
      # return the average
      avg_lpd_rescaled = lpd.detach().mean()
      return -avg_lpd_rescaled
  
def nlpd_marginal(Y_test_pred, Y_test, Y_std):
    
    test_lpds_per_point = []
    Y_test = Y_test.cpu().numpy()
    means = Y_test_pred.loc.detach().cpu().numpy()
    var = Y_test_pred.covariance_matrix.diag().detach().cpu().numpy()
    for i in np.arange(len(Y_test)):
        pp_lpd = stats.norm.logpdf(Y_test[i], loc = means[i], scale = np.sqrt(var[i])) - np.log(Y_std)
        test_lpds_per_point.append(pp_lpd)
    return -np.mean(test_lpds_per_point)

def rmse_missing(Y_test, Y_recon):
    
    return torch.sqrt(torch.mean(torch.Tensor([np.nanmean(np.square(Y_test - Y_recon))])))

def jaccard(list1, list2):
    
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union
