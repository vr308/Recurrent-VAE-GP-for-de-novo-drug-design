#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Config for project

"""

from pathlib import Path
import torch 
import matplotlib
import gpytorch 

matplotlib.rcParams.update({'font.size': 10})

TORCH_VERSION = torch.__version__
GPYTORCH_VERSION = gpytorch.__version__

AVAILABLE_GPU = torch.cuda.device_count()
GPU_ACTIVE = bool(AVAILABLE_GPU)
EPSILON = 1e-5
BASE_SEED = 42
RANDOM_SEEDS = [34,52,61,70,97]

BASE_PATH = Path(__file__).parent.parent
RESULTS_DIR = BASE_PATH / "results"
DATASET_DIR = BASE_PATH.parent / "datasets/data"
LOG_DIR = BASE_PATH / "logs"
TRAINED_MODELS = BASE_PATH / "trained_models"









