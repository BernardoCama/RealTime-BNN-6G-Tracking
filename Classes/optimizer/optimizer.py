import sys
import os
import torch
import math

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(cwd))

from Classes.BNN_algorithm.SGLD import SGLD_optimizer

class OPTIMIZER(object):
    
    DEFAULTS = {}   
    def __init__(self, parameters, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(OPTIMIZER.DEFAULTS, **params_dict)

        # Check regularization 
        has_weight_decay = hasattr(self, 'weight_decay')
        has_sigma_prior = hasattr(self, 'sigma_prior')
        if has_weight_decay and has_sigma_prior:
            # Give priority to sigma
            if self.sigma_prior != 0:
                self.weight_decay = 1 / (self.sigma_prior ** 2)
            elif self.weight_decay!= 0:
                self.sigma_prior = math.sqrt(1/self.weight_decay) 
        elif not has_weight_decay and not has_sigma_prior:
            self.weight_decay = 0
            self.sigma_prior = 0
        elif has_weight_decay and not has_sigma_prior:
            if self.weight_decay!= 0:
                self.sigma_prior = math.sqrt(1/self.weight_decay) 
            else:
                self.sigma_prior = 0
        elif not has_weight_decay and has_sigma_prior:
            if self.sigma_prior!= 0:
                self.weight_decay = 1 / (self.sigma_prior ** 2)
            else:
                self.weight_decay = 0

        if self.optimizer_name == 'SGD':
            self.optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adam': 
            self.optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'SGLD':
            self.optimizer = SGLD_optimizer(parameters, lr=self.lr, weight_decay=self.weight_decay, sigma_prior = self.sigma_prior, addnoise = 0)
        else:
            pass



