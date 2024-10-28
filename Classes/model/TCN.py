import os
import sys
import importlib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.metrics import (
    mean_absolute_error,
)
from torchprofile import profile_macs

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

from Classes.utils.model_utils import TemporalConvNet
from Classes.utils.utils import return_numpy, return_tensor
import Classes.utils.uncertainty_toolbox as uct


class MODEL(nn.Module):

    DEFAULTS = {} 
    def __init__(self, params):
        super(MODEL, self).__init__()

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(MODEL.DEFAULTS, **params_dict)

        # BNN algorithm instance
        self.BNN_algorithm_instance = getattr(importlib.import_module(self.package_BNN_algorithm), self.BNN_algorithm)(params)
        
        if self.BNN_algorithm == 'NN':

            self.tcn = TemporalConvNet(self.input_size[-1], self.num_channels, self.kernel_size, dropout=self.dropout)

            self.fcin = nn.Linear(self.num_channels[-1], self.neurons_per_layer[0])
            self.num_layers = len(self.neurons_per_layer)
            for layer in range(0, self.num_layers-1):
                name_layer = f'fc{layer}'
                neurons_in = self.neurons_per_layer[layer]
                neurons_exit = self.neurons_per_layer[layer+1]
                exec(f"self.{name_layer} = nn.Linear({neurons_in},{neurons_exit})")
            self.fcout = nn.Linear(self.neurons_per_layer[-1], self.output_size) 

            self.dropout = nn.Dropout(self.dropout)

    
    def forward(self, x, epoch = None, batch_idx = None, output_additional_data = None, train = 0, use_cuda = None):
        
        batch_shape = x.shape[0]
        if use_cuda == None:
            use_cuda = self.use_cuda
            
        self.x = return_tensor(x, use_cuda = use_cuda).view(batch_shape, -1, self.input_size[1])
        num_BS_per_pos = output_additional_data['num_BS_per_pos'].long()
        self.x = self.x[:,:num_BS_per_pos, :]
        
        # BNN algorithm
        if self.BNN_algorithm == 'NN':
        
            # x needs to have dimension (Batch, Channels, Seq length) in order to be passed into CNN
            self.x = self.tcn(self.x.transpose(1, 2)).transpose(1, 2)
            self.x = self.dropout(self.x)
            self.x = self.x[:, -1, :]  # Take the output from the last time step

            self.x = F.gelu(self.fcin(self.x))
            for layer in range(0, self.num_layers-1):
                name_layer = f'fc{layer}'
                exec(f"self.x = self.{name_layer}(self.x)")
                # exec(f"self.x = self.activation(self.{name_layer}(self.x))")
            output= self.fcout(self.x)

            return_model_output = {'y': output} 

        return return_model_output

    def loss_function(self, x, return_model_output, output_additional_data = None):

        if self.BNN_algorithm == 'NN':

            t = output_additional_data['UExyz_wrtcell']

            # Take the output from the last time step
            num_BS_per_pos = output_additional_data['num_BS_per_pos'].long()
            # t = t.squeeze()[:,-1, :]
            t = t.squeeze()[num_BS_per_pos-1, :]

            y = return_model_output['y']
            
            total_loss = torch.mean(torch.sqrt(torch.sum((t - y) ** 2, dim=1)))
            return_model_loss = {'total_loss': total_loss}

        return return_model_loss
    
    def evaluate(self, x, epoch = None, batch_idx = None, output_additional_data = None, train = 0, return_output = 0, return_model_output = None):
    
        batch_size = x.shape[0]
        output_size = self.output_size
        epsilon = 1e-6

        # Adjust input features for representation
        x = return_numpy(x) * self.params.x_train_std + self.params.x_train_mean

        # Model prediction
        y = np.array(return_numpy(return_model_output['y']))      # N_samples_param x batch_size x output_size

        # Not BNN
        if self.BNN_algorithm == 'NN':
            # Mean and std of the t prediction
            y_mean = y.reshape(-1, output_size)

            # Predict only y 
            if self.number_outputs_per_output_feature == 1:
                epistemic_cov = epsilon*np.repeat(np.eye(output_size)[np.newaxis, :, :], batch_size, axis=0)
                aleatoric_cov = epsilon*np.repeat(np.eye(output_size)[np.newaxis, :, :], batch_size, axis=0)

        # Total std of uncertainty
        total_unc_cov = aleatoric_cov + epistemic_cov + epsilon
        if self.number_outputs_per_output_feature == 1:
            total_unc_std = np.sqrt(total_unc_cov)
        else:
            total_unc_std =  np.sqrt(np.einsum('...ii->...', total_unc_cov))

        if output_additional_data is not None:

            # Take the output from the last time step
            num_BS_per_pos = output_additional_data['num_BS_per_pos'].long()
            try:
                t = return_numpy(output_additional_data['UExyz_wrtcell']).squeeze()[num_BS_per_pos-1, :]  
            except:
                t = return_numpy(output_additional_data['UExyz_wrtcell'])[:,num_BS_per_pos-1, :]

            #### METRICS
            # Accuracy metrics
            mae = mean_absolute_error(t.reshape(-1, output_size), y_mean.reshape(-1, output_size))
            # rmse = np.sqrt(mean_squared_error(t, y_mean))

            # Uncertainty metrics
            if self.number_outputs_per_output_feature > 1:
                mace = uct.mean_absolute_calibration_error_multi_output(y_mean, total_unc_cov, t, vectorized = True)
                # rmsce = uct.root_mean_squared_calibration_error_multi_output(y_mean, total_unc_cov, t, vectorized = True)
                # ma = uct.miscalibration_area_multi_output(y_mean, total_unc_cov, t, vectorized = True)
                # nll = uct.nll_gaussian_multi_output(y_mean, total_unc_cov, t)

        #### OUTPUT
        # List of variables to check and add if they exist
        variables_to_check = ['x', 't', 'y', 'y_al', 'y_ep', 'y_mean', 'epistemic_cov', 'aleatoric_cov', 'total_unc_cov', 'total_unc_std', 'mae', 'rmse', 'mace', 'rmsce', 'ma', 'nll']

        # Include output of the model
        if return_output:
            return_model_evaluate = {}
            for var_name in variables_to_check:
                if var_name in locals().keys():
                    return_model_evaluate={**return_model_evaluate, var_name: locals()[var_name]}
        # Do not include not_to_return
        else:
            not_to_return = ['x', 't', 'y', 'y_al', 'y_ep', 'y_mean', 'epistemic_cov', 'aleatoric_cov', 'total_unc_cov', 'total_unc_std']
            return_model_evaluate = {}
            for var_name in variables_to_check:
                if var_name in locals().keys() and var_name not in not_to_return:
                    return_model_evaluate={**return_model_evaluate, var_name: locals()[var_name]}

        return return_model_evaluate


    def print_MACs_FLOPs(self):

        num_macs = profile_macs(self, torch.zeros(1, 1, self.input_size[0], self.input_size[1])) 
        print("#MACs:", num_macs)
        print("#FLOPs:", num_macs*2)
