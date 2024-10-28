import os
import sys
import importlib
import numpy as np
import torch.nn as nn
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

from Classes.utils.model_utils import log_multivariate_gaussian_loss, MSE
from Classes.utils.utils import return_numpy, return_tensor, batch_cov_torch, batch_cov_numpy
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

        # y(x, theta)     +   (y_al(x, theta)   +   y_ep(x, theta))
        # y_al is a self.output_size x self.output_size matrix 
        # y_ep is a self.output_size x self.output_size matrix 
        self.output_size_y = int(self.output_size)
        self.output_size_y_al = int(self.output_size**2) # int((self.output_size**2 + self.output_size)/2)
        self.output_size_y_ep = int(self.output_size**2) # int((self.output_size**2 + self.output_size)/2)
        if self.number_outputs_per_output_feature == 1:
            self.real_output_size = int(self.output_size_y)
        if self.number_outputs_per_output_feature == 2:
            self.real_output_size = int(self.output_size_y + self.output_size_y_al)
        elif self.number_outputs_per_output_feature == 3:
            self.real_output_size = int(self.output_size_y + self.output_size_y_al + self.output_size_y_ep)
        
        if self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'SGLD':
            self.fcin = nn.Linear(self.input_size, self.neurons_per_layer[0])
            self.num_layers = len(self.neurons_per_layer)
            for layer in range(0, self.num_layers-1):
                name_layer = f'fc{layer}'
                neurons_in = self.neurons_per_layer[layer]
                neurons_exit = self.neurons_per_layer[layer+1]
                exec(f"self.{name_layer} = nn.Linear({neurons_in},{neurons_exit})")
            self.fcout = nn.Linear(self.neurons_per_layer[-1], self.real_output_size) 
        elif self.BNN_algorithm == 'SGLD':
            self.fcin = nn.Linear(self.input_size, self.neurons_per_layer[0])
            self.fcin.weight.data.uniform_(-0.01, -0.01)
            self.fcin.bias.data.uniform_(-0.01, -0.01)
            self.num_layers = len(self.neurons_per_layer)
            for layer in range(0, self.num_layers-1):
                name_layer = f'fc{layer}'
                neurons_in = self.neurons_per_layer[layer]
                neurons_exit = self.neurons_per_layer[layer+1]
                exec(f"self.{name_layer} = nn.Linear({neurons_in},{neurons_exit})")
                exec(f"self.{name_layer}.weight.data.uniform_(-0.01, -0.01)")
                exec(f"self.{name_layer}.bias.data.uniform_(-0.01, -0.01)")
            self.fcout = nn.Linear(self.neurons_per_layer[-1], self.real_output_size) 
            self.fcout.weight.data.uniform_(-0.01, -0.01)
            self.fcout.bias.data.uniform_(-0.01, -0.01)

        # self.activation = nn.ReLU(inplace = True)
        self.activation = torch.nn.functional.softplus

    
    def forward(self, x, epoch = None, batch_idx = None, output_additional_data = None, train = 0, use_cuda = None):
        
        batch_shape = x.shape[0]
        if use_cuda == None:
            use_cuda = self.use_cuda
            
        self.x = return_tensor(x, use_cuda = use_cuda).view(-1, self.input_size)
        
        # BNN algorithm
        if self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'SGLD':
        
            self.x = self.activation(self.fcin(self.x))
            for layer in range(0, self.num_layers-1):
                name_layer = f'fc{layer}'
                # print(f'\n {name_layer}')
                exec(f"self.x = self.activation(self.{name_layer}(self.x))")
            self.x = self.fcout(self.x)

            # Create a tensor of False for output_size_y
            mask_y = torch.zeros(self.output_size_y, dtype=torch.bool)
            if self.number_outputs_per_output_feature == 1:  
                mask = mask_y
            if self.number_outputs_per_output_feature == 2:  
                # Apply activation functions to uncertainties which has t be >= 0
                mask_al = torch.tensor([(i % (self.output_size_y_al - 1) == 0) for i in range(self.output_size_y_al)], dtype=torch.bool)
                mask = torch.cat([mask_y, mask_al])
            elif self.number_outputs_per_output_feature == 3:
                # Apply activation functions to uncertainties which has t be >= 0
                mask_al = torch.tensor([(i % (self.output_size_y_al - 1) == 0) for i in range(self.output_size_y_al)], dtype=torch.bool)
                mask_ep = torch.tensor([(i % (self.output_size_y_ep - 1) == 0) for i in range(self.output_size_y_ep)], dtype=torch.bool)
                mask = torch.cat([mask_y, mask_al, mask_ep])

            # Replicate the mask to match the batch size and feature size
            mask = mask.unsqueeze(0).repeat(batch_shape, 1)
            # Apply activation function where mask is True
            self.x[mask] = self.activation(self.x[mask])
            y = self.x

            if self.number_outputs_per_output_feature == 1:
               return_model_output = {'y': y[:, :self.output_size_y]} 
            elif self.number_outputs_per_output_feature == 2:
                return_model_output = {'y': y[:, :self.output_size_y], 'y_al':y[:, self.output_size_y:self.output_size_y+self.output_size_y_al]}
            elif self.number_outputs_per_output_feature == 3:
                return_model_output = {'y': y[:, :self.output_size_y], 
                                'y_al':y[:, self.output_size_y:self.output_size_y+self.output_size_y_al], 
                                'y_ep':y[:, self.output_size_y+self.output_size_y_al:self.output_size_y+self.output_size_y_al+self.output_size_y_ep]}

        return return_model_output

    def loss_function(self, x, return_model_output, output_additional_data = None):

        # BNN algorithm
        if self.BNN_algorithm_master == 'BDK':

            if self.BNN_algorithm == 'SGLD':
                y = return_model_output['y']
                y_al = return_model_output['y_al']
                t = output_additional_data['t']
                total_loss = log_multivariate_gaussian_loss(self.params, y, t, y_al)

                return_model_loss = {'total_loss': total_loss}
            
            elif self.BNN_algorithm == 'NN':

                y_S = return_model_output['y']
                y_S_al = return_model_output['y_al']

                # Do not preserve computational graph
                y_T = output_additional_data['y'].detach()
                y_T_al = output_additional_data['y_al'].detach()

                total_loss = log_multivariate_gaussian_loss(self.params, output_y = y_S, target_t = y_T, output_sigma2 = y_S_al, target_sigma2_T = y_T_al)

                return_model_loss = {'total_loss': total_loss}

        elif self.BNN_algorithm_master == 'BDKep':
    
            if self.BNN_algorithm == 'SGLD':
                y = return_model_output['y']
                y_al = return_model_output['y_al']
                t = output_additional_data['t']
                total_loss = log_multivariate_gaussian_loss(self.params, y, t, y_al)

                return_model_loss = {'total_loss': total_loss}
            
            elif self.BNN_algorithm == 'NN':

                # First phase of training (not enough samples)
                if not output_additional_data:
                    # Fictitious loss
                    total_loss = (return_model_output['y']).sum().detach().clone().requires_grad_()
                    return_model_loss = {'total_loss': total_loss} 
                    return return_model_loss

                y_S = return_model_output['y']
                y_S_al = return_model_output['y_al']
                y_S_ep = return_model_output['y_ep'].reshape(-1, self.output_size, self.output_size).permute(0, 2, 1)

                # Do not preserve computational graph
                y_T = [output['y'].detach() for output in output_additional_data]
                y_T_al = [output['y_al'].detach() for output in output_additional_data]

                # First part of loss
                y_T_mean = torch.mean(torch.stack(y_T), 0)
                y_T_al_mean = torch.mean(torch.stack(y_T_al), 0)
                total_loss_A = log_multivariate_gaussian_loss(self.params, output_y = y_S, target_t = y_T_mean, output_sigma2 = y_S_al, target_sigma2_T = y_T_al_mean)

                # Second part of loss
                y_T_cov = batch_cov_torch(torch.stack(y_T).permute(1, 0, 2))

                total_loss_B = 1/(2*self.var_xi_S) * MSE(y_S_ep, y_T_cov)
                
                total_loss = total_loss_A + total_loss_B

                return_model_loss = {'total_loss': total_loss, 'total_loss_A':total_loss_A, 'total_loss_B':total_loss_B}

        else:

            if self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'MCDropout' or self.BNN_algorithm == 'SGLD':

                t = output_additional_data['t']
                y = return_model_output['y']
                if self.number_outputs_per_output_feature == 1:
                    total_loss = log_multivariate_gaussian_loss(self.params, y, t)
                if self.number_outputs_per_output_feature >= 2:
                    y_al = return_model_output['y_al'] 
                    total_loss = log_multivariate_gaussian_loss(self.params, y, t, y_al)

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

        # Predict y and y_al 
        if self.number_outputs_per_output_feature >= 2:
            y_al = np.array(return_numpy(return_model_output['y_al'])) # N_samples_param x batch_size x output_size**2
        # Predict y, y_al and y_ep
        if self.number_outputs_per_output_feature == 3:
            y_ep = np.array(return_numpy(return_model_output['y_ep']))

        # Not BNN
        if self.BNN_algorithm == 'NN':
            # Mean and std of the t prediction
            y_mean = y.reshape(-1, output_size)

            # Predict only y 
            if self.number_outputs_per_output_feature == 1:
                epistemic_cov = epsilon*np.repeat(np.eye(output_size)[np.newaxis, :, :], batch_size, axis=0)
                aleatoric_cov = epsilon*np.repeat(np.eye(output_size)[np.newaxis, :, :], batch_size, axis=0)
            # Predict only y and y_al
            if self.number_outputs_per_output_feature == 2:
                epistemic_cov = epsilon*np.repeat(np.eye(output_size)[np.newaxis, :, :], batch_size, axis=0)

                aleatoric_cov = y_al.reshape(-1, output_size, output_size).transpose(0, 2, 1)
                aleatoric_cov = np.minimum(aleatoric_cov, 10e3)
                aleatoric_cov = np.maximum(aleatoric_cov, -10e3)
            # Predict y, y_al and y_ep
            elif self.number_outputs_per_output_feature == 3:
                epistemic_cov = y_ep.reshape(-1, output_size, output_size).transpose(0, 2, 1)
                epistemic_cov = np.minimum(epistemic_cov, 10e3)
                epistemic_cov = np.maximum(epistemic_cov, -10e3)

                aleatoric_cov = y_al.reshape(-1, output_size, output_size).transpose(0, 2, 1)
                aleatoric_cov = np.minimum(aleatoric_cov, 10e3)
                aleatoric_cov = np.maximum(aleatoric_cov, -10e3)


        # BNN -> compute predictive mean and var
        else:
        
            # Mean and cov of the t prediction
            y_mean = y.mean(axis = 0).reshape(-1, output_size)
            epistemic_cov = batch_cov_numpy(y.transpose(1, 0, 2))
            epistemic_cov = np.minimum(epistemic_cov, 10e3)
            epistemic_cov = np.maximum(epistemic_cov, -10e3)

            # Mean of predicted aleatoric_std
            aleatoric_cov = y_al.mean(axis = 0).reshape(-1, output_size, output_size).transpose(0, 2, 1)
            aleatoric_cov = np.minimum(aleatoric_cov, 10e3)
            aleatoric_cov = np.maximum(aleatoric_cov, -10e3)

        # Total std of uncertainty
        total_unc_cov = aleatoric_cov + epistemic_cov + epsilon
        if self.number_outputs_per_output_feature == 1:
            total_unc_std = np.sqrt(total_unc_cov)
        else:
            total_unc_std =  np.sqrt(np.einsum('...ii->...', total_unc_cov))

        if output_additional_data is not None:

            t = return_numpy(output_additional_data['t']).reshape(batch_size, -1)[:,:output_size]

            #### METRICS
            # Accuracy metrics
            mae = mean_absolute_error(t, y_mean)
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
        
        num_macs = profile_macs(self, torch.zeros(1, self.input_size)) 
        print("#MACs:", num_macs)
        print("#FLOPs:", num_macs*2)

