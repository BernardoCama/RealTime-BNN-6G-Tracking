import os
import sys
import importlib
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
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

from Classes.utils.model_utils import log_gaussian_loss, MSE
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

        # network with two hidden and one output layer
        # 1 output is the y(x, theta), 2 output is the log(var aleatoric uncert)
        if self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'MCDropout':
            self.layer1 = nn.Linear(self.input_size, self.neurons_per_layer[0])
            self.layer2 = nn.Linear(self.neurons_per_layer[0], self.number_outputs_per_output_feature*self.output_size)    
        elif self.BNN_algorithm == 'BBP':
            self.layer1 = self.BNN_algorithm_instance.BayesLinear(self.input_size, self.neurons_per_layer[0], self.BNN_algorithm_instance.Phi_pdf)
            self.layer2 = self.BNN_algorithm_instance.BayesLinear(self.neurons_per_layer[0], self.number_outputs_per_output_feature*self.output_size, self.BNN_algorithm_instance.Phi_pdf)
        elif self.BNN_algorithm == 'SGLD':
            self.layer1 = nn.Linear(self.input_size, self.neurons_per_layer[0]) 
            self.layer1.weight.data.uniform_(-0.01, -0.01)
            self.layer1.bias.data.uniform_(-0.01, -0.01)
            self.layer2 = nn.Linear(self.neurons_per_layer[0], self.number_outputs_per_output_feature*self.output_size)  
            self.layer2.weight.data.uniform_(-0.01, -0.01)
            self.layer2.bias.data.uniform_(-0.01, -0.01)
        else:
            self.layer1 = nn.Linear(self.input_size, self.neurons_per_layer[0])
            self.layer2 = nn.Linear(self.neurons_per_layer[0], self.number_outputs_per_output_feature*self.output_size)   
        
        self.activation = nn.ReLU(inplace = True)

    
    def forward(self, x, epoch = None, batch_idx = None, output_additional_data = None, train = 0, use_cuda = None):
        
        if use_cuda == None:
            use_cuda = self.use_cuda
            
        x = return_tensor(x).view(-1, self.input_size)
        
        # BNN algorithm
        if self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'SGLD':
        
            x = self.layer1(x)
            x = self.activation(x)
            y = self.layer2(x)

            if self.number_outputs_per_output_feature == 2:
                return_model_output = {'y': y[:, :1], 'y_al':y[:, 1:]}
            elif self.number_outputs_per_output_feature == 3:
                return_model_output = {'y': y[:, :1], 'y_al':y[:, 1:2], 'y_ep':y[:, 2:]}

        elif self.BNN_algorithm == 'MCDropout':

            x = self.layer1(x)
            x = self.activation(x)
            x = self.BNN_algorithm_instance.dropout_layer(x, p=self.drop_prob, training=True)
            y = self.layer2(x)

            return_model_output = {'y': y[:, :1], 'y_al':y[:, 1:]}
        
        elif self.BNN_algorithm == 'BBP':

            KL_loss_total = 0
            x, KL_loss = self.layer1(x)
            KL_loss_total = KL_loss_total + KL_loss
            x = self.activation(x)      
            y, KL_loss = self.layer2(x)
            KL_loss_total = KL_loss_total + KL_loss

            return_model_output = {'y': y[:, :1], 'y_al':y[:, 1:], 'KL_loss_total':KL_loss_total}

        return return_model_output

    def loss_function(self, x, return_model_output, output_additional_data = None):

        # BNN algorithm
        if self.BNN_algorithm_master == 'BDK':

            if self.BNN_algorithm == 'SGLD':
                y = return_model_output['y']
                y_al = return_model_output['y_al']
                t = output_additional_data['t']
                total_loss = log_gaussian_loss(y, t, y_al)

                return_model_loss = {'total_loss': total_loss}
            
            elif self.BNN_algorithm == 'NN':

                y_S = return_model_output['y']
                y_S_al = return_model_output['y_al']

                # Do not preserve computational graph
                y_T = output_additional_data['y'].detach()
                y_T_al = output_additional_data['y_al'].detach()

                total_loss = log_gaussian_loss(output = y_S, target = y_T, log_sigma2 = y_S_al, log_sigma2_T = y_T_al)

                return_model_loss = {'total_loss': total_loss}

        elif self.BNN_algorithm_master == 'BDKep':
    
            if self.BNN_algorithm == 'SGLD':
                y = return_model_output['y']
                y_al = return_model_output['y_al']
                t = output_additional_data['t']
                total_loss = log_gaussian_loss(y, t, y_al)

                return_model_loss = {'total_loss': total_loss}
            
            elif self.BNN_algorithm == 'NN':

                # First phase of training (not enough samples)
                if not output_additional_data:
                    # Fictitious loss
                    total_loss = (return_model_output['y']).sum().detach().clone().requires_grad_()
                    return_model_loss = {'total_loss': total_loss} 
                    return return_model_loss

                y_S = return_model_output['y'].reshape([-1, self.output_size])
                y_S_al = return_model_output['y_al'].reshape([-1, self.output_size])
                y_S_ep = return_model_output['y_ep'].reshape([-1, self.output_size])

                # Do not preserve computational graph
                y_T = [output['y'].detach() for output in output_additional_data]
                y_T_al = [output['y_al'].detach() for output in output_additional_data]

                # First part of loss
                y_T_mean = torch.mean(torch.stack(y_T), 0)
                y_T_al_mean = torch.mean(torch.stack(y_T_al), 0)
                total_loss_A = log_gaussian_loss(output = y_S, target = y_T_mean, log_sigma2 = y_S_al, log_sigma2_T = y_T_al_mean)

                # Second part of loss
                log_y_T_var = torch.log(torch.var(torch.stack(y_T), 0))
                total_loss_B = 1/(2*self.var_xi_S) * MSE(y_S_ep, log_y_T_var)
                
                total_loss = total_loss_A +  total_loss_B

                return_model_loss = {'total_loss': total_loss}

        else:

            if self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'MCDropout' or self.BNN_algorithm == 'SGLD':
                y = return_model_output['y']
                y_al = return_model_output['y_al']
                t = output_additional_data['t']
                total_loss = log_gaussian_loss(y, t, y_al)

                return_model_loss = {'total_loss': total_loss}

            elif self.BNN_algorithm == 'BBP':
                y = return_model_output['y']
                y_al = return_model_output['y_al']
                KL_loss_total = return_model_output['KL_loss_total']
                t = output_additional_data['t']

                KL_loss = KL_loss_total[-1]/self.num_train_batches
                total_loss = 0
                for i in range(self.N_samples_fitting):
                    total_loss = total_loss + log_gaussian_loss(y[i], t, y_al[i])
                total_loss = (total_loss+KL_loss)/(self.N_samples_fitting*self.batch_size)

                # print('total_loss: {}, KL_loss_total: {}'.format(total_loss, KL_loss_total))

                return_model_loss = {'total_loss': total_loss, 'KL_loss':KL_loss}

        return return_model_loss
    
    def evaluate(self, x, epoch = None, batch_idx = None, output_additional_data = None, train = 0, return_output = 0, return_model_output = None):
    
        # Adjust input features for representation
        x = x * self.params.x_train_std + self.params.x_train_mean

        # Model prediction
        y = np.array(return_model_output['y'])
        y_al = np.array(return_model_output['y_al']) 

        # Predict y, y_al and y_ep
        if self.number_outputs_per_output_feature == 3:
            y_ep = np.array(return_model_output['y_ep'])

        # Not BNN
        if self.BNN_algorithm == 'NN':
            # Mean and std of the t prediction
            y_mean = y.reshape(-1)

            # Predict only y and y_al
            if self.number_outputs_per_output_feature == 2:
                epistemic_std = np.array(0)
            # Predict y, y_al and y_ep
            elif self.number_outputs_per_output_feature == 3:
                epistemic_std =  (np.exp(y_ep)**0.5).reshape(-1)
                epistemic_std = np.minimum(epistemic_std, 10e3)

            # Mean of predicted aleatoric_std
            aleatoric_std = (np.exp(y_al)**0.5).reshape(-1) # y_al = log(aleatoric_var)
            aleatoric_std = np.minimum(aleatoric_std, 10e3)

        # BNN -> compute predictive mean and var
        else:

            # Mean and std of the t prediction
            y_mean = y.mean(axis = 0).reshape(-1)
            epistemic_std = (y.var(axis = 0)**0.5).reshape(-1)
            epistemic_std = np.minimum(epistemic_std, 10e3)

            # Mean of predicted aleatoric_std
            aleatoric_std = (np.exp(y_al).mean(axis = 0)**0.5).reshape(-1) # y_al = log(aleatoric_var)
            aleatoric_std = np.minimum(aleatoric_std, 10e3)

        # Total std of uncertainty
        total_unc_std = (aleatoric_std.reshape(-1)**2 + epistemic_std.reshape(-1)**2)**0.5

        if output_additional_data is not None:

            t = return_numpy(output_additional_data['t']).reshape(-1)

            #### METRICS
            # Accuracy metrics
            mae = mean_absolute_error(t, y_mean)
            rmse = np.sqrt(mean_squared_error(t, y_mean))

            # Uncertainty metrics
            mace = uct.mean_absolute_calibration_error(y_mean, total_unc_std, t)
            rmsce = uct.root_mean_squared_calibration_error(y_mean, total_unc_std, t)
            ma = uct.miscalibration_area(y_mean, total_unc_std, t)
            nll = uct.nll_gaussian(y_mean, total_unc_std, t)

        #### OUTPUT
        # List of variables to check and add if they exist
        variables_to_check = ['x', 't', 'y', 'y_al', 'y_ep', 'y_mean', 'epistemic_std', 'aleatoric_std', 'total_unc_std', 'mae', 'rmse', 'mace', 'rmsce', 'ma', 'nll']

        # Include output of the model
        if return_output:
            return_model_evaluate = {}
            for var_name in variables_to_check:
                if var_name in locals().keys():
                    return_model_evaluate={**return_model_evaluate, var_name: locals()[var_name]}
        # Do not include not_to_return
        else:
            not_to_return = ['x', 't', 'y', 'y_al', 'y_ep', 'y_mean', 'epistemic_std', 'aleatoric_std', 'total_unc_std']
            return_model_evaluate = {}
            for var_name in variables_to_check:
                if var_name in locals().keys() and var_name not in not_to_return:
                    return_model_evaluate={**return_model_evaluate, var_name: locals()[var_name]}

        return return_model_evaluate


    def print_MACs_FLOPs(self):
    
        num_macs = profile_macs(self, torch.zeros(1, self.input_size)) 
        print("#MACs:", num_macs)
        print("#FLOPs:", num_macs*2)