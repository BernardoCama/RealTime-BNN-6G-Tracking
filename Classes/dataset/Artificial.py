import sys
import os
import numpy as np
import GPy
import torch
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy

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
from Classes.utils.utils import print_name_and_value, dataloader_to_numpy, return_numpy
from Classes.plotting.plotting import plot_artificial_dataset, plot_uncertainty_artificial_dataset

class DATASET(object):

    DEFAULTS = {}   

    def __init__(self, params = {}):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(DATASET.DEFAULTS, **params_dict)

    def return_dataset(self):

        x = np.random.uniform(-3, 3, self.size_dataset)[:, None]
        x.sort(axis = 0)

        # TRAINING and VALIDATION DATASETS
        Gaussian_Kernel = GPy.kern.RBF(input_dim=1, variance=self.variance, lengthscale=self.lengthscale)
        if self.aleatoric == 'heteroscedastic':
            C = Gaussian_Kernel.K(x, x) + np.eye(self.size_dataset)*(x + 2)**2 * self.sig_noise**2
        else:
            C = Gaussian_Kernel.K(x, x) + np.eye(self.size_dataset)*self.sig_noise**2
        t = np.random.multivariate_normal(np.zeros((self.size_dataset)), C)[:, None]
        t = (t - t.mean())

        if self.BNN_algorithm == 'MCDropout':
            self.params.x_train_mean = 0 
            self.params.x_train_std = 1 

            self.x_train = deepcopy(x[75:325])
            self.t_train = deepcopy(t[75:325])

            self.x_valid = np.concatenate((x[:75], x[325:]))
            self.t_valid = np.concatenate((t[:75], t[325:]))
        
        # Standardize target output
        elif self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'BBP' or self.BNN_algorithm == 'SGLD' or self.BNN_algorithm == 'BDK' or self.BNN_algorithm == 'BDKep':
            # self.params.x_train_mean = 0 # x[75:325].mean()
            # self.params.x_train_std = 1 # x[75:325].std()
            self.params.x_train_mean = np.mean(np.linspace(-5, 5, 200))
            self.params.x_train_std = np.std(np.linspace(-5, 5, 200))

            self.x_train = (x[75:325] - self.params.x_train_mean)/self.params.x_train_std

            self.params.t_train_mean = 0 # t[75:325].mean()
            self.params.t_train_std = 1 #t[75:325].std()
            self.t_train = (t[75:325] - self.params.t_train_mean)/self.params.t_train_std

            self.x_valid = np.concatenate(((t[:75]-self.params.x_train_mean)/self.params.x_train_std, (t[325:]-self.params.x_train_mean)/self.params.x_train_std))
            self.t_valid = np.concatenate(((t[:75]-self.params.t_train_mean)/self.params.t_train_std, (t[325:]-self.params.t_train_mean)/self.params.t_train_std))

        self.dataset_output_names = ['t']
        self.params.dataset_output_names = self.dataset_output_names

        print_name_and_value(self.x_train.shape)
        print_name_and_value(self.t_train.shape)
        print_name_and_value(self.x_valid.shape)
        print_name_and_value(self.t_valid.shape)
        print_name_and_value(self.task)
        print_name_and_value(self.batch_size)

        train_dataset = TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.t_train).float())
        valid_dataset = TensorDataset(torch.tensor(self.x_valid).float(), torch.tensor(self.t_valid).float())
    
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle)
        self.val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle)

        self.params.num_train_batches = len(self.train_loader)
        self.params.num_val_batches = len(self.val_loader)
        
        print_name_and_value(self.params.num_train_batches)
        print_name_and_value(self.params.num_val_batches)

        self.params.update_all()
        return self.train_loader, self.val_loader

    def return_ground_truth (self, x_train, t_train, x_test):

        x_train = return_numpy(x_train).reshape(-1)
        t_train = return_numpy(t_train).reshape(-1)
        x_test = return_numpy(x_test).reshape(-1)

        Gaussian_Kernel = GPy.kern.RBF(input_dim=1, variance=self.variance, lengthscale=self.lengthscale)

        mean = []
        var = []
        for x in x_test:
            xs = np.array([[x] + list(x_train)]).T
            C = Gaussian_Kernel.K(xs, xs)
            
            data_var = np.eye(self.size_train_dataset)*(xs[1:]-2)**2*self.sig_noise**2
            pred_var = np.eye(1)*(xs[:1]+2)**2*self.sig_noise**2
            
            mean.append(C[:1, 1:].dot(np.linalg.inv(C[1:, 1:] + data_var)).dot(t_train))
            var.append(C[:1, :1] + pred_var - C[:1, 1:].dot(np.linalg.inv(C[1:, 1:] + data_var)).dot(C[:1, 1:].T))

        y_mean = np.array(mean)
        var = np.array(var)

        aleatoric_std = abs(x_test+2)*self.sig_noise
        total_unc_std = var**0.5
        epistemic_std = (total_unc_std**2 - aleatoric_std**2)**0.5

        x = x_test

        #### OUTPUT
        # List of variables to check and add if they exist
        variables_to_check = ['x', 't', 'y', 'y_al', 'y_ep', 'y_mean', 'epistemic_std', 'aleatoric_std', 'total_unc_std', 'mae', 'rmse', 'mace', 'rmsce', 'ma', 'nll']
        return_gt = {}
        for var_name in variables_to_check:
            if var_name in locals().keys():
                return_gt={**return_gt, var_name: locals()[var_name]}

        return return_gt

    def return_big_testing_dataset(self):

        if self.BNN_algorithm == 'MCDropout':

            x_test = torch.linspace(-5, 5, 200)

        elif self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'BBP' or self.BNN_algorithm == 'SGLD' or self.BNN_algorithm == 'BDK' or self.BNN_algorithm == 'BDKep':
            # x_test = torch.linspace(-5, 5, 200) 
            x_test = (torch.linspace(-5, 5, 200)-self.params.x_train_mean)/self.params.x_train_std 

        return x_test


    def show_dataset(self):

        # Show dataset
        if self.bool_plot_dataset:
            x_train, t_train = dataloader_to_numpy(self.train_loader)
            x_valid, t_valid = dataloader_to_numpy(self.val_loader)

            plot_artificial_dataset(x_train, t_train, x_valid, t_valid, self.params, file_name = 'train_test_dataset', xlabel_ = '$x$', ylabel_ = '$t$', title_ = 'Train/Valid dataset', logx = False, logy = False, xlim = [-5, 5], ylim = [-5, 7], save_eps = 1, ax = None, save_svg = 1, save_pdf = 1, plt_show = 1)


    def show_uncertainty_dataset(self, return_model_evaluate, train_loader = None, val_loader = None, file_name_uncertainty_dataset = 'uncertainty_dataset'):

        x_train, t_train = dataloader_to_numpy(train_loader)
        x_valid, t_valid = dataloader_to_numpy(val_loader)

        x_train = x_train * self.params.x_train_std + self.params.x_train_mean 
        x_valid = x_valid * self.params.x_train_std + self.params.x_train_mean 

        # Model prediction
        plot_uncertainty_artificial_dataset(x_train, t_train, x_valid, t_valid, return_model_evaluate, self.params, file_name = file_name_uncertainty_dataset, xlabel_ = '$x$', ylabel_ = '$t$', title_ = self.BNN_algorithm, logx = False, logy = False, xlim = [-5, 5], ylim = [-5, 7], save_eps = 1, ax = None, save_svg = 1, save_pdf = 1, plt_show = 0)

        # Ground truth from Gaussian Process
        # return_gt = self.return_ground_truth(x_train = x_train, t_train = t_train, x_test = return_model_evaluate['x'])
        # plot_uncertainty_artificial_dataset(x_train, t_train, x_valid, t_valid, return_gt, self.params, file_name = 'GT_dataset', xlabel_ = '$x$', ylabel_ = '$t$', title_ = 'Ground truth', logx = False, logy = False, xlim = [-5, 5], ylim = [-5, 7], save_eps = 1, ax = None, save_svg = 1, save_pdf = 1, plt_show = 0)



