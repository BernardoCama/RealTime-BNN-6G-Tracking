import os 
import sys
import numpy as np
import importlib
import torch
import platform

###########################################################################################
###########################################################################################
# DEFINITIONS DIRECTORIES - CLASSES - EXPERIMENTS - DATASETS - MODEL - OPTIMIZER - BNN ALGORITHM

# Directories
cwd = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
EXPERIMENTS_PAPER_DIR = os.path.join(cwd, 'Exp_paper')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(cwd))

# Importing Generic Classes
from Classes.utils.utils import mkdir
from Classes.solver.solver_BNN import Solver_BNN
from Classes.plotting.plotting import plot_box_plot_MAE




# RAY_TRACING
exp_name = 'NN_Ray_tracing2'
exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
saved_models_dir = os.path.join(exp_dir, 'Saved_models')
output_results_dir = os.path.join(exp_dir, 'Output_results')
Figures_dir = os.path.join(exp_dir, 'Figures')

exp_paper_name = 'L_choice'
exp_paper_dir = os.path.join(EXPERIMENTS_PAPER_DIR, exp_paper_name)
output_results_paper_dir = os.path.join(exp_paper_dir, 'Output_results')
Figures_paper_dir = os.path.join(exp_paper_dir, 'Figures')
mkdir(exp_paper_dir)
mkdir(output_results_paper_dir)
mkdir(Figures_paper_dir)

# General parameters
parameters_name = exp_name
parameters_dir = os.path.join(PARAMETERS_DIR, parameters_name, 'param.py')
package_params = parameters_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params = getattr(importlib.import_module(package_params), 'PARAMS')()
params({'cwd':cwd, 'DB_DIR':DB_DIR, 'CLASSES_DIR':CLASSES_DIR, 'EXPERIMENTS_DIR':EXPERIMENTS_DIR, 'PARAMETERS_DIR':PARAMETERS_DIR})
params({'exp_name':exp_name, 'exp_dir':exp_dir, 'saved_models_dir':saved_models_dir, 'output_results_dir':output_results_dir, 'Figures_dir':Figures_dir})
params({'parameters_name':parameters_name, 'parameters_dir':parameters_dir, 'package_params':package_params})


params({'exp_paper_dir':exp_paper_dir, 'output_results_paper_dir':output_results_paper_dir, 'Figures_paper_dir':Figures_paper_dir})
params({'optimizer_name': 'SGLD',
        'BNN_algorithm':'SGLD', 
        'N_samples_param': 10,
        'mix_epochs': 1, 
        'burnin_epochs': 600-10})

# Reproducibility
np.random.seed(params.seed)
torch.manual_seed(params.seed)

# OS
OS = platform.system()
params({'OS':OS})

# Dataset
dataset_name = params.dataset_name
dataset_dir = os.path.join(CLASSES_DIR, 'dataset', dataset_name + '.py')
db_dir = os.path.join(DB_DIR, dataset_name)
package_dataset = dataset_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params({'dataset_dir':dataset_dir, 'db_dir':db_dir, 'package_dataset':package_dataset})
dataset_class_instance = getattr(importlib.import_module(package_dataset), 'DATASET')(params)

# Resources
params({'use_cuda': torch.cuda.is_available()})

# Model
model_name = params.model_name
model_dir = os.path.join(CLASSES_DIR, 'model', model_name + '.py')
package_model = model_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params({'model_dir':model_dir, 'package_model':package_model})

# Optimizer
optimizer_dir = os.path.join(CLASSES_DIR, 'optimizer', 'optimizer' + '.py')
package_optimizer = optimizer_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params({'optimizer_dir':optimizer_dir, 'package_optimizer':package_optimizer})

# BNN algorithm
BNN_algorithm = params.BNN_algorithm
BNN_algorithm_dir = os.path.join(CLASSES_DIR, 'BNN_algorithm', BNN_algorithm + '.py')
package_BNN_algorithm = BNN_algorithm_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params({'BNN_algorithm_dir':BNN_algorithm_dir, 'package_BNN_algorithm':package_BNN_algorithm})


###########################################################################################
###########################################################################################
# IMPORTING DATASETS - SOLVER(MODEL)

# Dataset
# dataset_class_instance.batch_size = 1
train_loader, val_loader = dataset_class_instance.return_dataset()
dataset_class_instance.show_dataset(train_loader)
log_train_step = int(params.num_train_batches/1) # period in number of batches after which we check training statistics
log_valid_step = 1 # period in number of epochs after which we check validation statistics
model_save_step = int(params.num_train_batches/1) # period in number of batches after which we save the model parameters
params({'log_train_step':log_train_step, 'log_valid_step':log_valid_step, 'model_save_step':model_save_step})

# Model
solver = Solver_BNN(params)


###########################################################################################
###########################################################################################
# Update parameters before training
params.update_all()


###########################################################################################
###########################################################################################
# LOADING MODEL - VALIDATION - TESTING - VISUALIZATION

# SINGLE EPOCH TEST
best_epoch = 599

# cuda for testing
solver.set_cuda_usage(1)

L_list = [10, 20, 40, 100]
predict = 0
results = {'MAE': {}, 'MACE': {}}

if predict:
    for L in L_list:

        params({'optimizer_name': 'SGLD',
                'BNN_algorithm':'SGLD', 
                'N_samples_param': L,
                'mix_epochs': 1, 
                'burnin_epochs': best_epoch-L,

                # 'num_train_batches': 30,
                # 'model_save_step':30,
                'batch_size': 128})

        # Model
        solver = Solver_BNN(params)
        params.update_all()

        # Load pretrained model
        # params.num_train_batches = 30
        solver.load_pretrained_model(epoch=best_epoch, batch_id=params.num_train_batches) # params.num_epochs

        # Validation dataset performances 
        loss_metrics_valid = solver.test(train_loader, batchwise=1, return_statistics_per_batch = 1)

        results['MAE'][L] = loss_metrics_valid['mae_valid']
        results['MACE'][L] = loss_metrics_valid['mace_valid']

        # Save results
        solver.save_results(results, output_dir = params.output_results_paper_dir)

# Load results
results = solver.load_results(output_dir = params.output_results_paper_dir)

results['MAE'] = {L:np.array(results['MAE'][L]) for L in L_list}

# Visualize
plot_box_plot_MAE(results['MAE'], L_list, params, 'Exp_L_choice', xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = 0, xlim = None, ylim = [0, 2], fontsize = 18, labelsize = 16, save_eps = 1, ax = None, save_svg = 0, save_pdf = 1, save_jpg = 1, plt_show = 1)




