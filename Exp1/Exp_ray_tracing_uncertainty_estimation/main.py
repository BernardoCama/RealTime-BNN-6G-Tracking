import os 
import sys
import numpy as np
import importlib
import torch
import platform
import json

###########################################################################################
###########################################################################################
# DEFINITIONS DIRECTORIES - CLASSES - EXPERIMENTS - DATASETS - MODEL - OPTIMIZER - BNN ALGORITHM

# Directories
cwd = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'DB')
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
from Classes.plotting.plotting import plot_predicted_uncertainties_ray_tracing
from Classes.utils.utils import from_xyz_to_latlonh_matrix




# RAY_TRACING
exp_name = 'NN_Ray_tracing'
exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
saved_models_dir = os.path.join(exp_dir, 'Saved_models')
output_results_dir = os.path.join(exp_dir, 'Output_results')
Figures_dir = os.path.join(exp_dir, 'Figures')

exp_paper_name = 'Exp_ray_tracing_uncertainty_estimation'
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

predict = 0

# SINGLE EPOCH TEST
best_epoch = 600

# cuda for testing
solver.set_cuda_usage(0)

# Model
solver = Solver_BNN(params)

# Recover training params
loss_metrics_logs = np.load(os.path.join(params.output_results_dir,'loss_metrics_logs.npy'), allow_pickle = True).tolist()
# best_epoch = np.argmin(loss_metrics_logs[[key for key in loss_metrics_logs if 'mace' in key and 'valid' in key][1 if sum('mace' in key and 'valid' in key for key in loss_metrics_logs) > 1 else 0]])
best_epoch = 159      # specific epoch
params_training = json.load(open(os.path.join(solver.output_results_dir, 'params_defaults.txt'), 'r'))
solver.BNN_algorithm_instance.params.num_train_batches = params_training['num_train_batches']
solver.BNN_algorithm_instance.params.x_train_std = params_training['x_train_std']
solver.BNN_algorithm_instance.params.x_train_mean = params_training['x_train_mean']
solver.BNN_algorithm_instance.model_save_step = params_training['model_save_step']
solver.bool_save_training_info = 0 # avoid saving input variables
params.update_all()

# Load pretrained model
solver.load_pretrained_model(epoch=best_epoch, batch_id=params.num_train_batches) # params.num_epochs

# Validation dataset performances 
if predict:
    loss_metrics_valid = solver.test(train_loader, batchwise=1, return_statistics_per_batch = 1, return_output = 1)

    del loss_metrics_valid['dec']
    
    results = {}
    for k,v in loss_metrics_valid.items():
        try:
            results[k] = np.concatenate(np.array(v))
        except:
            pass
    results['UElatlonh'] = from_xyz_to_latlonh_matrix(results['UExyz_wrtBS'], results['BSlatlonh'])

    # Save results
    solver.save_results(results, output_dir = params.output_results_paper_dir)

# Load results
results = solver.load_results(output_dir = params.output_results_paper_dir)

# Visualize
y_al = results['y_al'].reshape(-1, 3, 3)
total_unc_std = np.sqrt(np.sum(np.diagonal(y_al, axis1=1, axis2=2), axis=1))
UElatlonh = results['UElatlonh']

plot_predicted_uncertainties_ray_tracing(UElatlonh[:,0], UElatlonh[:,1], total_unc_std, params, 'Exp_ray_tracing_uncertainty_estimation', xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 1, ax = None, save_svg = 0, save_pdf = 1, save_jpg = 1, plt_show = 1)





