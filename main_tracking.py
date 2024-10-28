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
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
EXPERIMENTS_DIR_TRACKING = os.path.join(cwd, 'Exp_tracking')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
PARAMETERS_DIR_TRACKING = os.path.join(cwd, 'Parameters_tracking')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR_TRACKING))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR_TRACKING))
sys.path.append(os.path.dirname(cwd))

# Importing Generic Classes
from Classes.utils.utils import mkdir
from Classes.solver.solver_BNN import Solver_BNN
from Classes.solver.solver_Tracking import Solver_Tracking


# Specific experiment
# ARTIFICIAL_2D
exp_tracking_name = 'EKF_vs_BNN_Artificial_2D'
# RAY_TRACING
# exp_tracking_name = 'EKF_vs_BNN_Ray_tracing'
exp_tracking_dir = os.path.join(EXPERIMENTS_DIR_TRACKING, exp_tracking_name)
saved_models_tracking_dir = os.path.join(exp_tracking_dir, 'Saved_models')
output_results_tracking_dir = os.path.join(exp_tracking_dir, 'Output_results')
Figures_tracking_dir = os.path.join(exp_tracking_dir, 'Figures')
mkdir(exp_tracking_dir)
mkdir(saved_models_tracking_dir)
mkdir(output_results_tracking_dir)
mkdir(Figures_tracking_dir)

# General parameters
# ARTIFICIAL_2D
# parameters_tracking_name = 'EKF_vs_BNN_Artificial_2D'
# RAY_TRACING
# parameters_tracking_name = 'EKF_vs_BNN_Ray_tracing'
parameters_tracking_name = exp_tracking_name
parameters_tracking_dir = os.path.join(PARAMETERS_DIR_TRACKING, parameters_tracking_name, 'param.py')
package_params_tracking = parameters_tracking_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params_tracking = getattr(importlib.import_module(package_params_tracking), 'PARAMS')()
params_tracking({'cwd':cwd, 'DB_DIR':DB_DIR, 'CLASSES_DIR':CLASSES_DIR, 'EXPERIMENTS_DIR':EXPERIMENTS_DIR, 'PARAMETERS_DIR':PARAMETERS_DIR, 'EXPERIMENTS_DIR_TRACKING':EXPERIMENTS_DIR_TRACKING, 'PARAMETERS_DIR_TRACKING':PARAMETERS_DIR_TRACKING})
params_tracking({'exp_tracking_name':exp_tracking_name, 'exp_tracking_dir':exp_tracking_dir, 'saved_models_tracking_dir':saved_models_tracking_dir, 'output_results_tracking_dir':output_results_tracking_dir, 'Figures_tracking_dir':Figures_tracking_dir})
params_tracking({'parameters_tracking_name':parameters_tracking_name, 'parameters_tracking_dir':parameters_tracking_dir, 'package_params_tracking':package_params_tracking})

# Reproducibility
np.random.seed(params_tracking.seed)
torch.manual_seed(params_tracking.seed)

# OS
OS = platform.system()
params_tracking({'OS':OS})

# Dataset
dataset_name = params_tracking.dataset_name
dataset_dir = os.path.join(CLASSES_DIR, 'dataset', dataset_name + '.py')
db_dir = os.path.join(DB_DIR, dataset_name)
package_dataset = dataset_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params_tracking({'dataset_dir':dataset_dir, 'db_dir':db_dir, 'package_dataset':package_dataset})
params_tracking.update_all()

###########################################################################################
###########################################################################################
# IMPORTING BASELINES

# Compared tracking models
# ARTIFICIAL_2D
baselines_tracking_classic_name_params = {'EKF':{'tracking_params':'EKF_Artificial_2D', 'training_params':''}}
baselines_tracking_BNN_name_params = {'BNN': {'tracking_params':'BNN_Artificial_2D', 'training_params':'SGLD_Artificial_2D'}}
# RAY_TRACING
# baselines_tracking_classic_name_params = {'EKF':{'tracking_params':'EKF_Ray_tracing', 'training_params':''}}
# baselines_tracking_BNN_name_params = {'BNN': {'tracking_params':'BNN_Ray_tracing', 'training_params':'NN_Ray_tracing'}}
# baselines_tracking_BNN_name_params = {'TCN': {'tracking_params':'TCN_Ray_tracing', 'training_params':'TCN_Ray_tracing'}}
baselines_name_params = {**baselines_tracking_classic_name_params, **baselines_tracking_BNN_name_params}
baselines = {}
for baseline_name, baseline_tracking_training_params_name in baselines_name_params.items():

    baseline_tracking_params_name = baseline_tracking_training_params_name['tracking_params']
    baseline_training_params_name = baseline_tracking_training_params_name['training_params']

    # Tracking parameters
    baseline_tracking_parameters_dir = os.path.join(PARAMETERS_DIR_TRACKING, baseline_tracking_params_name, 'param.py')
    baseline_tracking_package_params = baseline_tracking_parameters_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
    baseline_tracking_params = getattr(importlib.import_module(baseline_tracking_package_params), 'PARAMS')()
    baseline_tracking_params({'baseline_tracking_params_name':baseline_tracking_params_name, 'baseline_tracking_parameters_dir':baseline_tracking_parameters_dir, 'baseline_tracking_package_params':baseline_tracking_package_params})

    # Resources
    baseline_tracking_params({'use_cuda': torch.cuda.is_available()})

    # Update with general parameters tracking
    baseline_tracking_params({**baseline_tracking_params.DEFAULTS, **params_tracking.DEFAULTS})
    baseline_tracking_params.update_all()

    # If it is a NN, load trained model
    if baseline_name in baselines_tracking_BNN_name_params.keys():

        # Training params
        baseline_training_exp_name = baseline_training_params_name
        baseline_training_exp_dir = os.path.join(EXPERIMENTS_DIR, baseline_training_exp_name)
        baseline_training_saved_models_dir = os.path.join(baseline_training_exp_dir, 'Saved_models')
        baseline_training_output_results_dir = os.path.join(baseline_training_exp_dir, 'Output_results')
        baseline_training_Figures_dir = os.path.join(baseline_training_exp_dir, 'Figures')

        baseline_training_parameters_name = baseline_training_exp_name
        baseline_training_parameters_dir = os.path.join(PARAMETERS_DIR, baseline_training_parameters_name, 'param.py')
        baseline_training_package_params = baseline_training_parameters_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        baseline_training_params = getattr(importlib.import_module(baseline_training_package_params), 'PARAMS')()
        baseline_training_params({'cwd':cwd, 'DB_DIR':DB_DIR, 'CLASSES_DIR':CLASSES_DIR, 'EXPERIMENTS_DIR':EXPERIMENTS_DIR, 'PARAMETERS_DIR':PARAMETERS_DIR})
        baseline_training_params({'exp_name':baseline_training_exp_name, 'exp_dir':baseline_training_exp_dir, 'saved_models_dir':baseline_training_saved_models_dir, 'output_results_dir':baseline_training_output_results_dir, 'Figures_dir':baseline_training_Figures_dir})
        baseline_training_params({'parameters_name':baseline_training_parameters_name, 'parameters_dir':baseline_training_parameters_dir, 'package_params':baseline_training_package_params})

        # Resources
        baseline_training_params({'use_cuda': torch.cuda.is_available()})

        # Update with general parameters tracking
        baseline_training_params({**baseline_training_params.DEFAULTS, **params_tracking.DEFAULTS})
        baseline_training_params.update_all()

        # Model
        model_name = baseline_training_params.model_name
        model_dir = os.path.join(CLASSES_DIR, 'model', model_name + '.py')
        package_model = model_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        baseline_training_params({'model_dir':model_dir, 'package_model':package_model})

        # Optimizer
        optimizer_dir = os.path.join(CLASSES_DIR, 'optimizer', 'optimizer' + '.py')
        package_optimizer = optimizer_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        baseline_training_params({'optimizer_dir':optimizer_dir, 'package_optimizer':package_optimizer})

        # BNN algorithm
        BNN_algorithm = baseline_training_params.BNN_algorithm
        BNN_algorithm_dir = os.path.join(CLASSES_DIR, 'BNN_algorithm', BNN_algorithm + '.py')
        package_BNN_algorithm = BNN_algorithm_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        baseline_training_params({'BNN_algorithm_dir':BNN_algorithm_dir, 'package_BNN_algorithm':package_BNN_algorithm})

        # Model
        baseline_training_params({'type':'BNN'})
        solver_BNN = Solver_BNN(baseline_training_params)

        # Recover training params
        loss_metrics_logs = np.load(os.path.join(baseline_training_params.output_results_dir,'loss_metrics_logs.npy'), allow_pickle = True).tolist()
        # best_epoch = np.argmin(loss_metrics_logs[[key for key in loss_metrics_logs if 'mace' in key and 'valid' in key][1 if sum('mace' in key and 'valid' in key for key in loss_metrics_logs) > 1 else 0]])
        best_epoch = 600      # specific epoch    600
        params_training = json.load(open(os.path.join(solver_BNN.output_results_dir, 'params_defaults.txt'), 'r'))
        solver_BNN.BNN_algorithm_instance.params.num_train_batches = params_training['num_train_batches']
        solver_BNN.BNN_algorithm_instance.params.x_train_std = params_training['x_train_std']
        solver_BNN.BNN_algorithm_instance.params.x_train_mean = params_training['x_train_mean']
        solver_BNN.BNN_algorithm_instance.model_save_step = params_training['model_save_step']
        params_tracking({**params_tracking.DEFAULTS, **baseline_training_params.__dict__})

        # cuda for testing
        solver_BNN.set_cuda_usage(1 if baseline_training_params.use_cuda else 0)
        # Load pretrained model
        solver_BNN.load_pretrained_model(epoch=best_epoch, batch_id=solver_BNN.BNN_algorithm_instance.params.num_train_batches) # params.num_epochs

        # Tracking algorithm 
        Tracking_algorithm = baseline_tracking_params.Tracking_algorithm
        Tracking_algorithm_dir = os.path.join(CLASSES_DIR, 'Tracking_algorithm', Tracking_algorithm + '.py')
        package_Tracking_algorithm = Tracking_algorithm_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        baseline_tracking_params({'Tracking_algorithm_dir':Tracking_algorithm_dir, 'package_Tracking_algorithm':package_Tracking_algorithm})

        # Tracking algorithm instance
        baseline_tracking_params({'type':'BNN'})
        baseline_tracking_params({**baseline_tracking_params.DEFAULTS, **baseline_training_params.DEFAULTS})
        baseline_tracking_params.update_all()
        Tracking_algorithm_instance = getattr(importlib.import_module(package_Tracking_algorithm), Tracking_algorithm)(baseline_tracking_params)

        Tracking_algorithm_instance.set_model(solver_BNN)

    # If it is a classical tracking method
    else:

        # Tracking algorithm
        Tracking_algorithm = baseline_tracking_params.Tracking_algorithm
        Tracking_algorithm_dir = os.path.join(CLASSES_DIR, 'Tracking_algorithm', Tracking_algorithm + '.py')
        package_Tracking_algorithm = Tracking_algorithm_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        baseline_tracking_params({'Tracking_algorithm_dir':Tracking_algorithm_dir, 'package_Tracking_algorithm':package_Tracking_algorithm})

        # Tracking algorithm instance
        baseline_tracking_params({'type':'Classic'})
        Tracking_algorithm_instance = getattr(importlib.import_module(package_Tracking_algorithm), Tracking_algorithm)(baseline_tracking_params)

    # Update parameters
    baseline_tracking_params.update_all()

    baselines[baseline_name] = Tracking_algorithm_instance

###########################################################################################
###########################################################################################
# IMPORTING DATASETS - SOLVER (TRACKING)

# Dataset
dataset_class_instance = getattr(importlib.import_module(package_dataset), 'DATASET')(params_tracking)
test_loader = dataset_class_instance.return_dataset_tracking()
dataset_class_instance.show_dataset(test_loader)
# dataset_class_instance.show_dataset_tracking()

# Solver Tracking
solver_tracking = Solver_Tracking(params_tracking, dataset_class_instance)

# Update parameters
params_tracking.update_all()


###########################################################################################
###########################################################################################
# SET BASELINES - TESTING - VISUALIZATION

# Set baselines
solver_tracking.set_baselines(baselines)

# Testing
output_results_tracking = solver_tracking.test(test_loader)
solver_tracking.save_results(output_results_tracking)

# Load results
output_results_tracking = solver_tracking.load_results()

# Visualize
dataset_class_instance.show_tracking_results(output_results_tracking)






