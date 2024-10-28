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
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(cwd))

# Importing Generic Classes
from Classes.utils.utils import mkdir, convert_pdfs_to_images, convert_pdfs_to_images, create_movie
from Classes.solver.solver_BNN import Solver_BNN

# Specific experiment
# ARTIFICIAL
exp_name = 'NN_Artificial'
# exp_name = 'MCDropout_Artificial'
# exp_name = 'BBP_Artificial'
# exp_name = 'SGLD_Artificial'
# exp_name = 'BDK_Artificial'
# exp_name = 'BDKep_Artificial'

# ARTIFICIAL_2D
# exp_name = 'NN_Artificial_2D_Aleatoric'
# exp_name = 'SGLD_Artificial_2D'
# exp_name = 'SGLD_Artificial_2D_Aleatoric'
# exp_name = 'SGLD_Artificial_2D_Epistemic'
# exp_name = 'BDK_Artificial_2D_Aleatoric'
# exp_name = 'BDKep_Artificial_2D_Aleatoric'
# exp_name = 'BDKep_Artificial_2D_Epistemic'

# RAY_TRACING
# exp_name = 'NN_Ray_tracing'
# exp_name = 'NN_Ray_tracing2'
# exp_name = 'TCN_Ray_tracing'
# exp_name = 'SGLD_Ray_tracing'
# exp_name = 'BDKep_Ray_tracing'

exp_dir = os.path.join(EXPERIMENTS_DIR, exp_name)
saved_models_dir = os.path.join(exp_dir, 'Saved_models')
output_results_dir = os.path.join(exp_dir, 'Output_results')
Figures_dir = os.path.join(exp_dir, 'Figures')
mkdir(exp_dir)
mkdir(saved_models_dir)
mkdir(output_results_dir)
mkdir(Figures_dir)

# General parameters
# ARTIFICIAL
parameters_name = 'NN_Artificial'
# parameters_name = 'MCDropout_Artificial'
# parameters_name = 'BBP_Artificial'
# parameters_name = 'SGLD_Artificial'
# parameters_name = 'BDK_Artificial'
# parameters_name = 'BDKep_Artificial'

# ARTIFICIAL_2D
# parameters_name = 'NN_Artificial_2D_Aleatoric'
# parameters_name = 'SGLD_Artificial_2D'
# parameters_name = 'SGLD_Artificial_2D_Aleatoric'
# parameters_name = 'SGLD_Artificial_2D_Epistemic'
# parameters_name = 'BDK_Artificial_2D_Aleatoric'
# parameters_name = 'BDKep_Artificial_2D_Aleatoric'
# parameters_name = 'BDKep_Artificial_2D_Epistemic'

# RAY_TRACING
# parameters_name = 'NN_Ray_tracing'
# parameters_name = 'NN_Ray_tracing2'
# parameters_name = 'TCN_Ray_tracing'
# parameters_name = 'SGLD_Ray_tracing'
# parameters_name = 'BDKep_Ray_tracing'

parameters_name = exp_name
parameters_dir = os.path.join(PARAMETERS_DIR, parameters_name, 'param.py')
package_params = parameters_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
params = getattr(importlib.import_module(package_params), 'PARAMS')()
params({'cwd':cwd, 'DB_DIR':DB_DIR, 'CLASSES_DIR':CLASSES_DIR, 'EXPERIMENTS_DIR':EXPERIMENTS_DIR, 'PARAMETERS_DIR':PARAMETERS_DIR})
params({'exp_name':exp_name, 'exp_dir':exp_dir, 'saved_models_dir':saved_models_dir, 'output_results_dir':output_results_dir, 'Figures_dir':Figures_dir})
params({'parameters_name':parameters_name, 'parameters_dir':parameters_dir, 'package_params':package_params})

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
# TRAINING

# Update parameters before training
params.update_all()
solver.train(train_loader, val_loader=val_loader)


###########################################################################################
###########################################################################################
# LOADING MODEL - VALIDATION - TESTING - VISUALIZATION

# SINGLE EPOCH TEST
#  loss_metrics_logs = np.load(os.path.join(params.output_results_dir,'loss_metrics_logs.npy'), allow_pickle = True).tolist()
# best_epoch = np.argmin(loss_metrics_logs[[key for key in loss_metrics_logs if 'mace' in key and 'valid' in key][1 if sum('mace' in key and 'valid' in key for key in loss_metrics_logs) > 1 else 0]])
best_epoch = 296

# cuda for testing
solver.set_cuda_usage(0)
# Load pretrained model
solver.load_pretrained_model(epoch=best_epoch, batch_id=params.num_train_batches) # params.num_epochs

# Validation dataset performances with realiability diagram
solver.test(train_loader, batchwise=1, bool_plot_reliability_diagram=0)


# Visualize uncertainty 
test_loader = dataset_class_instance.return_big_testing_dataset()
solver.test(test_loader, batchwise=0, train_loader = train_loader, val_loader = val_loader, bool_plot_reliability_diagram=0, bool_plot_uncertainty_dataset=1)


# MULTIPLE EPOCH TEST
# Define the specific epochs for evaluation
evaluation_epochs = list(range(1, params.num_epochs, 10))

# MULTIPLE EPOCH TEST
for epoch in evaluation_epochs:
    solver.load_pretrained_model(epoch=epoch, batch_id=params.num_train_batches)
    test_loader = dataset_class_instance.return_big_testing_dataset()
    solver.test(test_loader, batchwise=0, train_loader=train_loader, val_loader=val_loader, 
                file_name_loss_metrics_valid=f'loss_metrics_valid_{epoch}',
                file_name_reliability_diagram=f'reliability_diagram_{epoch}', bool_plot_reliability_diagram=0, 
                file_name_uncertainty_dataset=f'uncertainty_dataset_{epoch}', bool_plot_uncertainty_dataset=1)


# Convert PDFs to JPEG for reliability diagram
# reliability_images_path = convert_pdfs_to_images(params.Figures_dir, 'reliability_diagram', evaluation_epochs)
# Convert PDFs to JPEG for uncertainty dataset
uncertainty_images_path = convert_pdfs_to_images(params, 'uncertainty_dataset', evaluation_epochs, convert = 1)


# Create movie for reliability diagram
# create_movie(reliability_images_path, 'reliability_diagram', os.path.join(params.Figures_dir, 'reliability_diagram_movie.mp4'))
# Create movie for uncertainty dataset
create_movie(uncertainty_images_path, 'uncertainty_dataset', os.path.join(params.Figures_dir, 'uncertainty_dataset_movie.mp4'), 
             evaluation_epochs, 
             fps = 5, 
             take_1_every = 1,
             num_max_images = 1000)




