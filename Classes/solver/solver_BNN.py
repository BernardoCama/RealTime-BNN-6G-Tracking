import sys
import os
import importlib
import torch
import time
import IPython
import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.set_loglevel("error")
import json
from torch.utils.data import DataLoader
from itertools import zip_longest


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
from Classes.utils.utils import to_var, dataloader_to_numpy, return_numpy, serialize_ndarray
from Classes.plotting.plotting import plot_uncertainty 



class Solver_BNN(object):

    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(Solver_BNN.DEFAULTS, **params_dict)

        # Create model
        self.build_model()

    def build_model(self):

        # BNN algorithm instance
        self.BNN_algorithm_instance = getattr(importlib.import_module(self.package_BNN_algorithm), self.BNN_algorithm)(self.params)

        # Define model
        self.BNN_algorithm_instance.set_model()
   
        # Optimizers
        self.BNN_algorithm_instance.set_optimizer()

        # Print networks
        if self.bool_print_network:

            self.BNN_algorithm_instance.print_network()

        # Set cuda if available
        self.BNN_algorithm_instance.set_model_cuda()

        # Update parameters
        self.params.update_all()

    def save_results(self, output_results, output_dir = None):

        if output_dir == None:
            output_dir = self.output_results_dir
    
        try:
            output_results_old = np.load(os.path.join(output_dir,'output_results.npy'), allow_pickle = True).tolist()
            output_results_new = output_results_old.update(output_results)
        except:
            output_results_new = output_results

        np.save(os.path.join(output_dir,'output_results.npy'), output_results_new, allow_pickle = True)

    def load_results(self, output_dir = None):
    
        if output_dir == None:
            output_dir = self.output_results_dir

        return np.load(os.path.join(output_dir,'output_results.npy'), allow_pickle = True).tolist()

    def load_pretrained_model(self, epoch, batch_id, cuda = None):

        self.BNN_algorithm_instance.load_models(epoch = epoch, batch_id = batch_id)

        self.BNN_algorithm_instance.set_model_cuda()

    def set_cuda_usage(self, use_cuda = None):

        if use_cuda == None:
            use_cuda = self.use_cuda
        self.use_cuda = use_cuda
        self.BNN_algorithm_instance.use_cuda = use_cuda
        self.BNN_algorithm_instance.set_cuda_usage(use_cuda)

    def train(self, train_loader, val_loader = None):
        self.train_loader = train_loader
        if val_loader is not None:
            self.val_loader = val_loader
        if val_loader is not None:
            val_iter = iter(val_loader)
        else:
            val_iter = iter([None] * len(train_loader))

        # Start with trained model if exists
        if self.bool_pretrained_model:
            start = int(self.bool_pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()

        for e in range(start, self.num_epochs):

            # Activate plotting of loss and accuracy metrics for last epoch
            if e == self.num_epochs - 1:
                self.bool_plot_training = 1

            # Logging
            loss_metrics_train = {}

            for i, ((input_data, *other_data), (val_batch)) in enumerate(zip_longest(self.train_loader, val_iter, fillvalue=None)):
            # for i, (input_data, *other_data) in enumerate((self.train_loader)):  # enumerate(tqdm(self.train_loader))

                iter_ctr += 1

                output_additional_data = {dataset_output_name:other_data[index_name] for index_name,(dataset_output_name) in enumerate(self.params.dataset_output_names)}
                input_data = to_var(input_data)
                output_additional_data = to_var(output_additional_data)

                input_all_data = {'input_data': input_data}
                output_all_additional_data = {'output_additional_data': output_additional_data}

                if val_loader is not None:
                    if val_batch is not None: 
                        input2_data, *other2_data = val_batch
                    else:
                        # Restart the validation loader from the beginning
                        val_iter = iter(val_loader)
                        input2_data, *other2_data = next(val_iter)
                    
                    output2_additional_data = {dataset_output_name:other2_data[index_name] for index_name,(dataset_output_name) in enumerate(self.params.dataset_output_names)}
                    input2_data = to_var(input2_data, use_cuda=0)
                    output2_additional_data = to_var(output2_additional_data, use_cuda=0)

                    input_all_data['input2_data'] = input2_data
                    output_all_additional_data['output2_additional_data'] = output2_additional_data

                return_model_step =  self.model_step(input_all_data, epoch = e, batch_idx = i, output_all_additional_data = output_all_additional_data, train = 1, bool_return_loss_metrics = 1, bool_return_accuracy_metrics = self.bool_return_train_accuracy_metrics)

                if return_model_step is not None:

                    # Logging
                    for k,v in return_model_step.items():
                        if k in loss_metrics_train.keys():
                            try:
                                loss_metrics_train[k].append(v.data.tolist())
                            except:
                                loss_metrics_train[k].append(v.tolist())
                        else:
                            try:
                                loss_metrics_train[k] = [v.data.tolist()]
                            except:
                                loss_metrics_train[k] = [v.tolist()]

                    # Print out log info
                    if (i+1) % self.log_train_step == 0:
                        elapsed = time.time() - start_time
                        total_time = ((self.num_epochs*self.num_train_batches)-(e*self.num_train_batches+i)) * elapsed/(e*self.num_train_batches+i+1)
                        epoch_time = (self.num_train_batches-i)* elapsed/(e*self.num_train_batches+i+1)
                        
                        epoch_time = str(datetime.timedelta(seconds=epoch_time))
                        total_time = str(datetime.timedelta(seconds=total_time))
                        elapsed = str(datetime.timedelta(seconds=elapsed))

                        tmplr = self.BNN_algorithm_instance.get_optimizer_lr()

                        log = "Training Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                            elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, self.num_train_batches, tmplr)

                        for tag, value in loss_metrics_train.items():
                            value = np.mean(value)
                            loss_metrics_train[tag] = value
                            log += ", {}: {:.4f}".format(tag, value)

                        IPython.display.clear_output()
                        print(log)

                        # Validation
                        if val_loader is not None and self.bool_validate_model:
                            if (e) % self.log_valid_step == 0:
                                loss_metrics_valid = self.test(self.val_loader, epoch = e, batch_idx = i, batchwise = 1)
                            else:
                                loss_metrics_valid = {}
                        else:
                            loss_metrics_valid = {}

                        # Train and validation metrics
                        loss_metrics = {**loss_metrics_train, **loss_metrics_valid}

                        columns_subplot = int(np.ceil(len(loss_metrics_train.keys())/3))

                        # Initialize logs if not already
                        if not hasattr(self, 'loss_metrics_logs'):
                            self.loss_metrics_logs = {}

                        # Calculate the number of subplots needed
                        unique_base_keys = set(key.rsplit('_', 1)[0] for key in loss_metrics.keys())
                        columns_subplot = int(np.ceil(len(unique_base_keys) / 3))

                        # Initialize plot
                        if self.bool_plot_training:
                            fig = plt.figure(figsize=(10, 10))

                        plt_ctr = 1

                        # Update logs and plot
                        for base_key in unique_base_keys:
                            train_key = f"{base_key}_train"
                            valid_key = f"{base_key}_valid"

                            if train_key in loss_metrics:
                                if train_key not in self.loss_metrics_logs:
                                    self.loss_metrics_logs[train_key] = []
                                self.loss_metrics_logs[train_key].append(loss_metrics[train_key])

                            if valid_key in loss_metrics:
                                if valid_key not in self.loss_metrics_logs:
                                    self.loss_metrics_logs[valid_key] = []
                                self.loss_metrics_logs[valid_key].append(loss_metrics[valid_key])

                            if self.bool_plot_training:
                                plt.subplot(3, columns_subplot, plt_ctr)

                                if train_key in self.loss_metrics_logs and len(self.loss_metrics_logs[train_key]) > 0:
                                    plt.plot(self.loss_metrics_logs[train_key], label=train_key, color='b')

                                if valid_key in self.loss_metrics_logs and len(self.loss_metrics_logs[valid_key]) > 0:
                                    plt.plot(self.loss_metrics_logs[valid_key], label=valid_key, color='orange')

                                plt.legend(fontsize='xx-small')
                                plt_ctr += 1



                        if self.bool_plot_training:
                            plt.savefig(os.path.join(self.Figures_dir,'training_epochs.pdf'), bbox_inches='tight')
                            plt.savefig(os.path.join(self.Figures_dir,'training_epochs.eps'), format='eps', bbox_inches='tight')
                        if self.bool_save_training_info:
                            np.save(os.path.join(self.output_results_dir,'loss_metrics_logs.npy'), self.loss_metrics_logs, allow_pickle = True)
                            # Save self.params.DEFAULTS as a JSON-style text file
                            with open(os.path.join(self.output_results_dir, 'params_defaults.txt'), 'w') as file:
                                json.dump(self.params.DEFAULTS, file, indent=4, default=serialize_ndarray)

                        if self.bool_plot_training:
                            plt.close(fig)

                        # Logging
                        loss_metrics = {}

                    # Save model checkpoints
                    if (i+1) % self.model_save_step == 0:

                        self.BNN_algorithm_instance.save_models(epoch = e, batch_id = i)

    def test(self, test_data_loader, epoch = None, batch_idx = None, 
             batchwise = 1, return_statistics_per_batch = 0, return_output = 0,
             train_loader = None, val_loader = None, 
             file_name_loss_metrics_valid = 'loss_metrics_valid',
             file_name_reliability_diagram = 'reliability_diagram', bool_plot_reliability_diagram = 0, 
             file_name_uncertainty_dataset = 'uncertainty_dataset', bool_plot_uncertainty_dataset = 0):

        self.BNN_algorithm_instance.set_model_mode(train=0)

        with torch.no_grad():

            start_time = time.time()

            # Logging
            iter_ctr = 0
            loss_metrics_valid = {}

            if batchwise:

                for i, (input_data, *other_data) in enumerate((test_data_loader)): # enumerate(tqdm(test_data_loader))
                    iter_ctr += 1

                    output_additional_data = {dataset_output_name:other_data[index_name]  for index_name,(dataset_output_name) in enumerate(self.params.dataset_output_names)}
                    input_data = to_var(input_data)

                    input_all_data = {'input_data': input_data}
                    output_all_additional_data = {'output_additional_data': output_additional_data}

                    if not return_output:
                        return_model_step = self.model_step(input_all_data, epoch = epoch, batch_idx = batch_idx, output_all_additional_data = output_all_additional_data, train = 0, bool_return_loss_metrics = self.bool_return_valid_loss_metrics, bool_return_accuracy_metrics = self.bool_return_valid_accuracy_metrics)
                    else:
                        return_model_step = self.BNN_algorithm_instance.forward_sample(input_all_data, epoch = epoch, batch_idx = batch_idx, output_all_additional_data = output_all_additional_data, train = 0) 
                        return_model_step.update(output_additional_data)
                        print(i)

                    if return_model_step is not None:

                        # Logging
                        for k,v in return_model_step.items():
                            if k in loss_metrics_valid.keys():
                                try:
                                    loss_metrics_valid[k].append(v.data.tolist())
                                except:
                                    loss_metrics_valid[k].append(v.tolist())
                            else:
                                try:
                                    loss_metrics_valid[k] = [v.data.tolist()]
                                except:
                                    loss_metrics_valid[k] = [v.tolist()]


                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Validation Elapsed {}]".format(elapsed)

                for tag, value in loss_metrics_valid.items():
                    value = np.mean(value)
                    if not return_statistics_per_batch:
                        loss_metrics_valid[tag] = value
                        log += ", {}: {:.4f}".format(tag, value)

            # Complete inference of the whole dataset in one step
            else:
                
                # Dataloader completed with x_test, t_test in batches
                if isinstance(test_data_loader, DataLoader):
                    # Concatenate batches in one single batch
                    test_data_loader = dataloader_to_numpy(test_data_loader)
                    x_test = test_data_loader.pop(0)
                    output_additional_data = {dataset_output_name:test_data_loader[index_name+1] for index_name,(dataset_output_name) in enumerate(self.params.dataset_output_names)}

                # x_test and t_test as single numpy arrays
                elif isinstance(test_data_loader, dict):
                    x_test = test_data_loader['x_test']
                    t_test = test_data_loader['t_test']
                    output_additional_data = {dataset_output_name:t_test[index_name] for index_name,(dataset_output_name) in enumerate(self.params.dataset_output_names)}

                # Only x_test as single numpy array
                elif isinstance(test_data_loader, np.ndarray):
                    x_test = test_data_loader
                    output_additional_data = None

                # As a list
                elif isinstance(test_data_loader, list):
                    x_test = test_data_loader[0]
                    output_additional_data = {dataset_output_name:test_data_loader[index_name+1] for index_name,(dataset_output_name) in enumerate(self.params.dataset_output_names)}

                # Only x_test as single tensor array
                elif isinstance(test_data_loader, torch.Tensor):
                    x_test = test_data_loader
                    output_additional_data = None

                input_all_data = {'input_data': x_test}
                output_all_additional_data = {'output_additional_data': output_additional_data}

                return_model_evaluate = self.BNN_algorithm_instance.evaluate(input_all_data, epoch = epoch, batch_idx = batch_idx, output_all_additional_data = output_all_additional_data, train = 0, return_output=1, use_cuda = self.use_cuda) 
                
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Validation Elapsed {}".format(elapsed)

                for tag, value in return_model_evaluate.items():
                    value = np.mean(return_numpy(value))
                    loss_metrics_valid[tag] = value
                    log += ", {}: {:.4f}".format(tag, value)

            # Plot reliability diagram
            if bool_plot_reliability_diagram:
                plot_uncertainty(return_model_evaluate, t_test, self.params, file_name = file_name_reliability_diagram, plot_save_str="row", xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 20, labelsize = 18, save_eps = 1, ax = None, save_svg = 1, save_pdf = 1, plt_show = 0)

            # Plot uncertainty predicted by the model
            if bool_plot_uncertainty_dataset:
                dataset_class_instance = getattr(importlib.import_module(self.package_dataset), 'DATASET')(self.params)
                dataset_class_instance.show_uncertainty_dataset(return_model_evaluate, train_loader = train_loader, val_loader = val_loader, file_name_uncertainty_dataset=file_name_uncertainty_dataset)

            if self.bool_save_training_info:
                np.save(os.path.join(self.output_results_dir, '{}.npy'.format(file_name_loss_metrics_valid)), loss_metrics_valid, allow_pickle = True)

            # Print statistics
            print(log)
                        
        return loss_metrics_valid

    def model_step(self, input_all_data, epoch = None, batch_idx = None, output_all_additional_data = None, train = 0, bool_return_loss_metrics = 1, bool_return_accuracy_metrics = 0):

        self.BNN_algorithm_instance.set_model_mode(train)

        # Forward
        return_model_output = self.BNN_algorithm_instance.forward_sample(input_all_data, epoch = epoch, batch_idx = batch_idx, output_all_additional_data = output_all_additional_data, train = train) 

        # Compute loss function
        if bool_return_loss_metrics:
            return_model_loss = self.BNN_algorithm_instance.loss_function(input_all_data, return_model_output, output_all_additional_data = output_all_additional_data)
        else:
            return_model_loss = {}

        if train and bool_return_loss_metrics:
            # Backward
            self.BNN_algorithm_instance.backward(return_model_loss)

        # Evaluate
        if bool_return_accuracy_metrics:
            with torch.no_grad():
                # self.BNN_algorithm_instance.set_model_mode(train=0)
                # input_data = to_var(input_data, use_cuda=False)
                return_model_evaluate = self.BNN_algorithm_instance.evaluate(input_all_data, epoch = epoch, batch_idx = batch_idx, output_all_additional_data = output_all_additional_data, 
                                                                            train = train, return_output=0, use_cuda = self.use_cuda) 
        else:
            return_model_evaluate = {}

        return_model_step = {**return_model_loss, **return_model_evaluate}
        if train:
            return_model_step = {key + '_train': value for key, value in return_model_step.items()}
        else:
            return_model_step = {key + '_valid': value for key, value in return_model_step.items()}
    
        return return_model_step


