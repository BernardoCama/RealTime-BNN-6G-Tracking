import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
import importlib

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


class MCDropout(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(MCDropout.DEFAULTS, **params_dict)

        self.params.BNN_algorithm_master = 'MCDropout'
        self.BNN_algorithm_master = self.params.BNN_algorithm_master
        self.params.update_all()

        self.dropout_layer = F.dropout

    # Set model to train or eval
    def set_model_mode(self, train = 0):

        if train:
            self.model.train()
        else:
            self.model.eval()

    # Set model to GPU
    def set_model_cuda(self, model = None, use_cuda = None):
        
        if use_cuda == None:
            use_cuda = self.use_cuda

        if model == None:
            if use_cuda:
                self.model.to("cuda:0", non_blocking=True)
            else:
                self.model.cpu()
        else:
            if use_cuda:
                model.to("cuda:0", non_blocking=True)
            else:
                model.cpu() 

    # Set cuda usage
    def set_cuda_usage(self, use_cuda = None):

        if use_cuda == None:
            use_cuda = self.use_cuda
        self.use_cuda = use_cuda
        self.model.use_cuda = use_cuda
        self.params.use_cuda = use_cuda

    # Set model
    def set_model(self):

        self.model = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params)
        
    # Load stored model
    def load_models(self, epoch = None, batch_id = None):
        
        self.model = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params)

        cuda_device = 'cuda:0' if (self.params.use_cuda and self.use_cuda) else 'cpu'
        self.model.load_state_dict(torch.load(os.path.join(self.saved_models_dir,'{}_{}_model.pth'.format(epoch, batch_id)), map_location=cuda_device))

        self.set_model_cuda(self.model)

        print('loaded trained models (epoch: {} - batch_idx: {})..!'.format(epoch, batch_id))

    # Save trained model
    def save_models(self, epoch = None, batch_id = None):

        torch.save(self.model.state_dict(), os.path.join(self.saved_models_dir, '{}_{}_model.pth'.format(epoch+1, batch_id+1)))

    # Print network structure
    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print(self.model_name)
        print(self.model)
        print("The number of parameters: {}".format(num_params))

        self.model.print_MACs_FLOPs()
        
    # Set optimizer
    def set_optimizer(self):
         
        self.optimizer = getattr(importlib.import_module(self.package_optimizer), 'OPTIMIZER')(self.model.parameters(), self.params).optimizer

    # Get current optimizer learning rate
    def get_optimizer_lr(self):

        lr_tmp = []

        for param_group in self.optimizer.param_groups:

            lr_tmp.append(param_group['lr'])

        tmplr = np.squeeze(np.array(lr_tmp))

        return tmplr

    # Forward or sample from posterior
    def forward_sample(self, input_all_data, epoch = None, batch_idx = None, output_all_additional_data = None, train = 0):
    
        x = input_all_data['input_data']
        output_additional_data = output_all_additional_data['output_additional_data']
        
        return_model_output = self.model.forward(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = 1, use_cuda = self.use_cuda)
        
        return return_model_output
    
    # Sample
    def sample(self, x, N_samples_param = 1000, epoch = None, batch_idx = None, output_additional_data = None, train = 0):
    
        for i in range(N_samples_param):
            return_model_output = self.model.forward(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = train, use_cuda = self.use_cuda)
            
            if i == 0:
                return_model_output_sample = {name:[] for name in return_model_output.keys()}

            for name in return_model_output.keys():
                if train:
                    return_model_output_sample[name].append(return_model_output[name])
                else:
                    return_model_output_sample[name].append(return_model_output[name].cpu().data.numpy())

        return return_model_output_sample
    
    # Testing performances
    def evaluate(self, input_all_data, epoch = None, batch_idx = None, output_all_additional_data = None, train = 0, return_output = 0, use_cuda = None):
        
        x = input_all_data['input_data']
        output_additional_data = output_all_additional_data['output_additional_data']
        
        self.set_cuda_usage(use_cuda)

        return_model_output = self.sample(x, N_samples_param = self.N_samples_param, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = 0)

        return_model_evaluate = self.model.evaluate(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = train, return_output = return_output, return_model_output = return_model_output)

        return return_model_evaluate
    
    # Compute loss function
    def loss_function(self, input_all_data, return_model_output, output_all_additional_data = None):
    
        x = input_all_data['input_data']
        output_additional_data = output_all_additional_data['output_additional_data']

        return_model_loss = self.model.loss_function(x, return_model_output, output_additional_data = output_additional_data)

        return return_model_loss
    
    # Backward pass
    def backward(self, return_model_loss):

        self.model.zero_grad()

        self.optimizer.zero_grad()

        return_model_loss['total_loss'].backward()

        if self.bool_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            
        self.optimizer.step()