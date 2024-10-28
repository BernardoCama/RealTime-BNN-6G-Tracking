import sys
import os
import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim import Adam
import importlib
import copy
import math

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

from Classes.utils.model_utils import LangevinLinear

class SGLD(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(SGLD.DEFAULTS, **params_dict)

        self.params.BNN_algorithm_master = 'SGLD'
        self.BNN_algorithm_master = self.params.BNN_algorithm_master
        self.params.update_all()

        self.LangevinLinear = LangevinLinear

        # Esemble of Neural Networks used for prediction
        self.models = []
        self.models_info = {'epoch':[], 'batch_idx':[], 'saved_models_path': []}

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
    
        self.models = []

        # Change default burnin_epochs
        if epoch is not None:
            burnin_epochs = epoch
            mix_epochs = min(math.ceil(burnin_epochs/10), self.mix_epochs)
            num_epochs = burnin_epochs + mix_epochs * self.N_samples_param + 1

            if num_epochs > self.num_epochs:
                num_epochs = self.num_epochs
                mix_epochs = math.floor((num_epochs - burnin_epochs - 1) / self.N_samples_param)

                # Use default values
                if mix_epochs <= 0:
                    num_epochs = self.num_epochs
                    mix_epochs = self.mix_epochs
                    burnin_epochs = self.burnin_epochs 

        # Use default burnin_epochs
        else:
            num_epochs = self.num_epochs
            mix_epochs = self.mix_epochs
            burnin_epochs = self.burnin_epochs

        model = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params)
        for e in range(burnin_epochs, num_epochs-1, mix_epochs):
            cuda_device = 'cuda:0' if (self.params.use_cuda and self.use_cuda) else 'cpu'
            model.load_state_dict(torch.load(os.path.join(self.saved_models_dir,'{}_{}_model.pth'.format(e, self.model_save_step)), map_location=cuda_device))
            self.models.append(copy.deepcopy(model))
            self.set_model_cuda(self.models[-1])
        
        print('loaded trained models (epochs: {}:{}:{})..!'.format(burnin_epochs, mix_epochs, num_epochs-1))

    # Save trained model
    def save_models(self, epoch = None, batch_id = None):

        torch.save(self.model.state_dict(), os.path.join(self.saved_models_dir, '{}_{}_model.pth'.format(epoch+1, batch_id+1)))

        # Save models by sampling from the posterior
        if (epoch >= self.burnin_epochs) and (epoch % self.mix_epochs == 0):
            new_model = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params)
            state_dict = self.model.state_dict()
            new_model.load_state_dict(state_dict)
            self.models.append(copy.deepcopy(new_model))
            self.set_model_cuda(self.models[-1])
            self.models_info['epoch'].append(epoch)
            self.models_info['batch_idx'].append(batch_id)
            self.models_info['saved_models_path'].append(os.path.join(self.saved_models_dir, '{}_{}_model.pth'.format(epoch+1, batch_id+1)))

            if len(self.models) > self.N_samples_param:
                self.models.pop(0)
                self.models_info['epoch'].pop(0)
                self.models_info['batch_idx'].pop(0)
                self.models_info['saved_models_path'].pop(0)

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

        if len(self.models) == 0:
            new_model = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params)
            state_dict = self.model.state_dict()
            new_model.load_state_dict(state_dict)
            self.models.append(copy.deepcopy(new_model))

            self.set_model_cuda(self.models[-1])

        for i in range(len(self.models)):
            return_model_output = self.models[i].forward(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = train, use_cuda = self.use_cuda)
            
            if i == 0:
                return_model_output_sample = {name:[] for name in return_model_output.keys()}

            for name in return_model_output.keys():
                if train:
                    return_model_output_sample[name].append(return_model_output[name])
                else:
                    if name == 'y':
                        # return_model_output_sample[name].append(return_model_output[name].cpu().data.numpy() * self.params.t_train_std + self.params.t_train_mean) 
                        return_model_output_sample[name].append(return_model_output[name].cpu().data.numpy()) 
                    elif name == 'y_al': 
                        # return_model_output_sample[name].append(np.log(return_model_output[name].exp().cpu().data.numpy() * self.params.t_train_std)) 
                        return_model_output_sample[name].append(return_model_output[name].cpu().data.numpy()) 
                    else:
                        return_model_output_sample[name].append(return_model_output[name].cpu().data.numpy())

        return return_model_output_sample
    
    # Testing performances
    def evaluate(self, input_all_data, epoch = None, batch_idx = None, output_all_additional_data = None, train = 0, return_output = 0, use_cuda = None):
        
        x = input_all_data['input_data']
        output_additional_data = output_all_additional_data['output_additional_data']
        
        self.set_cuda_usage(use_cuda)

        # Load models from .pth
        if epoch is None and len(self.models) == 0:
            self.load_models()
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


class SGLD_optimizer(Optimizer):
    """
    SGLD optimiser based on pytorch's SGD.
    Note that the weight decay is specified in terms of the gaussian prior sigma.
    """

    def __init__(self, params, lr, weight_decay=0, sigma_prior=1, addnoise=True):

        # L2 regularization in terms of either weight_decay or sigma_prior
        if (weight_decay == 0 or weight_decay is None) and sigma_prior!=0:
            weight_decay = 1 / (sigma_prior ** 2)

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, addnoise=addnoise)

        super(SGLD_optimizer, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if group['addnoise']:

                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                    p.data.add_(-group['lr'],
                                0.5 * d_p + langevin_noise)
                else:
                    p.data.add_(-group['lr'], 0.5 * d_p)

        return loss


class pSGLD(Optimizer):
    """
    RMSprop preconditioned SGLD using pytorch rmsprop implementation.
    """

    def __init__(self, params, lr, weight_decay=0, sigma_prior=0, alpha=0.99, eps=1e-8, centered=False, addnoise=True):

        if (weight_decay == 0 or weight_decay is None) and sigma_prior!=0:
            weight_decay = 1 / (sigma_prior ** 2)

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, weight_decay=weight_decay, alpha=alpha, eps=eps, centered=centered, addnoise=addnoise)
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                state['step'] += 1

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # sqavg x alpha + (1-alph) sqavg *(elemwise) sqavg
                square_avg.mul_(alpha).addcmul_(1 - alpha, d_p, d_p)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, d_p)
                    avg = square_avg.cmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                #                 print(avg.shape)
                if group['addnoise']:
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                    p.data.add_(-group['lr'],
                                0.5 * d_p.div_(avg) + langevin_noise / torch.sqrt(avg))

                else:
                    p.data.addcdiv_(-group['lr'], 0.5 * d_p, avg)
        return loss



class BayesianAdam(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, sigma_prior=1, addnoise=True):
        
        # L2 regularization in terms of either weight_decay or sigma_prior
        if weight_decay == 0 or weight_decay is None:
            weight_decay = 1 / (sigma_prior ** 2)

        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        self.addnoise = addnoise

        super(BayesianAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('BayesianAdam does not support sparse gradients')
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Adding weight decay term
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1

                if self.addnoise:
                    langevin_noise = p.data.new(p.data.size()).normal_(mean=0, std=1) / np.sqrt(group['lr'])
                    p.data.add_(-step_size, 0.5 * exp_avg / denom + langevin_noise)
                else:
                    p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
