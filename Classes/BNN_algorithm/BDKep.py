import sys
import os
import numpy as np
import torch
import importlib
import copy
import math

# Directories
cwd = os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(cwd)[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(cwd))


class BDKep(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(BDKep.DEFAULTS, **params_dict)

        self.params.BNN_algorithm_master = 'BDKep'
        self.BNN_algorithm_master = self.params.BNN_algorithm_master
        self.params.update_all()

        # Ensemble of Neural Networks used for prediction (Teacher)
        self.models_T = []
        self.models_T_info = {'epoch':[], 'batch_idx':[], 'saved_models_path': []}

        # Predictions of Teacher: first index -> Minibatch, second index -> Prediction 
        self.return_model_T_outputs_S_dataset = [[] for i in range(self.N_samples_param)]

        # Student batches
        self.input_S_dataset = []

        # Teacher params
        # self.params_T = getattr(importlib.import_module(params.package_params), 'PARAMS')()
        self.params_T = copy.deepcopy(self.params)
        self.params_T.role = 'Teacher'
        self.params_T.BNN_algorithm = 'SGLD'
        self.params_T.optimizer_name = self.params_T.optimizer_name_T
        self.params_T.sigma_prior = self.params_T.sigma_prior_T
        self.params_T.weight_decay = 1 / (self.params_T.sigma_prior ** 2)
        self.params_T.lr = self.params_T.lr_T
        self.params_T.number_outputs_per_output_feature = self.params.number_outputs_per_output_feature  

        BNN_algorithm = self.params_T.BNN_algorithm
        BNN_algorithm_dir = os.path.join(CLASSES_DIR, 'BNN_algorithm', BNN_algorithm + '.py')
        package_BNN_algorithm = BNN_algorithm_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        self.params_T.BNN_algorithm_dir = BNN_algorithm_dir
        self.params_T.package_BNN_algorithm = package_BNN_algorithm

        self.params_T.update_all()

        # Student params
        self.params_S = copy.deepcopy(self.params)
        self.params_S.role = 'Student'
        self.params_S.BNN_algorithm = 'NN'
        self.params_S.optimizer_name = self.params_S.optimizer_name_S
        self.params_S.sigma_prior = self.params_S.sigma_prior_S
        self.params_S.weight_decay = 1 / (self.params_S.sigma_prior ** 2)
        self.params_S.lr = self.params_S.lr_S
        self.params_S.number_outputs_per_output_feature = 3 # y, y_al, y_ep 

        BNN_algorithm = self.params_S.BNN_algorithm
        BNN_algorithm_dir = os.path.join(CLASSES_DIR, 'BNN_algorithm', BNN_algorithm + '.py')
        package_BNN_algorithm = BNN_algorithm_dir.split('Code')[1].replace(os.path.sep, '.')[1:-3]
        self.params_S.BNN_algorithm_dir = BNN_algorithm_dir
        self.params_S.package_BNN_algorithm = package_BNN_algorithm

        self.params_S.update_all()

        pass

    # Set model to train or eval
    def set_model_mode(self, train = 0):

        if train:
            self.model_T.train()
            self.model_S.train()
        else:
            self.model_T.eval()
            self.model_S.eval()

    # Set model to GPU
    def set_model_cuda(self, model = None, use_cuda = None):

        if use_cuda == None:
            use_cuda = self.use_cuda
            
        if model == None:
            if use_cuda:
                self.model_T.to("cuda:0", non_blocking=True)
                self.model_S.to("cuda:0", non_blocking=True)
            else:
                self.model_T.cpu()
                self.model_S.cpu()
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
        self.model_T.use_cuda = use_cuda
        self.model_S.use_cuda = use_cuda
        self.params_T.use_cuda = use_cuda
        self.params_S.use_cuda = use_cuda

    # Set model
    def set_model(self):

        self.model_T = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params_T)
        self.model_S = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params_S)

    # Load stored model
    def load_models(self, epoch = None, batch_id = None):
    
        # Load Student
        self.model_S = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params_S)
        cuda_device = 'cuda:0' if (self.params_S.use_cuda and self.use_cuda) else 'cpu'
        self.model_S.load_state_dict(torch.load(os.path.join(self.saved_models_dir,'{}_{}_model_S.pth'.format(epoch, batch_id)), map_location=cuda_device))
        print('loaded trained Student models (epoch: {} - batch_idx: {})..!'.format(epoch, batch_id))

        # Load Teacher
        self.models_T = []

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

        model_T = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params_T)
        for e in range(burnin_epochs, num_epochs-1, mix_epochs):
            cuda_device = 'cuda:0' if (self.params_T.use_cuda and self.use_cuda) else 'cpu'
            model_T.load_state_dict(torch.load(os.path.join(self.saved_models_dir,'{}_{}_model_T.pth'.format(e, self.model_save_step)), map_location=cuda_device))
            self.models_T.append(copy.deepcopy(model_T))
            self.set_model_cuda(self.models_T[-1])
            
        print('loaded trained Teacher models (epochs: {}:{}:{})..!'.format(burnin_epochs, mix_epochs, num_epochs-1))

    # Save trained model
    def save_models(self, epoch = None, batch_id = None):

        torch.save(self.model_T.state_dict(), os.path.join(self.saved_models_dir, '{}_{}_model_T.pth'.format(epoch+1, batch_id+1)))
        torch.save(self.model_S.state_dict(), os.path.join(self.saved_models_dir, '{}_{}_model_S.pth'.format(epoch+1, batch_id+1)))

        # Save models by sampling from the posterior
        if (epoch >= self.burnin_epochs) and (epoch % self.mix_epochs == 0):
            new_model_T = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params_T)
            state_dict = self.model_T.state_dict()
            new_model_T.load_state_dict(state_dict)
            self.models_T.append(copy.deepcopy(new_model_T))
            self.set_model_cuda(self.models_T[-1])
            self.models_T_info['epoch'].append(epoch)
            self.models_T_info['batch_idx'].append(batch_id)
            self.models_T_info['saved_models_path'].append(os.path.join(self.saved_models_dir, '{}_{}_model.pth'.format(epoch+1, batch_id+1)))

            if len(self.models_T) > self.N_samples_param:
                self.models_T.pop(0)
                self.models_T_info['epoch'].pop(0)
                self.models_T_info['batch_idx'].pop(0)
                self.models_T_info['saved_models_path'].pop(0)

    # Print network structure
    def print_network(self):

        num_params = 0
        for p in self.model_T.parameters():
            num_params += p.numel()
        print(self.model_name)
        print(self.model_T)
        print("The number of parameters: {}".format(num_params))

        num_params = 0
        for p in self.model_S.parameters():
            num_params += p.numel()
        print(self.model_name)
        print(self.model_S)
        print("The number of parameters: {}".format(num_params))

        self.model_S.print_MACs_FLOPs()

    # Set optimizer
    def set_optimizer(self):
         
        self.optimizer_T = getattr(importlib.import_module(self.package_optimizer), 'OPTIMIZER')(self.model_T.parameters(), self.params_T).optimizer
        self.optimizer_S = getattr(importlib.import_module(self.package_optimizer), 'OPTIMIZER')(self.model_S.parameters(), self.params_S).optimizer

    # Get current optimizer learning rate
    def get_optimizer_lr(self):

        lr_tmp = []

        for param_group in self.optimizer_T.param_groups:

            lr_tmp.append(param_group['lr'])

        tmplr = np.squeeze(np.array(lr_tmp))

        return tmplr
    

    # Forward or sample from posterior
    def forward_sample(self, input_all_data, epoch = None, batch_idx = None, output_all_additional_data = None, train = 0):
        
        x = input_all_data['input_data']
        output_additional_data = output_all_additional_data['output_additional_data']

        if train:
            x2 = input_all_data['input2_data']
            output2_additional_data = output_all_additional_data['output2_additional_data']

        # Teacher prediction
        return_model_T_output = self.model_T.forward(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = train, use_cuda = self.use_cuda)

        # Teacher and Student prediction on Student dataset
        # If we are in training, use Student dataset
        if train:

            # 1 Strategy
            # Student dataset is a noise dataset
            # random samples from a uniform distribution with a mean of 0 and a standard deviation of 1
            # a = 0 - (3 * 1) ** 0.5
            # b = 0 + (3 * 1) ** 0.5
            # x_noise = torch.rand_like(x) * (b - a) + a

            # 2 Strategy
            # x_noise = x

            # 3 Strategy
            x_noise = x2

            # Save Student batches and Teacher predictions
            with torch.no_grad():
                self.input_S_dataset.append(x_noise) 
                for idx, batch_x in enumerate(self.input_S_dataset):
                    return_model_T_output_S_dataset = self.model_T.forward(batch_x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = 0, use_cuda = self.use_cuda)
                    self.return_model_T_outputs_S_dataset[idx].append(return_model_T_output_S_dataset)

            # If we ahve completed the number of samples, i.e., prediction of the same batch, prepare for the loss function 
            num_batches_stored = len(self.return_model_T_outputs_S_dataset[0])
            if len(self.return_model_T_outputs_S_dataset[0]) >= self.N_samples_param:
                return_model_T_output_S_dataset = self.return_model_T_outputs_S_dataset.pop(0)
                x_noise = self.input_S_dataset.pop(0)
                self.return_model_T_outputs_S_dataset.append([])
            else:
                return_model_T_output_S_dataset = None
            
            # Student prediction
            return_model_S_output_S_dataset = self.model_S.forward(x_noise, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = 1, use_cuda = self.use_cuda)

        # If we are in validation, use Teacher dataset
        else:
            # Teacher prediction
            return_model_T_output_S_dataset = copy.deepcopy(return_model_T_output)
            # Student prediction
            return_model_S_output_S_dataset = self.model_S.forward(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = 0, use_cuda = self.use_cuda)

        return_model_output = {'return_model_T_output': return_model_T_output, 'return_model_T_output_S_dataset': return_model_T_output_S_dataset, 'return_model_S_output_S_dataset': return_model_S_output_S_dataset}

        return return_model_output

    # Sample
    def sample(self, x, N_samples_param = 1000, epoch = None, batch_idx = None, output_additional_data = None, train = 0):

        if len(self.models_T) == 0:
            new_model_T = getattr(importlib.import_module(self.package_model), 'MODEL')(self.params_T)
            state_dict = self.model_T.state_dict()
            new_model_T.load_state_dict(state_dict)
            self.models_T.append(copy.deepcopy(new_model_T))

            self.set_model_cuda(self.models_T[-1])
            
        for i in range(len(self.models_T)):
            return_model_output = self.models_T[i].forward(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = train, use_cuda = self.use_cuda)
            
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
        if epoch is None and len(self.models_T) == 0:
            self.load_models()

        # Teacher evaluate
        return_model_T_output = self.sample(x, N_samples_param = self.N_samples_param, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = 0)
        return_model_T_evaluate = self.model_T.evaluate(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = train, return_output = return_output, return_model_output = return_model_T_output)

        # Student evaluate
        return_model_S_output = self.model_S.forward(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = 0, use_cuda = self.use_cuda)
        return_model_S_evaluate = self.model_S.evaluate(x, epoch = epoch, batch_idx = batch_idx, output_additional_data = output_additional_data, train = train, return_output = return_output, return_model_output = return_model_S_output)

        # We are in the validation procedure of the training
        if not return_output:
            return_model_T_evaluate = {key + '_T': value for key, value in return_model_T_evaluate.items()}
            return_model_S_evaluate = {key + '_S': value for key, value in return_model_S_evaluate.items()}
            return_model_evaluate = {**return_model_T_evaluate, **return_model_S_evaluate}
        # We are in the testing procedure after training
        else:
            return_model_evaluate = return_model_S_evaluate

        return return_model_evaluate

    # Compute loss function
    def loss_function(self, input_all_data, return_model_output, output_all_additional_data = None):
    
        x = input_all_data['input_data']
        x2 = input_all_data['input2_data']
        output_additional_data = output_all_additional_data['output_additional_data']

        return_model_T_output = return_model_output['return_model_T_output']
        return_model_T_output_S_dataset = return_model_output['return_model_T_output_S_dataset']
        return_model_S_output_S_dataset = return_model_output['return_model_S_output_S_dataset']

        return_model_T_loss = self.model_T.loss_function(x, return_model_T_output, output_additional_data = output_additional_data)
        return_model_S_loss = self.model_S.loss_function(x2, return_model_S_output_S_dataset, output_additional_data = return_model_T_output_S_dataset)

        return_model_T_loss = {key + '_T': value for key, value in return_model_T_loss.items()}
        return_model_S_loss = {key + '_S': value for key, value in return_model_S_loss.items()}
        return_model_loss = {**return_model_T_loss, **return_model_S_loss}

        return return_model_loss
    
    # Backward pass
    def backward(self, return_model_loss):

        # Teacher backward
        self.model_T.zero_grad()
        self.optimizer_T.zero_grad()
        return_model_loss['total_loss_T'].backward()
        if self.bool_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model_T.parameters(), 5)
        self.optimizer_T.step()

        # Student backward
        self.model_S.zero_grad()
        self.optimizer_S.zero_grad()
        return_model_loss['total_loss_S'].backward()
        if self.bool_clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), 5)
        self.optimizer_S.step()

