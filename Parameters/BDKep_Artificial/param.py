from typing import Any
import math

class PARAMS(object):

    def __init__(self, config = {}):

        self.DEFAULTS = {

                'seed': 2,

                # DATASET GENERAL
                'dataset_name':'Artificial',
                'task': 'regression',
                'aleatoric': 'heteroscedastic',  # 'homoscedastic'
                'size_dataset': 400,
                'size_train_dataset':250,
                'size_valid_dataset':150,
                'batch_size': 250,

                # DATASET SPECIFIC
                'lengthscale':1,
                'variance':1, 
                'sig_noise': 0.3,

                # DATASET BOOL
                'bool_shuffle':0,

                # MODEL DEFINITION
                'model_name':'MLP_simple',
                'input_size': 1,
                'output_size': 1,
                'number_outputs_per_output_feature': 2,         # 1 y, 2 y_al (aleatoric), 3 y_ep (epistemic)
                'neurons_per_layer': [200],

                # OPTIMIZER
                'optimizer_name_T': 'SGLD',
                'optimizer_name_S': 'Adam',
                'sigma_prior_T': math.sqrt(1/2),                 # sigma_prior = sqrt(1/weight_decay) Teacher
                'sigma_prior_S': math.sqrt(1/2)*10,              # sigma_prior = sqrt(1/weight_decay) Student
                'lr_T': 1e-4,                                    # Learning rate Teacher
                'lr_S': 1e-2,                                    # Learning rate Student
                'bool_clip_grad_norm': 0,

                # BNN ALGORITHM
                'bool_use_BNN':1,                 # Use a BNN algorithm
                'BNN_algorithm':'BDKep',
                'var_xi_S':1,                     # Variance of the noise of the epistemic uncertainty prediction

                'N_samples_param': 50,            # Number networks of ensambles - Number of samples from the posterior (validation)
                'mix_epochs': 100,
                'burnin_epochs': 1000,

                # TRAINING PARAMETERS
                'num_epochs' : 100*50 + 1000 + 1,   # num_epochs = mix_epochs * N_samples_param + burnin_epochs + 1

                # TRAINING - STATISTICS BOOL
                'bool_validate_model':0,            # Validate the model
                'bool_return_train_accuracy_metrics':0,   # Compute accuracy metrics when performing training
                'bool_return_valid_loss_metrics':0,       # Compute loss metric when performing validation
                'bool_return_valid_accuracy_metrics':0,   # Compute accuracy metrics when performing validation
                'bool_plot_dataset':0,              # Plot dataset when created
                'bool_pretrained_model': None,      # Load pretrained model when start training
                'bool_print_network': 1,            # Print network composition
                'bool_save_training_info': 1,       # Save training statistics
                'bool_plot_training':0,             # Plot the training statistics during training
        }   

        self.DEFAULTS.update(config)
        self.__dict__.update(self.DEFAULTS, **config)

    def __call__(self, config = {}, *args: Any, **kwds: Any) -> Any:
        self.DEFAULTS.update(config)
        self.__dict__.update(self.DEFAULTS, **config)

    def update_all(self):
        # Update self.DEFAULTS with every self.variable_name
        for key, value in self.__dict__.items():
            if key != 'DEFAULTS':
                self.DEFAULTS[key] = value

        # Update self with all the variables in self.DEFAULTS
        self.__dict__.update(self.DEFAULTS)