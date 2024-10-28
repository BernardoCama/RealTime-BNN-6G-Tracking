from typing import Any
import math

class PARAMS(object): 

    def __init__(self, config = {}):

        self.DEFAULTS = {

                'seed': 2,

                # DATASET GENERAL
                'dataset_name':'Ray_tracing',
                'task': 'regression',
                'aleatoric': 'heteroscedastic',  # 'homoscedastic'
                'batch_size': 128,
                'DB_name': 'Dataset_Positioning',
                'load_dataset': 1,                  # 1 to load the already processed dataset
                'save_dataset': 0, 
                'recompute_statistics':0,
                'latlonh_reference': [42.3630833, -71.0923222, 25],

                # DATASET SPECIFIC
                'num_OFDM_symbols_per_pos': 36,
                'Ng': 352,
                'RXants': 64,

                # DATASET BOOL
                'bool_shuffle':1,

                # MODEL DEFINITION
                'model_name':'AEDL',
                'in_channels': 1,
                'out_channels': 1,
                'input_size': [352, 64],
                'output_size': 3,
                'number_outputs_per_output_feature': 2,         # 1 y, 2 y_al (aleatoric), 3 y_ep (epistemic)
                'neurons_per_layer': [16, 32, 64, 128, 256, 512, 256, 128, 64, 32, 16],
                'latent_dim': 8,
                'lamda_rec_err': 0.1, 

                # OPTIMIZER
                'optimizer_name': 'SGLD',
                'weight_decay': 0,#2                # weight_decay = 1 / (sigma_prior ** 2)
                'sigma_prior': 0,             # sigma_prior = sqrt(1/weight_decay)
                'lr': 1e-4,                       
                'bool_clip_grad_norm': 0,

                # BNN ALGORITHM
                'bool_use_BNN':1,                 # Use a BNN algorithm
                'BNN_algorithm':'SGLD',

                'N_samples_param': 10,            # Number networks of ensambles - Number of samples from the posterior (validation)
                'mix_epochs': 1,
                'burnin_epochs': 1,

                # TRAINING PARAMETERS
                'num_epochs' : 600,  #1*1 + 1 + 1,   # num_epochs = mix_epochs * N_samples_param + burnin_epochs + 1

                # TRAINING - STATISTICS BOOL
                'bool_validate_model':1,            # Validate the model
                'bool_return_train_accuracy_metrics':1,   # Compute accuracy metrics when performing training
                'bool_return_valid_loss_metrics':0,       # Compute loss metric when performing validation
                'bool_return_valid_accuracy_metrics':1,   # Compute accuracy metrics when performing validation
                'bool_plot_dataset':0,              # Plot dataset when created
                'bool_pretrained_model': None,      # Load pretrained model when start training
                'bool_print_network': 0,            # Print network composition
                'bool_save_training_info': 1,       # Save training statistics
                'bool_plot_training':1,             # Plot the training statistics during training
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