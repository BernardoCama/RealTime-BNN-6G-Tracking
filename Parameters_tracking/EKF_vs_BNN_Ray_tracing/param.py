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
                'size_dataset': 100,
                'batch_size': 1,
                'num_agents': 1,   # same as batch_size
                'DB_name': 'Dataset_Positioning',
                'load_dataset': 1,                  # 1 to load the already processed dataset
                'save_dataset': 0, 

                # DATASET SPECIFIC
                'dimension_dataset': 3, 
                'limit_t1':[-10, 10],
                'limit_t2':[-10, 10],
                'limit_vt1':[-1, 1],
                'limit_vt2':[-1, 1],
                'limit_at1':[-0.1, 0.1],
                'limit_at2':[-0.1, 0.1],
                'num_OFDM_symbols_per_pos': 36,
                'Ng': 352,  # 352, 416
                'RXants': 64,
                'latlonh_reference': [42.3630833, -71.0923222, 25],

                # REAL MEASUREMENTS -> FROM 5G SIMULATION

                # REAL MOTION -> FROM SUMO
                'T_between_timestep':1,  # [s]
                
                # DATASET BOOL
                'bool_shuffle':0,

                # STATE
                'number_state_variables': 3,              # t1, t2, t3, vt1, vt2, vt3, at1, at2, at3 ...
                'num_particles': 100,
                'Particle_filter': 0,

                # STATISTICS BOOL
                'bool_print_estimated_state': 1,

                # RIVEDERE
                'bool_validate_model':1,            # Validate the model
                'bool_return_train_accuracy_metrics':0,   # Compute accuracy metrics when performing training
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