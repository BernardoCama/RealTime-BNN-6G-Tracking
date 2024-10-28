from typing import Any
import math

class PARAMS(object): 

    def __init__(self, config = {}):

        self.DEFAULTS = {

                'seed': 2,

                # DATASET GENERAL
                'dataset_name':'Artificial_2D',
                'task': 'regression',
                'aleatoric': 'heteroscedastic',  # 'homoscedastic'
                'size_dataset': 100,
                'batch_size': 1,
                'num_agents': 1,   # same as batch_size

                # DATASET SPECIFIC
                'dimension_dataset': 2, 
                'limit_t1':[-10, 10],
                'limit_t2':[-10, 10],
                'limit_vt1':[-1, 1],
                'limit_vt2':[-1, 1],
                'limit_at1':[-0.1, 0.1],
                'limit_at2':[-0.1, 0.1],
                'density_t1_min': 0.1,  # [points/m]
                'density_t1_max': 40,   # [points/m]

                # REAL MEASUREMENTS (ALEATORIC)
                'comm_distance':1000,

                'std_x_inter_agent': 0,     # [m]

                'std_position_x_gnss': 0.1, # [m]
                'std_velocity_x_gnss': 0, 
                'std_acceleration_x_gnss': 0,  

                # REAL MOTION
                'noise_type':'Gaussian',
                'T_between_timestep':1,  # [s]

                'std_noise_position':0.1,
                'mean_velocity_t1':0.2,
                'mean_velocity_t2':-0.2,
                'std_noise_velocity':0,
                'mean_acceleration_t1':0,
                'mean_acceleration_t2':0,
                'std_noise_acceleration':0,

                'limit_behavior':'reflection',              # 'reflection', 'continue', 'none'
                'setting_trajectories': 'not_defined',       # 'star', 'spiral', 'not_defined'
                
                # DATASET BOOL
                'bool_shuffle':0,

                # STATE
                'number_state_variables': 2,              # t1, t2, vt1, vt2, at1, at2 ...
                'num_particles': 100,
                'Particle_filter': 0,

                # STATISTICS BOOL
                'bool_print_estimated_state': 1,

                # RIVEDERE
                'bool_validate_model':1,            # Validate the model
                'bool_return_train_accuracy_metrics':0,   # Compute accuracy metrics when performing training
                'bool_return_valid_loss_metrics':0,       # Compute loss metric when performing validation
                'bool_return_valid_accuracy_metrics':1,   # Compute accuracy metrics when performing validation
                'bool_plot_dataset':1,              # Plot dataset when created
                'bool_pretrained_model': None,      # Load pretrained model when start training
                'bool_print_network': 1,            # Print network composition
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