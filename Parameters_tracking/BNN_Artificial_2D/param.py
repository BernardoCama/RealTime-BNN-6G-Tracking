from typing import Any
import math

class PARAMS(object): 

    def __init__(self, config = {}):

        self.DEFAULTS = {

                'seed': 2,

                # DATASET GENERAL
                'dataset_name':'Artificial_2D',
                'task': 'regression',

                # TRACKING ALGORITHM         
                'Tracking_algorithm':'prediction_Classic_Update_BNN_Artificial_2D',

                # PREDICTION - MOTION MODEL
                'std_noise_position_motion_model': 0.5,     # the std on the motion model is the real one + std_noise_position_motion_model
                'std_noise_velocity_motion_model': 0,  # the std on the motion model is the real one + std_noise_velocity_motion_model

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