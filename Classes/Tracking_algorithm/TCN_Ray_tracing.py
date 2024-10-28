import sys
import os
import numpy as np
import copy
import torch
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

from Classes.utils.utils import from_xyz_to_latlonh_matrix, from_latlongh_to_xyz_matrix, rotate_axis



class TCN_Ray_tracing(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(TCN_Ray_tracing.DEFAULTS, **params_dict)

        self.params.update_all()        

    ###########################################################################################
    # PREDICTION -> Return original beliefs since the update part takes into account the motion of the target
    def prediction(self, beliefs, time_n):

        return beliefs
        
    ###########################################################################################
    # UPDATE
    # @lru_cache(maxsize=None)
    def update(self, input_data, output_additional_data, prediction_message, beliefs, time_n):

        output_size =self.solver_BNN.output_size
        num_BS_per_pos = int(output_additional_data['num_BS_per_pos'].numpy())

        # Select measurements of detected BSs
        pos_index = output_additional_data['pos_index'][:int(num_BS_per_pos), :].numpy()[0]
        print(f'Original pos index: {pos_index}')
        UExyz_global = output_additional_data['UExyz_global']
        UElatlonh = output_additional_data['UElatlonh']

        output_additional_data['UExyz_wrtcell'] = output_additional_data['UExyz_wrtcell'][:,:int(num_BS_per_pos), :]
        output_additional_data['BSxyz_wrtUE'] = output_additional_data['BSxyz_wrtUE'][:,:int(num_BS_per_pos), :]
        output_additional_data['BSxyz_global'] = output_additional_data['BSxyz_global'][:,:int(num_BS_per_pos), :]
        output_additional_data['BS_cell_angles'] = output_additional_data['BS_cell_angles'][:,:int(num_BS_per_pos), :]

        input_all_data = {'input_data': input_data}
        output_all_additional_data = {'output_additional_data': output_additional_data}

        # TCN prediction
        self.solver_BNN.BNN_algorithm_instance.set_model_mode(train=0)

        # Modify sequence length
        self.solver_BNN.BNN_algorithm_instance.model.input_size[0] = num_BS_per_pos
        with torch.no_grad():
            return_model_evaluate = self.solver_BNN.BNN_algorithm_instance.evaluate(input_all_data, output_all_additional_data = output_all_additional_data, train = 0, return_output=1, use_cuda = self.use_cuda)
        
        # for agent in range(self.num_agents):
        agent = 0

        # Prior
        beliefs[agent] = copy.copy(prediction_message[agent])
        mean1 = beliefs[agent][0]
        cov1 = beliefs[agent][1]

        y_mean = return_model_evaluate['y_mean'].reshape(-1, output_size)

        # Convert in xyz_global coordinate ref
        est_UExyz_wrtcell = y_mean 
        BS_cell_angles = output_additional_data['BS_cell_angles'].reshape(-1, 2)[num_BS_per_pos-1, :].numpy().reshape(-1, 2)
        est_UExyz_wrtBS = np.array([rotate_axis(est_UExyz_wrtcell[bs,:], az = BS_cell_angles[bs,0], el = BS_cell_angles[bs,1], roll = 0,  backtransformation=1) for bs in range(int(1))]).reshape(1, -1)
        est_UExyz_wrtUE = est_UExyz_wrtBS + output_additional_data['BSxyz_wrtUE'].reshape(int(num_BS_per_pos), -1)[num_BS_per_pos-1, :].numpy()
        est_UElatlonh = from_xyz_to_latlonh_matrix(est_UExyz_wrtUE, UElatlonh)
        est_UExyz_global = from_latlongh_to_xyz_matrix(est_UElatlonh, self.latlonh_reference)
        mean2 = est_UExyz_global

        beliefs[agent][0] = mean2.reshape(output_size, -1)
        beliefs[agent][1] = cov1

        print('Prediction mean: a {} {}'.format(agent, mean2.squeeze().tolist()))

        return beliefs

    ###########################################################################################
    # POSITION ESTIMATION
    def estimate_position(self, beliefs):
        
        if not self.Particle_filter:

            return self.estimate_position_no_particle(beliefs)

        else:

            return self.estimate_position_particle(beliefs)

    def estimate_position_particle(self, beliefs):
         
        predicted_t = {agent:[] for agent in range(self.num_agents)}
        predicted_C = {agent:[] for agent in range(self.num_agents)}

        for agent in range(self.num_agents):

            weights = (beliefs[agent][1]/np.sum(beliefs[agent][1])).transpose()

            prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*beliefs[agent][0]

            x_state_mean = np.sum(prod_particles_weights,1)/np.sum(beliefs[agent][1])

            C = np.cov(beliefs[agent][0]-x_state_mean.reshape([-1,1]))

            predicted_t[agent].append(np.array(x_state_mean))

            predicted_C[agent].append(C)

        return predicted_t, predicted_C

    def estimate_position_no_particle(self, beliefs):

        predicted_t = {agent:[] for agent in range(self.num_agents)}
        predicted_C = {agent:[] for agent in range(self.num_agents)}

        for agent in range(self.num_agents):

            x_state_mean = beliefs[agent][0]

            C = beliefs[agent][1]

            predicted_t[agent].append(np.array(x_state_mean))

            predicted_C[agent].append(C)

        return predicted_t, predicted_C


    ###########################################################################################
    # AUXILIARY
    # Set BNN model for update
    def set_model(self, solver_BNN):

        self.solver_BNN = solver_BNN
































