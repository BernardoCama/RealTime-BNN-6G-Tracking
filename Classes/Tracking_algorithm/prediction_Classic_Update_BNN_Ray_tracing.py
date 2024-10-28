import sys
import os
import numpy as np
import copy
import torch

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

from Classes.utils.model_utils import product_of_Gaussians
from Classes.utils.utils import from_xyz_to_latlonh_matrix, from_latlongh_to_xyz_matrix, rotate_axis



class prediction_Classic_Update_BNN_Ray_tracing(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(prediction_Classic_Update_BNN_Ray_tracing.DEFAULTS, **params_dict)

        self.params.update_all()

        self.F = np.array([[1, 0, 0                      , self.T_between_timestep, 0                             , 0                             , (self.T_between_timestep**2)/2,                              0,                              0],
                           [0, 1, 0                      , 0                      , self.T_between_timestep       , 0                             , 0                             , (self.T_between_timestep**2)/2,                              0],                           
                           [0, 0, 1                      , 0                      , 0                             , self.T_between_timestep       , 0                             , 0                             , (self.T_between_timestep**2)/2],
                           
                           [0, 0, 0                      , 1                      , 0                             , 0                             , self.T_between_timestep       , 0                             ,                              0],
                           [0, 0, 0                      , 0                      , 1                             , 0                             , 0                             , self.T_between_timestep       ,                              0],
                           [0, 0, 0                      , 0                      , 0                             , 1                             , 0                             , 0                             ,        self.T_between_timestep],

                           [0, 0, 0                      , 0                      , 0                             , 0                             , 1                             , 0                             ,                              0],
                           [0, 0, 0                      , 0                      , 0                             , 0                             , 0                             , 1                             ,                              0],
                           [0, 0, 0                      , 0                      , 0                             , 0                             , 0                             , 0                             ,                              1]])
        self.F = self.F[:self.number_state_variables, :self.number_state_variables]
        self.W = copy.copy(self.F)

        # Motion model
        motion_model = [(self.std_noise_position_motion_model)**2, 
                        (self.std_noise_position_motion_model)**2, 
                        (self.std_noise_position_motion_model)**2, 
                        (self.std_noise_velocity_motion_model)**2, 
                        (self.std_noise_velocity_motion_model)**2, 
                        (self.std_noise_velocity_motion_model)**2][:self.number_state_variables]
        self.motion_model = np.diag(motion_model)
        Q = self.W@self.motion_model@self.W.transpose()
        self.Q = Q[:self.number_state_variables, :self.number_state_variables]

        self.start_cooperation = 0
        

    ###########################################################################################
    # PREDICTION
    def prediction(self, beliefs, time_n):

        if time_n == 0:
            return beliefs
        
        if not self.Particle_filter:

            return self.prediction_no_particle(beliefs, time_n)
        
        else:

            return self.prediction_particle(beliefs, time_n)

    def prediction_no_particle(self, beliefs, time_n):

        prediction_message = {agent:0 for agent in range(self.num_agents)}

        for agent in range(self.num_agents):

            prediction_message[agent] = [self.F@beliefs[agent][0], 
                                              self.Q + self.F@beliefs[agent][1]@self.F.transpose()]
            
            if self.bool_print_estimated_state:

                print('Prediction mean: a {} {}'.format(agent, prediction_message[agent][0].squeeze().tolist()))

                print('Prediction var: a {}\n{}'.format(agent, prediction_message[agent][1]))

        return prediction_message

    def prediction_particle(self, beliefs, time_n):

        prediction_message = copy.deepcopy(beliefs)

        for agent in range(self.num_agents):

            # Add noise
            for i in range(self.self.number_state_variables):

                prediction_message[agent][0][i] = beliefs[agent][0][i] + np.sqrt(self.motion_model[i][i])*np.random.normal(size=(1, self.num_particles)) 
            
            # Motion prediction
            prediction_message[agent][0] = self.F@prediction_message[agent][0]

            prediction_message[agent][1] = (np.ones((1,self.num_particles))/self.num_particles)

            if agent == 0 and self.log_BP:
                weights = (prediction_message[agent][1]/np.sum(prediction_message[agent][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*prediction_message[agent][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(prediction_message[agent][1])
                mean = x_state_mean[:self.self.number_state_variables]
                print('Prediction mean: a {} {}'.format(agent, mean.squeeze().tolist()))

            return prediction_message
        

    ###########################################################################################
    # UPDATE
    # @lru_cache(maxsize=None)
    def update(self, input_data, output_additional_data, prediction_message, beliefs, time_n):

        output_size =self.solver_BNN.output_size
        num_BS_per_pos = int(output_additional_data['num_BS_per_pos'].numpy())


        # (num_meas (batch size) x dimension_input) ********************************************
        # Reshape such that the batch size are the measurements of all BSs
        input_data = input_data.permute(1, 0, 2, 3)[:int(num_BS_per_pos), :, :, :].squeeze().view(-1, self.input_size[0], self.input_size[1])

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

        # BNN prediction
        self.solver_BNN.BNN_algorithm_instance.set_model_mode(train=0)
        with torch.no_grad():
            return_model_evaluate = self.solver_BNN.BNN_algorithm_instance.evaluate(input_all_data, output_all_additional_data = output_all_additional_data, train = 0, return_output=1, use_cuda = self.use_cuda)
        
        # for agent in range(self.num_agents):
        agent = 0

        # Prior
        beliefs[agent] = copy.copy(prediction_message[agent])
        mean1 = beliefs[agent][0]
        cov1 = beliefs[agent][1]

        # Likelihood
        y_mean = return_model_evaluate['y_mean'].reshape(-1, output_size)
        total_unc_cov = return_model_evaluate['total_unc_cov'].reshape(-1, output_size, output_size)
        mean2 = y_mean
        cov2 = total_unc_cov 

        # Convert in xyz_global coordinate ref
        est_UExyz_wrtcell = mean2
        BS_cell_angles = output_additional_data['BS_cell_angles'].reshape(-1, 2).numpy()
        est_UExyz_wrtBS = np.array([rotate_axis(est_UExyz_wrtcell[bs,:], az = BS_cell_angles[bs,0], el = BS_cell_angles[bs,1], roll = 0,  backtransformation=1) for bs in range(int(num_BS_per_pos))]).reshape(int(num_BS_per_pos), -1)
        est_UExyz_wrtUE = est_UExyz_wrtBS + output_additional_data['BSxyz_wrtUE'].reshape(int(num_BS_per_pos), -1).numpy()
        est_UElatlonh = from_xyz_to_latlonh_matrix(est_UExyz_wrtUE, UElatlonh)
        est_UExyz_global = from_latlongh_to_xyz_matrix(est_UElatlonh, self.latlonh_reference)
        mean2 = est_UExyz_global

        # Product of likelihoods
        mean2_prod, cov2_prod = product_of_Gaussians(mean2, cov2)

        # Compute the inverse of the sum of the covariance matrices
        inv_sum_cov = np.linalg.inv(cov1 + cov2_prod)

        # Compute the mean and covariance of the product distribution
        mean_product = np.dot(cov2_prod, np.dot(inv_sum_cov, mean1)) + np.dot(cov1, np.dot(inv_sum_cov, mean2_prod))
        cov_product = np.dot(cov1, np.dot(inv_sum_cov, cov2_prod))

        # Retain the velocity components
        mean_product[output_size:] = mean1[output_size:]
        cov_product[output_size:, output_size:] = cov1[output_size:, output_size:]

        # Update posterior
        beliefs[agent][0] = mean_product
        beliefs[agent][1] = cov_product

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
































