import sys
import os
import numpy as np
import copy

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

from Classes.utils.model_utils import resampleSystematic, regularizeAgentParticles, compute_H, compute_h
from Classes.utils.utils import from_xyz_to_latlonh_matrix, from_latlongh_to_xyz_matrix


class EKF_Ray_tracing(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(EKF_Ray_tracing.DEFAULTS, **params_dict)

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

        # Measurement model
        self.var_TOA_meas_model = self.std_TOA_meas_model**2

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
        if not self.Particle_filter:
            return self.update_no_particle(output_additional_data, prediction_message, beliefs, time_n)
        else:
            # non-coop between agents
            return self.update_particle_no_agent_coop(output_additional_data, prediction_message, beliefs, time_n)
            # coop between agents
            # return self.update_particle_agent_coop(output_additional_data, prediction_message, beliefs, time_n)

    def update_no_particle(self, output_additional_data, prediction_message, beliefs, time_n):

        number_state_variables = self.number_state_variables    
        pos_index = output_additional_data['pos_index'].numpy()[0]
        num_BS_per_pos = int(output_additional_data['num_BS_per_pos'])
        num_BS_LOS = int(np.sum(output_additional_data['BS_LOS'][:,:int(num_BS_per_pos)].numpy()))

        # Consider only LOS
        num_BS_per_pos = min(num_BS_LOS, num_BS_per_pos)
        print(f'Original pos index: {pos_index}')

        UExyz_global = output_additional_data['UExyz_global']
        UElatlonh = output_additional_data['UElatlonh']

        BSxyz_wrtUE = output_additional_data['BSxyz_wrtUE'][:,:int(num_BS_per_pos), :].numpy()
        BSxyz_global = output_additional_data['BSxyz_global'][:,:int(num_BS_per_pos), :].numpy()

        true_ToA = output_additional_data['true_ToA'][:,:int(num_BS_per_pos)].numpy().reshape(-1, 1)
        est_ToA = output_additional_data['est_ToA'][:,:int(num_BS_per_pos)].numpy().reshape(-1, 1)

        true_AoA_wrtBS = output_additional_data['true_AoA_wrtBS'][:,:int(num_BS_per_pos),:].numpy().reshape(-1, 2)
        est_AoA_wrtBS = output_additional_data['est_AoA_wrtBS'][:,:int(num_BS_per_pos),:].numpy().reshape(-1, 2)

        # Wrt UE
        est_AoA_wrtBS[:,0] = np.mod(-est_AoA_wrtBS[:,0], 360)
        est_AoA_wrtBS[:,1] = - est_AoA_wrtBS[:,1]
        est_AoA_wrtBS_deg = est_AoA_wrtBS
        # Rad
        est_AoA_wrtBS_rad = np.deg2rad(est_AoA_wrtBS_deg)

        # GNSS measure
        est_GNSS_xyx_wrtUE = (np.random.normal(0, 1, 3)*1.5).reshape(3, 1)

        dim = self.dimension_dataset
        
        for i in range(self.num_agents):
            beliefs[i] = copy.copy(prediction_message[i])

            est_UElatlonh = from_xyz_to_latlonh_matrix(beliefs[i][0], self.latlonh_reference).reshape(3, 1)
            est_UExyx_wrtUE = from_latlongh_to_xyz_matrix(est_UElatlonh, UElatlonh).reshape(3, 1)

            H_toa = compute_H(est_UExyx_wrtUE, BSxyz_wrtUE, dim, method='toa')
            h_toa = compute_h(est_UExyx_wrtUE, BSxyz_wrtUE, dim, method='toa')
            
            H_aod = compute_H(est_UExyx_wrtUE, BSxyz_wrtUE, dim, method='aod')
            h_aod_deg = compute_h(est_UExyx_wrtUE, BSxyz_wrtUE, dim, method='aod')
            # Rad
            h_aod_rad = h_aod_deg
            h_aod_rad[:,0] = np.mod(h_aod_rad[:,0], 360)
            h_aod_rad = np.deg2rad(h_aod_rad)


            H_gnss = np.eye(3)
            h_gnss = est_UExyx_wrtUE.reshape(3, 1)

            # With only ToA
            # H = H_toa
            # h = h_toa
            # With ToA and GNSS
            H = np.concatenate((H_toa, H_gnss), 0)
            h = np.concatenate((h_toa, h_gnss.reshape(-1, 1)), 0)
            # With ToA and AoA
            # H = np.concatenate((H_toa, H_aod), 0)
            # h = np.concatenate((h_toa, h_aod_rad.reshape(-1, 1)), 0)

            # Measurements
            # With ToA and GNSS
            est = np.concatenate((est_ToA, est_GNSS_xyx_wrtUE.reshape(-1,1)), 0)
            # With ToA and AoA
            # est = np.concatenate((est_ToA, est_AoA_wrtBS_rad.reshape(-1,1)), 0)

            C = H@beliefs[i][1]@H.transpose()
            damp_factor = 0.01;
            if np.linalg.cond(C) <= 1e-4:
                C = C + damp_factor * np.eye(C.shape[0])

            # With only ToA
            # G = beliefs[i][1]@H.transpose()@np.linalg.inv(C + self.var_TOA_meas_model)
            # With ToA and GNSS
            R = np.diag(
                np.concatenate(
                    (np.ones((1, num_BS_per_pos))*self.var_TOA_meas_model, (np.ones((1, 3))*3))
                     , 1).squeeze(), 
                )
            G = beliefs[i][1]@H.transpose()@np.linalg.inv(C + R)
            # With ToA and AoA
            # R = np.diag(
            #     np.concatenate(
            #         (np.ones((1, num_BS_per_pos))*self.var_TOA_meas_model, (np.ones((1, num_BS_per_pos*2))*10/180*np.pi))
            #          , 1).squeeze(), 
            #     )
            # G = beliefs[i][1]@H.transpose()@np.linalg.inv(C + R)

            # With only ToA
            # est_UExyx_wrtUE = est_UExyx_wrtUE + G@(est_ToA - h)
            # With ToA and GNSS
            est_UExyx_wrtUE = est_UExyx_wrtUE + G@(est - h)
            # With ToA and AoA
            # est_UExyx_wrtUE = est_UExyx_wrtUE + G@(est - h)
            est_UElatlonh = from_xyz_to_latlonh_matrix(est_UExyx_wrtUE, UElatlonh).reshape(3, 1)
            beliefs[i][0] = from_latlongh_to_xyz_matrix(est_UElatlonh, self.latlonh_reference).reshape(3, 1)
            beliefs[i][1] = beliefs[i][1] - G@H@beliefs[i][1]   

        return beliefs

    def update_particle_no_agent_coop(self, output_additional_data, prediction_message, beliefs, time_n):

        UExyz_global = output_additional_data['UExyz_global']
        x_gnss = output_additional_data['x_gnss']
        num_measurements_gnss = x_gnss.shape[-1]

        measurement_gnss_message = {agent:np.ones((1,self.num_particles)) for agent in range(self.num_agents)}

        for t in range(self.T_message_steps):
            for i in range(self.num_agents): # rx

                # evaluate likelihood gnss measurements
                x_gnss_agent = np.array(x_gnss[i]).reshape([-1,1])
                likelihood_gnss = self.compute_likelihood('GNSS', x_gnss_agent - beliefs[i][0][:self.num_measurements_gnss])
                if np.sum(likelihood_gnss) < 10^-20:
                    likelihood_gnss = np.ones((1,self.num_particles))
                    raise('State meas not consistent')
                measurement_gnss_message[i] = likelihood_gnss

                # Strategy: only gnss
                beliefs[i][1] = beliefs[i][1] * measurement_gnss_message[i]
                beliefs[i][1] = beliefs[i][1]/np.sum(beliefs[i][1])

                if i == 0 and self.log_BP:
                    weights = (beliefs[i][1]/np.sum(beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(beliefs[i][1])
                    mean = x_state_mean[:self.self.number_state_variables]
                    print('Update1 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

            # resampling
            for i in range(self.num_agents):

                beliefs[i][0][:,:] = beliefs[i][0][:,resampleSystematic(beliefs[i][1], self.num_particles).astype(int).squeeze()]
                beliefs[i][1] = np.array(np.ones((1,self.num_particles))/self.num_particles)

                if i == 0 and self.log_BP and self.log_BP:
                    weights = (beliefs[i][1]/np.sum(beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(beliefs[i][1])
                    mean = x_state_mean[:self.self.number_state_variables]
                    print('Update2 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

            # regularize
            for i in range(self.num_agents):
                beliefs[i][0][:,:] = regularizeAgentParticles(beliefs[i][0][:,:])

                if i == 0 and self.log_BP:
                    weights = (beliefs[i][1]/np.sum(beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(beliefs[i][1])
                    mean = x_state_mean[:self.self.number_state_variables]
                    print('Update3 mean: t {} a {} {}\n'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.num_agents): # rx
            beliefs[i][1] = beliefs[i][1]/np.sum(beliefs[i][1])

        return beliefs

    # TEMP_BELIEFS are in LOG 
    def update_particle_agent_coop(self, output_additional_data, prediction_message, beliefs, time_n):
        UExyz_global = output_additional_data['UExyz_global']
        x_gnss = output_additional_data['x_gnss']
        x_inter_agent = output_additional_data['x_inter_agent']
        connectivity_matrix = output_additional_data['connectivity_matrix']
        num_measurements_gnss = x_gnss.shape[-1]

        measurement_gnss_message = {agent:np.ones((1,self.num_particles)) for agent in range(self.num_agents)}
        measurement_inter_agent_message = [ [ 0 for j in range(self.num_agents)] for i in range(self.num_agents)]

        for t in range(self.T_message_steps):
            if t == 0:
                self.temp_beliefs = copy.deepcopy(beliefs)

            for i in range(self.num_agents): # rx

                real_pos = t[i]

                # evaluate likelihood gnss measurements
                x_gnss_agent = np.array(x_gnss[i]).reshape([-1,1])
                likelihood_gnss = self.compute_likelihood('GNSS', x_gnss_agent - beliefs[i][0][:self.num_measurements_gnss])
                if np.sum(likelihood_gnss) < 10^-20:
                    likelihood_gnss = np.ones((1,self.num_particles))
                    raise('State meas not consistent')
                measurement_gnss_message[i] = likelihood_gnss   

                # Avoid computation of inter-agent measurements
                for j in range(self.num_agents): # tx
                    if connectivity_matrix[i][j]:

                        # evaluate the range between corresponding agents' particles
                        delta_x = beliefs[i][0][0] - beliefs[j][0][0]
                        delta_y = beliefs[i][0][1] - beliefs[j][0][1]
                        ranges_ij = np.sqrt(delta_x**2 + delta_y**2)
                        likelihood_range = self.compute_likelihood('INTER-AGENT', ranges_ij-x_inter_agent[i][j])
                        if np.sum(likelihood_range) < 10^-50:
                            likelihood_range = np.ones((1,self.num_particles))
                            raise('Direct RANGE not consistent')
                        measurement_inter_agent_message[i][j] = likelihood_range.reshape([1,-1])

                        # visualize likelihood
                        # plot_particles(beliefs[i][0][0,:].reshape([-1,]), beliefs[i][0][1,:].reshape([-1,]), measurement_inter_agent_message[i][j].reshape([-1,]))

                # Start with inter-agent measurements after few seconds
                # Strategy: inter-meas + gnss
                if self.start_cooperation or (time_n > 0):# and np.linalg.norm(real_pos-gnss[0:2]) < 2):
                    self.temp_beliefs[i][1] = np.log(beliefs[i][1]) + np.sum([np.log(measurement_inter_agent_message[i][j]) for j in range(self.num_agents) if connectivity_matrix[i][j]],0) + np.log(measurement_gnss_message[i])
                    self.start_cooperation = 1
                # Strategy: only gnss
                else:
                    self.temp_beliefs[i][1] = np.log(beliefs[i][1]) + np.log(measurement_gnss_message[i])

                # scale first
                self.temp_beliefs[i][1] = self.temp_beliefs[i][1] - np.max(self.temp_beliefs[i][1])
                self.temp_beliefs[i][1] = self.temp_beliefs[i][1] - np.log(np.sum(np.exp(self.temp_beliefs[i][1])))

                if i == 0 and self.log_BP:
                    weights = (beliefs[i][1]/np.sum(beliefs[i][1])).transpose()
                    prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*beliefs[i][0]
                    x_state_mean = np.sum(prod_particles_weights,1)/np.sum(beliefs[i][1])
                    mean = x_state_mean[:self.self.number_state_variables]
                    # print('Update1 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.num_agents): # rx
            # print('Time n: {}, a: {}, beliefs_max {}, beliefs_min {}'.format(time_n, i, np.max(self.temp_beliefs[i][1]), np.min(self.temp_beliefs[i][1])))
            self.temp_beliefs[i][1] = self.temp_beliefs[i][1] - np.max(self.temp_beliefs[i][1])
            beliefs[i][1] = np.exp(self.temp_beliefs[i][1] - np.log(np.sum(np.exp(self.temp_beliefs[i][1]))))

        # resampling
        for i in range(self.num_agents):

            beliefs[i][0][:,:] = beliefs[i][0][:,resampleSystematic(beliefs[i][1], self.num_particles).astype(int).squeeze()]
            beliefs[i][1] = np.array(np.ones((1,self.num_particles))/self.num_particles)

            if i == 0 and self.log_BP and self.log_BP:
                weights = (beliefs[i][1]/np.sum(beliefs[i][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*beliefs[i][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(beliefs[i][1])
                mean = x_state_mean[:self.self.number_state_variables]
                # print('Update2 mean: t {} a {} {}'.format(t, i, mean.squeeze().tolist()))

        # regularize
        for i in range(self.num_agents):
            beliefs[i][0][:,:] = regularizeAgentParticles(beliefs[i][0][:,:])

            if i == 0 and self.log_BP:
                weights = (beliefs[i][1]/np.sum(beliefs[i][1])).transpose()
                prod_particles_weights = np.repeat(weights, self.self.number_state_variables, 1).transpose()*beliefs[i][0]
                x_state_mean = np.sum(prod_particles_weights,1)/np.sum(beliefs[i][1])
                mean = x_state_mean[:self.self.number_state_variables]
                print('Update3 mean: t {} a {} {}\n'.format(t, i, mean.squeeze().tolist()))

        for i in range(self.num_agents): # rx
            beliefs[i][1] = beliefs[i][1]/np.sum(beliefs[i][1])


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
    # Print network structure
    def print_network(self):

        print('Measurement model:')
        print('R_GNSS = {}'.format(self.Cov_gnss))
        # print('R_TOA = {}'.format(self.var_TOA))

    def compute_likelihood(self, type_, argument):
        if type_ == 'INTER-AGENT':
            return np.exp(-0.5*(argument/np.sqrt(self.var_inter_agent_meas_model))**2)/np.sqrt(2*np.pi*self.var_inter_agent_meas_model)
        if type_ == 'TOA':
            return np.exp(-0.5*(argument/np.sqrt(self.var_toa_meas_model))**2)/np.sqrt(2*np.pi*self.var_toa_meas_model)
        elif type_ == 'GNSS':
            return (1 / np.sqrt((2 * np.pi) ** self.Cov_gnss.shape[0] * np.linalg.det(self.Cov_gnss))) * \
                        np.exp(-0.5 * np.sum((argument) ** 2 * np.tile(self.diag_sigma_gnss.T ** (-2), (self.num_particles, 1)).T, axis=0))
































