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

from Classes.utils.model_utils import resampleSystematic, regularizeAgentParticles


class EKF(object):
    
    DEFAULTS = {}   
    def __init__(self, params):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(EKF.DEFAULTS, **params_dict)

        self.params.update_all()

        self.F = np.array([[1, 0, self.T_between_timestep, 0                      , (self.T_between_timestep**2)/2, 0                             ],
                        [0, 1, 0                      , self.T_between_timestep, 0                             , (self.T_between_timestep**2)/2],
                        [0, 0, 1                      , 0                      , self.T_between_timestep       , 0                             ],
                        [0, 0, 0                      , 1                      , 0                             , self.T_between_timestep       ],
                        [0, 0, 0                      , 0                      , 1                             , 0                             ],
                        [0, 0, 0                      , 0                      , 0                             , 1                             ]])
        self.F = self.F[:self.number_state_variables, :self.number_state_variables]
        self.W = copy.copy(self.F)

        # Motion model
        motion_model = [self.std_noise_position**2 + (self.std_noise_position_motion_model)**2, 
                        self.std_noise_position**2 + (self.std_noise_position_motion_model)**2, 
                        self.std_noise_velocity**2 + (self.std_noise_velocity_motion_model)**2, 
                        self.std_noise_velocity**2 + (self.std_noise_velocity_motion_model)**2][:self.number_state_variables]
        self.motion_model = np.diag(motion_model)
        Q = self.W@self.motion_model@self.W.transpose()
        self.Q = Q[:self.number_state_variables, :self.number_state_variables]

        # Measurement model
        diag_var_gnss = np.array([self.std_position_x_gnss**2 + (self.std_noise_position_gnss_measurement_model)**2, 
                                self.std_position_x_gnss**2 + (self.std_noise_position_gnss_measurement_model)**2,
                                self.std_velocity_x_gnss**2 + (self.std_noise_velocity_gnss_measurement_model)**2,
                                self.std_velocity_x_gnss**2 + (self.std_noise_velocity_gnss_measurement_model)**2][:self.num_meas_gnss]).astype(float)
        self.diag_sigma_gnss = np.sqrt(diag_var_gnss)
        self.Cov_gnss = np.diag(diag_var_gnss)

        self.var_inter_agent_meas_model = self.std_x_inter_agent**2 + self.std_inter_agent_meas_model**2

        self.start_cooperation = 0
        

    ###########################################################################################
    # PREDICTION
    def prediction(self, beliefs, time_n):
    
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

        t = output_additional_data['t']
        x_gnss = output_additional_data['x_gnss']
        num_measurements_gnss = x_gnss.shape[-1]
        number_state_variables = self.number_state_variables

        # No message passing, only gnss
        for i in range(self.num_agents):
            beliefs[i] = copy.copy(prediction_message[i])

            # evaluate likelihood gnss measurements
            x_gnss_agent = np.array(x_gnss[i]).reshape([-1,1])[:num_measurements_gnss,:]
            H = np.eye(num_measurements_gnss)
            # diff = max(0, number_state_variables - num_measurements_gnss)
            G = beliefs[i][1]@H.transpose()@np.linalg.inv(H@beliefs[i][1]@H.transpose() + self.Cov_gnss)

            beliefs[i][0] = beliefs[i][0] + G@(x_gnss_agent - H@beliefs[i][0])
            beliefs[i][1] = beliefs[i][1] - G@H@beliefs[i][1]   

        return beliefs

    def update_particle_no_agent_coop(self, output_additional_data, prediction_message, beliefs, time_n):

        t = output_additional_data['t']
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
        t = output_additional_data['t']
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
































