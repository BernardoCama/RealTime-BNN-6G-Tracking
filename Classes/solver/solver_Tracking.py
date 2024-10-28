import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("error")
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
from Classes.utils.utils import return_numpy
from Classes.utils.model_utils import compute_metrics_positioning

class Solver_Tracking(object):
    
    DEFAULTS = {}   
    def __init__(self, params, dataset_instance):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(Solver_Tracking.DEFAULTS, **params_dict)

    def create_beliefs(self, first_true_position):
    
        num_particles = self.num_particles
        number_state_variables = self.number_state_variables
        dimension_dataset = self.dimension_dataset
        beliefs = {}
        if self.Particle_filter:
            # beliefs[agent] = [particles, weights], particles = [x, y, vx, vy, ax, ay]

            for agent in range(self.num_agents):
                beliefs[agent] = [[[] for i in range(number_state_variables)],[[] for i in range(number_state_variables)]]
            
                noise_angle = (np.random.uniform(-0.5, 0.5 ,(1, num_particles)))*2.*np.pi
                noise_range = np.random.uniform(0,1,(1, num_particles)) 
                try:
                    pos = np.array(first_true_position[agent][0][:dimension_dataset]).reshape(-1,1) + 5*np.random.normal(size=(dimension_dataset, num_particles))
                except:
                    pass
                beliefs[agent][0][0] = pos[0]
                beliefs[agent][0][1] = pos[1]
                if dimension_dataset>=3:
                    beliefs[agent][0][2] = pos[2]

                if dimension_dataset==2:
                    if number_state_variables >= 4:    
                        vel = (np.random.uniform(-0.5, 0.5 ,(dimension_dataset, num_particles)))*2*5
                        beliefs[agent][0][2] = vel[0]
                        beliefs[agent][0][3] = vel[1]

                        if number_state_variables >= 6:    
                            acc = (np.random.uniform(-0.5, 0.5 ,(dimension_dataset, num_particles)))*2*10
                            beliefs[agent][0][4] = acc[0]
                            beliefs[agent][0][5] = acc[1]
                elif dimension_dataset==3:
                    if number_state_variables >= 6:    
                        vel = (np.random.uniform(-0.5, 0.5 ,(dimension_dataset, num_particles)))*2*5
                        beliefs[agent][0][3] = vel[0]
                        beliefs[agent][0][4] = vel[1]
                        beliefs[agent][0][5] = vel[2]

                        if number_state_variables >= 9:    
                            acc = (np.random.uniform(-0.5, 0.5 ,(dimension_dataset, num_particles)))*2*10
                            beliefs[agent][0][6] = acc[0]
                            beliefs[agent][0][7] = acc[1]
                            beliefs[agent][0][8] = acc[2]

                beliefs[agent][0] = np.array(beliefs[agent][0])
                beliefs[agent][1] = np.array(np.ones((1,num_particles))/num_particles)
        else:
            if dimension_dataset==2:
                beliefs = {agent:[np.pad(np.random.uniform(self.limit_t1[0], self.limit_t1[1], (2,1)) +  5*np.random.normal(size=(2, 1)), [(0,4), (0,0)]),np.diag([99**2,100**2,98,97,9,8])] for agent in range(self.num_agents)}
            elif dimension_dataset==3:
                beliefs = {agent:[np.pad(first_true_position.numpy().reshape(3, 1)+ 10*np.random.normal(size=(3, 1)), [(0,6), (0,0)]),np.diag([99**2,100**2,100**2, 99,98,97,9,8,7])] for agent in range(self.num_agents)}

            beliefs = {agent:[beliefs[agent][0][:number_state_variables],beliefs[agent][1][:number_state_variables, :number_state_variables]]   for agent in range(self.num_agents)}

        return beliefs

    def set_baselines(self, baselines):
    
        self.baselines = baselines

    def save_results(self, output_results_tracking):

        try:
            output_results_tracking_old = np.load(os.path.join(self.output_results_tracking_dir,'output_results_tracking.npy'), allow_pickle = True).tolist()
            output_results_tracking_old.update(output_results_tracking)
            output_results_tracking_new = output_results_tracking_old
        except:
            output_results_tracking_new = output_results_tracking

        np.save(os.path.join(self.output_results_tracking_dir,'output_results_tracking.npy'), output_results_tracking_new, allow_pickle = True)

    def load_results(self):
    
        return np.load(os.path.join(self.output_results_tracking_dir,'output_results_tracking.npy'), allow_pickle = True).tolist()

    def test(self, test_data_loader):

        """
        output_results_tracking:
            - NAME_BASELINE:
                - 'beliefs':
                    without particles:
                        time_n x agent x [mean, cov]
                    with particles:
                        time_n x agent x [particles_position, particles_weights]
                - 'prediction_message'
                ...
        """

        # Create state representation
        beliefs = self.create_beliefs(next(iter(test_data_loader))[-1]) # 't' is the last element
        # Prediction messages
        prediction_message = {agent:[] for agent in range(self.num_agents)}
        # Predicted output
        t_hat = {agent:[] for agent in range(self.num_agents)}
        t_hat_cov = {agent:[] for agent in range(self.num_agents)}
        # Target
        t = {agent:[] for agent in range(self.num_agents)}

        generic = {agent:[] for agent in range(self.num_agents)}

        # Results
        output_results_tracking = {baseline_name: {'beliefs': [copy.deepcopy(beliefs)], 
                                                   'prediction_message':[copy.deepcopy(prediction_message)],
                                                   't_hat':[copy.deepcopy(t_hat)],
                                                   't_hat_cov':[copy.deepcopy(t_hat_cov)],
                                                   't':[copy.deepcopy(t)],} for baseline_name in self.baselines.keys()}

        for baseline_name, baseline_Tracking_algorithm in self.baselines.items():
                
            # input_data: 1 (batch=1) x num_agents x dimensions_input -> see the agent dimension as the real batch
            for time_n, (input_data, *other_data) in enumerate((test_data_loader)):

                output_additional_data = {dataset_output_name:other_data[index_name] for index_name,(dataset_output_name) in enumerate(self.params.dataset_output_names)}

                # Initialize beliefs when changing agent
                if time_n != 0 and output_additional_data['UE_index'][0].numpy()[0] != output_results_tracking[baseline_name]['UE_index'][-1][-1][-1]:
                    beliefs = self.create_beliefs(output_additional_data['UExyz_global']) 
                else:
                    beliefs = output_results_tracking[baseline_name]['beliefs'][-1]   # posterior time_n - 1

                # Prediction (in xyz_global coordinate ref)
                prediction_message = baseline_Tracking_algorithm.prediction(beliefs, time_n)             # prior time_n
                output_results_tracking[baseline_name]['prediction_message'].append(copy.deepcopy(prediction_message))

                # Update (in xyz_global coordinate ref)
                beliefs = output_results_tracking[baseline_name]['beliefs'][-1]                         # posterior time_n - 1
                prediction_message = output_results_tracking[baseline_name]['prediction_message'][-1]   # prior time_n

                beliefs = baseline_Tracking_algorithm.update(input_data, output_additional_data, prediction_message, beliefs, time_n)  # posterior time_n
                output_results_tracking[baseline_name]['beliefs'].append(copy.deepcopy(beliefs))
                
                # Compute state
                t_hat, t_hat_cov = baseline_Tracking_algorithm.estimate_position(beliefs)
                output_results_tracking[baseline_name]['t_hat'].append(copy.deepcopy(t_hat))
                output_results_tracking[baseline_name]['t_hat_cov'].append(copy.deepcopy(t_hat_cov))

                # Save real position
                try:
                    t = {agent:t_agent for agent, t_agent in enumerate(return_numpy(output_additional_data['t']))}
                except:
                    t = {agent:t_agent for agent, t_agent in enumerate(return_numpy(output_additional_data['UExyz_global']))}
                print('Target: a {}\n{}'.format(0, t[0]))
                output_results_tracking[baseline_name]['t'].append(copy.deepcopy(t)) 

                # Save auxiliary data
                for k in list(output_additional_data.keys()):
                    if k in output_results_tracking[baseline_name].keys():
                        output_results_tracking[baseline_name][k].append(copy.deepcopy(output_additional_data[k].numpy()))
                    else:
                        output_results_tracking[baseline_name][k] = [copy.deepcopy(output_additional_data[k].numpy())]

            # Compute metrics
            metrics = compute_metrics_positioning(t = output_results_tracking[baseline_name]['t'], t_hat = output_results_tracking[baseline_name]['t_hat'], params = self.params)
            output_results_tracking[baseline_name]['metrics'] = metrics

        return output_results_tracking
