import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy
from copy import copy
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
from Classes.utils.utils import print_name_and_value, dataloader_to_numpy
from Classes.plotting.plotting import plot_artificial_2D_dataset, plot_uncertainty_artificial_2D_dataset, plot_artificial_2D_dataset_tracking, plot_tracking_results_artificial_2D_dataset

class DATASET(object):

    DEFAULTS = {}   

    def __init__(self, params = {}):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(DATASET.DEFAULTS, **params_dict)

        if 't1_min' not in self.params.__dict__:
            self.t1_min = self.limit_t1[0]
            self.t1_max = self.limit_t1[1]
            self.t2_min = self.limit_t2[0]
            self.t2_max = self.limit_t2[1]

    ###########################################################################################
    ###########################################################################################
    # RETURN DATASETS

    def return_dataset(self):

        # Test Aleatoric uncertainty
        if self.sig_noise_t1_min != self.sig_noise_t1_max:

            # TRAINING
            # Generate true position t
            t1 = np.random.uniform(self.t1_min, self.t1_max, self.size_dataset)[:, None]
            t1.sort(axis = 0)
            self.original_indices_t1 = np.argsort(t1, axis=0).flatten()
            t2 = np.random.uniform(self.t2_min, self.t2_max, self.size_dataset)[:, None]
            t = np.hstack([t1, t2])  # Make it (self.size_dataset, 2)

            # Compute the standard deviation of the noise as a function of t1
            t1_values = t[:, 0]
            self.sig_noise_t1 = self.sig_noise_t1_min + (self.sig_noise_t1_max - self.sig_noise_t1_min) * (t1_values - self.t1_min) / (self.t1_max - self.t1_min)
            # Generate noise and add it to 't' to get 'x'
            self.noise = np.random.normal(0, self.sig_noise_t1[:, None], (self.size_dataset, 2))
            x = t + self.noise

            self.params.x_train_mean = np.mean(x, 0)
            self.params.x_train_std = np.std(x, 0)
            self.x_train = (x - self.params.x_train_mean)/self.params.x_train_std
            self.params.t_train_mean = 0 
            self.params.t_train_std = 1
            self.t_train = (t - self.params.t_train_mean)/self.params.t_train_std

            if self.bool_shuffle:
                perm = np.random.permutation(len(self.x_train))
                self.x_train = self.x_train[perm]
                self.t_train = self.t_train[perm]

            # VALIDATION
            num_points_per_dim = int(np.sqrt(self.size_dataset)/2)
            t1 = np.linspace(self.t1_min, self.t1_max, num_points_per_dim)
            t2 = np.linspace(self.t2_min, self.t2_max, num_points_per_dim)
            # Create 2D grid
            t1_grid, t2_grid = np.meshgrid(t1, t2)
            # Flatten and stack to make it self.size_dataset x 2
            t_valid = np.stack([t1_grid.flatten(), t2_grid.flatten()], axis=1)

            # Compute the standard deviation of the noise as a function of t1
            t1_values = t_valid[:, 0]
            self.sig_noise_t1_valid = self.sig_noise_t1_min + (self.sig_noise_t1_max - self.sig_noise_t1_min) * (t1_values - self.t1_min) / (self.t1_max - self.t1_min)
            # Generate noise and add it to 't' to get 'x'
            num_tot_points = t1_values.shape[0]
            noise_valid = np.random.normal(0, self.sig_noise_t1_valid[:, None], (num_tot_points, 2))
            x_valid = t_valid + noise_valid
            x_valid = (x_valid-self.params.x_train_mean)/self.params.x_train_std 
            
            self.x_valid = x_valid
            self.t_valid = t_valid

        # Test Epistemic uncertainty
        else:

            # TRAINING
            # Create a linear space representing the desired density
            linear_density = np.linspace(0, 1, self.size_dataset)
            # Modify the linear space to get a cumulative distribution that is linear
            transformed_density = (linear_density)**(4)
            # transformed_density = linear_density
            # Create density_t1 based on the transformed_density
            self.density_t1 = np.flip((self.density_t1_min + (self.density_t1_max - self.density_t1_min) * transformed_density))
            # Get t1 values using the cumulative sum of density_t1
            t1 = np.cumsum(self.density_t1)
            t1 = (t1 - np.min(t1)) / (np.max(t1) - np.min(t1))  # Normalize
            t1 = t1 * (self.t1_max - self.t1_min) + self.t1_min  # Scale to [t1_min, t1_max]
            t1 = t1[:, None]  # Convert to column vector
            
            t2 = np.random.uniform(self.t2_min, self.t2_max, self.size_dataset)[:, None]
            t = np.hstack([t1, t2])  # Combine t1 and t2

            # The statistics of the noise added is the same for every t and it is equal to self.sig_noise_t1_min = self.sig_noise_t1_max
            self.sig_noise_t1 = np.ones(self.size_dataset) * self.sig_noise_t1_min
            self.noise = np.random.normal(0, self.sig_noise_t1[:, None], (self.size_dataset, 2))
            x = t + self.noise
            
            self.params.x_train_mean = np.mean(x, 0)
            self.params.x_train_std = np.std(x, 0)
            self.x_train = (x - self.params.x_train_mean) / self.params.x_train_std
            self.params.t_train_mean = 0 
            self.params.t_train_std = 1
            self.t_train = (t - self.params.t_train_mean) / self.params.t_train_std
            
            if self.bool_shuffle:
                perm = np.random.permutation(len(self.x_train))
                self.x_train = self.x_train[perm]
                self.t_train = self.t_train[perm]

            # VALIDATION
            # Repeat the same process to generate t1 values for the validation set
            linear_density = np.linspace(0, 1, self.size_dataset)
            transformed_density = (linear_density)**(4)
            # transformed_density = linear_density
            density_t1 = np.flip(self.density_t1_min + (self.density_t1_max - self.density_t1_min) * transformed_density)
            t1 = np.cumsum(density_t1)
            t1 = (t1 - np.min(t1)) / (np.max(t1) - np.min(t1))  # Normalize
            t1 = t1 * (self.t1_max - self.t1_min) + self.t1_min  # Scale to [t1_min, t1_max]
            t1 = t1[:, None]  # Convert to column vector

            t2 = np.random.uniform(self.t2_min, self.t2_max, self.size_dataset)[:, None]
            t = np.hstack([t1, t2])  # Combine t1 and t2

            # The statistics of the noise added is the same for every t and it is equal to self.sig_noise_t1_min = self.sig_noise_t1_max
            self.sig_noise_t1_valid = np.ones(self.size_dataset) * self.sig_noise_t1_min
            noise_valid = np.random.normal(0, self.sig_noise_t1_valid[:, None], (self.size_dataset, 2))
            x = t + noise_valid

            self.x_valid = (x - self.params.x_train_mean) / self.params.x_train_std
            self.t_valid = (t - self.params.t_train_mean) / self.params.t_train_std


        self.dataset_output_names = ['t']
        self.params.dataset_output_names = self.dataset_output_names

        print_name_and_value(self.x_train.shape)
        print_name_and_value(self.t_train.shape)
        print_name_and_value(self.x_valid.shape)
        print_name_and_value(self.t_valid.shape)
        print_name_and_value(self.task)
        print_name_and_value(self.batch_size)

        train_dataset = TensorDataset(torch.tensor(self.x_train).float(), torch.tensor(self.t_train).float())
        valid_dataset = TensorDataset(torch.tensor(self.x_valid).float(), torch.tensor(self.t_valid).float())
    
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle, pin_memory=True)
        self.val_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle, pin_memory=True)

        self.params.num_train_batches = len(self.train_loader)
        self.params.num_val_batches = len(self.val_loader)
        
        print_name_and_value(self.params.num_train_batches)
        print_name_and_value(self.params.num_val_batches)

        self.params.update_all()
        return self.train_loader, self.val_loader

    def return_dataset_tracking(self):

        # Timestep
        self.n = 0

        if self.noise_type == 'Gaussian':
            self.noise_class = np.random.normal
        elif self.noise_type == 'Laplacian':
            self.noise_class = np.random.laplace
        else:
            self.noise_class = np.random.normal

        # compute positions, velocities and accelerations of each agent
        self.define_initial_t()
        try:
            self.mutual_distances = [scipy.spatial.distance.cdist(np.array([self.positions[agent] for agent in range(self.num_agents)]).squeeze(),np.array([self.positions[agent] for agent in range(self.num_agents)]).squeeze()).tolist()]
        except:
            self.mutual_distances = [0]
        
        self.connectivity_matrix = [(np.array(self.mutual_distances)<self.comm_distance).squeeze()*1 - np.eye(self.num_agents)]
        self.t = {agent:[[self.positions[agent][self.n][0], 
                        self.positions[agent][self.n][1], 
                        self.velocities[agent][self.n][0], 
                        self.velocities[agent][self.n][1], 
                        self.accelerations[agent][self.n][0], 
                        self.accelerations[agent][self.n][1]]] for agent in range(self.num_agents)}
        
        self.x_inter_agent = [self.mutual_distances[0]+self.noise_class(0, self.std_x_inter_agent, (self.num_agents,self.num_agents))]
        
        self.x_gnss = {agent:[[self.positions[agent][0][0] + self.noise_class(0, self.std_position_x_gnss, (1,)).item(), 
                            self.positions[agent][0][1] + self.noise_class(0, self.std_position_x_gnss, (1,)).item(), 
                            self.velocities[agent][0][0] + self.noise_class(0, self.std_velocity_x_gnss, (1,)).item(), 
                            self.velocities[agent][0][1] + self.noise_class(0, self.std_velocity_x_gnss, (1,)).item(), 
                            self.accelerations[agent][0][0] + self.noise_class(0, self.std_acceleration_x_gnss, (1,)).item(), 
                            self.accelerations[agent][0][1] + self.noise_class(0, self.std_acceleration_x_gnss, (1,)).item()]] for agent in range(self.num_agents)}
        
        # x_n = F x_n-1 + W w_n-1
        # x: pos, vel, acc. w: pos_noise, vel_noise, acc_noise
        self.F = np.array([[1, 0, self.T_between_timestep, 0                      , (self.T_between_timestep**2)/2, 0                             ],
                        [0, 1, 0                      , self.T_between_timestep, 0                             , (self.T_between_timestep**2)/2],
                        [0, 0, 1                      , 0                      , self.T_between_timestep       , 0                             ],
                        [0, 0, 0                      , 1                      , 0                             , self.T_between_timestep       ],
                        [0, 0, 0                      , 0                      , 1                             , 0                             ],
                        [0, 0, 0                      , 0                      , 0                             , 1                             ]])
        self.W = copy(self.F)

        for n in range(self.size_dataset):

            self.compute_next_step()

        # Create torch dataloader (batch 1). x are NN measurements, additional data are Kalman measurements
        x_test_NN = np.swapaxes(np.array(np.array([v for k,v in self.x_gnss.items()])), 0, 1)[:,:,:2]
        if hasattr(self, 'x_train_mean'):
            x_test_NN = (x_test_NN - self.x_train_mean) / self.x_train_std
        test_dataset = TensorDataset(torch.tensor(x_test_NN).float(),  # Measurements for NN
                                      torch.tensor(np.array(self.x_inter_agent)).float(),                                              # Measurements for Classic
                                      torch.tensor(np.swapaxes(np.array(np.array([v for k,v in self.x_gnss.items()])), 0, 1)[:,:,:2]).float(), # Measurements for Classic
                                      torch.tensor(np.array(self.connectivity_matrix)).float(),                                        # Auxiliary connectivity matrix between agents
                                      torch.tensor(np.swapaxes(np.array(np.array([v for k,v in self.t.items()])), 0, 1)).float(),      # GT
                                      )    

        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=0, pin_memory=True)

        self.params.num_val_batches = len(self.test_loader)
        
        self.dataset_output_names = ['x_inter_agent', 'x_gnss', 'connectivity_matrix', 't',]
        self.params.dataset_output_names = self.dataset_output_names

        print_name_and_value(self.params.num_val_batches)

        self.params.update_all()

        return self.test_loader

    def return_big_testing_dataset(self):

        if self.BNN_algorithm == 'NN' or self.BNN_algorithm == 'BBP' or self.BNN_algorithm == 'SGLD' or self.BNN_algorithm == 'BDK' or self.BNN_algorithm == 'BDKep':

            num_points_per_dim = int(np.sqrt(self.size_dataset))  
            x1 = np.linspace(self.t1_min, self.t1_max, num_points_per_dim)
            x2 = np.linspace(self.t2_min, self.t2_max, num_points_per_dim)
            # Create 2D grid
            x1_grid, x2_grid = np.meshgrid(x1, x2)
            # Flatten and stack to make it self.size_dataset x 2
            x_test = np.stack([x1_grid.flatten(), x2_grid.flatten()], axis=1)
            x_test = (x_test-self.params.x_train_mean)/self.params.x_train_std 

        return x_test



    ###########################################################################################
    ###########################################################################################
    # SHOW DATASETS

    def show_dataset(self, loader = None):

        # Show dataset
        if self.bool_plot_dataset:
            x_train, t_train = dataloader_to_numpy(self.train_loader)
            x_valid, t_valid = dataloader_to_numpy(self.val_loader)
            x_train = x_train * self.params.x_train_std + self.params.x_train_mean 
            x_valid = x_valid * self.params.x_train_std + self.params.x_train_mean 

            # Plot noise magnitude
            # plot_added_noise_artificial_2D_dataset(t_train[:,0], self.noise, self.params, 'magnitude_noise', save_pdf=0, save_jpg=0, save_svg=0, save_eps=0, plt_show=0)
            # plot_std_noise_artificial_2D_dataset(t_train[:,0], self.noise, x_train, self.params, 'std_noise', save_pdf=1, save_jpg=1, save_svg=0, save_eps=1, plt_show=0)

            # Plot 
            plot_artificial_2D_dataset(self, x_train, t_train, x_valid, t_valid, self.params, file_name = 'train_test_dataset', xlabel_ = 't1', ylabel_ = 't2', title_ = '2D Scatter Plot with Desired Noise Level', logx = False, logy = False, xlim = None, ylim = None, save_eps = 1, ax = None, save_svg = 0, save_pdf = 1, save_jpg=1, plt_show = 0)
            # plot_artificial_2D_dataset(self, x_valid, t_valid, x_valid, t_valid, self.params, file_name = 'train_test_dataset', xlabel_ = 't1', ylabel_ = 't2', title_ = '2D Scatter Plot with Desired Noise Level', logx = False, logy = False, xlim = None, ylim = None, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg=0, plt_show = 0)

    def show_uncertainty_dataset(self, return_model_evaluate, train_loader = None, val_loader = None, file_name_uncertainty_dataset = 'uncertainty_dataset'):

        x_train, t_train = dataloader_to_numpy(train_loader)
        x_valid, t_valid = dataloader_to_numpy(val_loader)

        x_train = x_train * self.params.x_train_std + self.params.x_train_mean 
        x_valid = x_valid * self.params.x_train_std + self.params.x_train_mean 

        # Model prediction
        plot_uncertainty_artificial_2D_dataset(x_train, t_train, x_valid, t_valid, return_model_evaluate, self.params, file_name = file_name_uncertainty_dataset, xlabel_ = 't1', ylabel_ = 't2', title_ = self.BNN_algorithm, logx = False, logy = False, xlim = None, ylim = None, save_eps = 0, ax = None, save_svg = 0, save_jpg=1, save_pdf = 0, plt_show = 0)

    def show_dataset_tracking(self):

        plot_artificial_2D_dataset_tracking(self, self.t, self.params, file_name = 'testing_trajectory', xlabel_ = 't1', ylabel_ = 't2', title_ = '2D testing trajectory', logx = False, logy = False, xlim = self.limit_t1, ylim = self.limit_t2, save_eps = 1, ax = None, save_svg = 0, save_pdf = 1, save_jpg=1, plt_show = 0)

    def show_tracking_results(self, output_results_tracking):

        plot_tracking_results_artificial_2D_dataset(output_results_tracking,  self.params, file_name = 'tracking_results', xlabel_ = 't1', ylabel_ = 't2', title_ = '2D testing tracking', logx = False, logy = False, xlim = self.limit_t1, ylim = self.limit_t2, save_eps = 1, ax = None, save_svg = 0, save_pdf = 1, save_jpg=1, plt_show = 0)
 


    ###########################################################################################
    ###########################################################################################
    # AUXILIRAY
        
    def compute_next_step(self): 

        new_positions_agents_list = []
        for agent in range(self.num_agents):

            noise_position, noise_velocities, noise_accelerations = self.add_noise_deterministic_components(agent)
            new_t = self.F@np.array(self.t[agent][-1]) + self.W@np.concatenate((noise_position, noise_velocities, noise_accelerations))
            new_t_list = new_t.tolist()

            # Check if agent is outside the area
            if self.limit_behavior == 'reflection':
                if new_t_list[0] < self.limit_t1[0]:
                    while new_t_list[0] < self.limit_t1[0]:
                        new_t_list[0] = self.limit_t1[0] + abs(self.limit_t1[0]-new_t_list[0])
                    new_t_list[2] = - new_t_list[2]
                    new_t_list[4] = - new_t_list[4]
                elif new_t_list[0] > self.limit_t1[1]:
                    while new_t_list[0] > self.limit_t1[1]:
                        new_t_list[0] = self.limit_t1[1] - abs(self.limit_t1[1]-new_t_list[0])
                    new_t_list[2] = - new_t_list[2]
                    new_t_list[4] = - new_t_list[4]
                if new_t_list[1] < self.limit_t2[0]:
                    while new_t_list[1] < self.limit_t2[0]:
                        new_t_list[1] = self.limit_t2[0] + abs(self.limit_t2[0]-new_t_list[1])
                    new_t_list[3] = - new_t_list[3]
                    new_t_list[5] = - new_t_list[5]
                elif new_t_list[1] > self.limit_t2[1]:
                    while new_t_list[1] > self.limit_t2[1]:
                        new_t_list[1] = self.limit_t2[1] - abs(self.limit_t2[1]-new_t_list[1])
                    new_t_list[3] = - new_t_list[3]
                    new_t_list[5] = - new_t_list[5]
            elif self.limit_behavior == 'continue':
                if new_t_list[0] < self.limit_t1[0]:
                    while new_t_list[0] < self.limit_t1[0]:
                        new_t_list[0] = self.limit_t1[1] - abs(self.limit_t1[0]-new_t_list[0])
                if new_t_list[0] > self.limit_t1[1]:
                    while new_t_list[0] > self.limit_t1[1]:
                        new_t_list[0] = self.limit_t1[0] + abs(self.limit_t1[1]-new_t_list[0])
                if new_t_list[1] < self.limit_t2[0]:
                    while new_t_list[1] < self.limit_t2[0]:
                        new_t_list[1] = self.limit_t2[1] - abs(self.limit_t2[0]-new_t_list[1])
                if new_t_list[1] > self.limit_t2[1]:
                    while new_t_list[1] > self.limit_t2[1]:
                        new_t_list[1] = self.limit_t2[0] + abs(self.limit_t2[1]-new_t_list[1])

            # Limit velocity
            if new_t_list[2] < self.limit_vt1[0]:
                new_t_list[2] = self.limit_vt1[0]
            if new_t_list[2] > self.limit_vt1[1]:
                new_t_list[2] = self.limit_vt1[1]
        
            if new_t_list[3] < self.limit_vt2[0]:
                new_t_list[3] = self.limit_vt2[0]
            if new_t_list[3] > self.limit_vt2[1]:
                new_t_list[3] = self.limit_vt2[1]

            # Limit acceleration
            if new_t_list[4] < self.limit_at1[0]:
                new_t_list[4] = self.limit_at1[0]
            if new_t_list[4] > self.limit_at1[1]:
                new_t_list[4] = self.limit_at1[1]
        
            if new_t_list[4] < self.limit_at2[0]:
                new_t_list[4] = self.limit_at2[0]
            if new_t_list[4] > self.limit_at2[1]:
                new_t_list[4] = self.limit_at2[1]

            new_positions_agents_list.append(new_t_list[0:2])
            self.t[agent].append(new_t_list)
            self.positions[agent].append(new_t_list[0:2])
            self.velocities[agent].append(new_t_list[2:4])
            self.accelerations[agent].append(new_t_list[4:6])

            self.x_gnss[agent].append([self.positions[agent][-1][0] + self.noise_class(0, self.std_position_x_gnss, (1,)).item(), 
                                    self.positions[agent][-1][1] + self.noise_class(0, self.std_position_x_gnss, (1,)).item(), 
                                    self.velocities[agent][-1][0] + self.noise_class(0, self.std_velocity_x_gnss, (1,)).item(), 
                                    self.velocities[agent][-1][1] + self.noise_class(0, self.std_velocity_x_gnss, (1,)).item(), 
                                    self.accelerations[agent][-1][0] + self.noise_class(0, self.std_acceleration_x_gnss, (1,)).item(), 
                                    self.accelerations[agent][-1][1] + self.noise_class(0, self.std_acceleration_x_gnss, (1,)).item()])

        new_positions_agents = np.array(new_positions_agents_list)

        # Compute real distances
        new_mutual_distances = scipy.spatial.distance.cdist(new_positions_agents,new_positions_agents).tolist()
        try:
            self.mutual_distances.append(new_mutual_distances)
        except:
            self.mutual_distances.append(0)

        # Compute connectivity matrix
        self.connectivity_matrix.append((np.array(new_mutual_distances)<self.comm_distance).squeeze()*1 - np.eye(self.num_agents))

        self.x_inter_agent.append(new_mutual_distances+self.noise_class(0, self.std_x_inter_agent, (self.num_agents,self.num_agents)))

        # Update timestep
        self.n = self.n + 1


    def add_noise_deterministic_components(self, agent):
        # Random component
        noise_position = self.noise_class(0, self.std_noise_position, 2) 
        noise_velocities = self.noise_class(0, self.std_noise_velocity, 2) 
        noise_accelerations = self.noise_class(0, self.std_noise_acceleration, 2) 

        # Deterministic component
        if self.setting_trajectories == 'spiral' and self.n >3: # self.n >20:

            # Perfect spiral
            # angle = np.arctan2(self.positions[agent][-1][1], self.positions[agent][-1][0])
            # angle_deg = angle*180/np.pi
            # distance_from_center = np.linalg.norm([self.positions[agent][-1][0],self.positions[agent][-1][1]])
            # mod_vel = np.max((np.linalg.norm([self.mean_velocity_t1, self.mean_velocity_t2]), 1))
            # orthogonal_vel = np.array([-np.sin(angle), np.cos(angle)])* mod_vel #*(1-distance_from_center/self.limit_t1[0])
            # # remove velocity
            # orthogonal_vel -= np.array([self.velocities[agent][-1][0],self.velocities[agent][-1][1]])
            # noise_velocities += orthogonal_vel

            # Golden spiral
            angle = np.arctan2(self.positions[agent][-1][1], self.positions[agent][-1][0])
            angle_deg = angle*180/np.pi
            distance_from_center = np.linalg.norm([self.positions[agent][-1][0],self.positions[agent][-1][1]])
            mod_vel = np.max((np.linalg.norm([self.mean_velocity_t1, self.mean_velocity_t2]), 1))
            orthogonal_vel = np.array([-np.sin(angle), np.cos(angle)])* mod_vel #*(1-distance_from_center/self.limit_t1[0])

            if self.n + agent > 10 + 16:
                # Remove velocity
                orthogonal_vel -= np.array([self.velocities[agent][-1][0],self.velocities[agent][-1][1]])
            noise_velocities += orthogonal_vel


        return noise_position, noise_velocities, noise_accelerations

    def define_initial_t(self):
    
        # random initial position, velocity defined, acceleration defined 
        if self.setting_trajectories == 'not_defined':
            self.positions = {agent:[[np.random.uniform(self.limit_t1[0],self.limit_t1[1]), np.random.uniform(self.limit_t2[0],self.limit_t2[1])]] for agent in range(self.num_agents)}
            self.velocities = {agent:[[self.mean_velocity_t1, self.mean_velocity_t2]] for agent in range(self.num_agents)}
            self.accelerations = {agent:[[self.mean_acceleration_t1, self.mean_acceleration_t2]] for agent in range(self.num_agents)}
        elif self.setting_trajectories == 'star' or self.setting_trajectories == 'spiral':
            angle_directions = np.arange(0,360, 360/self.num_agents) * math.pi/180
            self.positions = {agent:[[np.random.uniform(0,0), np.random.uniform(0,0)]] for agent in range(self.num_agents)}
            self.velocities = {agent:[[abs(self.mean_velocity_t1)*np.cos(angle_directions[agent]), abs(self.mean_velocity_t1)*np.sin(angle_directions[agent])]] for agent in range(self.num_agents)}
            self.accelerations = {agent:[[abs(self.mean_acceleration_t1)*np.cos(angle_directions[agent]), abs(self.mean_acceleration_t1)*np.sin(angle_directions[agent])]] for agent in range(self.num_agents)}    