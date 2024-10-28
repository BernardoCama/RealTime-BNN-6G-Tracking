import sys
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy
from copy import copy
import mat73

# Directories
cwd = os.path.split(os.path.abspath(__file__))[0]
DB_DIR =  os.path.join(os.path.split(os.path.split(os.path.split(os.path.split(cwd)[0])[0])[0])[0], 'DB')
CLASSES_DIR = os.path.join(cwd, 'Classes')
EXPERIMENTS_DIR = os.path.join(cwd, 'Exp')
PARAMETERS_DIR = os.path.join(cwd, 'Parameters')
sys.path.append(os.path.dirname(CLASSES_DIR))
sys.path.append(os.path.dirname(EXPERIMENTS_DIR))
sys.path.append(os.path.dirname(PARAMETERS_DIR))
sys.path.append(os.path.dirname(cwd))
from Classes.utils.utils import print_name_and_value, dataloader_to_numpy, from_latlongh_to_xyz_matrix, create_sequences
from Classes.plotting.plotting import plot_uncertainty_artificial_2D_dataset, plot_tracking_results_ray_tracing2, plot_ray_tracing_dataset

class DATASET(object):

    DEFAULTS = {}   

    def __init__(self, params = {}):

        self.params = params
        if not isinstance(params, dict):
            params_dict = params.DEFAULTS
        self.__dict__.update(DATASET.DEFAULTS, **params_dict)

    ###########################################################################################
    ###########################################################################################
    # RETURN DATASETS

    def return_dataset(self):

        self.DB_training_path = os.path.join(DB_DIR, self.DB_name, 'Data', 'training')
        DB_training_path_mat = os.path.join(self.DB_training_path, 'dataset.mat')
        DB_training_path_pt = os.path.join(self.DB_training_path, 'dataset.pt')

        self.DB_testing_path = os.path.join(DB_DIR, self.DB_name, 'Data', 'testing')
        DB_testing_path_mat = os.path.join(self.DB_testing_path, 'dataset.mat')
        DB_testing_path_pt = os.path.join(self.DB_testing_path, 'dataset.pt')

        self.DB_tracking_path = os.path.join(DB_DIR, self.DB_name, 'Data', 'tracking')
        DB_tracking_path_mat = os.path.join(self.DB_tracking_path, 'dataset.mat')
        DB_tracking_path_pt = os.path.join(self.DB_tracking_path, 'dataset.pt')

        # REMOVE
        # DB_training_path_mat = DB_testing_path_mat

        if not self.load_dataset:

            ####################################################################################################
            #################### TRAINING ######################################################################
            ####################################################################################################
            # import datasets
            try:
                raw_dataset = np.array(mat73.loadmat(DB_training_path_mat)['dataset_tot'])
            except: 
                raw_dataset = scipy.io.loadmat(DB_training_path_mat)['dataset'].flatten()
            print('Loaded training dataset')
            
            numPos_train = raw_dataset.shape[0]

            # Measurements
            ADCPM_train = np.concatenate(np.array([raw_dataset[pos]['measurements']['ADCPM'].reshape(-1, self.Ng, self.num_OFDM_symbols_per_pos, self.RXants)[:,:,0,:]   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            # ADCPM_train = np.concatenate(np.array([raw_dataset[pos]['measurements']['ADCPM'].reshape(-1, self.Ng, self.num_OFDM_symbols_per_pos, self.RXants)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0).transpose(0, 2, 1, 3).reshape(-1, self.Ng, self.RXants)

            est_ToA_train = np.concatenate(np.array([raw_dataset[pos]['measurements']['est_ToA'].reshape(-1, 1) for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            est_AoA_wrtBS_train = np.concatenate(np.array([raw_dataset[pos]['measurements']['est_AoA_wrtBS'].reshape(-1, 2)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            est_AoA_wrtcell_train = np.concatenate(np.array([raw_dataset[pos]['measurements']['est_AoA_wrtcell'].reshape(-1, 2)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)

            # Target
            UExyz_wrtBS_train = np.concatenate(np.array([raw_dataset[pos]['target']['UExyz_wrtBS'].reshape(-1, 3)  for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            UExyz_wrtcell_train = np.concatenate(np.array([raw_dataset[pos]['target']['UExyz_wrtcell'].reshape(-1, 3)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            
            # Auxiliary
            BS_detected_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BS_detected'].reshape(-1, 1)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BSlatlonh_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BSlatlonh'].reshape(-1, 3)  for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BSxyz_wrtUE_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BSxyz_wrtUE'].reshape(-1, 3)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BS_LOS_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BS_LOS'].reshape(-1, 1)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BS_cell_angles_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BS_cell_angles'].reshape(-1, 2)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            pos_index_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['pos_index'].reshape(-1, 1)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            true_ToA_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['true_ToA'].reshape(-1, 1)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            true_AoA_wrtBS_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['true_AoA_wrtBS'].reshape(-1, 2)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            true_AoA_wrtcell_train = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['true_AoA_wrtcell'].reshape(-1, 2)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)

            print('Deleting raw training dataset')
            del raw_dataset
            print('Deleted raw training dataset')
            
            # shuffle
            print('Random shuffling training')
            (rng_state := np.random.get_state(), 
             np.random.shuffle(ADCPM_train), np.random.set_state(rng_state), np.random.shuffle(est_ToA_train), np.random.set_state(rng_state), np.random.shuffle(est_AoA_wrtBS_train), np.random.set_state(rng_state), np.random.shuffle(est_AoA_wrtcell_train), np.random.set_state(rng_state),
             np.random.shuffle(UExyz_wrtBS_train), np.random.set_state(rng_state), np.random.shuffle(UExyz_wrtcell_train), np.random.set_state(rng_state),
             np.random.shuffle(BS_detected_train), np.random.set_state(rng_state), np.random.shuffle(BSlatlonh_train), np.random.set_state(rng_state), np.random.shuffle(BSxyz_wrtUE_train), np.random.set_state(rng_state), np.random.shuffle(BS_LOS_train), np.random.set_state(rng_state), np.random.shuffle(BS_cell_angles_train), np.random.set_state(rng_state), np.random.shuffle(pos_index_train), np.random.set_state(rng_state), np.random.shuffle(true_ToA_train), np.random.set_state(rng_state), np.random.shuffle(true_AoA_wrtBS_train), np.random.set_state(rng_state), np.random.shuffle(true_AoA_wrtcell_train), np.random.set_state(rng_state))
            print('Random shuffling training complete')


            ####################################################################################################
            #################### TESTING #######################################################################
            ####################################################################################################
            # import datasets
            try:
                raw_dataset = np.array(mat73.loadmat(DB_testing_path_mat)['dataset_tot'])
            except: 
                raw_dataset = scipy.io.loadmat(DB_testing_path_mat)['dataset_tot'].flatten()
            print('Loaded testing dataset')

            numPos_test = raw_dataset.shape[0]

            # Measurements
            ADCPM_test = np.concatenate(np.array([raw_dataset[pos]['measurements']['ADCPM'].reshape(-1, self.Ng, self.num_OFDM_symbols_per_pos, self.RXants)[:,:,0,:]   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            # ADCPM_test = np.concatenate(np.array([raw_dataset[pos]['measurements']['ADCPM'].reshape(-1, self.Ng, self.num_OFDM_symbols_per_pos, self.RXants)   for pos in range(numPos_train) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0).transpose(0, 2, 1, 3).reshape(-1, self.Ng, self.RXants)
            
            est_ToA_test = np.concatenate(np.array([raw_dataset[pos]['measurements']['est_ToA'].reshape(-1, 1) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            est_AoA_wrtBS_test = np.concatenate(np.array([raw_dataset[pos]['measurements']['est_AoA_wrtBS'].reshape(-1, 2)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            est_AoA_wrtcell_test = np.concatenate(np.array([raw_dataset[pos]['measurements']['est_AoA_wrtcell'].reshape(-1, 2)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)

            # Target
            UExyz_wrtBS_test = np.concatenate(np.array([raw_dataset[pos]['target']['UExyz_wrtBS'].reshape(-1, 3)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            UExyz_wrtcell_test = np.concatenate(np.array([raw_dataset[pos]['target']['UExyz_wrtcell'].reshape(-1, 3)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            
            # Auxiliary
            BS_detected_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BS_detected'].reshape(-1, 1)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BSlatlonh_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BSlatlonh'].reshape(-1, 3)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BSxyz_wrtUE_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BSxyz_wrtUE'].reshape(-1, 3)    for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BS_LOS_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BS_LOS'].reshape(-1, 1)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            BS_cell_angles_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['BS_cell_angles'].reshape(-1, 2)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            pos_index_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['pos_index'].reshape(-1, 1)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            true_ToA_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['true_ToA'].reshape(-1, 1)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            true_AoA_wrtBS_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['true_AoA_wrtBS'].reshape(-1, 2)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)
            true_AoA_wrtcell_test = np.concatenate(np.array([raw_dataset[pos]['auxiliary']['true_AoA_wrtcell'].reshape(-1, 2)   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]), 0)

            print('Deleting raw testing dataset')
            del raw_dataset
            print('Deleted raw testing dataset')
            
            # shuffle
            print('Random shuffling testing')
            (rng_state := np.random.get_state(), 
             np.random.shuffle(ADCPM_test), np.random.set_state(rng_state), np.random.shuffle(est_ToA_test), np.random.set_state(rng_state), np.random.shuffle(est_AoA_wrtBS_test), np.random.set_state(rng_state), np.random.shuffle(est_AoA_wrtcell_test), np.random.set_state(rng_state),
             np.random.shuffle(UExyz_wrtBS_test), np.random.set_state(rng_state), np.random.shuffle(UExyz_wrtcell_test), np.random.set_state(rng_state),
             np.random.shuffle(BS_detected_test), np.random.set_state(rng_state), np.random.shuffle(BSlatlonh_test), np.random.set_state(rng_state), np.random.shuffle(BSxyz_wrtUE_test), np.random.set_state(rng_state), np.random.shuffle(BS_LOS_test), np.random.set_state(rng_state), np.random.shuffle(BS_cell_angles_test), np.random.shuffle(pos_index_test), np.random.set_state(rng_state), np.random.set_state(rng_state), np.random.shuffle(true_ToA_test), np.random.set_state(rng_state), np.random.shuffle(true_AoA_wrtBS_test), np.random.set_state(rng_state), np.random.shuffle(true_AoA_wrtcell_test), np.random.set_state(rng_state))
            print('Random shuffling testing complete')


        # Load dataset
        else:
            train_dataset = torch.load(DB_training_path_pt)
            test_dataset = torch.load(DB_testing_path_pt)
    
        if self.recompute_statistics:
            # compute mean and var of training dataset, normalize data
            temp_loader = DataLoader(torch.tensor(ADCPM_train).double(), batch_size=ADCPM_train.shape[0])
            data = next(iter(temp_loader))
            self.x_train_mean = data[0].mean() # torch.mean(data[0], 0, keepdim=True) #data[0].mean()
            self.x_train_std =  data[0].std() # torch.std(data[0], 0, keepdim=True) # data[0].std()
            self.x_train_mean = self.x_train_mean.cpu().detach().numpy()
            self.x_train_std = self.x_train_std.cpu().detach().numpy()
            np.save(self.DB_training_path + '/mean_std_train.npy', [self.x_train_mean, self.x_train_std], allow_pickle = True)
            del temp_loader
        else:
            mean_std = np.load(self.DB_training_path + '/mean_std_train.npy',  allow_pickle = True).tolist()
            self.x_train_mean = mean_std[0]
            self.params.x_train_mean = self.x_train_mean
            self.x_train_std = mean_std[1]
            self.params.x_train_std = self.x_train_std
            print(f'mean train: {self.x_train_mean}', f'std train: {self.x_train_std}')

        if not self.load_dataset:
            x_train = (ADCPM_train - self.x_train_mean)/self.x_train_std
            x_test = (ADCPM_test - self.x_train_mean)/self.x_train_std

            print_name_and_value(x_train.shape)
            print_name_and_value(x_test.shape)

            train_dataset = TensorDataset(torch.tensor(x_train).float(), 
                                        torch.tensor(est_ToA_train).double(), 
                                        torch.tensor(est_AoA_wrtBS_train).double(), 
                                        torch.tensor(est_AoA_wrtcell_train).double(), 

                                        torch.tensor(UExyz_wrtBS_train).double(),
                                        torch.tensor(UExyz_wrtcell_train).double(),

                                        torch.tensor(BS_detected_train).double(),
                                        torch.tensor(BSlatlonh_train).double(),
                                        torch.tensor(BSxyz_wrtUE_train).double(),
                                        torch.tensor(BS_LOS_train).double(),
                                        torch.tensor(BS_cell_angles_train).double(),
                                        torch.tensor(pos_index_train.astype(float)).double(),
                                        torch.tensor(true_ToA_train).double(),
                                        torch.tensor(true_AoA_wrtBS_train).double(),
                                        torch.tensor(true_AoA_wrtcell_train).double(),)
            test_dataset = TensorDataset(torch.tensor(x_test).float(), 
                                        torch.tensor(est_ToA_test).double(), 
                                        torch.tensor(est_AoA_wrtBS_test).double(), 
                                        torch.tensor(est_AoA_wrtcell_test).double(), 

                                        torch.tensor(UExyz_wrtBS_test).double(),
                                        torch.tensor(UExyz_wrtcell_test).double(),

                                        torch.tensor(BS_detected_test).double(),
                                        torch.tensor(BSlatlonh_test).double(),
                                        torch.tensor(BSxyz_wrtUE_test).double(),
                                        torch.tensor(BS_LOS_test).double(),
                                        torch.tensor(BS_cell_angles_test).double(),
                                        torch.tensor(pos_index_test.astype(float)).double(),
                                        torch.tensor(true_ToA_test).double(),
                                        torch.tensor(true_AoA_wrtBS_test).double(),
                                        torch.tensor(true_AoA_wrtcell_test).double(),)

        if self.save_dataset:
            torch.save(train_dataset, DB_training_path_pt)
            torch.save(test_dataset, DB_testing_path_pt)

        self.dataset_output_names = ['est_ToA', 'est_AoA_wrtBS', 'est_AoA_wrtcell', 
                                     'UExyz_wrtBS', 'UExyz_wrtcell', 
                                     'BS_detected', 'BSlatlonh', 'BSxyz_wrtUE', 'BS_LOS', 'BS_cell_angles', 'pos_index', 'true_ToA', 'true_AoA_wrtBS', 'true_AoA_wrtcell']
        self.params.dataset_output_names = self.dataset_output_names

        print_name_and_value(self.batch_size)

        # reorder -> For sequences
        if not self.bool_shuffle:

            if hasattr(self, 'bool_tracking_dataset'):
                if self.bool_tracking_dataset:

                    train_dataset = torch.load(DB_tracking_path_pt)
                    # Extract all tensors
                    (
                        x_train_tensor, est_ToA_train_tensor, est_AoA_wrtBS_train_tensor, est_AoA_wrtcell_train_tensor, 
                        UE_index_train_tensor, UElatlonh_train_tensor, UExyz_global_train_tensor, UExyz_wrtBS_train_tensor, UExyz_wrtcell_train_tensor, UEvxvy_wrtBS_test_train_tensor,
                        BS_detected_train_tensor, num_BS_per_pos_train_tensor, BSlatlonh_train_tensor, BSxyz_wrtUE_train_tensor, BSxyz_global_train_tensor, 
                        BS_LOS_train_tensor, BS_cell_angles_train_tensor, pos_index_train_tensor,
                        true_ToA_train_tensor, true_AoA_wrtBS_train_tensor, true_AoA_wrtcell_train_tensor
                    ) = train_dataset.tensors

                    # Create a new TensorDataset with sorted tensors
                    train_dataset = TensorDataset(
                        x_train_tensor,
                        est_ToA_train_tensor,
                        est_AoA_wrtBS_train_tensor,
                        est_AoA_wrtcell_train_tensor,
                        UE_index_train_tensor,
                        UElatlonh_train_tensor,
                        UExyz_global_train_tensor,

                        UExyz_wrtBS_train_tensor,
                        UExyz_wrtcell_train_tensor,
                        UEvxvy_wrtBS_test_train_tensor,
                        
                        BS_detected_train_tensor,
                        num_BS_per_pos_train_tensor,
                        BSlatlonh_train_tensor,
                        BSxyz_wrtUE_train_tensor,
                        BSxyz_global_train_tensor,
                        BS_LOS_train_tensor,
                        BS_cell_angles_train_tensor,
                        pos_index_train_tensor,
                        true_ToA_train_tensor,
                        true_AoA_wrtBS_train_tensor,
                        true_AoA_wrtcell_train_tensor
                    )

                self.dataset_output_names = ['est_ToA', 'est_AoA_wrtBS', 'est_AoA_wrtcell', 
                                            'UE_index', 'UElatlonh', 'UExyz_global', 'UExyz_wrtBS', 'UExyz_wrtcell', 'UEvxvy_wrtBS_test',
                                            'BS_detected', 'num_BS_per_pos', 'BSlatlonh', 'BSxyz_wrtUE', 'BSxyz_global', 'BS_LOS', 'BS_cell_angles', 'pos_index', 'true_ToA', 'true_AoA_wrtBS', 'true_AoA_wrtcell']
                self.params.dataset_output_names = self.dataset_output_names


            else:

                
                # TRAINING
                # Extract all tensors
                (
                    x_train_tensor, est_ToA_train_tensor, est_AoA_wrtBS_train_tensor,
                    est_AoA_wrtcell_train_tensor, UExyz_wrtBS_train_tensor, UExyz_wrtcell_train_tensor,
                    BS_detected_train_tensor, BSlatlonh_train_tensor, BSxyz_wrtUE_train_tensor,
                    BS_LOS_train_tensor, BS_cell_angles_train_tensor, pos_index_train_tensor,
                    true_ToA_train_tensor, true_AoA_wrtBS_train_tensor, true_AoA_wrtcell_train_tensor
                ) = train_dataset.tensors

                # Sort based on pos_index_train
                sorted_indices = torch.argsort(pos_index_train_tensor, 0)

                # Reorder all tensors
                x_train_tensor = x_train_tensor[sorted_indices]
                est_ToA_train_tensor = est_ToA_train_tensor[sorted_indices]
                est_AoA_wrtBS_train_tensor = est_AoA_wrtBS_train_tensor[sorted_indices]
                est_AoA_wrtcell_train_tensor = est_AoA_wrtcell_train_tensor[sorted_indices]

                UExyz_wrtBS_train_tensor = UExyz_wrtBS_train_tensor[sorted_indices]
                UExyz_wrtcell_train_tensor = UExyz_wrtcell_train_tensor[sorted_indices]

                BS_detected_train_tensor = BS_detected_train_tensor[sorted_indices]
                BSlatlonh_train_tensor = BSlatlonh_train_tensor[sorted_indices]
                BSxyz_wrtUE_train_tensor = BSxyz_wrtUE_train_tensor[sorted_indices]
                BS_LOS_train_tensor = BS_LOS_train_tensor[sorted_indices]
                BS_cell_angles_train_tensor = BS_cell_angles_train_tensor[sorted_indices]
                pos_index_train_tensor = pos_index_train_tensor[sorted_indices]
                true_ToA_train_tensor = true_ToA_train_tensor[sorted_indices]
                true_AoA_wrtBS_train_tensor = true_AoA_wrtBS_train_tensor[sorted_indices]
                true_AoA_wrtcell_train_tensor = true_AoA_wrtcell_train_tensor[sorted_indices]

                # Create sequences
                x_train_tensor = create_sequences(x_train_tensor, self.S_len)
                est_ToA_train_tensor = create_sequences(est_ToA_train_tensor, self.S_len)
                est_AoA_wrtBS_train_tensor = create_sequences(est_AoA_wrtBS_train_tensor, self.S_len)
                est_AoA_wrtcell_train_tensor = create_sequences(est_AoA_wrtcell_train_tensor, self.S_len)
                UExyz_wrtBS_train_tensor = create_sequences(UExyz_wrtBS_train_tensor, self.S_len)
                UExyz_wrtcell_train_tensor = create_sequences(UExyz_wrtcell_train_tensor, self.S_len)
                BS_detected_train_tensor = create_sequences(BS_detected_train_tensor, self.S_len)
                BSlatlonh_train_tensor = create_sequences(BSlatlonh_train_tensor, self.S_len)
                BSxyz_wrtUE_train_tensor = create_sequences(BSxyz_wrtUE_train_tensor, self.S_len)
                BS_LOS_train_tensor = create_sequences(BS_LOS_train_tensor, self.S_len)
                BS_cell_angles_train_tensor = create_sequences(BS_cell_angles_train_tensor, self.S_len)
                pos_index_train_tensor = create_sequences(pos_index_train_tensor, self.S_len)
                true_ToA_train_tensor = create_sequences(true_ToA_train_tensor, self.S_len)
                true_AoA_wrtBS_train_tensor = create_sequences(true_AoA_wrtBS_train_tensor, self.S_len)
                true_AoA_wrtcell_train_tensor = create_sequences(true_AoA_wrtcell_train_tensor, self.S_len)

                # Create a new TensorDataset with sorted tensors
                train_dataset = TensorDataset(
                    x_train_tensor,
                    est_ToA_train_tensor,
                    est_AoA_wrtBS_train_tensor,
                    est_AoA_wrtcell_train_tensor,

                    UExyz_wrtBS_train_tensor,
                    UExyz_wrtcell_train_tensor,
                    
                    BS_detected_train_tensor,
                    BSlatlonh_train_tensor,
                    BSxyz_wrtUE_train_tensor,
                    BS_LOS_train_tensor,
                    BS_cell_angles_train_tensor,
                    pos_index_train_tensor,
                    true_ToA_train_tensor,
                    true_AoA_wrtBS_train_tensor,
                    true_AoA_wrtcell_train_tensor
                )

                # TESTING
                # Extract all tensors
                (
                    x_test_tensor, est_ToA_test_tensor, est_AoA_wrtBS_test_tensor,
                    est_AoA_wrtcell_test_tensor, UExyz_wrtBS_test_tensor, UExyz_wrtcell_test_tensor,
                    BS_detected_test_tensor, BSlatlonh_test_tensor, BSxyz_wrtUE_test_tensor,
                    BS_LOS_test_tensor, BS_cell_angles_test_tensor, pos_index_test_tensor,
                    true_ToA_test_tensor, true_AoA_wrtBS_test_tensor, true_AoA_wrtcell_test_tensor
                ) = test_dataset.tensors

                # Sort based on pos_index_test
                sorted_indices_test = torch.argsort(pos_index_test_tensor, 0)

                # Reorder all tensors
                x_test_tensor = x_test_tensor[sorted_indices_test]
                est_ToA_test_tensor = est_ToA_test_tensor[sorted_indices_test]
                est_AoA_wrtBS_test_tensor = est_AoA_wrtBS_test_tensor[sorted_indices_test]
                est_AoA_wrtcell_test_tensor = est_AoA_wrtcell_test_tensor[sorted_indices_test]

                UExyz_wrtBS_test_tensor = UExyz_wrtBS_test_tensor[sorted_indices_test]
                UExyz_wrtcell_test_tensor = UExyz_wrtcell_test_tensor[sorted_indices_test]

                BS_detected_test_tensor = BS_detected_test_tensor[sorted_indices_test]
                BSlatlonh_test_tensor = BSlatlonh_test_tensor[sorted_indices_test]
                BSxyz_wrtUE_test_tensor = BSxyz_wrtUE_test_tensor[sorted_indices_test]
                BS_LOS_test_tensor = BS_LOS_test_tensor[sorted_indices_test]
                BS_cell_angles_test_tensor = BS_cell_angles_test_tensor[sorted_indices_test]
                pos_index_test_tensor = pos_index_test_tensor[sorted_indices_test]
                true_ToA_test_tensor = true_ToA_test_tensor[sorted_indices_test]
                true_AoA_wrtBS_test_tensor = true_AoA_wrtBS_test_tensor[sorted_indices_test]
                true_AoA_wrtcell_test_tensor = true_AoA_wrtcell_test_tensor[sorted_indices_test]

                # Create sequences
                x_test_tensor = create_sequences(x_test_tensor, self.S_len)
                est_ToA_test_tensor = create_sequences(est_ToA_test_tensor, self.S_len)
                est_AoA_wrtBS_test_tensor = create_sequences(est_AoA_wrtBS_test_tensor, self.S_len)
                est_AoA_wrtcell_test_tensor = create_sequences(est_AoA_wrtcell_test_tensor, self.S_len)
                UExyz_wrtBS_test_tensor = create_sequences(UExyz_wrtBS_test_tensor, self.S_len)
                UExyz_wrtcell_test_tensor = create_sequences(UExyz_wrtcell_test_tensor, self.S_len)
                BS_detected_test_tensor = create_sequences(BS_detected_test_tensor, self.S_len)
                BSlatlonh_test_tensor = create_sequences(BSlatlonh_test_tensor, self.S_len)
                BSxyz_wrtUE_test_tensor = create_sequences(BSxyz_wrtUE_test_tensor, self.S_len)
                BS_LOS_test_tensor = create_sequences(BS_LOS_test_tensor, self.S_len)
                BS_cell_angles_test_tensor = create_sequences(BS_cell_angles_test_tensor, self.S_len)
                pos_index_test_tensor = create_sequences(pos_index_test_tensor, self.S_len)
                true_ToA_test_tensor = create_sequences(true_ToA_test_tensor, self.S_len)
                true_AoA_wrtBS_test_tensor = create_sequences(true_AoA_wrtBS_test_tensor, self.S_len)
                true_AoA_wrtcell_test_tensor = create_sequences(true_AoA_wrtcell_test_tensor, self.S_len)

                # Create a new TensorDataset with sorted tensors
                test_dataset = TensorDataset(
                    x_test_tensor,
                    est_ToA_test_tensor,
                    est_AoA_wrtBS_test_tensor,
                    est_AoA_wrtcell_test_tensor,

                    UExyz_wrtBS_test_tensor,
                    UExyz_wrtcell_test_tensor,

                    BS_detected_test_tensor,
                    BSlatlonh_test_tensor,
                    BSxyz_wrtUE_test_tensor,
                    BS_LOS_test_tensor,
                    BS_cell_angles_test_tensor,
                    pos_index_test_tensor,
                    true_ToA_test_tensor,
                    true_AoA_wrtBS_test_tensor,
                    true_AoA_wrtcell_test_tensor
                )

            # REMOVE
            test_dataset = train_dataset

            # Shuffle the sequence
            self.bool_shuffle = 1
    
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle, pin_memory=True)

        self.params.num_train_batches = len(train_loader)
        self.params.num_val_batches = len(test_loader)
        
        print_name_and_value(self.params.num_train_batches)
        print_name_and_value(self.params.num_val_batches)

        self.params.update_all()
        return train_loader, test_loader

    def return_dataset_tracking(self):

        """
        Dimension dataset: N_samples x N_agents (1 in this case) x N_measurements x Dimension_variable
        """

        # Timestep
        self.n = 0

        DB_tracking_path = os.path.join(DB_DIR, self.DB_name, 'Data', 'tracking')
        DB_tracking_path_mat = os.path.join(DB_tracking_path, 'dataset.mat')
        DB_tracking_path_pt = os.path.join(DB_tracking_path, 'dataset.pt')
    
        self.DB_training_path = os.path.join(DB_DIR, self.DB_name, 'Data', 'training')
        DB_training_path_mat = os.path.join(self.DB_training_path, 'dataset.mat')
        DB_training_path_pt = os.path.join(self.DB_training_path, 'dataset.pt')

        if not self.load_dataset:

            # import datasets
            try:
                raw_dataset = np.array(mat73.loadmat(DB_tracking_path_mat)['dataset_tot']) # dataset_tot
            except: 
                raw_dataset = scipy.io.loadmat(DB_tracking_path_mat)['dataset_tot'].flatten()
            print('Loaded tracking dataset')

            numPos_test = raw_dataset.shape[0]
            num_BS_per_pos_test = np.array([raw_dataset[pos]['auxiliary']['BS_detected'].size if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys() else 0 for pos in range(numPos_test)])

            # Measurements
            # first OFDM symbol
            # N_samples x N_measurements x Dimension_variable
            ADCPM_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['measurements']['ADCPM']).reshape(1, self.RXants, self.num_OFDM_symbols_per_pos, self.Ng, -1)[:,:,0,:,:].astype(float),((0, 0),(0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 3, 2, 1)
            est_ToA_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['measurements']['est_ToA']).reshape([1,-1]),((0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0)
            est_AoA_wrtBS_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['measurements']['est_AoA_wrtBS']).reshape([1,2,-1]).astype(float),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            est_AoA_wrtcell_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['measurements']['est_AoA_wrtcell']).reshape([1,2,-1]).astype(float),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)

            # Target 
            UE_index_test = np.concatenate([raw_dataset[pos]['target']['UE_index'].reshape([1,-1]) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0)
            UElatlonh_test =  np.array([raw_dataset[pos]['target']['UElatlonh'].reshape([3,-1])   for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]).squeeze()
            UExyz_global_test = from_latlongh_to_xyz_matrix(UElatlonh_test, self.latlonh_reference)
            UExyz_wrtBS_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['target']['UExyz_wrtBS']).reshape([1,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            UExyz_wrtcell_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['target']['UExyz_wrtcell']).reshape([1,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            UEvxvy_wrtBS_test = np.array([raw_dataset[pos]['target']['UEvxvy_wrtBS'].reshape([2,-1])    for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()]).squeeze()
            
            # Auxiliary
            BS_detected_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['BS_detected']).reshape([1,-1]),((0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0)
            BSlatlonh_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['BSlatlonh']).reshape([1,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            BSxyz_wrtUE_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['BSxyz_wrtUE']).reshape([1,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            BSxyz_global_test = np.concatenate([np.pad(np.transpose(from_latlongh_to_xyz_matrix(raw_dataset[pos]['auxiliary']['BSlatlonh'], self.latlonh_reference)).reshape([1,3,-1]),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            BS_LOS_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['BS_LOS']).reshape([1,-1]),((0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0)
            BS_cell_angles_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['BS_cell_angles']).reshape([1,2,-1]).astype(float),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            pos_index_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['pos_index']).reshape([1,-1]),((0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0)
            true_ToA_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['true_ToA']).reshape([1,-1]),((0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0)
            true_AoA_wrtBS_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['true_AoA_wrtBS']).reshape([1,2,-1]).astype(float),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)
            true_AoA_wrtcell_test = np.concatenate([np.pad(np.transpose(raw_dataset[pos]['auxiliary']['true_AoA_wrtcell']).reshape([1,2,-1]).astype(float),((0, 0), (0, 0), (0, np.max(num_BS_per_pos_test)-num_BS_per_pos_test[pos])),mode='constant',constant_values=(np.nan,)) for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()], 0).transpose(0, 2, 1)

            # Recover num BS in the positions considered
            num_BS_per_pos_test = np.array([raw_dataset[pos]['auxiliary']['BS_detected'].size for pos in range(numPos_test) if raw_dataset[pos] is not None and 'measurements' in raw_dataset[pos].keys()])

            print('Deleting raw testing dataset')
            del raw_dataset
            print('Deleted raw testing dataset')

        # Load dataset
        else:
            test_dataset = torch.load(DB_tracking_path_pt)

        mean_std = np.load(self.DB_training_path + '/mean_std_train.npy',  allow_pickle = True).tolist()
        self.x_train_mean = mean_std[0]
        self.params.x_train_mean = self.x_train_mean
        self.x_train_std = mean_std[1]
        self.params.x_train_std = self.x_train_std
        print(f'mean train: {self.x_train_mean}', f'std train: {self.x_train_std}')

        if not self.load_dataset:
            x_test = (ADCPM_test - self.x_train_mean)/self.x_train_std

            print_name_and_value(x_test.shape)

            test_dataset = TensorDataset(torch.tensor(x_test).float(), 
                                        torch.tensor(est_ToA_test).double(), 
                                        torch.tensor(est_AoA_wrtBS_test).double(), 
                                        torch.tensor(est_AoA_wrtcell_test).double(), 

                                        torch.tensor(UE_index_test).double(),
                                        torch.tensor(UElatlonh_test).double(),
                                        torch.tensor(UExyz_global_test).double(),
                                        torch.tensor(UExyz_wrtBS_test).double(),
                                        torch.tensor(UExyz_wrtcell_test).double(),
                                        torch.tensor(UEvxvy_wrtBS_test).double(),

                                        torch.tensor(BS_detected_test).double(),
                                        torch.tensor(num_BS_per_pos_test).double(),
                                        torch.tensor(BSlatlonh_test).double(),
                                        torch.tensor(BSxyz_wrtUE_test).double(),
                                        torch.tensor(BSxyz_global_test).double(),                                    
                                        torch.tensor(BS_LOS_test).double(),
                                        torch.tensor(BS_cell_angles_test).double(),
                                        torch.tensor(pos_index_test.astype(float)).double(),
                                        torch.tensor(true_ToA_test).double(),
                                        torch.tensor(true_AoA_wrtBS_test).double(),
                                        torch.tensor(true_AoA_wrtcell_test).double(),)

        if self.save_dataset:
            torch.save(test_dataset, DB_tracking_path_pt)

        self.dataset_output_names = ['est_ToA', 'est_AoA_wrtBS', 'est_AoA_wrtcell', 
                                     'UE_index', 'UElatlonh', 'UExyz_global', 'UExyz_wrtBS', 'UExyz_wrtcell', 'UEvxvy_wrtBS_test',
                                     'BS_detected', 'num_BS_per_pos', 'BSlatlonh', 'BSxyz_wrtUE', 'BSxyz_global', 'BS_LOS', 'BS_cell_angles', 'pos_index', 'true_ToA', 'true_AoA_wrtBS', 'true_AoA_wrtcell']
        self.params.dataset_output_names = self.dataset_output_names

        print_name_and_value(self.batch_size)
    
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=0, pin_memory=True)

        self.params.num_val_batches = len(test_loader)
        
        print_name_and_value(self.params.num_val_batches)

        self.params.update_all()


        # x_n = F x_n-1 + W w_n-1
        # x: pos, vel, acc. w: pos_noise, vel_noise, acc_noise
        self.F = np.array([[1, 0, 0                      , self.T_between_timestep, 0                             , 0                             , (self.T_between_timestep**2)/2,                              0,                              0],
                           [0, 1, 0                      , 0                      , self.T_between_timestep       , 0                             , 0                             , (self.T_between_timestep**2)/2,                              0],                           
                           [0, 0, 1                      , 0                      , 0                             , self.T_between_timestep       , 0                             , 0                             , (self.T_between_timestep**2)/2],
                           
                           [0, 0, 0                      , 1                      , 0                             , 0                             , self.T_between_timestep       , 0                             ,                              0],
                           [0, 0, 0                      , 0                      , 1                             , 0                             , 0                             , self.T_between_timestep       ,                              0],
                           [0, 0, 0                      , 0                      , 0                             , 1                             , 0                             , 0                             ,        self.T_between_timestep],

                           [0, 0, 0                      , 0                      , 0                             , 0                             , 1                             , 0                             ,                              0],
                           [0, 0, 0                      , 0                      , 0                             , 0                             , 0                             , 1                             ,                              0],
                           [0, 0, 0                      , 0                      , 0                             , 0                             , 0                             , 0                             ,                              1]])
        self.W = copy(self.F)


        return test_loader

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

            mean_std = np.load(self.DB_training_path + '/mean_std_train.npy',  allow_pickle = True).tolist()
            self.x_train_mean = mean_std[0]
            self.params.x_train_mean = self.x_train_mean
            self.x_train_std = mean_std[1]
            self.params.x_train_std = self.x_train_std

            # Plot 
            plot_ray_tracing_dataset(self, loader, self.params, file_name = 'ray_tracing_dataset', xlabel_ = 't1', ylabel_ = 't2', title_ = '', logx = False, logy = False, xlim = None, ylim = None, save_eps = 1, ax = None, save_svg = 0, save_pdf = 1, save_jpg=1, plt_show = 0)

    def show_uncertainty_dataset(self, return_model_evaluate, train_loader = None, val_loader = None, file_name_uncertainty_dataset = 'uncertainty_dataset'):

        x_train, t_train = dataloader_to_numpy(train_loader)
        x_valid, t_valid = dataloader_to_numpy(val_loader)

        mean_std = np.load(self.DB_training_path + '/mean_std_train.npy',  allow_pickle = True).tolist()
        self.x_train_mean = mean_std[0]
        self.params.x_train_mean = self.x_train_mean
        self.x_train_std = mean_std[1]
        self.params.x_train_std = self.x_train_std

        x_train = x_train * self.params.x_train_std + self.params.x_train_mean 
        x_valid = x_valid * self.params.x_train_std + self.params.x_train_mean 

        # Model prediction
        plot_uncertainty_artificial_2D_dataset(x_train, t_train, x_valid, t_valid, return_model_evaluate, self.params, file_name = file_name_uncertainty_dataset, xlabel_ = 't1', ylabel_ = 't2', title_ = self.BNN_algorithm, logx = False, logy = False, xlim = None, ylim = None, save_eps = 0, ax = None, save_svg = 0, save_jpg=1, save_pdf = 0, plt_show = 0)

    def show_dataset_tracking(self):

        return 

    def show_tracking_results(self, output_results_tracking):

        plot_tracking_results_ray_tracing2(output_results_tracking,  self.params, file_name = 'tracking_results2', 
                                          xlabel_ = 't1', ylabel_ = 't2', title_ = '', 
                                          logx = False, logy = 0, xlim = None, ylim = [0.01, 10], 
                                          save_eps = 1, ax = None, save_svg = 0, save_pdf = 1, save_jpg=1, plt_show = 1)

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

