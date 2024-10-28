import os
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("error")
from scipy.stats import gaussian_kde
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from matplotlib.gridspec import GridSpec
from functools import reduce
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from math import radians, sin, cos, sqrt, atan2
import seaborn as sns

import Classes.utils.uncertainty_toolbox as uct
from Classes.utils.utils import format_dBm, asCartesian2D, from_xyz_to_latlonh_matrix

##########################################
############### ARTIFICIAL ###############
##########################################
def plot_artificial_dataset(x_train, t_train, x_valid, t_valid, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

    # Plot training points
    ax.scatter(x_train, t_train, s = 10, marker = 'x', color = 'black', alpha = 0.5)
    # ax.legend(['training points'], shadow=False, fontsize=fontsize, loc='best')  

    # Plot validation points
    ax.scatter(x_valid, t_valid, s = 10, marker = 'o', color = 'blue', alpha = 0.5)
    ax.legend(['training points', 'validation points'], shadow=False, fontsize=fontsize, loc='best')  
    
    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log', nonposy='clip')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)



def plot_uncertainty_artificial_dataset(x_train, t_train, x_valid, t_valid, return_model_evaluate, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Save variables returned from the model evaluate as local variables
    for var_name, var_value in return_model_evaluate.items(): globals()[var_name] = var_value.reshape(-1)

    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

    # Plot training points
    ax.scatter(x_train, t_train, s = 10, marker = 'x', color = 'black', alpha = 0.5)
    # Plot validation points
    # ax.scatter(x_valid, t_valid, s = 10, marker = 'o', color = 'blue', alpha = 0.5)
    
    # Plot uncertainties
    ax.fill_between(x, y_mean + aleatoric_std, y_mean + total_unc_std, color = c[0], alpha = 0.3, label = 'Epistemic + Aleatoric')
    ax.fill_between(x, y_mean - total_unc_std, y_mean - aleatoric_std, color=c[0], alpha=0.3, label='_nolegend_')
    ax.fill_between(x, y_mean - aleatoric_std, y_mean + aleatoric_std, color = c[1], alpha = 0.4, label = 'Aleatoric')
    # Plot mean prediction
    ax.plot(x, y_mean, color = 'black', linewidth = 1)

    # ax.legend(['training points', 'validation points', 'Epistemic and Aleatoric', 'Aleatoric', 'Mean prediction'], shadow=False, fontsize=fontsize, loc='best')  
    ax.legend(['training points', 'Epistemic and Aleatoric', 'Aleatoric', 'Mean prediction'], shadow=False, fontsize=fontsize, loc='best')  


    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log', nonposy='clip')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'),bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)




##############################################################################################################################
############# ARTIFICIAL 2D ##############
##############################################################################################################################
def plot_artificial_2D_dataset(dataset, x_train, t_train, x_valid, t_valid, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):
    
    sig_noise_t1 = dataset.sig_noise_t1
    density_t1 = dataset.density_t1


    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

    if params.sig_noise_t1_min != params.sig_noise_t1_max:
        # Aleatoric uncertainty: use the standard deviation of the noise for the color map
        sig_noise_t1 = dataset.sig_noise_t1
        norm = plt.Normalize(params.sig_noise_t1_min, params.sig_noise_t1_max)
        cmap = plt.get_cmap('jet')
        c_label = 'Std Noise [m]'
    else:
        # Epistemic uncertainty: use density for the color map

        density = density_t1
        norm = plt.Normalize(min(density), max(density))

        cmap = plt.get_cmap('jet_r')
        c_label = 'Density [points/m^2]'
        density = max(density) - density

    sc = ax.scatter(x_train[:, 0], x_train[:, 1], c=sig_noise_t1 if params.sig_noise_t1_min != params.sig_noise_t1_max else density, s=10, marker='o', cmap=cmap, norm=norm, alpha=0.5)
    ax.legend(['training points'], shadow=False, fontsize=fontsize, loc='best')

    # Add colorbar for the colormap
    cbar = plt.colorbar(sc, orientation='vertical')
    cbar.set_label(c_label)

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log', nonposy='clip')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

def plot_added_noise_artificial_2D_dataset(t1_values, added_noise, params, file_name, xlabel_ = 't1 Values', ylabel_ = 'Actual Added Noise', title_ = 'Actual Added Noise as a Function of t1', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

    ax.scatter(t1_values, added_noise[:, 0], s=1, c='green')

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log', nonposy='clip')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    
    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'),bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

def plot_std_noise_artificial_2D_dataset(t1_values, added_noise, x_train, params, file_name, xlabel_='t1 Bin Centers', ylabel_='Estimated Standard Deviation of Added Noise', title_='Estimated Standard Deviation of Added Noise as a Function of t1', logx=False, logy=False, xlim=None, ylim=None, fontsize=18, labelsize=18, save_eps=0, ax=None, save_svg=0, save_pdf=0, save_jpg=0, plt_show=1):
    
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

    # Define the number of bins
    num_bins = 100
    # Create bins for t1
    bin_edges = np.linspace(params.t1_min, params.t1_max, num_bins + 1)
    # Initialize an array to hold the standard deviation for each bin
    metric_bins = np.zeros(num_bins)
    
    # Loop through each bin and calculate the metric (standard deviation of the noise or density)
    for i in range(num_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        mask = (t1_values >= lower_bound) & (t1_values < upper_bound)
        
        if params.sig_noise_t1_min != params.sig_noise_t1_max:
            # Aleatoric uncertainty: calculate the standard deviation of the noise
            metric_bins[i] = np.std(added_noise[mask, 0])
        else:
            # Epistemic uncertainty: calculate the density of training points
            tree = cKDTree(x_train[mask])
            density = tree.query_ball_point(x_train[mask], r=0.1, return_length=True)
            metric_bins[i] = np.mean(density) if density.size > 0 else 0

    # Compute the center of each bin for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.scatter(bin_centers, metric_bins, c='purple')

    # Update ylabel for the case of epistemic uncertainty
    if params.sig_noise_t1_min == params.sig_noise_t1_max:
        ylabel_ = 'Mean Density of Training Points'

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log', nonposy='clip')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    plt.tick_params(labelsize=labelsize)
    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)

    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'),bbox_inches='tight')
    if save_jpg:
        plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

def plot_uncertainty_artificial_2D_dataset(x_train, t_train, x_valid, t_valid, return_model_evaluate, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Save variables returned from the model evaluate as local variables
    for var_name, var_value in return_model_evaluate.items(): globals()[var_name] = var_value

    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

    if params.sig_noise_t1_min != params.sig_noise_t1_max:
        
        # Plot uncertainties
        norm = plt.Normalize(params.sig_noise_t1_min, params.sig_noise_t1_max)
        cmap = plt.get_cmap('jet')

        aleatoric_std =  np.sqrt(np.einsum('...ii->...', aleatoric_cov))

    else:

        norm = plt.Normalize(params.sig_noise_t1_min, params.sig_noise_t1_max*10)
        cmap = plt.get_cmap('jet')

        epistemic_std =  np.sqrt(np.einsum('...ii->...', epistemic_cov))


    sc = ax.scatter(x[:, 0], x[:, 1], c=aleatoric_std if params.sig_noise_t1_min != params.sig_noise_t1_max else epistemic_std, s=10, marker='o', cmap=cmap, alpha=0.5) # norm=norm
    ax.legend(['validation points'], shadow=False, fontsize=fontsize, loc='best')  

    # Add colorbar for the colormap
    cbar = plt.colorbar(sc, orientation='vertical')
    cbar.set_label('Std prediction [m]')

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log', nonposy='clip')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'),bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

def plot_artificial_2D_dataset_tracking(dataset, t_test, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):
    
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

    t_test = np.array([v for k,v in t_test.items()])
    for a in range(len(t_test)):
        sc = ax.plot(t_test[a,:, 0], t_test[a,:, 1], marker='o')
        ax.legend([f'Target {a}'], shadow=False, fontsize=fontsize, loc='best')

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log', nonposy='clip')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    log_path = params.Figures_tracking_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

def plot_tracking_results_artificial_2D_dataset(output_results_tracking, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    METHODS = output_results_tracking.keys()
    number_state_variables = params.number_state_variables
    num_agents = params.num_agents
    num_instants = params.size_dataset

    fig = plt.figure(figsize=(25,10), constrained_layout=True)
    gs = GridSpec(int(number_state_variables/2), 2, figure=fig)
    ax_scenario = fig.add_subplot(gs[:, 0])
    ax_error = [fig.add_subplot(gs[var, 1]) for var in range(int(number_state_variables/2))]
    plt.rcParams.update({'font.size': 22}) 
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['GT'] + ['GNSS'] + list(METHODS) 
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    temp_line_style_indexes_per_method = ['-', '-', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}

    # Plot scenario
    for i, name_method in enumerate(METHODS):

        # num_agents x num_instants x dimensions
        t = np.array([np.array([np.array(output_results_tracking[name_method]['t'][n][agent]).reshape(-1, 1) for n in range(1, len(output_results_tracking[name_method]['t']))]) for agent in range(num_agents)]).reshape(num_agents, num_instants+1, -1)
        # num_agents x num_instants x state dimensions
        t_hat = np.array([np.array([np.array(output_results_tracking[name_method]['t_hat'][n][agent]).reshape(-1, 1) for n in range(1, len(output_results_tracking[name_method]['t_hat']))]) for agent in range(num_agents)]).reshape(num_agents, num_instants+1, -1)
        # num_agents x num_instants
        AE_pos_per_agent_per_timestep = np.array([ output_results_tracking[name_method]['metrics']['AE_pos_per_agent_per_timestep'][agent] for agent in range(num_agents)]).reshape(num_agents,-1)
        # num_instants x 1
        RMSE_pos_per_timestep = np.sqrt(np.mean(AE_pos_per_agent_per_timestep**2, 0)) # mean between agents
        # num_agents x num_instants
        AE_vel_per_agent_per_timestep = np.array([ output_results_tracking[name_method]['metrics']['AE_vel_per_agent_per_timestep'][agent] for agent in range(num_agents)])
        # num_instants x 1
        RMSE_vel_per_timestep = np.sqrt(np.mean(AE_vel_per_agent_per_timestep**2, 0)) # mean between agents

        # plot GT
        if i == 0:
            pos_GT = t
            for agent in range(num_agents):
                ax_scenario.plot(pos_GT[agent,:,0], pos_GT[agent,:,1], linestyle=line_style_indexes_per_method['GT'], color=colors[color_indexes_per_method['GT']])

        # plot method
        pos_method = t_hat
        for agent in range(num_agents):
            ax_scenario.plot(pos_method[agent,:,0], pos_method[agent,:,1], linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])

        ax_error[0].plot(range(len(RMSE_pos_per_timestep)), reduce(lambda x, _: x, range(1), RMSE_pos_per_timestep), linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
        print('{}: Mean RMSE pos [m]: {:.3f}'.format(name_method, np.sqrt(np.mean(AE_pos_per_agent_per_timestep**2)))) # mean between agents and timesteps
        print('{}: Mean MAE pos [m]: {:.3f}'.format(name_method, np.mean(AE_pos_per_agent_per_timestep))) # mean between agents and timesteps
        if len(ax_error) >= 2:
            ax_error[1].plot(range(len(RMSE_vel_per_timestep)), (RMSE_vel_per_timestep), linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
            print('{}: Mean RMSE vel [m]: {:.3f}'.format(name_method, np.sqrt(np.mean(AE_vel_per_agent_per_timestep**2)))) # mean between agents and timesteps
            print('{}: Mean MAE vel [m]: {:.3f}'.format(name_method, np.mean(AE_vel_per_agent_per_timestep))) # mean between agents and timesteps
            if len(ax_error) >= 3:
                ax_error[2].plot(range(len(RMSE_acc_per_timestep)), (RMSE_acc_per_timestep), linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
                print('{}: Mean RMSE acc [m]: {:.3f}'.format(name_method, np.sqrt(np.mean(AE_acc_per_agent_per_timestep**2)))) # mean between agents and timesteps
                print('{}: Mean MAE acc [m]: {:.3f}'.format(name_method, np.mean(AE_acc_per_agent_per_timestep))) # mean between agents and timesteps

    # ax_scenario.grid(alpha=0.3)
    ax_scenario.axis('equal')
    ax_scenario.legend(legend_labels, loc='best', shadow=False)

    if xlim is not None and ylim is not None:
        ax_scenario.set(xlim=xlim, ylim=ylim)

    if ylabel_ is not None and xlabel_ is not None:       
        ax_scenario.set(xlabel=xlabel_)
        ax_scenario.xaxis.label.set_size(fontsize)
        ax_scenario.set(ylabel=ylabel_)
        ax_scenario.yaxis.label.set_size(fontsize)

    # Plot errors
    ax_error[0].legend([k for k in METHODS], loc='best', shadow=False)
    ax_error[0].set(xlabel='Time n')
    ax_error[0].set(ylabel='RMSE pos [m]')
    # ax_error[0].grid(alpha=0.3)
    if len(ax_error) >= 2:
        ax_error[1].legend([k for k in METHODS], loc='best', shadow=False)
        ax_error[1].set(xlabel='Time n')
        ax_error[1].set(ylabel='RMSE vel [m/s]')
        # ax_error[1].grid(alpha=0.3)
        if len(ax_error) >= 3:
            ax_error[2].legend([k for k in METHODS], loc='best', shadow=False)
            ax_error[2].set(xlabel='Time n')
            ax_error[2].set(ylabel='RMSE acc [m/s^2]')
            
    ax_scenario.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)
    # plt.rcParams.update({'font.size': 30})

    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    log_path = params.Figures_tracking_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)


##############################################################################################################################
############# RAY TRACING ##############
##############################################################################################################################

def plot_ray_tracing_dataset(dataset, loader, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    with torch.no_grad():

        for i, (input_data, *other_data) in enumerate((loader)): 

            output_additional_data = {dataset_output_name:other_data[index_name]  for index_name,(dataset_output_name) in enumerate(params.dataset_output_names)}

            # input_all_data = {'input_data': input_data}
            # output_all_additional_data = {'output_additional_data': output_additional_data}

            for sample in range(95, params.batch_size):
                
                print(sample)

                x = np.squeeze(input_data[sample].data.cpu().numpy()) * params.x_train_std + params.x_train_mean
                # UElatlonh = np.squeeze(output_additional_data['UElatlonh'][sample].data.cpu().numpy())
                UExyz_wrtBS = np.squeeze(output_additional_data['UExyz_wrtBS'][sample].data.cpu().numpy())
                BS_LOS = np.squeeze(output_additional_data['BS_LOS'][sample].data.cpu().numpy())
                BS_detected = np.squeeze(output_additional_data['BS_detected'][sample].data.cpu().numpy())
                BSlatlonh = np.squeeze(output_additional_data['BSlatlonh'][sample].data.cpu().numpy())
                pos_index = np.squeeze(output_additional_data['pos_index'][sample].data.cpu().numpy())
                UElatlonh = np.squeeze(from_xyz_to_latlonh_matrix(UExyz_wrtBS, BSlatlonh))

                # params.Figures_dir, params.Figures_tracking_dir
                plot_reconstructed_samples(input_=x, sample_number=sample, log_path = params.Figures_dir, file_name = f'ADCPM_BS_detected_{BS_detected}_BSlatlonh_{BSlatlonh}_BS_LOS_{BS_LOS}_pos_index_{pos_index}_UElatlonh_{UElatlonh}', remove_axis_ticks=0, flat_image=0, plt_show=0, polar=0, cmap='jet')

def plot_reconstructed_samples(input_, sample_number, log_path, title_ = None, file_name = '', output_ = None, save_pdf = 1, save_png = 1, save_eps = 0, save_svg = 0, save_jpg=0, plt_show = 0, fontsize = 20, remove_axis_ticks = 0, flat_image = 0, polar = 0, cmap='parula'):
    
    cmap = 'jet'

    if not polar:
            
        X = np.arange(0, input_.shape[1], 1)
        Y = np.arange(0, input_.shape[0], 1)
        X, Y = np.meshgrid(X, Y)
        #X, Y = np.mgrid[0:input_.shape[0], 0:input_.shape[1]]
        X.shape
        Y.shape
        fig = plt.figure(figsize=(20,20))
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')

        # POSITIONING PAPER for no background but keeping axes
        # Make the grid and labels opaque
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        # Make the panes (background) transparent
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Make the figure background transparent
        fig.patch.set_alpha(0)

        surf = ax.plot_surface(X, Y, input_, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)


        # POSITIONING PAPER
        ax.grid(False)

        # ax.set_zlim(-1.01, 1.01)
        # fig.colorbar(surf, shrink=0.5, aspect=5)

        if remove_axis_ticks:
            plt.axis('off')
        else:
            plt.minorticks_off()
            ax.zaxis.set_major_formatter(plt.FuncFormatter(format_dBm))
            plt.xticks(fontsize=fontsize, rotation=0)
            plt.yticks(fontsize=fontsize, rotation=0)
            ax.zaxis.set_tick_params(labelsize=fontsize)

            # POSITIONING PAPER)
            # labels = ax.get_xticks().tolist()
            # Reverse the order of the labels
            # labels = [int(label) for label in labels[::-1]]
            # Set the new labels
            # ax.set_xticklabels(labels)
            ax.invert_xaxis()

        if flat_image:
            # Flat image
            plt.show(block=False)
            plt.pause(1)
            vmin, vmax =  surf.get_clim()
            plt.pause(1)
            surf.set_clim(vmin, vmax/3) 
            ax.view_init(90, 0)
        else:

            # adjust colormap (POSITIONING PAPER)
            plt.show(block=False)
            plt.pause(1)
            vmin, vmax =  surf.get_clim()
            plt.pause(1)
            surf.set_clim(vmin, vmax/3) 

            # vertical for input image (POSITIONING PAPER)
            # ax.view_init(13, -117)
            ax.view_init(30, -40)

    else:

        x_dim = int(input_.shape[0]) 
        y_dim = int(input_.shape[1]) 

        x = np.arange(0, x_dim, 1)
        y = np.arange(0, y_dim, 1)
        X,Y = np.meshgrid(y, x) 
        interp = LinearNDInterpolator(np.stack((X.flatten(), Y.flatten()), 1), input_.flatten())

        max_ind = np.argmax(input_, 0)[0]       # index of delays with highest power
        max_phi_theta = np.argmax(input_, 1)[0] # index of angles with highest power
        max_ = np.max(input_.flatten())

        # interpolate angles
        y = np.arange(0, y_dim, 0.1)
        X,Y = np.meshgrid(y, x) 
        input_ = interp(X, Y)

        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111, projection='3d')

        X_cart, Y_cart = asCartesian2D(Y, X)

        surf = ax.plot_surface(X_cart, Y_cart, input_, rstride=1, cstride=1, cmap=cmap, linewidth=0, antialiased=False)
        
        plt.show(block=False)
        plt.pause(1)
        vmin, vmax =  surf.get_clim()
        plt.pause(1)
        surf.set_clim(vmin, vmax/2) 
        if remove_axis_ticks:
            plt.axis('off')
        
        ax.view_init(30, -40)


    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

    plt.clf()
    plt.close()

def plot_sparse_reconstructed_samples (input_, sample_number, label, log_path, file_name = '', output_ = None, save_eps = 0, plt_show = 0, fontsize = 20, polar = 0):
    

    try:

        x_dim = int(np.sqrt(input_.shape[1]))
        y_dim = x_dim
        z_dim = input_.shape[0]

        input_ = input_.reshape((z_dim, x_dim, y_dim))
        x = np.arange(0, x_dim, 1)
        y = np.arange(0, y_dim, 1)
        z = np.arange(0, z_dim, 1)
        X,Y,Z = np.meshgrid(y, z, x)
        interp = LinearNDInterpolator(np.stack((X.flatten(), Y.flatten(), Z.flatten()), 1), input_.flatten())

        max_ind = np.argmax(input_, 0)[0][0]
        max_phi = np.argmax(input_, 1)[0][0]
        max_theta = np.argmax(input_, 2)[0][0]
        max_ = np.max(input_.flatten())
        if polar == 0:

            x = np.arange(0, x_dim, 0.1)
            y = np.arange(0, y_dim, 0.1) 
            z = np.arange(max_ind-7, max_ind+7, 0.1) 
            X,Y,Z = np.meshgrid(y, z, x)
            input_ = interp(X, Y, Z)
            input_[input_<7] = np.nan
            X = X[~np.isnan(input_)]
            Y = Y[~np.isnan(input_)]
            Z = Z[~np.isnan(input_)]
            input_ = input_[~np.isnan(input_)]

            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')

            p = ax.scatter(X, Y, Z, c=input_, 
                        marker='o', cmap=parula_color(), 
                        alpha = 1, #(input_-np.min(input_))/(np.max(input_)-np.min(input_)), 
                        vmin=0, vmax=max_)
            p._facecolors[:, 3] = np.clip((input_-np.min(input_))/(np.max(input_)-np.min(input_)) + 0.1, 0, 1)  # 0.2
            c = p._facecolors

            ax.clear()

            # cbar = fig.colorbar(p, fraction=0.046, pad=0.04)
            cbar = fig.colorbar(p, fraction=0.046, pad=0.1)
            
            p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=c, 
                        marker='o', cmap=parula_color(), 
                        vmin=0, vmax=max_,
                        edgecolors='none')
            plt.ylim([max_ind-7, max_ind+7]) # [55, 65]


            h, yedges, zedges = np.histogram2d(Y.flatten(), Z.flatten(), bins=50)
            h = h.transpose()
            normalized_map = plt.cm.Blues(h/h.max())
            yy, zz = np.meshgrid(yedges, zedges)
            xpos = ax.get_xlim()[0]-0 # Plane of histogram
            xflat = np.full_like(yy, xpos) 
            p = ax.plot_surface(xflat, yy, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

            h, xedges, zedges = np.histogram2d(X.flatten(), Z.flatten(), bins=50)
            h = h.transpose()
            normalized_map = plt.cm.Blues(h/h.max())
            xx, zz = np.meshgrid(xedges, zedges)
            ypos = ax.get_ylim()[1]+0 # Plane of histogram
            yflat = np.full_like(xx, ypos) 
            p = ax.plot_surface(xx, yflat, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

            h, xedges, yedges = np.histogram2d(X.flatten(), Y.flatten(), bins=50)
            h = h.transpose()
            normalized_map = plt.cm.Blues(h/h.max())
            xx, yy = np.meshgrid(xedges, yedges)
            zpos = ax.get_zlim()[0]-0 # Plane of histogram
            zflat = np.full_like(xx, zpos) 
            p = ax.plot_surface(xx, yy, zflat, facecolors=normalized_map, rstride=1, cstride=1, shade=False)

            plt.minorticks_off()
            ax.zaxis.set_major_formatter(plt.FuncFormatter(format_dBm))
            plt.xticks(fontsize=fontsize, rotation=0)
            plt.yticks(fontsize=fontsize, rotation=0)

            ax.zaxis.set_tick_params(labelsize=fontsize)

            a=ax.get_xticks().tolist()
            a = [str(int(float(el)*360/x_dim ))  for el in a]
            ax.set_xticklabels(a)
            ax.set_xlim([0.1, 6.9])

            a=ax.get_yticks().tolist()
            a = [str(int(float(el)*(8.6)))  for el in a]
            ax.set_yticklabels(a)
            ax.set_ylim([52.1, 65.9])

            a=ax.get_zticks().tolist()
            a = [str(int(float(el)*180/y_dim ))  for el in a]
            ax.set_zticklabels(a, ha='left' )#va='bottom', )
            ax.set_zlim([-0.9, 7.9])

            ax.view_init(27, -53)
           
            plt.savefig(log_path + f'/{file_name}_sparse_input_sample_{sample_number}_label_{label}_energy_{sample_energy}.pdf', bbox_inches='tight')
            if save_eps:
                plt.savefig(log_path + f'/{file_name}_sparse_input_sample_{sample_number}_label_{label}_energy_{sample_energy}.eps', format='eps', bbox_inches='tight')
            if plt_show:
                plt.show()

        else:

            x = np.arange(max(0, max_theta-2), min(x_dim, max_theta+2), 0.1)
            y = np.arange(max(0, max_phi-2), min(y_dim, max_phi+2), 0.1)
            z = np.arange(0, min(z_dim, max_ind), 0.1)
            X,Y,Z = np.meshgrid(y, z, x)
            input_ = interp(X, Y, Z)
            input_[input_<0] = np.nan
            X = X[~np.isnan(input_)]
            Y = Y[~np.isnan(input_)]
            Z = Z[~np.isnan(input_)]
            input_ = input_[~np.isnan(input_)]

            fig = plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111, projection='3d')

            X_cart, Y_cart, Z_cart = asCartesian(Y, Z, X)

            p = ax.scatter(X_cart, Y_cart, Z_cart, c=input_, 
                        marker='o', cmap=parula_color(), 
                        alpha = 1, #(input_-np.min(input_))/(np.max(input_)-np.min(input_)), 
                        vmin=0, vmax=max_)
            p._facecolors[:, 3] = np.clip((input_-np.min(input_))/(np.max(input_)-np.min(input_)) - 0.1, 0, 1)  # 0.2
            c = p._facecolors

            ax.clear()

            cbar = fig.colorbar(p)
            p = ax.scatter(X_cart.flatten(), Y_cart.flatten(), Z_cart.flatten(), c=c, 
                        marker='o', cmap=parula_color(), 
                        vmin=0, vmax=max_,
                        edgecolors='none')

            # azimuth
            R = np.linspace(0, max_ind, 100)
            h = 0
            u_pi = np.linspace(0,  np.pi, 100)
            u_2pi = np.linspace(0,  2*np.pi, 100)
            x_ = np.outer(R, np.cos(u_2pi))
            y_ = np.outer(R, np.sin(u_2pi))
            zflat = np.full_like(x_, h) 
            ax.plot_surface(x_,-y_,zflat, rstride=1, cstride=1, shade=False, color='tab:blue', alpha=0.5, linewidth=0)
            ax.plot(max_ind*np.cos(u_2pi), max_ind*np.sin(u_2pi), h, color='tab:blue')

            # ax.set_xticks(locs)
            # ax.set_xticklabels(labels)

            # elevation
            x_ = np.outer(R, np.cos(u_pi))
            y_ = np.outer(R, np.sin(u_pi))
            zflat = np.full_like(x_, h) 
            yflat = zflat
            h = np.full_like(max_ind*np.cos(u_pi), 0) 
            z_ = y_
            ax.plot_surface(x_,yflat,z_, rstride=1, cstride=1, shade=False, color='tab:blue', alpha=0.5, linewidth=0)
            ax.plot(max_ind*np.cos(u_pi), h, max_ind*np.sin(u_pi), color='tab:blue')


            ax.set_zlim(0, max_ind)

            plt.xticks(fontsize=fontsize, rotation=0)
            plt.yticks(fontsize=fontsize, rotation=0)
            ax.zaxis.set_tick_params(labelsize=fontsize)
            plt.axis('off')
            ax.view_init(27, -53)        
            plt.savefig(log_path + f'/{file_name}_sparse_polar_input_sample_{sample_number}_label_{label}.pdf', bbox_inches='tight')
            if save_eps:
                plt.savefig(log_path + f'/{file_name}_sparse_polar_input_sample_{sample_number}_label_{label}.eps', format='eps', bbox_inches='tight')
            if plt_show:
                plt.show()
            
    except:
        pass

def plot_box_plot_MAE(results, labels, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    # Set properties for outliers
    flierprops = dict(marker='d', markerfacecolor='#3f3f3f', markersize=5, linestyle='none')

    num_box = len(labels)

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    # Create boxplots
    bp = []
    for box in range(num_box):
        bp.append(ax.boxplot(results[labels[box]], positions=[box], widths=0.6, patch_artist=True, flierprops=flierprops)) # boxprops=dict(facecolor="C1")

    # Calculate median values
    medians = [np.median(results[labels[box]]) for box in range(num_box)]

    # Draw a line connecting the medians of the second, third, and fourth boxplots
    ax.plot(list(range(num_box)), medians, color='red', linestyle='-', linewidth=2)

    # Setting labels for x-axis
    plt.xticks(list(range(num_box)),  [f"L = {labels[box]}" for box in range(num_box)])

    plt.xticks(rotation=0, ha='right')
    plt.subplots_adjust(bottom=0.30)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel_ is not None:
        plt.ylabel(ylabel_, fontsize=fontsize)  
    if xlabel_ is not None:       
        plt.xlabel(xlabel_, fontsize=fontsize)
    if logx:
        plt.xscale('log', nonposy='clip')
    if logy:
        plt.yscale('log')
    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    # ax.set_aspect('equal', adjustable='box')
    plt.tick_params(labelsize=labelsize)


    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    # ax.grid()
    # plt.gca().yaxis.grid(alpha=0.3)
    # plt.gca().xaxis.grid(alpha=0.3)
    # plt.rcParams.update({'font.size': 30})

    log_path = params.Figures_paper_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

def plot_predicted_uncertainties_ray_tracing(lats, lons, total_unc_std, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):

    if ax is None:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)

    # BS .pts file
    filename = os.path.join(params.DB_DIR, 'Dataset_Positioning', 'Positions', 'BaseStations.pts')
    base_stations = np.loadtxt(filename, skiprows=5)  # Adjust the skiprows parameter to skip header lines
    base_station_lats = base_stations[:, 2]
    base_station_lons = base_stations[:, 1]

    # Plot the Base Station positions on the same axis
    ax.scatter(base_station_lons, base_station_lats, c='white', marker='^', s=50, zorder=5)  # Setting zorder to make sure they are on top

    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Radius of the Earth in kilometers
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) * sin(dlat / 2) + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) * sin(dlon / 2)
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance

    # Calculate the distance in kilometers between the minimum and maximum latitudes and longitudes
    # Original
    aspect_ratio = (max(lons) - min(lons)) / (max(lats) - min(lats)) 

    # Setting the same aspect ratio for both plots
    extent = [min(lons), max(lons), 
              min(lats), max(lats)]    

    # Calculate the counts for each bin
    hist_count, _, _ = np.histogram2d(lons, lats, bins=[200, 200])

    # Calculate the weighted histogram
    hist, _, _ = np.histogram2d(lons, lats, bins=[200, 200], weights=total_unc_std)

    # Avoid division by zero
    hist_count[hist_count == 0] = 1

    # Normalize the bins
    normalized_hist = hist /(hist_count ** 2.1)

    # If you have regions with no data and you want them to be black, you can set them to a high value
    normalized_hist[hist_count == 1] = np.max(normalized_hist) * 8

    # Plotting
    color_min = 0.1 # hist_count.min()   # 6
    color_max = 10# hist_count.max() # 100
    log_norm = LogNorm(vmin=color_min, vmax=color_max)
    # log_norm = LogNorm()
    c = ax.imshow(normalized_hist.T, origin='lower', extent=extent, aspect='equal', cmap='hot_r', norm=log_norm)

    cbar = fig.colorbar(c, ax=ax, pad=0.01)
    cbar.ax.tick_params(labelsize=14)

    # Adding labels and title to the plot
    # Increase size of x and y labels
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Latitude', fontsize=16)
    # Increase size of x and y ticks
    ax.tick_params(axis='both', which='major',  labelsize=16)
    # Convert decimal degrees to DMS format for tick labels
    lat_dms = ["{:.0f}°{:.0f}'{:.0f}\"N".format(int(abs(d)), int(abs(d)*60) % 60, int(abs(d)*3600) % 60) for d in ax.get_yticks()]
    lon_dms = ["{:.0f}°{:.0f}'{:.0f}\"W".format(int(abs(d)), int(abs(d)*60) % 60, int(abs(d)*3600) % 60) for d in ax.get_xticks()]
    # Set custom tick labels
    ax.set_xticklabels(lon_dms)  
    ax.set_yticklabels(lat_dms)
    # For ax
    # Capture original x-limits
    original_xlim_ax2 = ax.get_xlim()
    # Subset the x-ticks and their labels
    subset_xticks_ax2 = ax.get_xticks()[::2]
    subset_lon_dms_ax2 = lon_dms[::2]  # Assuming lat_dms for ax1 and ax are the same
    # Set x-ticks and their labels
    ax.set_xticks(subset_xticks_ax2)
    ax.set_xticklabels(subset_lon_dms_ax2)
    # Set x-limits back to original
    ax.set_xlim(original_xlim_ax2)

    # ax.set_title('Point Density Heatmap (log scale)')
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_aspect(aspect_ratio)


    # Display the plot
    plt.tight_layout()
    # plt.show()

    log_path = params.Figures_paper_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

# Original, num BS LOS in Error per timestep axes
def plot_tracking_results_ray_tracing(output_results_tracking, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 20, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):
    
    METHODS = output_results_tracking.keys()
    number_state_variables = params.number_state_variables
    num_agents = params.num_agents
    num_instants = params.size_dataset

    fig = plt.figure(figsize=(25,10), constrained_layout=True)
    gs = GridSpec(int(number_state_variables/2), 2, figure=fig)
    ax_scenario = fig.add_subplot(gs[:, 0])
    ax_error = [fig.add_subplot(gs[var, 1]) for var in range(int(number_state_variables/2))]
    plt.rcParams.update({'font.size': 22}) 
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['GT'] + ['BS'] + list(METHODS) 
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    color_indexes_per_method['BS'] = 'red'
    temp_line_style_indexes_per_method = ['-', '-', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}

    # Legend handles
    handles_ax0, labels_ax0 = [], []

    # Plot scenario
    for i, name_method in enumerate(METHODS):

        # num_agents x num_instants x dimensions
        t = np.array([np.array([np.array(output_results_tracking[name_method]['t'][n][agent]).reshape(-1, 1) for n in range(1, len(output_results_tracking[name_method]['t']))]) for agent in range(num_agents)]).reshape(num_agents, -1, 3)
        # num_agents x num_instants x state dimensions
        t_hat = np.array([np.array([np.array(output_results_tracking[name_method]['t_hat'][n][agent]).reshape(-1, 1) for n in range(1, len(output_results_tracking[name_method]['t_hat']))]) for agent in range(num_agents)]).reshape(num_agents,  -1, 3)
        # num_agents x num_instants
        AE_pos_per_agent_per_timestep = np.array([ output_results_tracking[name_method]['metrics']['AE_pos_per_agent_per_timestep'][agent] for agent in range(num_agents)]).reshape(num_agents,-1)
        # num_instants x 1
        RMSE_pos_per_timestep = np.sqrt(np.mean(AE_pos_per_agent_per_timestep**2, 0)) # mean between agents
        # num_agents x num_instants
        AE_vel_per_agent_per_timestep = np.array([ output_results_tracking[name_method]['metrics']['AE_vel_per_agent_per_timestep'][agent] for agent in range(num_agents)])
        # num_instants x 1
        RMSE_vel_per_timestep = np.sqrt(np.mean(AE_vel_per_agent_per_timestep**2, 0)) # mean between agents
    
        bs_positions_unique = np.unique(np.concatenate(output_results_tracking[name_method]['BSxyz_global'], 1).squeeze(), axis=0)
        bs_positions_unique = bs_positions_unique[~np.isnan(bs_positions_unique).any(axis=1)]

        BS_LOS = np.nansum(np.array(output_results_tracking[name_method]['BS_LOS']).squeeze(), 1)
        cmap = cm.get_cmap('jet')  
        norm = Normalize(vmin=BS_LOS.min(), vmax=BS_LOS.max())

        # plot GT
        if i == 0:

            pos_GT = t.copy()
            break_indices = {}  # To keep track of where each agent's trajectory should break
            for agent in range(num_agents):
                break_indices[agent] = []
                for idx in range(1, len(pos_GT[agent])):
                    distance = np.linalg.norm(pos_GT[agent, idx, :2] - pos_GT[agent, idx-1, :2])
                    if distance > 200:
                        pos_GT[agent, idx, :2] = np.nan  # Set to nan to break the line
                        break_indices[agent].append(idx)  # Record the index where the break occurs
                ax_scenario.plot(pos_GT[agent, :, 0], pos_GT[agent, :, 1], linestyle=line_style_indexes_per_method['GT'], color=colors[color_indexes_per_method['GT']])

            # Plot base station positions
            ax_scenario.plot(bs_positions_unique[:,0], bs_positions_unique[:,1], marker='^', color='red', linestyle='', markersize=8)

        pos_method = t_hat.copy()
        for agent in range(num_agents):
            for idx in break_indices[agent]:  # Use the same indices where pos_GT was broken
                pos_method[agent, idx, :2] = np.nan  # Set to nan to break the line
            ax_scenario.plot(pos_method[agent, :, 0], pos_method[agent, :, 1], linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])


        line, = ax_error[0].plot(range(len(RMSE_pos_per_timestep)), reduce(lambda x, _: x, range(1), RMSE_pos_per_timestep), linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
        print('{}: Mean RMSE pos [m]: {:.3f}'.format(name_method, np.sqrt(np.mean(AE_pos_per_agent_per_timestep**2)))) # mean between agents and timesteps
        print('{}: Mean MAE pos [m]: {:.3f}'.format(name_method, np.mean(AE_pos_per_agent_per_timestep))) # mean between agents and timesteps
        if len(ax_error) >= 2:
            line, = ax_error[1].plot(range(len(RMSE_vel_per_timestep)), (RMSE_vel_per_timestep), linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
            print('{}: Mean RMSE vel [m]: {:.3f}'.format(name_method, np.sqrt(np.mean(AE_vel_per_agent_per_timestep**2)))) # mean between agents and timesteps
            print('{}: Mean MAE vel [m]: {:.3f}'.format(name_method, np.mean(AE_vel_per_agent_per_timestep))) # mean between agents and timesteps
            if len(ax_error) >= 3:
                line, = ax_error[2].plot(range(len(RMSE_acc_per_timestep)), (RMSE_acc_per_timestep), linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
                print('{}: Mean RMSE acc [m]: {:.3f}'.format(name_method, np.sqrt(np.mean(AE_acc_per_agent_per_timestep**2)))) # mean between agents and timesteps
                print('{}: Mean MAE acc [m]: {:.3f}'.format(name_method, np.mean(AE_acc_per_agent_per_timestep))) # mean between agents and timesteps

        # Add the handle and label to the lists
        handles_ax0.append(line)
        labels_ax0.append(name_method)

        lines1, labels1 = ax_error[0].get_legend_handles_labels()

    # Plot BS in visibility
    ax2 = ax_error[0].twinx()
    line2, = ax2.plot(range(len(BS_LOS)), BS_LOS, color='blue')
    ax2.set_ylabel('Num. BS LoS', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    original_y_ticks = np.arange(min(BS_LOS), max(BS_LOS) + 1, 1)
    ax2.set_ylim(min(BS_LOS), max(BS_LOS)*3)
    ax2.set_yticks(original_y_ticks)

    # Add the handle and label for the secondary y-axis
    handles_ax0.append(line2)
    labels_ax0.append('Num. BS LoS')

    # ax_scenario.grid(alpha=0.3)
    ax_scenario.axis('equal')

    # For zoom out figure
    ax_scenario.legend(legend_labels, loc='best', shadow=False)
    # For zoom in figure
    # NO LEGEND

    if xlim is not None:
        ax_error[0].set(xlim=xlim)
    if ylim is not None:
        ax_error[0].set(ylim=ylim)

    if ylabel_ is not None and xlabel_ is not None:       
        ax_scenario.set(xlabel=xlabel_)
        ax_scenario.xaxis.label.set_size(fontsize)
        ax_scenario.set(ylabel=ylabel_)
        ax_scenario.yaxis.label.set_size(fontsize)
    
    ax_scenario.tick_params(axis='both', which='major', labelsize=labelsize)

    # Set the font size for the labels and ticks to match ax_scenario
    ax2.xaxis.label.set_size(ax_scenario.xaxis.label.get_size())
    ax2.yaxis.label.set_size(ax_scenario.yaxis.label.get_size())
    ax2.tick_params(axis='both', which='major', labelsize=ax_scenario.xaxis.get_ticklabels()[0].get_size())

    ax_error[0].xaxis.label.set_size(ax_scenario.xaxis.label.get_size())
    ax_error[0].yaxis.label.set_size(ax_scenario.yaxis.label.get_size())
    ax_error[0].tick_params(axis='both', which='major', labelsize=ax_scenario.xaxis.get_ticklabels()[0].get_size())

    # Plot errors
    # ax_error[0].legend([k for k in METHODS], loc='best', shadow=False)
    ax_error[0].legend(handles_ax0, labels_ax0, loc='upper right', shadow=False)
    ax_error[0].set(xlabel='Time n')
    ax_error[0].set(ylabel='RMSE pos [m]')
    # ax_error[0].grid(alpha=0.3)
    if logy is not None and logy:
        ax_error[0].set_yscale('log')
    if len(ax_error) >= 2:
        ax_error[1].legend([k for k in METHODS], loc='best', shadow=False)
        ax_error[1].set(xlabel='Time n')
        ax_error[1].set(ylabel='RMSE vel [m/s]')
        # ax_error[1].grid(alpha=0.3)
        if len(ax_error) >= 3:
            ax_error[2].legend([k for k in METHODS], loc='best', shadow=False)
            ax_error[2].set(xlabel='Time n')
            ax_error[2].set(ylabel='RMSE acc [m/s^2]')
            
    ax_scenario.set_aspect('equal', adjustable='box')
    # plt.tick_params(labelsize=labelsize)
    # plt.rcParams.update({'font.size': 30})

    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    log_path = params.Figures_tracking_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

    ################################# CDF PLOT
    fig_cdf, ax_cdf = plt.subplots(figsize=(12, 8))
    for i, name_method in enumerate(METHODS):
        AE_pos_flattened = np.array([ output_results_tracking[name_method]['metrics']['AE_pos_per_agent_per_timestep'][agent] for agent in range(num_agents)]).reshape(-1)
        
        # Kernel Density Estimation
        kde = gaussian_kde(AE_pos_flattened, bw_method=0.01)
        x_range = np.linspace(min(AE_pos_flattened), max(AE_pos_flattened), 500)  # Adjust the number of points for smoothness
        # Compute the CDF using the estimated PDF
        cdf_kde = np.cumsum(kde(x_range)) * (x_range[1] - x_range[0])
        # Ensure the CDF starts from (0,0) by adding a zero at the beginning
        x_range = np.insert(x_range, 0, 0)
        cdf_kde = np.insert(cdf_kde, 0, 0)
        # Plotting the smooth CDF
        ax_cdf.plot(x_range, cdf_kde, label=name_method, linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])

        # Find the closest value to 1 in x_range
        closest_to_one = np.abs(x_range - 1).argmin()
        cdf_at_one_meter = cdf_kde[closest_to_one] 
        median_error = np.median(AE_pos_flattened)
        error_95 = np.percentile(AE_pos_flattened, 95)
        
        print(f"{name_method}: Median Error: {median_error:.3f}, 95% Error: {error_95:.3f}, CDF at 1m: {cdf_at_one_meter:.3f}")


    ax_cdf.set_xlabel('Absolute Error [m]')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.legend(loc='best')
    # ax_cdf.grid(True)
    ax_cdf.set_xscale('log')

    ax_cdf.tick_params(axis='both', which='major', labelsize=labelsize)
    # Set the font size for axis labels
    ax_cdf.xaxis.label.set_size(fontsize)
    ax_cdf.yaxis.label.set_size(fontsize)
    ax_cdf.set_xlim([0.1, 100])
    ax_cdf.set_ylim([0, 1])

    # Save the CDF plot
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'CDF_{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'CDF_{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_jpg:
        plt.savefig(os.path.join(log_path, f'CDF_{file_name}.jpg'), bbox_inches='tight', dpi=300)

    plt.close(fig_cdf)

def plot_tracking_results_ray_tracing2(output_results_tracking, params, file_name, xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False, xlim = None, ylim = None, fontsize = 20, labelsize = 18, save_eps = 0, ax = None, save_svg = 0, save_pdf = 0, save_jpg = 0, plt_show = 1):
    
    METHODS = output_results_tracking.keys()
    number_state_variables = params.number_state_variables
    num_agents = params.num_agents
    num_instants = params.size_dataset

    fig = plt.figure(figsize=(25,10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig)
    ax_scenario = fig.add_subplot(gs[:, 0])
    ax_error = [fig.add_subplot(gs[var, 1]) for var in range(2)]
    plt.rcParams.update({'font.size': 22}) 
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors_default = prop_cycle.by_key()['color']

    legend_labels = ['GT'] + ['BS'] + list(METHODS) 
    temp_color_indexes_per_method = range(-1, len(legend_labels)-1)
    color_indexes_per_method = {method:temp_color_indexes_per_method[i] for i,method in enumerate(legend_labels)}
    color_indexes_per_method['BS'] = 'red'
    temp_line_style_indexes_per_method = ['-', '-', '-.', ':', '--', (0,(3, 1, 1, 1))]
    line_style_indexes_per_method = {method:temp_line_style_indexes_per_method[i] for i,method in enumerate(legend_labels)}

    # Legend handles
    handles_ax0, labels_ax0 = [], []

    # Plot scenario
    for i, name_method in enumerate(METHODS):

        # num_agents x num_instants x dimensions
        t = np.array([np.array([np.array(output_results_tracking[name_method]['t'][n][agent]).reshape(-1, 1) for n in range(1, len(output_results_tracking[name_method]['t']))]) for agent in range(num_agents)]).reshape(num_agents, -1, 3)
        # num_agents x num_instants x state dimensions
        t_hat = np.array([np.array([np.array(output_results_tracking[name_method]['t_hat'][n][agent]).reshape(-1, 1) for n in range(1, len(output_results_tracking[name_method]['t_hat']))]) for agent in range(num_agents)]).reshape(num_agents,  -1, 3)
        # num_agents x num_instants
        AE_pos_per_agent_per_timestep = np.array([ output_results_tracking[name_method]['metrics']['AE_pos_per_agent_per_timestep'][agent] for agent in range(num_agents)]).reshape(num_agents,-1)
        # num_instants x 1
        RMSE_pos_per_timestep = np.sqrt(np.mean(AE_pos_per_agent_per_timestep**2, 0)) # mean between agents
        # num_agents x num_instants
        AE_vel_per_agent_per_timestep = np.array([ output_results_tracking[name_method]['metrics']['AE_vel_per_agent_per_timestep'][agent] for agent in range(num_agents)])
        # num_instants x 1
        RMSE_vel_per_timestep = np.sqrt(np.mean(AE_vel_per_agent_per_timestep**2, 0)) # mean between agents
    
        bs_positions_unique = np.unique(np.concatenate(output_results_tracking[name_method]['BSxyz_global'], 1).squeeze(), axis=0)
        bs_positions_unique = bs_positions_unique[~np.isnan(bs_positions_unique).any(axis=1)]

        BS_LOS = np.nansum(np.array(output_results_tracking[name_method]['BS_LOS']).squeeze(), 1)
        cmap = cm.get_cmap('jet')  
        norm = Normalize(vmin=BS_LOS.min(), vmax=BS_LOS.max())

        # plot GT
        if i == 0:

            pos_GT = t.copy()
            break_indices = {}  # To keep track of where each agent's trajectory should break
            for agent in range(num_agents):
                break_indices[agent] = []
                for idx in range(1, len(pos_GT[agent])):
                    distance = np.linalg.norm(pos_GT[agent, idx, :2] - pos_GT[agent, idx-1, :2])
                    if distance > 200:
                        pos_GT[agent, idx, :2] = np.nan  # Set to nan to break the line
                        break_indices[agent].append(idx)  # Record the index where the break occurs
                ax_scenario.plot(pos_GT[agent, :, 0], pos_GT[agent, :, 1], linestyle=line_style_indexes_per_method['GT'], color=colors[color_indexes_per_method['GT']])

            # Plot base station positions
            ax_scenario.plot(bs_positions_unique[:,0], bs_positions_unique[:,1], marker='^', color='red', linestyle='', markersize=8)

        pos_method = t_hat.copy()
        for agent in range(num_agents):
            for idx in break_indices[agent]:  # Use the same indices where pos_GT was broken
                pos_method[agent, idx, :2] = np.nan  # Set to nan to break the line
            ax_scenario.plot(pos_method[agent, :, 0], pos_method[agent, :, 1], linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])


        line, = ax_error[0].plot(range(len(RMSE_pos_per_timestep)), reduce(lambda x, _: x, range(1), RMSE_pos_per_timestep), linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])
        print('{}: Mean RMSE pos [m]: {:.3f}'.format(name_method, np.sqrt(np.mean(AE_pos_per_agent_per_timestep**2)))) # mean between agents and timesteps
        print('{}: Mean MAE pos [m]: {:.3f}'.format(name_method, np.mean(AE_pos_per_agent_per_timestep))) # mean between agents and timesteps

        # Add the handle and label to the lists
        handles_ax0.append(line)
        labels_ax0.append(name_method)

        lines1, labels1 = ax_error[0].get_legend_handles_labels()

    # Plot BS in visibility
    line2, = ax_error[1].plot(range(len(BS_LOS)), BS_LOS, color='blue')
    ax_error[1].set_ylabel('Num. BS LoS', color='blue')
    ax_error[1].tick_params(axis='y', labelcolor='blue')

    # ax_scenario.grid(alpha=0.3)
    ax_scenario.axis('equal')

    if xlim is not None:
        ax_error[0].set(xlim=xlim)
    if ylim is not None:
        ax_error[0].set(ylim=ylim)

    if ylabel_ is not None and xlabel_ is not None:       
        ax_scenario.set(xlabel=xlabel_)
        ax_scenario.xaxis.label.set_size(fontsize)
        ax_scenario.set(ylabel=ylabel_)
        ax_scenario.yaxis.label.set_size(fontsize)
    
    ax_scenario.tick_params(axis='both', which='major', labelsize=labelsize)

    # Set the font size for the labels and ticks to match ax_scenario
    ax_error[1].xaxis.label.set_size(ax_scenario.xaxis.label.get_size())
    ax_error[1].yaxis.label.set_size(ax_scenario.yaxis.label.get_size())
    ax_error[1].tick_params(axis='both', which='major', labelsize=ax_scenario.xaxis.get_ticklabels()[0].get_size())

    ax_error[0].xaxis.label.set_size(ax_scenario.xaxis.label.get_size())
    ax_error[0].yaxis.label.set_size(ax_scenario.yaxis.label.get_size())
    ax_error[0].tick_params(axis='both', which='major', labelsize=ax_scenario.xaxis.get_ticklabels()[0].get_size())

    # Plot errors
    # ax_error[0].legend([k for k in METHODS], loc='best', shadow=False)
    ax_error[0].legend(handles_ax0, labels_ax0, loc='upper right', shadow=False)
    ax_error[0].set(xlabel='Time n')
    ax_error[0].set(ylabel='RMSE pos [m]')
    # ax_error[0].grid(alpha=0.3)

    # ax_error[1].legend(line2, labels_ax0, loc='upper right', shadow=False)
    ax_error[1].set(xlabel='Time n')
    ax_error[1].set(ylabel='Num. BS LoS')
    # ax_error[1].grid(alpha=0.3)

    if logy is not None and logy:
        ax_error[0].set_yscale('log')
            
    ax_scenario.set_aspect('equal', adjustable='box')
    # plt.tick_params(labelsize=labelsize)
    # plt.rcParams.update({'font.size': 30})

    if title_ is not None:       
        plt.title(title_, fontsize=fontsize)

    log_path = params.Figures_tracking_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), format='svg',bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

    ################################# CDF PLOT
    fig_cdf, ax_cdf = plt.subplots(figsize=(12, 8))
    for i, name_method in enumerate(METHODS):
        AE_pos_flattened = np.array([ output_results_tracking[name_method]['metrics']['AE_pos_per_agent_per_timestep'][agent] for agent in range(num_agents)]).reshape(-1)
        
        # Kernel Density Estimation
        kde = gaussian_kde(AE_pos_flattened, bw_method=0.01)
        x_range = np.linspace(min(AE_pos_flattened), max(AE_pos_flattened), 500)  # Adjust the number of points for smoothness
        # Compute the CDF using the estimated PDF
        cdf_kde = np.cumsum(kde(x_range)) * (x_range[1] - x_range[0])
        # Ensure the CDF starts from (0,0) by adding a zero at the beginning
        x_range = np.insert(x_range, 0, 0)
        cdf_kde = np.insert(cdf_kde, 0, 0)
        # Plotting the smooth CDF
        ax_cdf.plot(x_range, cdf_kde, label=name_method, linestyle=line_style_indexes_per_method[name_method], color=colors[color_indexes_per_method[name_method]])

        # Find the closest value to 1 in x_range
        closest_to_one = np.abs(x_range - 1).argmin()
        cdf_at_one_meter = cdf_kde[closest_to_one] 
        median_error = np.median(AE_pos_flattened)
        error_95 = np.percentile(AE_pos_flattened, 95)
        
        print(f"{name_method}: Median Error: {median_error:.3f}, 95% Error: {error_95:.3f}, CDF at 1m: {cdf_at_one_meter:.3f}")


    ax_cdf.set_xlabel('Absolute Error [m]')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.legend(loc='best')
    # ax_cdf.grid(True)
    ax_cdf.set_xscale('log')

    ax_cdf.tick_params(axis='both', which='major', labelsize=labelsize)
    # Set the font size for axis labels
    ax_cdf.xaxis.label.set_size(fontsize)
    ax_cdf.yaxis.label.set_size(fontsize)
    ax_cdf.set_xlim([0.1, 100])
    ax_cdf.set_ylim([0, 1])

    # Save the CDF plot
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'CDF_{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'CDF_{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_jpg:
        plt.savefig(os.path.join(log_path, f'CDF_{file_name}.jpg'), bbox_inches='tight', dpi=300)

    plt.close(fig_cdf)



##############################################################################################################################
################ GENERAL #################
##############################################################################################################################

# Plot after testing
def plot_uncertainty(prediction, t, params, file_name, plot_save_str="row", xlabel_ = '', ylabel_ = '', title_ = '', logx = False, logy = False,  xlim = None, ylim = None, fontsize = 18, labelsize = 18, save_eps = 0, ax = None, save_svg = 1, save_pdf = 1, save_jpg = 0, plt_show = 1):
    
    pred_mean = prediction['y_mean']

    if pred_mean.shape[1] == 1:
        # Single output feature
        fig, axs = plt.subplots(1, 1, figsize=(17, 8))
        pred_std = prediction['total_unc_std']
        axs = uct.plot_calibration(pred_mean.reshape(-1), pred_std.reshape(-1), t.reshape(-1), ax=axs)
    else:
        # Multiple output features
        num_features = pred_mean.shape[1]
        pred_cov = prediction['total_unc_cov']
        fig, axs = plt.subplots(1, num_features, figsize=(17 * num_features, 8))
        for i in range(num_features):
            axs[i] = uct.plot_calibration(pred_mean[:, i], np.sqrt(pred_cov[:, i, i]), t[:, i], ax=axs[i])


    # Adjust subplots spacing
    fig.subplots_adjust(wspace=0.25)

    # Save figure
    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), format='eps', bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'),bbox_inches='tight')
    if save_jpg:
            plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
    if plt_show:
        plt.show()
    plt.close(fig)

def plot_metrics(metrics_dict, params, file_name, num_columns=3, figsize=(15, 15), 
                 xlabel_='Epochs', ylabel_='Value', title_=None, logx=False, logy=False, 
                 xlim=None, ylim=None, fontsize=18, labelsize=18, save_eps=0, ax=None, 
                 save_svg=0, save_pdf=0, save_jpg=0, plt_show=1):
    
    # Calculate the number of subplots needed
    num_metrics = len(metrics_dict.keys())
    num_rows = int(np.ceil(num_metrics / num_columns))
    
    # Initialize the plot
    if ax is None:
        fig, axes = plt.subplots(num_rows, num_columns, figsize=figsize)
        axes = axes.flatten()
    else:
        axes = ax

    # Plot each metric
    for i, key in enumerate(metrics_dict.keys()):
        ax = axes[i]
        ax.plot(metrics_dict[key], label=f"{key}")
        ax.set_title(f"{key}")
        ax.set_xlabel(xlabel_, fontsize=fontsize)
        ax.set_ylabel(ylabel_, fontsize=fontsize)
        
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
            
        ax.legend()
        ax.tick_params(labelsize=labelsize)

    # Hide any extra subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis('off')

    # Overall title
    if title_:
        plt.suptitle(title_, fontsize=fontsize)
        
    plt.tight_layout()

    # Save the plot
    log_path = params.Figures_dir
    if save_pdf:
        plt.savefig(os.path.join(log_path, f'{file_name}.pdf'), bbox_inches='tight')
    if save_eps:
        plt.savefig(os.path.join(log_path, f'{file_name}.eps'), bbox_inches='tight')
    if save_svg:
        plt.savefig(os.path.join(log_path, f'{file_name}.svg'), bbox_inches='tight')
    if save_jpg:
        plt.savefig(os.path.join(log_path, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)

    # Show the plot
    if plt_show:
        plt.show()
    plt.close(fig)





