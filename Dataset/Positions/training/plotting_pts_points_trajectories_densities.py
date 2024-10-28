import matplotlib.pyplot as plt
import geopandas as gpd
import os 
import sys
import numpy as np
from matplotlib.colors import LogNorm
import mat73
import scipy

cwd = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.dirname(cwd))

fontsize = 16


# Step 1: Read the file and extract the data points
file_path = os.path.join(cwd, 'VehicleRoute_AllVehicles.pts')

with open(file_path, 'r') as file:
    lines = file.readlines()

# Step 2: Extract latitude and longitude values from the data points
data = [line.strip().split(' ') for line in lines if line[0].isdigit()]
lons = [float(line[1]) for line in data]
lats = [float(line[2]) for line in data]

# Step 3: Use geopandas and matplotlib to plot the geographical data
gdf = gpd.GeoDataFrame(
    {'geometry': gpd.points_from_xy(lons, lats)},
    crs="EPSG:4326"
)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Setting the same aspect ratio for both plots
aspect_ratio = (max(lons) - min(lons)) / (max(lats) - min(lats))


# BS .pts file
filename = os.path.join(os.path.split(cwd)[0], 'BaseStations.pts')
base_stations = np.loadtxt(filename, skiprows=5)  # Adjust the skiprows parameter to skip header lines
base_station_lats = base_stations[:, 2]
base_station_lons = base_stations[:, 1]


# Plotting the scatter plot with just the points
fig1, ax1 = plt.subplots(figsize=(16, 8))
world.plot(ax=ax1, color='white', edgecolor='black')
gdf.plot(ax=ax1, color='red', markersize=5)


# Plot the Base Station positions on the same axis
ax1.scatter(base_station_lons, base_station_lats, c='black', marker='^', s=50, zorder=5)  # Setting zorder to make sure they are on top

ax1.set_xlim(min(lons), max(lons))
ax1.set_ylim(min(lats), max(lats))
# Increase size of x and y labels
ax1.set_xlabel('Longitude', fontsize=16)
ax1.set_ylabel('Latitude', fontsize=16)
# Increase size of x and y ticks
ax1.tick_params(axis='both', which='major', labelsize=16)
# Convert decimal degrees to DMS format for tick labels
lat_dms = ["{:.0f}°{:.0f}'{:.0f}\"N".format(int(abs(d)), int(abs(d)*60) % 60, int(abs(d)*3600) % 60) for d in ax1.get_yticks()]
lon_dms = ["{:.0f}°{:.0f}'{:.0f}\"W".format(int(abs(d)), int(abs(d)*60) % 60, int(abs(d)*3600) % 60) for d in ax1.get_xticks()]
# Set custom tick labels
ax1.set_xticklabels(lon_dms)
ax1.set_yticklabels(lat_dms)
# For ax1
# Capture original x-limits
original_xlim_ax1 = ax1.get_xlim()
# Subset the x-ticks and their labels
subset_xticks_ax1 = ax1.get_xticks()[::2]
subset_lon_dms_ax1 = lon_dms[::2]
# Set x-ticks and their labels
ax1.set_xticks(subset_xticks_ax1)
ax1.set_xticklabels(subset_lon_dms_ax1)
# Set x-limits back to original
ax1.set_xlim(original_xlim_ax1)

# ax1.set_title('Geographical Plot of Vehicle Route')
ax1.grid(False)
ax1.set_aspect(aspect_ratio)
plt.tight_layout()
# plt.show()

file_name = 'Training_trajectories_1'
plt.savefig(os.path.join(cwd, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.svg'), format='svg',bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)


##############################################################################################################################
##############################################################################################################################
### DENSITIES OF TRAINING POINTS IN PYTHON

# Creating a 2D histogram for density plot with more bins for granular representation
hist, xedges, yedges = np.histogram2d(lons, lats, bins=[200, 200])

# Creating the heatmap using the 2D histogram data with a logarithmic color scale
fig2, ax2 = plt.subplots(figsize=(16, 8))
c = ax2.imshow(hist.T, origin='lower', extent=[min(lons), max(lons), min(lats), max(lats)], aspect='auto', cmap='hot', norm=LogNorm())


# BS .pts file
filename = os.path.join(os.path.split(cwd)[0], 'BaseStations.pts')
base_stations = np.loadtxt(filename, skiprows=5)  # Adjust the skiprows parameter to skip header lines
base_station_lats = base_stations[:, 2]
base_station_lons = base_stations[:, 1]

# Plot the Base Station positions on the same axis
ax2.scatter(base_station_lons, base_station_lats, c='white', marker='^', s=50, zorder=5)  # Setting zorder to make sure they are on top


# Adding a colorbar to represent the density of points with scale/unit of measure
cbar = fig2.colorbar(c, ax=ax2, pad=0.01)  # pad=0.01 brings the colorbar closer
cbar.ax.tick_params(labelsize=14)  # Increases the tick label size
# cbar.set_label('Point Density (log scale) - Number of points per bin')

ax2.set_xlim(ax1.get_xlim())
ax2.set_ylim(ax1.get_ylim())
# Adding labels and title to the plot
# Increase size of x and y labels
ax2.set_xlabel('Longitude', fontsize=16)
ax2.set_ylabel('Latitude', fontsize=16)
# Increase size of x and y ticks
ax2.tick_params(axis='both', which='major',  labelsize=16)
# Convert decimal degrees to DMS format for tick labels
lat_dms = ["{:.0f}°{:.0f}'{:.0f}\"N".format(int(abs(d)), int(abs(d)*60) % 60, int(abs(d)*3600) % 60) for d in ax2.get_yticks()]
lon_dms = ["{:.0f}°{:.0f}'{:.0f}\"W".format(int(abs(d)), int(abs(d)*60) % 60, int(abs(d)*3600) % 60) for d in ax2.get_xticks()]
# Set custom tick labels
ax2.set_xticklabels(lon_dms)  
ax2.set_yticklabels(lat_dms)
# For ax2
# Capture original x-limits
original_xlim_ax2 = ax2.get_xlim()
# Subset the x-ticks and their labels
subset_xticks_ax2 = ax2.get_xticks()[::2]
subset_lon_dms_ax2 = lon_dms[::2]  # Assuming lat_dms for ax1 and ax2 are the same
# Set x-ticks and their labels
ax2.set_xticks(subset_xticks_ax2)
ax2.set_xticklabels(subset_lon_dms_ax2)
# Set x-limits back to original
ax2.set_xlim(original_xlim_ax2)

# ax2.set_title('Point Density Heatmap (log scale)')
ax2.grid(False, which='both', linestyle='--', linewidth=0.5)
ax2.set_aspect(aspect_ratio)


# Display the plot
plt.tight_layout()
# plt.show()

file_name = 'Training_trajectories_2'
plt.savefig(os.path.join(cwd, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.svg'), format='svg',bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)



##############################################################################################################################
##############################################################################################################################
### HISTOGRAMS OF VELOCITIES OF TRAINING POINTS IN PYTHON
file_path = os.path.join(cwd, 'sumoTrace_training.mat')
try:
    data = mat73.loadmat(file_path)
except: 
    data = scipy.io.loadmat(file_path)
velx = data['ueVelocities'][:, 0]
vely = data['ueVelocities'][:, 1]

# Compute the mean and std of velocities along the x and y axes
mean_velx = np.mean(velx)
std_velx = np.std(velx)
mean_vely = np.mean(vely)
std_vely = np.std(vely)
# Compute the absolute velocity for each point and then its mean and std
absolute_velocities = np.sqrt(velx**2 + vely**2)
mean_abs_velocity = np.mean(absolute_velocities)
std_abs_velocity = np.std(absolute_velocities)
min_abs_velocity = np.min(absolute_velocities)
max_abs_velocity = np.max(absolute_velocities)
print(f"Mean velocity along X: {mean_velx:.2f}, Std: {std_velx:.2f}")
print(f"Mean velocity along Y: {mean_vely:.2f}, Std: {std_vely:.2f}")
print(f"Mean absolute velocity: {mean_abs_velocity:.2f}, Std: {std_abs_velocity:.2f}")
print(f"Min absolute velocity: {min_abs_velocity:.2f}, Max: {max_abs_velocity:.2f}")


# TODO PLOT 1 HISTOGRAM OF VELOCITIES ON X AND Y ON THE SAME HISTOGRAM
plt.figure(figsize=(10, 6))
plt.hist(velx, bins=30, alpha=0.5, label='X-axis Velocities', density=True, edgecolor='black')
plt.hist(vely, bins=30, alpha=0.5, label='Y-axis Velocities', density=True, edgecolor='black')

# Adding labels and title
plt.xlabel('Velocity [km/h]', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.title('Density Histogram of Velocities in X and Y Directions', fontsize=16)
plt.legend(fontsize=14)

# Save the plot if needed
file_name = 'Velocity_Histogram'
plt.savefig(os.path.join(cwd, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
# plt.show()


# TODO PLOT 2 HISTOGRAM OF ABSOLUTE VELOCITIES 
plt.figure(figsize=(10, 6))
plt.hist(absolute_velocities, bins=30, alpha=0.5, label='Absolute Velocities', density=True, edgecolor='black')

# Adding labels and title
plt.xlabel('Velocity [km/h]', fontsize=16)
plt.ylabel('Density', fontsize=16)
# plt.title('Density Histogram of Absolute Velocities', fontsize=16)
plt.legend(fontsize=14)

# Save the plot if needed
file_name = 'Velocity_absolute_Histogram'
plt.savefig(os.path.join(cwd, f'{file_name}.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.eps'), format='eps', bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.svg'), format='svg', bbox_inches='tight')
plt.savefig(os.path.join(cwd, f'{file_name}.jpg'), bbox_inches='tight', dpi=300)
# plt.show()
