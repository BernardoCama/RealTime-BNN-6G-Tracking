% MATLAB Code for Geographical Plotting
clc, clear all, close all

% Open the file
fileID = fopen('VehicleRoute_AllVehicles.pts','r');

% Skip the header lines. The number of lines to skip depends on your file.
fgetl(fileID);  % Repeat this line as many times as the number of header lines.

% Read the data
data_cell = textscan(fileID, '%f %f %f');  % Update format based on your actual data
fclose(fileID);

% Convert cell array to matrix
data = cell2mat(data_cell);

% Extract latitude and longitude
lats = data(:,2);
lons = data(:,3);

% Calculate aspect ratio
aspect_ratio = (max(lons) - min(lons)) / (max(lats) - min(lats));

% Create a figure for the scatter plot
fig1 = figure('Position', [905, 94, 754*1.1, 684*1.1]);
scatter(lons, lats, 'r.')
xlim([min(lons) max(lons)]);
ylim([min(lats) max(lats)]);
xlabel('Longitude');
ylabel('Latitude');
title('Geographical Plot of Vehicle Route');
grid on;
set(gca, 'FontSize', 20);

% Save the figure
filename1 = 'Vehicle_Route_Plot';
saveas(fig1, fullfile(pwd, sprintf('%s.png', filename1)));
saveas(fig1, fullfile(pwd, sprintf('%s.pdf', filename1)));


% Create a figure for the density plot
fig2 = figure('Position', [905, 94, 754*1.1, 684*1.1]);

% Create a 2D histogram for the density plot
[hist, xedges, yedges] = histcounts2(lons, lats, [200, 200]);

% Create the density plot using imagesc
imagesc(xedges, yedges, hist');
set(gca, 'YDir', 'normal');
colormap('hot');
c = colorbar;
c.Label.String = 'Point Density (log scale)';
caxis([min(hist(:)) max(hist(:)) + 1]);
set(gca, 'ColorScale', 'log');

% Add labels and title
xlabel('Longitude');
ylabel('Latitude');
title('Point Density Heatmap (log scale)');
grid on;
set(gca, 'FontSize', 20);

% Save the figure
filename2 = 'Density_Plot';
saveas(fig2, fullfile(pwd, sprintf('%s.png', filename2)));
saveas(fig2, fullfile(pwd, sprintf('%s.pdf', filename2)));

