clc, clear all, close all

% Create a figure with fixed dimensions
fig = figure('Position', [905, 94, 754*1.1, 684*1.1]);  % Change 1600 and 800 to your desired dimensions
% disp(['Current figure size: ', num2str(get(gcf, 'Position'))]);


mat_filepath = 'sumoTrace_testing.mat';
bs_filepath = '../BaseStations.pts';
lat_limits = [42.359859, 42.371962];
lon_limits = [-71.111399, -71.077748];
scatter = 1;
vehicles = [11, 12];

% Load the MAT file
mat_data = load(mat_filepath);

% Load the base stations file
bs_positions = dlmread(bs_filepath, '', 5, 0);
bs_positions = bs_positions(:,2:3);
bs_Id = (1:size(bs_positions,1))';

% Extract the variables
timeAxis = mat_data.timeAxis;
ueId = mat_data.ueId;
uePositions = mat_data.uePositions;
ueVelocities = mat_data.ueVelocities;
absolute_velocities = sqrt(sum(ueVelocities.^2, 2));

% Create a geographical plot with a set of distinct colors
% figure('Position', [100, 100, 1600, 800]);
% ax = axesm('mercator', 'MapLatLimit', lat_limits, 'MapLonLimit', lon_limits);

unique_ueId = unique(ueId);
unique_absolute_velocities = unique(absolute_velocities);
%colors = parula(length(unique_ueId));
colors = jet(length(unique_absolute_velocities)); % use absolute_velocities for individual color mapping


if ~scatter
    for i = 1:length(unique_ueId)
        uid = unique_ueId(i);
        if ismember(uid, vehicles)
            filtered_positions = uePositions((ueId == uid) & ...
                                            (uePositions(:,1) >= lat_limits(1)) & ...
                                            (uePositions(:,1) <= lat_limits(2)) & ...
                                            (uePositions(:,2) >= lon_limits(1)) & ...
                                            (uePositions(:,2) <= lon_limits(2)), :);
            filtered_velocities = absolute_velocities((ueId == uid) & ...
                                            (uePositions(:,1) >= lat_limits(1)) & ...
                                            (uePositions(:,1) <= lat_limits(2)) & ...
                                            (uePositions(:,2) >= lon_limits(1)) & ...
                                            (uePositions(:,2) <= lon_limits(2)), :);
            if size(filtered_positions, 1) > 1
                % geoplot(filtered_positions(:,1), filtered_positions(:,2), 'Color', colors(i,:), 'DisplayName', num2str(uid));
                % scatter = geoscatter(filtered_positions(:,1), filtered_positions(:,2), 36, filtered_velocities, '.');
                hold on;
            end
        end
    end
end

if scatter
    for i = 1:length(unique_ueId)
        uid = unique_ueId(i);
        if ismember(uid, vehicles)
            filtered_positions = uePositions((ueId == uid) & ...
                                            (uePositions(:,1) >= lat_limits(1)) & ...
                                            (uePositions(:,1) <= lat_limits(2)) & ...
                                            (uePositions(:,2) >= lon_limits(1)) & ...
                                            (uePositions(:,2) <= lon_limits(2)), :);
            filtered_velocities = absolute_velocities((ueId == uid) & ...
                                            (uePositions(:,1) >= lat_limits(1)) & ...
                                            (uePositions(:,1) <= lat_limits(2)) & ...
                                            (uePositions(:,2) >= lon_limits(1)) & ...
                                            (uePositions(:,2) <= lon_limits(2)), :);
            % geoscatter(filtered_positions(:,1), filtered_positions(:,2), [], colors(i,:), '.', 'DisplayName', num2str(uid));
            scatter = geoscatter(filtered_positions(:,1), filtered_positions(:,2), 36, filtered_velocities, '.');
            hold on;
        end
    end
end

colormap('jet'); % You can choose any colormap you like
c = colorbar;
c.Label.String = 'Absolute Velocity [km/h]';
c.FontSize = 12;

% Plot base station positions
bs_positions_unique = unique(bs_positions, 'rows');
geoscatter(bs_positions_unique(:,2), bs_positions_unique(:,1), 100, 'r', '^', 'filled', 'DisplayName', 'BS');

% Set labels and title
% title('Geographical Representation of UE Trajectories');
hold on;
htmlGray = [128 128 128]/255;
d1 = geoplot(NaN, NaN, '.', 'Color', htmlGray);
d2 = geoplot(NaN, NaN, '^', 'Color', 'r', 'MarkerFaceColor', 'r');

% Manually define the legend using the dummy plots
[~, objh] =legend([d1, d2], {'UE', 'BS'}, 'FontSize', 20, 'Location', 'northwest');
objhl = findobj(objh, 'type', 'line');
set(objhl(2), 'Markersize', 19);
set(objhl(2), 'Markersize', 16);


% Hide the dummy plots
% set([d1, d2], 'Visible', 'off');
gx = gca;
gx.Legend.TextColor = 'black';

geobasemap streets-dark
set(gca,'FontSize',20)


filename = 'Testing_trajectories';


% Adjust Paper Position and Size for saving
set(fig, 'PaperUnits', 'inches');
paperPosition = get(fig, 'PaperPosition');
widthHeightRatio = paperPosition(3) / paperPosition(4);
paperHeight = 11;  % Use 11-inch paper
paperWidth = paperHeight * widthHeightRatio;
set(fig, 'PaperSize', [paperWidth paperHeight]);
set(fig, 'PaperPositionMode', 'manual');
set(fig, 'PaperPosition', [0 0 paperWidth paperHeight]);

% Save figure as PNG and PDF
filename = 'Testing_trajectories_colorbar';
% print(fig, '-dpng', fullfile(pwd, sprintf('%s.png', filename)));
% print(fig, '-dpdf', fullfile(pwd, sprintf('%s.pdf', filename)));

ax = gca;
exportgraphics(ax,fullfile(pwd, sprintf('%s.png', filename)),'Resolution',300) 
exportgraphics(ax,fullfile(pwd, sprintf('%s.pdf', filename)),'Resolution',300) 


