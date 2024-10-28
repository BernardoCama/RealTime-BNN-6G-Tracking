clc, clear all, close all

type = 'training'; % testing, training

name_dataset = 'Positioning';
DB = sprintf('../../../DB/Dataset_%s', name_dataset);
[status, msg, msgID] = mkdir(sprintf('%s', DB));
Data_dir = sprintf('../../../DB/Dataset_%s/Data/%s', name_dataset, type);
[status, msg, msgID] = mkdir(sprintf('%s', Data_dir));



filesAndFolders = dir(Data_dir);  
Folders_pos = filesAndFolders(arrayfun(@(x) x.isdir && startsWith(x.name, 'pos_'), filesAndFolders));
numOfFolders_pos = length(Folders_pos);

dataset_tot = cell(1);

for i = 1:40000
    folderName = sprintf('pos_%d', i);
    posIdx = i;
    
    try
        load(sprintf('%s/%s/dataset.mat', Data_dir, folderName))
        dataset_tot{posIdx}.target = dataset.target;
        dataset_tot{posIdx}.auxiliary = dataset.auxiliary;
        dataset_tot{posIdx}.measurements = dataset.measurements;
    end
    
    fprintf('Processed posIdx = %d\n', posIdx);
end


save(sprintf('%s/dataset.mat', Data_dir), 'dataset_tot', '-v7.3');