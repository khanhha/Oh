%ptCloud = pcread('G:\\Projects\\Oh\\data\\Armadillo.ply');
ptCloud = pcread('G:\\Projects\\Oh\\data\\oh_none_repaired.ply');
gridStep = 6;
ptCloudA = pcdownsample(ptCloud,'gridAverage',gridStep);
%ptCloudA = pcdownsample(ptCloud,'nonuniformGridSample',16);
pcwrite(ptCloudA, 'G:\\Projects\\Oh\\data\\oh_none_repaired_matlab_uniform.ply')
figure;
pcshow(ptCloudA);