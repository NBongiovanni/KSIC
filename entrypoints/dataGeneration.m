%open_system(new_system('abcdef123456'));
bdclose('abcdef123456');
clear variables;
close all;
clc;

% === Début du bootstrap du path ===
% 1) Dossier contenant ce script (= racine du projet)
projectRoot = fileparts(mfilename('fullpath'));

% 2) Ajouter tous les sous-dossiers à MATLAB path,
%    en excluant les __*__ si besoin
allPaths = genpath(projectRoot);
addpath(allPaths);
% === Fin du bootstrap du path ===

% Parameters settings
rng(1);
typeConfig = "inventedDrone";
sysParams = defineSysParams(typeConfig);
sysParams.projectRoot = projectRoot;
resultsDir  = getDateTime();
sysParams.resultsDir = resultsDir;

datasetTypeSequence = {"train_1", "train_2", "val", "test"};
noiseSequence = {false, false, false, false};
nTrajsSequence = {6000, 10, 200, 10};
gainsPID = {1, 1, 1, 1};
plotResultsSequence = {false, false, false, false};

for i=1:4
    fprintf(datasetTypeSequence{i} + " dataset\n")
    ctrlParams = defineCtrlParams(datasetTypeSequence{i}, gainsPID{i});
    sysParams.datasetType = datasetTypeSequence{i};
    sysParams.nTrajs = nTrajsSequence{i};
    generateDataset( ...
        sysParams.datasetType, ...
        sysParams, ...
        ctrlParams, ...
        plotResultsSequence{i}, ...
        noiseSequence{i} ...
        );
end