function saveResults2D(sysParams, states, inputs, statesRef, metadata)
    arguments
        sysParams
        states
        inputs
        statesRef
        metadata
    end

    subFolderName = sysParams.datasetType;
    
    % 4) Construire le chemin absolu vers results/<horodatage>/<datasetType>
    resultsPath = fullfile( ...
        sysParams.projectRoot, ...
        'results', ...
        sysParams.resultsDir, ...
        subFolderName);

    if ~exist(resultsPath, 'dir')
        mkdir(resultsPath);
    end

    % --- Sauvegarde des données ---
    save(fullfile(resultsPath, 'states.mat'),    "states");
    save(fullfile(resultsPath, 'statesRef.mat'), "statesRef");
    save(fullfile(resultsPath, 'inputs.mat'),    "inputs");

    % --- Sauvegarde du metadata.json ---
    % On suppose que "metadata" est une struct
    metadataFile = fullfile(resultsPath, 'metadata.json');

    % Conversion struct -> JSON (nécessite jsonencode, R2016b+)
    jsonStr = jsonencode(metadata);

    % Optionnel : rendre le JSON un peu plus lisible
    jsonStr = prettyJson(jsonStr);

    fid = fopen(metadataFile, 'w');
    if fid == -1
        error("Impossible d'ouvrir %s pour écriture.", metadataFile);
    end
    fwrite(fid, jsonStr, 'char');
    fclose(fid);
end

function out = prettyJson(in)
    % Petit "pretty print" très simple pour JSON
    out = regexprep(in, ',', ",\n");
    out = regexprep(out, '{', "{\n");
    out = regexprep(out, '}', "\n}");
end
