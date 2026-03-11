function generateDataset( ...
        type, ...
        sysParams, ...
        ctrlParams, ...
        plotBool, ...
        addNoise ...
        )
    time = 0:sysParams.dt:sysParams.lenSequ;    
    states = zeros(sysParams.nTrajs, length(time), 6);
    statesRef = zeros(sysParams.nTrajs, length(time), 3);
    inputs = zeros(sysParams.nTrajs, length(time)-1, 2);
    
    outliersCounter = 0;
    for i = 1:sysParams.nTrajs
        disp(i);
        xInit = genPlanarICs(sysParams, i);
        xRef = genPlanarRefs(sysParams, time, i, xInit);
        
        if addNoise
            if mod(i, 2) == 0
                noise = true;
            else
                noise = false;
            end
        else
            noise = false;
        end
    
        [x, xfullRef, u] = planarControl( ...
            sysParams, ...
            ctrlParams, ...
            xRef, ...
            xInit, ...
            noise ...
            );

        while isTrajectoryWithinBounds(x, type, sysParams) == false
            outliersCounter = outliersCounter + 1;
            xInit = genPlanarICs(sysParams, i);
            xRef = genPlanarRefs(sysParams, time, i, xInit);
            [x, xfullRef, u] = planarControl( ...
                sysParams, ...
                ctrlParams, ...
                xRef, ...
                xInit, ...
                noise ...
                );
        end
            
        states(i, :, :) = x;
        statesRef(i, :, :) = xfullRef;
        inputs(i, :, :) = u;
    end
    
    smoothedStates = movmean(states, 5, 2);
    idx      = 1:5:size(smoothedStates, 2);
    statesDS   = smoothedStates(:, idx, :);

    smoothedInputs = movmean(inputs, 5, 2);
    idx      = 1:5:size(smoothedInputs, 2);
    inputsDS   = smoothedInputs(:, idx, :);

    smoothedRefs = movmean(statesRef, 5, 2);
    idx      = 1:5:size(smoothedRefs, 2);
    statesRefsDS   = smoothedRefs(:, idx, :);

    idx      = 1:5:size(time, 2);
    timeDS   = time(:, idx, :);
    
    summarizeDataset(states, inputs, outliersCounter)

    metadata = struct();
    metadata.datasetType   = sysParams.datasetType;
    metadata.dt            = sysParams.dt;
    metadata.lenSequ       = sysParams.lenSequ;
    metadata.nTrajs        = sysParams.nTrajs;
    metadata.noise         = addNoise;
    metadata.downsample    = 5;
    metadata.smoothing     = true;
    metadata.dateGenerated = datetime('now');

    saveResults(sysParams, statesDS, inputsDS, statesRefsDS, metadata);

    if plotBool
        plotResults( ...
            sysParams, ...
            timeDS, ...
            statesDS, ...
            inputsDS, ...
            statesRefsDS, ...
            sysParams.nTrajs ...
            )
    end