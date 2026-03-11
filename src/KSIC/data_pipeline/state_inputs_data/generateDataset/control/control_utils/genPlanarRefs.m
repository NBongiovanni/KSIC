function ref = genPlanarRefs(params, time, trajIdx, stateInit)
    xMax = params.xInitMax;
    zMax = params.zInitMax;

    [nRows, nSteps] = size(time);
    ref = zeros(nSteps, 2);
    nTrajs = params.nTrajs;

    if params.datasetType == "test"
        if trajIdx == 1
            ref(:, 1) = -0.4;
            ref(:, 2) = -0.1;
        elseif trajIdx < 3
            ref(:, 2) = rand()*zMax*2 - zMax;
        elseif trajIdx < 5
            ref(:, 1) = rand()*xMax*2 - xMax;

        else
            ref(:, 1) = rand()*xMax*2 - xMax;
            ref(:, 2) = rand()*zMax*2 - zMax;
        end
    else
        if trajIdx < nTrajs/10
            ref(:, 1) = stateInit(1);
            ref(:, 2) = stateInit(2);
        elseif trajIdx < 3*nTrajs/10
            ref(:, 2) = rand()*zMax*2 - zMax;
        elseif trajIdx < 6*nTrajs/10
            ref(:, 1) = rand()*xMax*2 - xMax;
        else
            ref(:, 1) = rand()*xMax*2 - xMax;
            ref(:, 2) = rand()*zMax*2 - zMax;
        end
    end
end
