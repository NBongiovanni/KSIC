function stateInit = genPlanarICs(params, trajIdx)
    if params.datasetType == "test"
        stateInit = genPlanarTestICs(params, trajIdx);
    else
        stateInit = genPlanarRandICs(params, trajIdx);
    end
end


function stateInit = genPlanarTestICs(params, trajIdx)
    xMax = params.xInitMax;
    zMax = params.zInitMax;
    if trajIdx == 1
        stateInit = transpose(cat( ...
        1, ...
        0.4, ...
        0.1, ...
        0, ...
        0, ...
        0, ...
        0 ...
        ));
    elseif trajIdx < 3
        stateInit = genRandomICs(0, zMax, 0, 0, 0, 0);
    elseif trajIdx < 5
        stateInit = genRandomICs(xMax, 0, 0, 0, 0, 0);
    else
        stateInit = genRandomICs(xMax, zMax, 0, 0, 0, 0);
    end
end


function stateInit = genPlanarRandICs(params, trajIdx)
    xMax = params.xInitMax;
    zMax = params.zInitMax;
    thMax = params.thetaInitMax;
    xdMax = params.xdotInitMax;
    zdMax = params.zdotInitMax;
    thdMax = params.thetadotInitMax;
    nTrajs = params.nTrajs;

    if trajIdx < nTrajs/10
        stateInit = genRandomICs(xMax, zMax, 0, 0, 0, 0);
    elseif trajIdx < 3*nTrajs/10
        stateInit = genRandomICs(0, zMax, 0, 0, zdMax, 0);
    elseif trajIdx < 4*nTrajs/10
        stateInit = genRandomICs(xMax, 0, 0, xdMax, 0, 0);
    elseif trajIdx < 6*nTrajs/10
        stateInit = genRandomICs(xMax, 0, thMax, xdMax, 0, thdMax);
    else
        stateInit = genRandomICs(xMax, zMax, thMax, xdMax, zdMax, thdMax);
    end
end


function stateInit = genRandomICs( ...
    xMax, ...
    zMax, ...
    thetaMax, ...
    xdotMax, ...
    zdotMax, ...
    thetadotMax)

    xInit = rand()*xMax*2 - xMax;
    zInit = rand()*zMax*2 - zMax;
    thetaInit = rand()*thetaMax*2 - thetaMax;
    xdotInit = rand()*xdotMax*2 - xdotMax;
    zdotInit = rand()*zdotMax*2 - zdotMax;
    thetadotInit = rand()*thetadotMax*2 - thetadotMax;  

    stateInit = transpose(cat( ...
        1, ...
        xInit, ...
        zInit, ...
        thetaInit, ...
        xdotInit, ...
        zdotInit, ...
        thetadotInit ...
        ));
end