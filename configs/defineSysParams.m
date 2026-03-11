function sysParams = defineSysParams(typeConfig)
    if typeConfig == "inventedDrone"
        sysParams = defineConfigInventedDrone();
    elseif typeConfig == "crazyflie"
        sysParams = defineConfigCrazyflie();
    end
end


function sysParams = defineConfigInventedDrone()
    sysParams.droneType = "planarQuad";
    sysParams.trajectoryType = "planar";
    sysParams.mass = 1;        % Masse (kg)
    sysParams.I_y = 0.15;      % Moment d'inertie autour de y (kg.m^2)
    sysParams.lenSequ = 5;

    % Maximum values used for dataset generation 
    % (outliers are removed)
    sysParams.xMax = 0.9;
    sysParams.zMax = 0.9;
    sysParams.thetaMax = 20*pi/180;
    sysParams.xdotInitMax = 0;
    sysParams.zdotInitMax = 0;
    sysParams.thetadotMax = 100*pi/180;

    % Used for setpoints and C.I generation:
    sysParams.xInitMax = 0.8;
    sysParams.zInitMax = 0.8;
    sysParams.thetaInitMax = 20*pi/180;
    sysParams.xdotInitMax = 0.2;
    sysParams.zdotInitMax = 0.2;
    sysParams.thetadotInitMax = 20*pi/180;

    sysParams.nTrajsDisp = 10;

    sysParams.g = 9.81;       % Accélération due à la gravité (m/s^2)
    sysParams.l = 0.2;        % Demi-longueur du quadrotor (m)
    sysParams.dt = 0.01;
end


function sysParams = defineConfigCrazyflie()
    sysParams.mass = 0.027;        % Masse (kg)
    sysParams.I_y = 1.395*10^(-5);  % Moment d'inertie autour de y (kg.m^2)
    sysParams.lenSequ = 2.5;
    sysParams.noise = false;
    sysParams.g = 9.81;       % Accélération due à la gravité (m/s^2)
    sysParams.l = 30*0.001;        % Demi-longueur du quadrotor (m)

    % Maximum values used for setpoints and C.I generation
    % (setpoint only on xmax, zmax)
    sysParams.xMax = 40;
    sysParams.zMax = 40;
    sysParams.theta_max = pi/5;
    sysParams.xdotMax = 1;
    sysParams.zdotMax = 1;
    sysParams.thetadotMax = 1.5;
    
    sysParams.TMax = 2;
    sysParams.TMin = -sysParams.TMax;

    sysParams.PMax = sysParams.mass * sysParams.g * 1;
    sysParams.PMin = sysParams.mass * sysParams.g * 0.2;

    sysParams.animate = 0;
    sysParams.nTrajsDisp = 5;

    sysParams.hoverThrust = sysParams.mass * sysParams.g;
    sysParams.dt = 0.01;
end



