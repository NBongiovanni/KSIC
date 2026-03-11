function [state, stateRef, inputs] = planarControl(...
    sysParams, ...
    ctrlParams, ...
    positionRef, ...
    stateInit, ...
    noise ...
    )

    % Parameters settings
    dt = sysParams.dt;
    time = 0:dt:sysParams.lenSequ;
    mass = sysParams.mass;
    I_y = sysParams.I_y;
    g = sysParams.g;
    l = sysParams.l;

    if sysParams.datasetType == "test"
        tauRef = 0.1;
    else
        tauMin = 0.01;
        tauMax = 0.2;
        tauRef = rand()*(tauMax - tauMin) + tauMin;
    end
    
    TMax = ctrlParams.TMax;
    TMin = ctrlParams.TMin;
    FMax = ctrlParams.FMax;
    FMin = ctrlParams.FMin;

    PMax = ctrlParams.PMax;
    PMin = ctrlParams.PMin;
    
    thetaRefMax = ctrlParams.thetaRefMax;
    thetaRefMin = ctrlParams.thetaRefMin;

    % States and inputs initialization
    x = zeros(size(time));
    z = zeros(size(time));
    theta = zeros(size(time));
    xdot = zeros(size(time));
    zdot = zeros(size(time));
    thetadot = zeros(size(time));
    thetaRef = zeros(size(time));

    xRef = transpose(positionRef(:, 1));
    zRef = transpose(positionRef(:, 2));

    xRefFilt = zeros(size(time));
    zRefFilt = zeros(size(time));
    xRefFilt(1) = stateInit(1);
    zRefFilt(1) = stateInit(2);
    
    F = zeros(length(time)-1, 1);
    T = zeros(length(time)-1, 1);

    if noise
        [noiseF, noiseT] = genNoise(time);
    else
        noiseF = zeros(size(time));
        noiseT = zeros(size(time));
    end
    
    % Initial conditions
    x(1) = stateInit(1);
    z(1) = stateInit(2);
    theta(1) = stateInit(3);
    xdot(1) = stateInit(4);
    zdot(1) = stateInit(5);
    thetadot(1) = stateInit(6);
    
    % Erreurs cumulées pour les intégrateurs
    errorSumX = 0;
    errorSumZ = 0;
    errorSumTheta = 0;

    % Boucle de simulation
    for k = 1:length(time) - 1

        alpha = dt / (tauRef + dt);
        xRefFilt(k+1) = xRefFilt(k) + alpha * (xRef(k) - xRefFilt(k));
        zRefFilt(k+1) = zRefFilt(k) + alpha * (zRef(k) - zRefFilt(k));

        % Erreurs de position
        errorX = xRefFilt(k) - x(k);
        errorZ = zRefFilt(k) - z(k);
        
        % Contrôleur PID pour la position x (boucle externe)
        errorSumX = errorSumX + errorX * dt;
        uThetaRef = (-1) * (ctrlParams.KpX * errorX ...
                              + ctrlParams.KdX * (0 - xdot(k)) ...
                              + ctrlParams.KiX * errorSumX);
        
        % Limitation de la consigne d'angle
        uThetaRef = min(max(uThetaRef, thetaRefMin), thetaRefMax);
        thetaRef(k) = uThetaRef;
        
        % Contrôleur PID pour la position z (boucle externe)
        errorSumZ = errorSumZ + errorZ * dt;
        F(k) = mass * (ctrlParams.KpZ * errorZ ...
                       + ctrlParams.KdZ * (0 - zdot(k)) ...
                       + ctrlParams.KiZ * errorSumZ + g) ...
                       + noiseF(k);
        
        % Force entre 0 et 2 fois le poids du quadrotor
        F(k) = min(max(F(k), FMin), FMax);
        
        % Erreur d'attitude (boucle interne)
        errorTheta = uThetaRef - theta(k);
        errorSumTheta = errorSumTheta + errorTheta * dt;

        T(k) = ctrlParams.KpTheta * errorTheta ...
             + ctrlParams.KdTheta * (0 - thetadot(k)) ...
             + ctrlParams.KiTheta * errorSumTheta ...
             + noiseT(k);

        TMax_1 = TMax;
        TMax_2 = l*(2*PMax - F(k));
        TMax_3 = l*(F(k) - 2*PMin);
        TMax_f = min(min(TMax_1, TMax_2), TMax_3);

        TMin_1 = TMin;
        TMin_2 = -TMax_2;
        TMin_3 = -TMax_3;
        TMin_f = max(max(TMin_1, TMin_2), TMin_3);

        T(k) = min(max(T(k), TMin_f), TMax_f);
    
        % Mise à jour des états (intégration)
        xdot(k+1) = xdot(k) + dt * ((-1)*F(k) * sin(theta(k))/mass);
        zdot(k+1) = zdot(k) + dt * (F(k) * cos(theta(k))/mass - g);
        thetadot(k+1) = thetadot(k) + dt * T(k)/I_y;
        
        x(k+1) = x(k) + dt * xdot(k);
        z(k+1) = z(k) + dt * zdot(k);
        theta(k+1) = theta(k) + dt * thetadot(k);
    end

    % Output of the function
    state = transpose(cat(1, x, z, theta, xdot, zdot, thetadot));
    stateRef = transpose(cat(1, xRefFilt, zRefFilt, thetaRef));
    inputs = cat(2, F, T);
end

