function test = isTrajectoryWithinBounds(x, type, params)
    if type ~= "test"
        thetaMax = params.thetaMax;
    else
        thetaMax = params.thetaMax*10;
    end

    if (max(abs(x(:, 1))) > params.xMax) ...
        || (max(abs(x(:, 2))) > params.zMax) ...
        || (max(abs(x(:, 3))) > thetaMax)
        test = false;
    else
        test = true;
    end
end

