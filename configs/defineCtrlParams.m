function ctrlParams = defineCtrlParams(typeTuning, phase)
    ctrlParams.tuning_PID = typeTuning;
    
    % Gains du contrôleur (inner and outer loop)
    if typeTuning == "train_1" || typeTuning == "train_2" || typeTuning == "val"
        if phase == 1
            scalar = 1;
        elseif phase == 2
            scalar = 3;
        end

        ctrlParams.KpX = 2*scalar;
        ctrlParams.KiX = 0;
        ctrlParams.KdX = 2*scalar;
        
        ctrlParams.KpZ = 5.0*scalar;
        ctrlParams.KiZ = 0.;
        ctrlParams.KdZ = 3.0*scalar;
        
        ctrlParams.KpTheta = 20.0*scalar;
        ctrlParams.KiTheta = 0.;
        ctrlParams.KdTheta = 8.*scalar;

        ctrlParams.TMax = 2*scalar;
        ctrlParams.TMin = -ctrlParams.TMax;
        ctrlParams.FMax = 13*scalar;
        ctrlParams.FMin = 1;
        ctrlParams.PMax = 11*scalar;
        ctrlParams.PMin = 1;
        ctrlParams.thetaRefMax = 0.75*pi/6*scalar;
        ctrlParams.thetaRefMin = -ctrlParams.thetaRefMax;

        

    elseif typeTuning == "test"
        scalar = 1.7;
        ctrlParams.KpX = 1.3*scalar;
        ctrlParams.KiX = 0;
        ctrlParams.KdX = 2*scalar;
        
        ctrlParams.KpZ = 5.0*scalar;
        ctrlParams.KiZ = 0.;
        ctrlParams.KdZ = 3.0*scalar;
        
        ctrlParams.KpTheta = 23.0*scalar; % planar: 20??
        ctrlParams.KiTheta = 0.;
        ctrlParams.KdTheta = 9.0*scalar;

        ctrlParams.TMax = 3;
        ctrlParams.TMin = -ctrlParams.TMax;
        ctrlParams.FMax = 14;
        ctrlParams.FMin = 1;
        ctrlParams.PMax = 12;
        ctrlParams.PMin = 1;
        ctrlParams.thetaRefMax = 1.3*pi/6;
        ctrlParams.thetaRefMin = -ctrlParams.thetaRefMax;
    end
end



