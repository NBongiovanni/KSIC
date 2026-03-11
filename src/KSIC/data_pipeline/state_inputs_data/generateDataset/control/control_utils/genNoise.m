function [noiseF, noiseT] = genNoise(time)
    noiseF = zeros(size(time));
    noiseT = zeros(size(time));
    
    NT = 0;
    NF = 0;
    for k = 1:length(time) - 1
        if mod(k, 5) == 0
            NT = randn()*0.3;
            NF = randn()*0.3;
        end
        noiseT(k) = NT;
        noiseF(k) = NF;
    end
end

