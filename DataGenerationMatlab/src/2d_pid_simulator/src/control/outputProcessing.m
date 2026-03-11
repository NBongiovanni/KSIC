function outputControl = outputProcessing(params, counter, states, statesRef, inputs)

    if params.typeDataset == "test"
        roundCounter = counter;
    else
        roundCounter = floor(counter/100)*100;
    end
    
    statesV2 = states(1:roundCounter, :, :);
    statesRefV2 = statesRef(1:roundCounter, :, :);
    inputsV2 = inputs(1:roundCounter, :, :);
    
    outputControl.roundCounter = roundCounter;

end