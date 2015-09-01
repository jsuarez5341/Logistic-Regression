function [parameters, testingPercentAccuracy, testingPercentClassificationAccuracy] = LogRegDriver(dist, numSamples, numTestingSamples, numCVSamples, polyDegreeArray, cTolDec, cConstTol, regStrength, nDivs, twoColor, vidDebug, graphDebug)
%LOGREGDRIVER(dist, numSamples, numTestingSamples, numCVSamples, polyDegreeArray, cTolDec, cConstTol, regStrength, nDivs, twoColor, vidDebug, graphDebug)
%Is an all-in-one, nonlinear logistic regression software package.
%Data regularization, model selection, and activation 3D gradient plots are
%included.
%Returns:[parameters, testingPercentAccuracy, testingPercentClassificationAccuracy]
%dist: a string representing the desired dataset. See PolyGenData.
%numSamples: An even integer used to denote the total number of samples.
%These samples will be split among two catagories.
%numTestingSamples: used to evaluate the learned set of parameters.
%Increase for higher accuracy. Reccomended usage: 20-30 percent of
%numSamples.
%numCVSamples: used to pick a model. Increase for higher chance of picking
%the best model. Reccomended usage: 20-30 percent of numSamples.
%PolyDegreeArray: Either a single whole number or an array of whole
%numbers. All values in this parameter will be used to test the
%corrosponding polynomial model.
%cTolDec: Used to determine required percent accuracy for a model. Must be
%between 0 and 1.
%cConstTol: Used to determine how flat the certainty gradient must be before
%the gradient is considered to be zero. Decrease to increase model
%accuracy.
%regStrength: Used to manage data regularization. Experiment with this
%parameter for best results.
%nDivs: Used to manage graph clarity.
%twoColor: Use 1 to plot learned activation function in only two colors.
%Zero otherwise.
%vidDebug: 0 or 1. Will show learning algorithm at work. Only for visualization
%purposes. This will make code run very slowly.
%graphDebug: 0 or 1. Used to determine whether or not all graphs will be
%displayed.

%Used for model selection
qualityVec=[];
parameterCellArray=[];


for i=polyDegreeArray
    %Generate a dataset
    [data, xyLabels, xExps, yExps]=PolyGenData(dist, i, numSamples);
    
    %Perform logistic regression (binary classification)
    [parameters, quality]=LogReg(data, xyLabels, cTolDec, cConstTol, regStrength, xExps, yExps, vidDebug, nDivs, twoColor);
    qualityVec = [qualityVec, quality];
    parameterCellArray = [parameterCellArray; {parameters}];
    if graphDebug
        %Bounds
        xMin=min(xyLabels(:, 1))-1;
        xMax=max(xyLabels(:, 1))+1;
        yMin=min(xyLabels(:, 2))-1;
        yMax=max(xyLabels(:, 2))+1;
        %Display all graphs
        figure(i+1), PlotIntensity(xyLabels, parameters, [xMin, xMax, yMin, yMax], xExps, yExps, nDivs, twoColor);
    end
    
end

%Cross-Validation
CVQualityVec=[];
h=@(features, params) 1./(1+exp(-features*params'));
for i=polyDegreeArray
    %CV data
    data=PolyGenData(dist, i, numCVSamples);
    predictedActivations=h(data(:, 1:end-1), parameterCellArray{find(polyDegreeArray==i), :});
    missclassificationErrors = abs(data(:, end) - predictedActivations)>=0.5;
    CVQualityVec=[CVQualityVec, (numCVSamples-sum(missclassificationErrors))./numCVSamples];
end

%Choose best fit
goodDegreesIndicies=find(CVQualityVec>cTolDec);
if length(goodDegreesIndicies)>0
    bestDegreeIndex=goodDegreesIndicies(1); %pick the lowest good degree
else
    error('No fits found. Check that parameters are realistic.');
end

%Evaluate learned parameters on testing set
data=PolyGenData(dist, polyDegreeArray(bestDegreeIndex), numTestingSamples);

predictedActivations=h(data(:, 1:end-1), parameterCellArray{find(bestDegreeIndex), :});
testingPercentAccuracy = (numTestingSamples-sum(abs(data(:, end)-predictedActivations)))./numTestingSamples;
testingPercentClassificationAccuracy =  sum(abs(data(:, end)-predictedActivations)<=0.5)./numTestingSamples;

%Only return one set of parameters
parameters=parameterCellArray{bestDegreeIndex, :};

%Graph handling
display(strcat('Chose a polynomial of degree= ', num2str(polyDegreeArray(bestDegreeIndex))));
display(strcat('Testing percent accuracy=', num2str(100*testingPercentAccuracy)));
display(strcat('Testing percent classification accuracy=', num2str(100*testingPercentClassificationAccuracy)));

end

