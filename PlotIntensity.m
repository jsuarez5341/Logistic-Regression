function [] = PlotIntensity(xyLabels, parameters, bounds, xExps, yExps, nDivs, twoColor)
%PLOTINTENSITY(xyLabels, parameters, bounds, xExps, yExps)
%Returns nothing. Plots the learned hypothesis.
%xyLabels: a three-column vector of x, y, and the binary data labels
%parameters: row vector of parameters matching data
%xExps: row vector containing a vector of exponents to attach to x
%yExps: row vector containing a vector of exponents to attach to y
%twoColor: 0 or 1. Used to control the colormap of the plot.

%Extract from bounds
xMin=bounds(1);
xMax=bounds(2);
yMin=bounds(3);
yMax=bounds(4);

%Parse input
if nargin==5
    nDivs=149; %this is actually nDivs-1
else
    nDivs=nDivs-1;
end

%Slice up the graph for coloring
[X, Y]=meshgrid(xMin:(xMax-xMin)./nDivs:xMax, yMin:(yMax-yMin)./nDivs:yMax);

%Sigmoid activation
h = @(featureSet) 1./(1+exp(-featureSet*parameters'));


sizeConst=size(X,1).*size(X,2); %Number of elements in cMap

%Defines a color as an intensity between 0 and 1, given by the activation
%function applied to the features given by an x-y pair and the
%corresponding polynomial terms 
cMap=h(bsxfun(@power, reshape(X, sizeConst, 1), xExps) .* bsxfun(@power, reshape(Y, sizeConst, 1), yExps));

%Plot color map and view from above
surf(X, Y, reshape( cMap, size(X,1), size(X,2) ) , 'EdgeColor', 'None'), ... 
xlabel('x'), ylabel('y'), zlabel('Certainty that data class is 1'), title('Logistic Regression for Nonlinear Binary Classification');
view(2);

if nargin==7 && twoColor %First arg checked first
    colormap(summer(2)); %Blue on green, magenta on yellow
end

%Scatter plot data on top of the color map. Note that the points will
%appear slightly above the rest of the plot, but do not contain z data.
hold on;
for i=1:length(xyLabels(:,3))
    if xyLabels(i,3)==1
        plot3(xyLabels(i,1), xyLabels(i,2), 1.1, 'Marker', '.', 'MarkerSize', 20, 'Color', 'm'); %label = 1
    else
        plot3(xyLabels(i,1), xyLabels(i,2), 1.1, 'Marker', '.', 'MarkerSize', 20, 'Color', 'c'); %label = 0
    end
end
axis([xMin, xMax, yMin, yMax]);

end