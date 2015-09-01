%Run-file for logistic regression

%Cleanup
clear;
clc;
close all hidden;

%Give user a second to adjust size
figure(1);
%pause(.2);

%Fix display fonts
set(0,'DefaultAxesFontName', 'Ariel');
set(0,'DefaultAxesFontSize', 16);
set(0,'DefaultTextFontname', 'Ariel');
set(0,'DefaultTextFontSize', 16);

%Function input parameters
dist=                   'ring';
numSamples=             2000;
numTestingSamples=      200;
numCVSamples=           200;
polyDegreeArray=        2;
cTolDec=                0.95;
cConstTol=              0.1;
regStrength=            0;
nDivs=                  150;
twoColor=               0;
vidDebug=               1;
graphDebug=             1;


%Script
[parameters, testingPercentAccuracy, testingPercentClassificationAccuracy] =... 
LogRegDriver(dist, numSamples, numTestingSamples, numCVSamples, ...
             polyDegreeArray, cTolDec, cConstTol, regStrength, ...
             nDivs, twoColor, vidDebug, graphDebug);
