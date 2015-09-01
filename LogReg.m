function [ parameters, quality ] = LogReg(data, xyLabels, cTolDec, cConstTol, regStrength, xExps, yExps, vidDebug, nDivs, twoColor)
%LOGREG(data, data, xyLabels, cTolDec, cConstTol, regStrength, xExps, yExps, vidDebug, nDivs, twoColor)
%Performs logistic regression to learn a nonlinear classification boundary.
%Returns: [parameters, quality] where parameters is a row vector of learned
%parameters and quality is a two-element vector containing the summed
%activations learned for the dataset and the total number of data points.
%data: formatted data matrix including data by columns and labels in the
%final column. See PolyGenData.
%xyLabels: easy-access matrix containing three columns of x, y, and labels
%cTolDec: Used to determine required percent accuracy for a model. Must be
%between 0 and 1.
%cConstTol: Used to determine how flat the certainty gradient must be before
%the gradient is considered to be zero. Decrease to increase model
%accuracy.
%regStrength: Used to manage data regularization. Experiment with this
%parameter for best results.
%xExps and yExps: row vectors that can be paired to give all possible exponent
%combinations x^n*y^m
%vidDebug: 0 or 1. Will show learning algorithm at work. Only for visualization
%purposes. This will make code run very slowly.
%nDivs: Used to manage graph clarity.
%twoColor: Use 1 to plot learned activation function in only two colors.
%Zero otherwise.

%y = labels
y=xyLabels(:,3);

%x = features
x=data(:,1:end-1);

%t = parameters (initialized randomly)
t=rand(1, size(x,2));
newt=t;

%Equal to the number of rows in data
numSamples = size(data, 1);

%Learning constant information
alpha=1;
alphaMult=2;
alphaDiv=2;

%Sigmoid activation
h=@(features, params) 1./(1+exp(-features*params'));

newh=h(x,t);            %Computed activation column vector
c=-inf;                 %Certainty (to maximize)

%Check input
if nargin==2
    cTol=.995.*numSamples;    %Desired boundary accuracy
else
    cTol=cTolDec.*numSamples;
end

if nargin==3
    cConstTol=.01;          %When c is changing by this little, grad slope is flat
end

if nargin==4
    lambda=1;
else
    lambda=regStrength;
end

j=0;
iMax=250;               %Controls random walk code
done=0;                 %Tells the program when to exit

while ~done
    j=j+1; %Used for video debug
    
    %Gradient (analytically derived from log certainty. Note that the
    %log certainty and the certainty share a maximum, and should be negative of convex
    grad = (alpha./length(y)).*  (((y-newh)'*x) + lambda.*sum(  abs(newt(2:end)).^.25 ));
    
    newt=grad + t; %Gradient ascent
    
    %Check certainty
    newh=h(x,newt);
    
    %There is only one term here. The other will cancel (y=0 or y=1)
    newc=sum(y.*newh+(1-y).*(1-newh));
    
    if newc >= c %So far so good
        if (newc-c)<cConstTol %Gradient is flat
            if newc >= cTol
                done=1;
            else
                %Something went wrong. Parameters found yield too much error.
                %Consider checking polynomial degree.
                %Will execute random walk to look for a better maximum.
                c=newc;
                aBase=1+max(abs(t)); %Seems like a reasonable guess
                for i=1:iMax
                    alpha = aBase.*(rand-.5);
                    grad = (alpha./length(y)).*  (((y-newh)'*x) + lambda.*sum(  abs(newt(2:end)).^.25 ));
                    
                    newt=t + grad;
                    newh=h(x,newt);
                    newc=sum(y.*newh+(1-y).*(1-newh));
                    if newc > (c+1)
                        %Found a better maximum
                        t=newt;
                        done=0;
                        break;
                    else %Perhaps the data is bad. Could be the polynomial degree.
                        if i==iMax
                            warning('Learned function may be a poor fit. Check plot, input, and polynomial degree');
                        end
                        done=1;
                    end
                    
                end
                
            end
        else %Normal case. Gradient not flat.
            c=newc;
            t = t + grad;
            if c>cTol
                done=1;
                break;
            end
            %Account for gradient reduction inherent in the curve
            alpha=alpha*alphaMult;
        end
    else %Overshot maximum.
        alpha = alpha./alphaDiv; %Reduce step size
    end
    
    
    %-----Code for animation-----
    if nargin>=8 && vidDebug && c==newc
        clf
        PlotIntensity(xyLabels, t, [-4, 4, -4, 4], xExps, yExps, nDivs, twoColor);
        pause(.1);
    end
    %----------------------------
    
end
newh=h(x, t); %Don't use the potentially bad value from before

quality = [sum(y.*newh+(1-y).*(1-newh)), length(y)];
parameters=t;

%Allow a moment for user to view figure
if nargin>=8 && vidDebug
    clf
    PlotIntensity(xyLabels, t, [-4, 4, -4, 4], xExps, yExps, nDivs, twoColor); %last update
    pause(.75);
end
end