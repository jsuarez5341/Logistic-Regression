function [ data, xylabel, xExps, yExps ] = PolyGenData(dist, degree, numSamples)
%POLYGENDATA(distribution, degree, numSamples)
%Generates a random dataset according to a predefined distribution.
%Features are stored column-wise, with labels in the last column.
%Returns: [data, [x y labels], xExps, yExps] where xExps and yExps are
%row vectors that can be paired to give all possible exponent combinations x^n*y^m
%distribution: choose "2cluster", "4cluster" "ring" or "tiny"
%degree: highest polynomial term combination to include (x^d*y^d)
%numSamples: even integer. First half labeled zero, second half
%labeled one

%Check inputs----------
if nargin~=3 && ~strcmp(dist, 'tiny')
    error('3 parameters required, unless the first parameter is "tiny"');
end

if mod(numSamples,2)~=0
    warning('Input an even number of samples. Sample count reduced by one');
end
%----------------------

numSamples=2.*floor(numSamples./2); %Fix odd numSamples input
numClass=numSamples./2;             %Split into 2 classes
degree=floor(abs(degree));          %Positive int


if strcmp(dist, 'tiny')
    warning('Ignoring numSamples parameter (tiny dataset requested)');
    x= [ -1 -1  0  1  1  1 ]';
    y= [  1  0  0  1 -1  0 ]';
    %Will assign labels [0 0 0 1 1 1]
    numSamples=6;
    numClass=3;
    
elseif strcmp(dist, '2cluster')
    x=[rand(numClass,1); rand(numClass,1)+1.25];
    y=[rand(numClass,1); rand(numClass,1)+1.25];
    
elseif strcmp(dist, '4cluster')
    x=[rand(numClass./2,1); rand(numClass./2,1)+2; rand(numClass./2,1); rand(numClass./2,1)+2;];
    y=[rand(numClass./2,1); rand(numClass./2,1); rand(numClass./2,1)+2; rand(numClass./2,1)+2;];
    
elseif strcmp(dist, 'ring')
    
    %Inner Circle
    x = rand(numClass,1);
    y = 1-x;
    x=sqrt(x);
    y=sqrt(y);
    x= (2*round(rand(numClass,1))-1).*x;
    y= (2*round(rand(numClass,1))-1).*y;
    x=rand(numClass,1).*x;
    y=rand(numClass,1).*y;
    
    %Save data
    xInner=x;
    yInner=y;
    
    %Next ring layer
    thetas=2.*pi.*rand(numClass,1);
    x=(3+2.*rand(numClass,1)).*cos(thetas);
    y=(3+2.*rand(numClass,1)).*sin(thetas);
    %Add to data matrix
    x=[xInner; x];
    y=[yInner; y];
else
    error('Pick a valid distribution');
end

%Generate polynomial terms
xExps=reshape(meshgrid(0:degree,    0:degree)' , 1, (degree+1).^2);
yExps=reshape(meshgrid((0:degree)', 0:degree)  , 1, (degree+1).^2);

%Simple method of applying powers element-wise to a matrix
data=bsxfun(@power, x, xExps).* bsxfun(@power, y, yExps);

%Assume first half labeled 0, second half labeled 1
data(1:numClass, end+1) = 0;
data(numClass+1:numSamples, end) = 1; %use new end

%Handy matrix for later use, of columns x, y, and labels
xylabel=[data(:,2), data(:,degree+2), data(:,end)];


end

