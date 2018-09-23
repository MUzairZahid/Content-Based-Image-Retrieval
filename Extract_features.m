function queryImageFeature=Extract_features(queryImage,imgInfo,name)
%This function extract the following features from input image:
% LOW IMAGE FEATURES
% * Color histogram:  HSV space is chosen, each H, S, V component is uniformly
% quantized into 8, 2 and 2 bins respectively
%
% * Color auto-correlogram: The image is quantized into 4x4x4 = 64 colors 
% in the RGB space 64
% 
% * Color moments: The first two moments (mean and standard deviation) from the R,
% G, B color channels are extracted
%
% SEMANTIC IMAGE FEATURES
% * Gabor wavelet: Gabor wavelet filters spanning four scales and six orientations
% are applied to the image.
%
% * Wavelet moments: Applying the wavelet transform to the image with a 3-level
% decomposition, the mean and the standard deviation of the
% transform coefficients are used to form the feature vector

%The output is a vector of the features values



WB=waitbar(0.5,'Extracting image features...');
hsvHist = HSV_Histo(queryImage);
autoCorrelogram = auto_Cor(queryImage);
color_moments = Color_moment(queryImage);
img = double(rgb2gray(queryImage))/255; % image must be turned to gray to use gabor filter
[meanAmplitude, msEnergy] = wavelet_g(img, 4, 6); % 4 scales, 6 orientations
wavelet_moments = wavelet_trans(queryImage, imgInfo.ColorType);
% construct the queryImage feature vector
queryImageFeature = [hsvHist autoCorrelogram color_moments meanAmplitude,...
    msEnergy wavelet_moments str2double(name)];
pause(0.25)
waitbar(1,WB,'Done')
close(WB)
end

function hsvHist = HSV_Histo(image)
[rows, cols, ~] = size(image);
% totalPixelsOfImage = rows*cols*numOfBands;
image = rgb2hsv(image);
% split image into h, s & v planes
h = image(:, :, 1);
s = image(:, :, 2);
v = image(:, :, 3);
numberOfLevelsForH = 8;
numberOfLevelsForS = 2;
numberOfLevelsForV = 2;
% Find the max.
maxValueForH = max(h(:));
maxValueForS = max(s(:));
maxValueForV = max(v(:));
% create final histogram matrix of size 8x2x2
hsvHist = zeros(8, 2, 2);
% create col vector of indexes for later reference
index = zeros(rows*cols, 3);
% Put all pixels into one of the "numberOfLevels" levels.
count = 1;
for row = 1:size(h, 1)
    for col = 1 : size(h, 2)
        quantizedValueForH(row, col) = ceil(numberOfLevelsForH * h(row, col)/maxValueForH);
        quantizedValueForS(row, col) = ceil(numberOfLevelsForS * s(row, col)/maxValueForS);
        quantizedValueForV(row, col) = ceil(numberOfLevelsForV * v(row, col)/maxValueForV);
        
        % keep indexes where 1 should be put in matrix hsvHist
        index(count, 1) = quantizedValueForH(row, col);
        index(count, 2) = quantizedValueForS(row, col);
        index(count, 3) = quantizedValueForV(row, col);
        count = count+1;
    end
end
% put each value of h,s,v to matrix 8x2x2
for row = 1:size(index, 1)
    if (index(row, 1) == 0 || index(row, 2) == 0 || index(row, 3) == 0)
        continue;
    end
    hsvHist(index(row, 1), index(row, 2), index(row, 3)) = ...
        hsvHist(index(row, 1), index(row, 2), index(row, 3)) + 1;
end
% normalize hsvHist to unit sum
hsvHist = hsvHist(:)';
hsvHist = hsvHist/sum(hsvHist);
end

function auto_Cor = auto_Cor(image)
% quantize image into 64 colors = 4x4x4, in RGB space
[img_no_dither, map] = rgb2ind(image, 64, 'nodither');
rgb = ind2rgb(img_no_dither, map); % rgb = double(rgb)
distances = [1 3 5 7];
auto_Cor = correlogram(rgb, map, distances);
auto_CorFix = zeros(1,64);
for i = 1:size(auto_Cor, 2)
    auto_CorFix(i) = auto_Cor(i);
end
auto_Cor = reshape(auto_CorFix, [4 4 4]);
% consturct final correlogram using distances
auto_Cor(:, :, 1) = auto_Cor(:, :, 1)*distances(1);
auto_Cor(:, :, 2) = auto_Cor(:, :, 2)*distances(2);
auto_Cor(:, :, 3) = auto_Cor(:, :, 3)*distances(3);
auto_Cor(:, :, 4) = auto_Cor(:, :, 4)*distances(4);
% reform it to vector format
auto_Cor = reshape(auto_Cor, 1, 64);
end

% check if point is a valid pixel
function valid = is_valid(X, Y, point)
if point(1) < 0 || point(1) >= X
    valid = 0;
end
if point(2) < 0 || point(2) >= Y
    valid = 0;
end
valid = 1;
end

% find pixel neighbors
function Cn = get_neighbors(X, Y, x, y, dist)
cn1 = [x+dist, y+dist];
cn2 = [x+dist, y];
cn3 = [x+dist, y-dist];
cn4 = [x, y-dist];
cn5 = [x-dist, y-dist];
cn6 = [x-dist, y];
cn7 = [x-dist, y+dist];
cn8 = [x, y+dist];

points = {cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8};
Cn = cell(1, length(points));

for ii = 1:length(points)
    valid = is_valid(X, Y, points{1, ii});
    if (valid)
        Cn{1, ii} = points{1, ii};
    end
end

end

% get correlogram
function colors_percent = correlogram(photo, Cm, K)
[X, Y, ~] = size(photo);
colors_percent = [];

%     for k = 1:length(K) % loop over distances
for k = 1:K % loop over distances
    countColor = 0;
    
    color = zeros(1, length(Cm));
    
    for x = 2:floor(X/10):X % loop over image width
        for y = 2:floor(Y/10):Y % loop over image height
            Ci = photo(x, y);
            %                Cn = get_neighbors(X, Y, x, y, K(k));
            Cn = get_neighbors(X, Y, x, y, k);
            
            for jj = 1:length(Cn) % loop over neighbor pixels
                Cj = photo( Cn{1, jj}(1), Cn{1, jj}(2) );
                
                for m = 1:length(Cm) % loop over map colors
                    if isequal(Cm(m), Ci) && isequal(Cm(m), Cj)
                        countColor = countColor + 1;
                        color(m) = color(m) + 1;
                    end
                end
            end
        end
    end
    
    for ii = 1:length(color)
        color(ii) = double( color(ii) / countColor );
    end
    
    colors_percent = color;
end
end

function Color_moment = Color_moment(image)
% extract color channels
R = double(image(:, :, 1));
G = double(image(:, :, 2));
B = double(image(:, :, 3));
% compute 2 first color moments from each channel
meanR = mean( R(:) );
stdR  = std( R(:) );
meanG = mean( G(:) );
stdG  = std( G(:) );
meanB = mean( B(:) );
stdB  = std( B(:) );
% construct output vector
Color_moment = zeros(1, 6);
Color_moment(1, :) = [meanR stdR meanG stdG meanB stdB];
end

function[gaborSquareEnergy, gaborMeanAmplitude] = wavelet_g(varargin)

% Get arguments and/or default values
[im, nscale, norient, minWaveLength, mult, sigmaOnf,  dThetaOnSigma,k, ...
    polarity] = checkargs(varargin(:));

v = version; Octave = v(1) < '5';  % Crude Octave test
epsilon         = .0001;         % Used to prevent division by zero.

% Calculate the standard deviation of the angular Gaussian function
% used to construct filters in the frequency plane.
thetaSigma = pi/norient/dThetaOnSigma;

[rows,cols] = size(im);
imagefft = fft2(im);                % Fourier transform of image
zero = zeros(rows,cols);

totalEnergy = zero;                 % Matrix for accumulating weighted phase
% congruency values (energy).
totalSumAn  = zero;                 % Matrix for accumulating filter response
% amplitude values.
orientation = zero;                 % Matrix storing orientation with greatest
% energy for each pixel.
estMeanE2n = [];
EO = cell(nscale, norient);         % Cell array of convolution results
ifftFilterArray = cell(1, nscale);  % Cell array of inverse FFTs of filters


% Pre-compute some stuff to speed up filter construction

% Set up X and Y matrices with ranges normalised to +/- 0.5
% The following code adjusts things appropriately for odd and even values
% of rows and columns.
if mod(cols,2)
    xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
else
    xrange = [-cols/2:(cols/2-1)]/cols;
end

if mod(rows,2)
    yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
else
    yrange = [-rows/2:(rows/2-1)]/rows;
end

[x,y] = meshgrid(xrange, yrange);

radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
theta = atan2(-y,x);              % Matrix values contain polar angle.
% (note -ve y is used to give +ve
% anti-clockwise angles)

radius = ifftshift(radius);       % Quadrant shift radius and theta so that filters
theta  = ifftshift(theta);        % are constructed with 0 frequency at the corners.
radius(1,1) = 1;                  % Get rid of the 0 radius value at the 0
sintheta = sin(theta);
costheta = cos(theta);
lp = lowpassfilter([rows,cols], .4, 10);   % Radius .4, 'sharpness' 10

logGabor = cell(1, nscale);

for s = 1:nscale
    wavelength = minWaveLength*mult^(s-1);
    fo = 1.0/wavelength;                  % Centre frequency of filter.
    logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));
    logGabor{s} = logGabor{s}.*lp;        % Apply low-pass filter
    logGabor{s}(1, 1) = 0;                 % Set the value at the 0 frequency point of the filter
    % back to zero (undo the radius fudge).
end

% Then construct the angular filter components...
spread = cell(1, norient);

for o = 1:norient
    angl = (o-1)*pi/norient;           % Filter angle.
    ds = sintheta * cos(angl) - costheta * sin(angl);    % Difference in sine.
    dc = costheta * cos(angl) + sintheta * sin(angl);    % Difference in cosine.
    dtheta = abs(atan2(ds,dc));                          % Absolute angular distance.
    spread{o} = exp((-dtheta.^2) / (2 * thetaSigma^2));  % Calculate the
    % angular filter component.
end

count = 1;
gaborSquareEnergy = [];
gaborMeanAmplitude = [];
% The main loop...
for o = 1:norient,                   % For each orientation.
    if Octave,fflush(1); end
    
    sumAn_ThisOrient  = zero;
    Energy_ThisOrient = zero;
    
    for s = 1:nscale,                  % For each scale.
        
        filter = logGabor{s} .* spread{o};  % Multiply radial and angular
        % components to get filter.
        
        ifftFilt = real(ifft2(filter))*sqrt(rows*cols);  % Note rescaling to match power
        ifftFilterArray{s} = ifftFilt;                   % record ifft2 of filter
        
        % Convolve image with even and odd filters returning the result in EO
        EO{s, o} = ifft2(imagefft .* filter);
        An  = abs(EO{s,o});                        % Amplitude of even & odd filter response.
        sumAn_ThisOrient = sumAn_ThisOrient + An; % Sum of amplitude responses.
        gaborSquareEnergy(count) = sum(sum( An.^2 ) );
        gaborMeanAmplitude(count) = mean2( An );
        count = count + 1;
        if s==1
            EM_n = sum(sum(filter.^2)); % Record mean squared filter value at smallest
        end                             % scale. This is used for noise estimation.
        
    end                                 % ... and process the next scale
    
    % calculate the phase symmetry measure
    
    if polarity == 0     % look for 'white' and 'black' spots
        for s = 1:nscale,
            Energy_ThisOrient = Energy_ThisOrient ...
                + abs(real(EO{s,o})) - abs(imag(EO{s,o}));
        end
        
    elseif polarity == 1  % Just look for 'white' spots
        for s = 1:nscale,
            Energy_ThisOrient = Energy_ThisOrient ...
                + real(EO{s,o}) - abs(imag(EO{s,o}));
        end
        
    elseif polarity == -1  % Just look for 'black' spots
        for s = 1:nscale,
            Energy_ThisOrient = Energy_ThisOrient ...
                - real(EO{s,o}) - abs(imag(EO{s,o}));
        end
        
    end
    medianE2n  = median(reshape(abs(EO{1,o}).^2,1,rows*cols));
    meanE2n    = -medianE2n/log(0.5);
    estMeanE2n = [estMeanE2n meanE2n];
    
    noisePower = meanE2n/EM_n; % Estimate of noise power.
    
    % Now estimate the total energy^2 due to noise
    % Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))
    
    EstSumAn2 = zero;
    for s = 1:nscale
        EstSumAn2 = EstSumAn2+ifftFilterArray{s}.^2;
    end
    
    EstSumAiAj = zero;
    for si = 1:(nscale - 1)
        for sj = (si + 1):nscale
            EstSumAiAj = EstSumAiAj + ifftFilterArray{si} .* ifftFilterArray{sj};
        end
    end
    
    EstNoiseEnergy2 = 2*noisePower*sum(sum(EstSumAn2)) + 4*noisePower*sum(sum(EstSumAiAj));
    tau = sqrt(EstNoiseEnergy2/2);                % Rayleigh parameter
    EstNoiseEnergy = tau*sqrt(pi/2);              % Expected value of noise energy
    EstNoiseEnergySigma = sqrt( (2-pi/2)*tau^2 );
    T =  EstNoiseEnergy + k*EstNoiseEnergySigma;  % Noise threshold
    T = T/1.7;
    
    % Apply noise threshold
    Energy_ThisOrient = max(Energy_ThisOrient - T, zero);
    
    % Update accumulator matrix for sumAn and totalEnergy
    totalSumAn  = totalSumAn + sumAn_ThisOrient;
    totalEnergy = totalEnergy + Energy_ThisOrient;
    if(o == 1),
        maxEnergy = Energy_ThisOrient;
    else
        change = Energy_ThisOrient > maxEnergy;
        orientation = (o - 1).*change + orientation.*(~change);
        maxEnergy = max(maxEnergy, Energy_ThisOrient);
    end
    
end  % For each orientation
averageDirectionalEnergy = zero;
for sc = 1:nscale
    clear XA;
    clear XE;
    scale_current = sc;
    % for a fixed scale, iterate thru each orientation
    for ori=1:norient
        XA(:, :, ori) = abs( EO{scale_current, ori} );
        XE(:, :, ori) = abs( real(EO{scale_current, ori}) ) - abs( imag(EO{scale_current, ori}) );
    end
    appr_r_XA = reshape( XA, [ size(XA,1)*size(XA,2) norient ] );
    appr_r_median_XA = median( appr_r_XA, 2 );
    mA = reshape( appr_r_median_XA, [size(XA,1) size(XA,2) ] );
    
    appr_r_XE = reshape( XE, [ size(XE,1)*size(XE,2) norient ] );
    appr_r_median_XE = median( appr_r_XE, 2 );
    mE = reshape( appr_r_median_XE, [size(XE,1) size(XE,2) ] );
    
    
    %     figure,imagesc( tmp )
    %     colormap(gray), title( sprintf( 'scale : %d', sc ) );
end
A = sum( mA, 3 );
E = sum( mE, 3 );

averageDirectionalEnergy = E ./ (A + epsilon);

% Normalize totalEnergy by the totalSumAn to obtain phase symmetry
phaseSym = totalEnergy ./ (totalSumAn + epsilon);

% Convert orientation matrix values to degrees
orientation = orientation * (180 / norient);
end

%------------------------------------------------------------------
% CHECKARGS
%
% Function to process the arguments that have been supplied, assign
% default values as needed and perform basic checks.

function [im, nscale, norient, minWaveLength, mult, sigmaOnf, ...
    dThetaOnSigma,k, polarity] = checkargs(arg);

nargs = length(arg);

if nargs < 1
    error('No image supplied as an argument');
end

% Set up default values for all arguments and then overwrite them
% with with any new values that may be supplied
im              = [];
nscale          = 5;     % Number of wavelet scales.
norient         = 6;     % Number of filter orientations.
minWaveLength   = 3;     % Wavelength of smallest scale filter.
mult            = 2.1;   % Scaling factor between successive filters.
sigmaOnf        = 0.55;  % Ratio of the standard deviation of the
% Gaussian describing the log Gabor filter's
% transfer function in the frequency domain
% to the filter center frequency.
dThetaOnSigma   = 1.2;   % Ratio of angular interval between filter orientations
% and the standard deviation of the angular Gaussian
% function used to construct filters in the
% freq. plane.
k               = 2.0;   % No of standard deviations of the noise
% energy beyond the mean at which we set the
% noise threshold point.

polarity        = 0;     % Look for both black and white spots of symmetrry


% Allowed argument reading states
allnumeric   = 1;       % Numeric argument values in predefined order
keywordvalue = 2;       % Arguments in the form of string keyword
% followed by numeric value
readstate = allnumeric; % Start in the allnumeric state

if readstate == allnumeric
    for n = 1:nargs
        if isa(arg{n}, 'char')
            readstate = keywordvalue;
            break;
        else
            if     n == 1, im            = arg{n};
            elseif n == 2, nscale        = arg{n};
            elseif n == 3, norient       = arg{n};
            elseif n == 4, minWaveLength = arg{n};
            elseif n == 5, mult          = arg{n};
            elseif n == 6, sigmaOnf      = arg{n};
            elseif n == 7, dThetaOnSigma = arg{n};
            elseif n == 8, k             = arg{n};
            elseif n == 9, polarity      = arg{n};
            end
        end
    end
end

% Code to handle parameter name - value pairs
if readstate == keywordvalue
    while n < nargs
        
        if ~isa(arg{n},'char') || ~isa(arg{n+1}, 'double')
            error('There should be a parameter name - value pair');
        end
        
        if     strncmpi(arg{n},'im'      ,2), im =        arg{n+1};
        elseif strncmpi(arg{n},'nscale'  ,2), nscale =    arg{n+1};
        elseif strncmpi(arg{n},'norient' ,2), norient =   arg{n+1};
        elseif strncmpi(arg{n},'minWaveLength',2), minWavelength = arg{n+1};
        elseif strncmpi(arg{n},'mult'    ,2), mult =      arg{n+1};
        elseif strncmpi(arg{n},'sigmaOnf',2), sigmaOnf =  arg{n+1};
        elseif strncmpi(arg{n},'dthetaOnSigma',2), dThetaOnSigma =  arg{n+1};
        elseif strncmpi(arg{n},'k'       ,1), k =         arg{n+1};
        elseif strncmpi(arg{n},'polarity',2), polarity =  arg{n+1};
        else   error('Unrecognised parameter name');
        end
        
        n = n+2;
        if n == nargs
            error('Unmatched parameter name - value pair');
        end
        
    end
end

if isempty(im)
    error('No image argument supplied');
end

if ~isa(im, 'double')
    im = double(im);
end

if nscale < 1
    error('nscale must be an integer >= 1');
end

if norient < 1
    error('norient must be an integer >= 1');
end

if minWaveLength < 2
    error('It makes little sense to have a wavelength < 2');
end

if polarity ~= -1 && polarity ~= 0 && polarity ~= 1
    error('Allowed polarity values are -1, 0 and 1')
end
end

function waveletMoments = wavelet_trans(image, spaceColor)
if (strcmp(spaceColor, 'truecolor') == 1)
    imgGray = double(rgb2gray(image))/255;
    imgGray = imresize(imgGray, [256 256]);
elseif (strcmp(spaceColor, 'grayscale') == 1)
    imgGray = imresize(image, [256 256]);
end

coeff_1 = dwt2(imgGray', 'coif1');
coeff_2 = dwt2(coeff_1, 'coif1');
coeff_3 = dwt2(coeff_2, 'coif1');
coeff_4 = dwt2(coeff_3, 'coif1');

% construct the feaute vector
meanCoeff = mean(coeff_4);
stdCoeff = std(coeff_4);

waveletMoments = [meanCoeff stdCoeff];

end

function f = lowpassfilter(sze, cutoff, n)
% Impliments a low-pass butterworth filter.
if cutoff < 0 || cutoff > 0.5
    error('cutoff frequency must be between 0 and 0.5');
end

if rem(n,1) ~= 0 || n < 1
    error('n must be an integer >= 1');
end

if length(sze) == 1
    rows = sze; cols = sze;
else
    rows = sze(1); cols = sze(2);
end

if mod(cols, 2)
    xrange = (-(cols-1)/2:(cols-1)/2)/(cols-1);
else
    xrange = (-cols/2:(cols/2-1))/cols;
end

if mod(rows, 2)
    yrange = (-(rows-1)/2:(rows-1)/2)/(rows-1);
else
    yrange = (-rows/2:(rows/2-1))/rows;
end

[x, y] = meshgrid(xrange, yrange);
radius = sqrt(x.^2 + y.^2);        % A matrix with every pixel = radius relative to centre.
f = ifftshift( 1 ./ (1.0 + (radius ./ cutoff).^(2*n)) );   % The filter
end
