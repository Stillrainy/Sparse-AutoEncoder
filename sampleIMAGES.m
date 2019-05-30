function patches = sampleIMAGES(patchsize,numpatches)
% sampleIMAGES
% Returns 10000 patches for training

load IMAGES;    % load images from disk 



% Initialize patches with zeros.  Your code will fill in this matrix--one
% column per patch, 10000 columns. 
patches = zeros(patchsize*patchsize, numpatches);

%% ---------------------------------------------------------------
ind = floor(rand(1,numpatches)*size(IMAGES,3))+1;
for i=1:numpatches
    Y = floor(rand()*(size(IMAGES,1)-patchsize-1))+1;
    X = floor(rand()*(size(IMAGES,2)-patchsize-1))+1;
    I= IMAGES(Y:Y+patchsize-1,X:X+patchsize-1,ind(i));
    patches(:,i)=I(:);
end




%% ---------------------------------------------------------------
% For the autoencoder to work well we need to normalize the data
% Specifically, since the output of the network is bounded between [0,1]
% (due to the sigmoid activation function), we have to make sure 
% the range of pixel values is also bounded between [0,1]
patches = normalizeData(patches);

end


%% ---------------------------------------------------------------
function patches = normalizeData(patches)

% Squash data to [0.1, 0.9] since we use sigmoid as the activation
% function in the output layer

% Remove DC (mean of images). 
patches = bsxfun(@minus, patches, mean(patches));

% Truncate to +/-3 standard deviations and scale to -1 to 1
pstd = 3 * std(patches(:));
patches = max(min(patches, pstd), -pstd) / pstd;

% Rescale from [-1,1] to [0.1,0.9]
patches = (patches + 1) * 0.4 + 0.1;

end
