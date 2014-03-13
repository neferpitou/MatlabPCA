function [] = eigenfaces(input_image)
%EIGENFACES Performs PCA to find the image most similar to input_image
% EIGENFACES(A) performs principal component analysis (PCA) on the given
% input image based on images in a predetermined subdirectory. This is done
% by getting the eigenvectors of the standard covariance matrix and looking
% for the lowest variance between the feature vectors. It will either
% return a match of the image most similar to the input image, or it will
% display that no match can be found.

image_dims = size(input_image);
input_dir = strcat(pwd, '\faces');

filenames = dir(fullfile(input_dir, '*.png'));
num_images = numel(filenames);
product = prod(image_dims);

% mat2gray is required to make sure image is both grayscale and double
input_image = imresize(mat2gray(input_image), image_dims);
images = zeros(product, num_images);

% Read in every image from the directory and store them columnwise
% in images
for n = 1:num_images
    filename = fullfile(input_dir, filenames(n).name);
    img = imresize(mat2gray(imread(filename)), image_dims);
    images(:, n) = img(:);
end

if num_images <= 1
    disp('Error: Not enough images passed to do PCA');
    quit();
end

% Subtract each image's data from the mean of all images. We do not divide
% by the standard deviation because we want to pick up on the variances
% between each image
std_img = bsxfun(@minus, images, mean(images, 2));

figure;
for n = 1:num_images
    % Calculating the coefficients of the principal components and their 
    % respective variances is done by finding the eigenfunctions of the 
    % sample covariance matrix. The matrix V contains the coefficients for 
    % the principal components. The diagonal elements of D store the 
    % variance of the respective principal components.
   [V ~] = eig( cov(std_img(n)) );
   
   % multiply the standardized data by the principal component coefficients
   % to get the principal components of images
   evectors = bsxfun(@times, std_img, V);
   
   % Show the generated eigenfaces for each input image
   subplot(2, ceil(num_images/2), n);
   imshow(reshape(evectors(:,n), image_dims));
end
pause;

% Project the images into the subspace to generate the feature vectors
features = evectors' * std_img;

% calculate the similarity of the input to each training image
feature_vec = evectors' * bsxfun(@minus, input_image(:), mean(images, 2));

% Compare the similarity of all the images read in
[match_score, match_num] = ...
    max(arrayfun(@(n) 1 / (1 + norm(features(:,n) - feature_vec)),...
        1:num_images));

% if there are no particularly good images, just report that no matching
% faces could be found.
if match_score < 0.0005
   disp('No matching faces found.'); 
else
% display the result
figure
imshow([input_image reshape(images(:,match_num), image_dims)]);
title(sprintf('Input is most similar to %s.\n Confidence level: %f', ...
    filenames(match_num).name, match_score*100));
end