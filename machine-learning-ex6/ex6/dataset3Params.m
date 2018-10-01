function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.3;
sigma = 0.1;
end
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
data_list = [0.01;0.03;0.1;0.3;1.3;10;30];
eval = zeros(1, 3);
mean_min = 1000;

for i = 1: length(data_list)
  C = data_list(i);
  for j = 1: length(data_list)
    sigma = data_list(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    mean_val = mean(double(predictions ~= yval));
    fprintf('[%d, %d]mean_val=%d, sigma=%d, C=%d\n', i, j, mean_val, sigma, C);
    if mean_val  < mean_min
      mean_min = mean_val;
      eval = [C, sigma, mean_val];
    end
    % fprintf('eval=%d %d %d\n', eval(:));
    
  end
  
end

C = eval(1);
sigma = eval(2);

fprintf('eval=%d %d %d\n', eval(:));







% =========================================================================

%end
