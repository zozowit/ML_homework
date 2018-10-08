function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
fprintf('X is %d x %d, centroids is %d x %d\n', size(X), size(centroids));
m = size(X, 1);

% Go though all the training set
for i=1:m
  % initial the square error
  s_err_min = 100;
  xi = X(i,:);
  
  % find the smallest square error
  for j=1:K
    % compute the || X - MU |^2
    s_err = sum((xi - centroids(j,:) ).^2);
    if s_err < s_err_min
      idx(i,:) = j;
      s_err_min = s_err;
    endif
  endfor
endfor






% =============================================================

end

