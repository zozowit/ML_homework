function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

% Implement the g'(z) = d/dz * g(z) = g(z)*(1-g(z))
% test code begin
% dgz should be 0.25 when z is 0 and it could work with vector and matrices
% z = zeros(2, 3)
g = sigmoid(z) .* (1 - sigmoid(z))
% test code end













% =============================================================




end
