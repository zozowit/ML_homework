function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% fprintf('m is %d, Theta1 is %d x %d, Theta2 is %d x %d\n', m, size(Theta1), size(Theta2))
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% layer 2 hiden layer
% add the bias unit
X = [ones(m, 1) X];
% fprintf('X is %d by %d\n', size(X));

% get the hidden layer
Z2 = Theta1 * X';

% get activation of layer 2
A2 = sigmoid(Z2);

size(A2);

%layer 3 output layer
m2 = size(A2, 2);
A2 = [ones(1, m2);A2];

size(A2);

Z3 = Theta2 * A2;

%HX is a maxtric of 10 x 5000
HX = sigmoid(Z3);
size(HX);

% set vector y here to Yv 5000 * 10
ny = size(y, 1);
Yv = zeros(ny, num_labels);
% fprintf('Yv is %d x %d\n', size(Yv));

for i = 1:ny
  Yv(i, y(i)) = 1;
end
% fprintf('y=%d %d %d %d %d %d %d %d %d %d \n', y([1:5],:)');
% fprintf('Yv=%d %d %d %d %d %d %d %d %d %d \n', Yv([1:5],:)');

% get the J

% Version1 begin it's correct
% Vector
% HX is 10 x 5000
Jsum = 0;
for i = 1:m
    Jik = -Yv(i,:) * log(HX(:,i)) - (1 - Yv(i,:)) * log(1 - HX(:,i));
    Jsum = Jsum + Jik;    
end

J = 1 / m * Jsum;
% Version1 end

% Version2 begin it's correct
% for loop 
% set HX to 5000 x 10
%HX = HX';
%Jsum = 0;
%for i = 1:m
%  for j = 1:num_labels
%    Jij = -Yv(i, j) * log(HX(i, j)) - (1 - Yv(i, j)) * log(1 - HX(i, j));
%    Jsum = Jsum + Jij;    
%end

%J = 1 / m * Jsum;
% Version2 end

% Do the regularization
% first column should not be regularized.

[mT1, nT1] = size(Theta1);
[mT2, nT2] = size(Theta2);
Theta1Sum = sum(sum(Theta1(:,[2:nT1]).^2));
Theta2Sum = sum(sum(Theta2(:,[2:nT2]).^2));
ThetaSum = Theta1Sum + Theta2Sum
R = lambda / (2 * m) * ThetaSum

J = J + R

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Go through each train set X_t_th
% 
for t = 1:m
  % first layer
  % X is 5000 x 401, get each row as a training set
  a_1 = X(t,:);
  
  % second layer 
  % a_2 is (row of Theta1) x (column+1 of Theta1)
  z_2 = Theta1 * a_1';
  a_2 = sigmoid(z_2);
  m_2 = size(a_2, 2);
  a_2 = [ones(1, m_2);a_2];
  
  % third layer
  z_3 = Theta2 * a_2;
  a_3 = sigmoid(z_3);
  
  % get delta in third layer
  delta_3 = a_3 - (Yv(t,:))';
  % % fprintf('delta_3 is %d x %d, Theta2 is %d x %d\n', size(delta_3), size(Theta2));
  
  % get delta of layer 2
  g_t_z = sigmoidGradient(z_2);
  g_t_z = [1 ; g_t_z];
  % % fprintf('g_t_z is %d x %d\n', size(g_t_z));
  delta_2 = Theta2' * delta_3 .* g_t_z;
  
  % skip the delta_2[0] to make delta_2 from 26 x 401 to 25 x401
  % to match to the Theta1_grad
  delta_2 = delta_2(2 : end);
  
  % accumulate the gradient
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  Theta1_grad = Theta1_grad + delta_2 * a_1; % a_1 has been a row
end

Theta2_grad = 1 / m * Theta2_grad;
Theta1_grad = 1 / m * Theta1_grad;

%  fprintf('Theta1_grad is %d x %d, Theta2_grad is %d x %d\n', size(Theta1_grad), size(Theta2_grad));


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad(:,[2:end]) = Theta1_grad(:,[2:end]) + lambda / m .* Theta1(:,[2:end]);
Theta2_grad(:,[2:end]) = Theta2_grad(:,[2:end]) + lambda / m .* Theta2(:,[2:end]);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
