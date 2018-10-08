function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%
% x should be a vector, that's why x = X(i, :)', because each row of X is a example
% Ureduce is n x k, Ureduce' is k x n
% x is n x 1, X' is n x m
% Ureduce' * x is k x 1, Ureduce' * X' is K x m

fprintf('project X is %d x %d, K is %d\n', size(X), K);
fprintf('U is %d x %d\n', size(U));

Ureduce = U(:, [1:K]);
fprintf('Ureduce is %d x %d\n', size(Ureduce));

% Z here is K x m , each column is an example
Z = Ureduce' * X';
Z = Z';
fprintf('Z is %d x %d\n', size(Z));


% =============================================================

end
