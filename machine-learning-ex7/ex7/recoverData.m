function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%  
% fprintf('recover K is %d\n', K);
% fprintf('U is %d x %d, Z is %d x %d\n', size(U), size(Z));
             
Ureduce = U(:, [1:K]);
% fprintf('Ureduce is %d x %d\n', size(Ureduce));

% Ureduce is n x K
% Z' is K x m
% X_rec is n x m, each column is an example
X_rec = Ureduce * Z';

X_rec = X_rec';


% =============================================================

end
