function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


hypothesis = zeros(m, 1);
hypothesis = sigmoid(X * theta);	% '-' has already in sigmoid function, no need to add it here
J = -1 / m * (y' * log(hypothesis) + (1 - y)' * log(1 - hypothesis));

grad = 1 / m * X' * (hypothesis - y);
%grad(1) = 1 / m * sum((hypothesis - y)' * X(:, 1));
%grad(2) = 1 / m * sum((hypothesis - y)' * X(:, 2));
%grad(3) = 1 / m * sum((hypothesis - y)' * X(:, 3));


% =============================================================

end
