function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
hypothesis = zeros(m, 1);
hypothesis = sigmoid(X * theta);
theta(1, 1) = 0;
J = -1 / m * (y' * log(hypothesis) + (1 - y)' * log(1 - hypothesis)) + lambda / (2 * m) * sum(theta .^ 2);
grad = 1 / m * X' * (hypothesis - y) + lambda / m * theta;

%grad(1) = 1 / m * sum((hypothesis - y)' * X(:, 1));
%grad(2) = 1 / m * sum((hypothesis - y)' * X(:, 2)) + lambda / m * theta(2);
%grad(3) = 1 / m * sum((hypothesis - y)' * X(:, 3)) + lambda / m * theta(3);
%grad(4) = 1 / m * sum((hypothesis - y)' * X(:, 4)) + lambda / m * theta(4);
%grad(5) = 1 / m * sum((hypothesis - y)' * X(:, 5)) + lambda / m * theta(5);


% =============================================================

end
