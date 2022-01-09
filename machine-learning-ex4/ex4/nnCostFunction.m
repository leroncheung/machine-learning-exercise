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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));	% Theta1 is a 25 by 401 matrix
Theta2_grad = zeros(size(Theta2));	% Theta2 is a 10 * 26 matrix

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1), X];	% insert one column of '1', m * (n + 1), as 5000*401 matrix
z2 = Theta1 * X';	% z2 is a 25 * 5000 matrix
a2 = sigmoid(z2);   % a2 is a 25 * 5000 matrix
x2 = [ones(m, 1), a2'];	% x2 is a 5000 * 26 matrix
z3 = Theta2 * x2';	% z3 is a 10 * 5000 matrix
a3 = sigmoid(z3);
hypothesis = a3;	% h(\theta)

Y = zeros(m, num_labels);	% Y is a output matrix of m * num_labels, 5000 * 10
for iter = 1:m,
	Y(iter, y(iter)) = 1;
end

J = 1 / m * sum(sum(-Y .* log(hypothesis') - (1 - Y) .* log(1 - hypothesis')));	% Y and log(hypothesis') must be the same size: 5000*10

regular = lambda / (2 * m) * (sum(sum(Theta1(:, 2:end) .^ 2 )) + sum(sum(Theta2(:, 2:end) .^ 2)));

J = J + regular;

for idx = 1:m,	% m examples
	x1 = X(idx, :);	% X has already insert one term at line 64, a1 is a 1 * 401 matrix
	z2 = Theta1 * x1';	% 25 * 1
	a2 = sigmoid(z2);
	x2 = [1 a2'];		% 1 * 26
	z3 = Theta2 * x2';	% 10 * 1
	a3 = sigmoid(z3);
	y = Y(idx, :);		% 1 * 10
	delta3 = a3 - y';	% 10 * 1
	delta2 = Theta2(:, 2:end)' * delta3 .* sigmoidGradient(z2);	% 25 * 1
	Theta1_grad = Theta1_grad + delta2 * x1;	% 25 * 401
	Theta2_grad = Theta2_grad + delta3 * x2;	% 10 * 26
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
Theta1_grad = Theta1_grad + lambda / m * Theta1;
Theta2_grad = Theta2_grad + lambda / m * Theta2;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
