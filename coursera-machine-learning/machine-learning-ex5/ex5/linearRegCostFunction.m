function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% sigmoid function
% z = X * theta;
% h = (1 ./ ( 1 .+ e .^ (-z)));

h = X * theta;

theta = [0; theta(2:end)];
j_theta_summation = ((theta .^ 2)' * ones(rows(theta), 1));
j_reg_term = (lambda / (2 * m)) * j_theta_summation;

% logistic_fn = ((-y' * log(h)) - ((1 - y)' * log(1 - h)));
% J = ((1 / m) * logistic_fn) + j_reg_term;

j_summation = ((h - y) .^ 2)'  * ones(rows(h), 1);
J = ((1 / (2 * m)) * j_summation) + j_reg_term;

g_reg_term = (lambda / m) .* theta;
grad = ((1 / m) * (X' * (h - y))) + g_reg_term;


% =========================================================================

grad = grad(:);

end
