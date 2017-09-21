clear ; close all; clc
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
m = length(y);

theta = zeros(size(X, 2), 1);
lambda = 1;


h = sigmoid(theta' * X')';
j_theta_summation = ((theta .^ 2)' * ones(rows(theta), 1))
j_reg_term = (lambda / (2 * m)) * j_theta_summation;
logistic_fn = ((-y' * log(h)) - ((1 - y)' * log(1 - h)));

J = ((1 / m) * logistic_fn) + j_reg_term
