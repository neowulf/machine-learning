function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);
    % Compute train / val errors when training linear
    % regression with regularization parameter lambda
    % You should store the result in error_train(i)
    % and error_val(i)

    %%%
    %  REUSE learningcurve.m
    %%%

    % Compute train/cross validation errors using training examples
    % X(1:i, :) and y(1:i), storing the result in
    % error_train(i) and error_val(i)

    ts_x = X; %(1:i, :);
    ts_y = y; %(1:i);

    m = size(ts_x, 1);

    % Train on the training subset data
    theta = trainLinearReg([ones(m, 1) ts_x], ts_y, lambda);

    % Training error - error on the training subset data
    error_train(i) = linearRegCostFunction([ones(m, 1) ts_x], ts_y, theta, 0);

    % Cross Validation error - error on the full cross validation data
    cv_x = Xval;
    cv_y = yval;
    m = size(cv_x, 1);
    error_val(i) = linearRegCostFunction([ones(m, 1) cv_x], cv_y, theta, 0);

end


% =========================================================================

end
