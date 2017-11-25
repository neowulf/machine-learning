function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% params to be used for C and sigma
params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
params_m = size(params, 2);

results = [];
index = 0;	
	for C_index = 1:params_m
		for sigma_index = 1:params_m
		
			C = params(C_index);
			sigma = params(sigma_index);
		
			model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
			predictions = svmPredict(model, Xval);
			prediction_error = mean(double(predictions ~= yval));
		
			index = index + 1
			results = [results; C sigma prediction_error];
		
			%%% index = index + 1;
			%%% C = params(C_index);
			%%% sigma = params(sigma_index);
			%%% 
			%%% % Number of training examples
			%%% m = 50; %size(X, 1);
	        %%% 
			%%% % You need to return these values correctly
			%%% error_train = zeros(m, 1);
			%%% error_val   = zeros(m, 1);
	        %%% 
			%%% for i = 2:5
		    %%% 
			%%%     ts_x = X(1:i, :);
			%%%     ts_y = y(1:i);
	        %%% 
			%%%     ts_m = size(ts_x, 1)
	        %%% 
			%%%     % Train on the training subset data
			%%%     model= svmTrain(ts_x, ts_y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	        %%% 
			%%%     % Training error - error on the training subset data
			%%%     predictions = svmPredict(model, Xval);
			%%%     error_train(i) = mean(double(predictions ~= yval));
	        %%% 
			%%%     % Cross Validation error - error on the full cross validation data
			%%%     cv_x = Xval;
			%%%     cv_y = yval;
			%%%     cv_m = size(cv_x, 1);
			%%%     predictions = svmPredict(model, cv_x);
			%%%     error_val(i) = mean(double(predictions ~= cv_y));
			%%% end
	        %%% 
			%%% % plot(1:m, error_train, 1:m, error_val);
			%%% 
			%%% index
			%%% C_index
			%%% sigma_index
			%%% subplot(params_m, params_m, index);
			%%% plot(1:m, error_train, 1:m, error_val);
			%%% 
			%%% % title('Learning curve for SVM');
			%%% % legend('Train', 'Cross Validation');
			%%% % xlabel('Number of training examples');
			%%% % ylabel('Error');
			%%% axis([0 m 0 1]);
	        %%% 
			%%% % fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
			%%% % fprintf('## \tC = %d\tsigma = %d\n', C, sigma);
			%%% % for i = 1:m
			%%% %     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
			%%% % end
		
		end
	end
% =========================================================================

results

sorted_results = sortrows(results, 3);
C = sorted_results(1,1);
sigma = sorted_results(1,2);

end
