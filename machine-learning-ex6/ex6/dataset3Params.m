function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

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

%C_options = [0.01, 0.03, 0.1, 1, 3, 10, 30];
%c = size(C_options,2);
%
%sigma_options = [0.01, 0.03, 0.1, 1, 3, 10, 30] .^ 0.5;
%s = size(sigma_options, 2);
%
%errors = zeros(c, s); % errors(i, j) represents the error of a model train with
%                      % C_options(i) and sigma_options(j)
%
%% test all combinations of C and sigma
%for i=1:c, 
%    for j=1:s,
%        fprintf('C = %f, sigma = %f', C_options(i), sigma_options(j))
%
%        model = svmTrain(X, y, C_options(i), ... 
%                    @(x1, x2) gaussianKernel(x1, x2, sigma_options(j)));
%
%        pred = svmPredict(model, Xval); % test model against cross-val set
%    
%        errors(i, j) = mean(double(pred != yval));
%        
%        fprintf('error = %f \n\n', errors(i, j))
%    end;
%end;
%
%% locate index with least error
%[max_err, index] = min(errors(:));
%[row, col] = ind2sub(size(errors), index); 
%
%C = C_options(row)
%sigma = sigma_options(col)

% =========================================================================

end
