function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % WITHOUT VECTORIZATION

    j1 = 0;
    j2 = 0;
    for i = 1:m,
        h = [X(i), X(i, 2)]  * theta;
        diff = h - y(i);
        error1 = diff  * X(i, 1);
        error2 = diff * X(i, 2);
        j1 += error1;
        j2 += error2;
    end

    j1 = j1;
    j2 = j2;

    theta_change1 = 1/m * alpha * j1;
    theta_change2 = 1/m * alpha * j2;

    % simultaneous update
    theta(1) = theta(1) - theta_change1;
    theta(2) = theta(2) - theta_change2;

    %h = X * theta;
    %h1 = h(1)
    %diff = h - y;
    %diff1 = diff(1)
    %error = X' * diff
    %theta_change = 1/m * alpha * (error);
    %theta = theta - theta_change;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
