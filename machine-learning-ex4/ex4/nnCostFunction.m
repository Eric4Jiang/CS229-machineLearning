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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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

% PART 1 - Forward propagation
y_matrix = eye(num_labels)(y,:); % Expand matrix to vector

X = [ones(m, 1) X]; % Add 1's to X (bias unit)

a1 = X; % set all training features as input layers

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2]; % Bias unit for layer 2

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% for every example, 
for i=1:m,
    cost_vec = -y_matrix(i,:) .* log(a3(i,:)) - (1-y_matrix(i,:)) .* log(1-a3(i,:));
    J += sum(cost_vec);
%    for k=1:num_labels,
%        J += -y_matrix(i, k)*log(a3(i, k)) - (1-y_matrix(i,k))*log(1-a3(i,k));
%    end
end

J = J/m;

% ------------------------- Cost Regularization -----------------------------------

theta1_temp = Theta1(:, (2:end));
theta2_temp = Theta2(:, (2:end));

theta1_reg = sum(sumsq(theta1_temp));
theta2_reg = sum(sumsq(theta2_temp));

J += lambda / (2*m) * (theta1_reg + theta2_reg);

% =========================== Gradient ==========================================
% feed forward already completed for layers 2 and 3
% find errors of layer 2 and 3, input layer not needed
g3 = a3 - y_matrix; % 5000 x 10
g2 = g3 * theta2_temp .* sigmoidGradient(z2); % 5000 x 26

% size(Theta1)
% size(Theta2)

delta1 = g2' * a1; % total error for each theta node value for layer 1 (25 x 401)
delta2 = g3' * a2; % total error for layer 2 (10 x 26)

Theta1_grad = (1/m) * delta1;
Theta2_grad = (1/m) * delta2;

% --------------------------- Gradient Regularization ---------------------------

reg1 = (lambda/m) * Theta1;
reg1(:, 1) = 0; % remove bias from regularization
reg2 = (lambda/m) * Theta2;
reg2(:, 1) = 0;

%size(Theta1_grad)
%size(theta1_temp)
%size(reg1)

Theta1_grad += reg1;
Theta2_grad += reg2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
