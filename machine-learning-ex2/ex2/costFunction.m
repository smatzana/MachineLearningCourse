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


%sigmoid
hypothesis=sigmoid(X * theta); % 100 x 1
part1=-(log(hypothesis).* y);
part2=(-y.+1).*log((-hypothesis).+1); % 100 * 1
J=sum(part1-part2)/m;

theta0 = (1/m)*( (sigmoid(X*theta) .- y)' * X(:, 1));
theta1 = (1/m)*( (sigmoid(X*theta) .- y)' * X(:, 2));
theta2 = (1/m)*( (sigmoid(X*theta) .- y)' * X(:, 3));
grad = [theta0; theta1; theta2];

hypothesis.-y; % 100 * 1

% =============================================================

end
