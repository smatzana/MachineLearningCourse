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



hypothesis=sigmoid(X * theta); % 100 x 1
part1=-(log(hypothesis).* y);
part2=(-y.+1).*log((-hypothesis).+1); % 100 * 1
smallTheta=theta(2:1:end);
J=sum(part1-part2)/m + (lambda/(2*m))*(sum(smallTheta.*smallTheta));
bigTheta=[0;smallTheta];
grad=(1/m)*((sigmoid(X*theta) .- y)' * X )' + (lambda/m)*bigTheta;

% =============================================================

end
