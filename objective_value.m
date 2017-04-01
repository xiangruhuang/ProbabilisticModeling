function [val] = objective_value(X, Y, h)
%
M = size(Y, 2);
N = size(X, 2);
d = size(X, 1);
%
Y_aug = kron(Y, ones(1, N));
X_aug = kron(ones(1, M), X);
%
mat_d = (2*Y_aug - (h+2)*X_aug)/h;
mat_dxy = X_aug - Y_aug;
norm_dxy = sum(mat_dxy.*mat_dxy);
weight_dxy = exp(-norm_dxy/h);
%
mat_d = mat_d.*kron(ones(d,1), weight_dxy);
%
mat_dif = zeros(d, M);
for i = 1 : d
    mat_dif(i, :) = sum(reshape(mat_d(i,:), [N, M]));
end
%
val = sum(sum(mat_dif.*mat_dif))/M/N;
