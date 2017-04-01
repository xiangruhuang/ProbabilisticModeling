function [X_opt] = gauss_newton_opt(X_init, Y, h, maxNumIters)
% This function applies regularized-GN method to optimize the following
% objective function:
% E_{x' \sim p}\|E_{x \sim q} (\nabla logp(x) k(x, x') + \nabla_{x} k(x,x')\|^2
% In our setting, x' is given by M samples denoted as the matrix Y
% the changing proability distribution q is characterized by a set of
% discrete samples encoded by matrix X_init.
% Arugments:
%  'X_init': a matrix of dimension 2\times N that encodes the initial q
%  'Y':      a matrix of diemsnion 2\times M that are sampled from 'p'
%  'h':      controls the sharpness of the kernel k(x,x') =exp(-\|x-x'\|^2/h)
%  'maxNumIters': maximum number iterations of the Gauss-Newton method
% Output:
%  'X_opt':  a discrete set of samples that encode the optimized q
M = size(Y, 2);
N = size(X_init, 2);
d = size(X_init, 1);
dim = 2*N;
% The regularization matrix applied to the diagonal 
Es = sparse(1:dim, 1:dim, ones(1, dim));
% Set the initial samples
X_opt = X_init;
for iter = 1 : 200
    % The following steps compute the first order approximation of each
    % term (There are M of such terms)
    Y_aug = kron(Y, ones(1, N));
    X_aug = kron(ones(1, M), X_opt);   
    % The following matrix is used to compute the residual
    mat_terms = (2*Y_aug - (h+2)*X_aug)/h;
    % The following matrix is used to compute the weights
    mat_dxy = Y_aug - X_aug;
    norm_dxy = sum(mat_dxy.*mat_dxy);
    weight_dxy = exp(-norm_dxy/h);
    % Weight each term using the Gaussian kernel
    mat_terms = mat_terms.*kron(ones(d,1), weight_dxy);
    % A matrix that stores the residuals
    mat_dif = zeros(d, M);
    for i = 1 : d
        mat_dif(i, :) = sum(reshape(mat_terms(i,:), [N, M]));
    end
    % Reshape the residuals into a column vector
    vec_g = reshape(mat_dif, [d*M, 1]);
    % Now we compute the jacobi matrix
    Mat_J = (1+2/h)*kron(ones(M,N), eye(d));
    for i = 1:d
        rows = i:d:(i+(M-1)*d);
        for j = 1:d
            cols = j:d:(j+(N-1)*d);
            tp = mat_terms(i,:).*mat_dxy(j,:);
            tp = reshape(tp, [N, M]);
            Mat_J(rows, cols) = Mat_J(rows, cols) - 2*tp'/h;
        end
    end
    % Weight of each block of the jacobian matrix
    weight_dxy = reshape(weight_dxy, [N, M]);
    weight_dxy = weight_dxy';
    Mat_J = Mat_J.*kron(weight_dxy, ones(d,d));
    % Compute the Hessian of the Gaussian Newton method
    H = Mat_J'*Mat_J;
    b = Mat_J'*vec_g;
    % Make matrix H sparse
    lambda = mean(diag(H))/4; 
    [rows, cols, vals] = find(H);
    ids = find(abs(vals) > 1e-8*lambda)';
    rows = rows(ids);
    cols = cols(ids);
    vals = vals(ids);
    H = sparse(rows, cols, vals, 2*N, 2*N);
    dx = (H+lambda*Es)\b;
    dx = reshape(dx, [d, N]);
    % Perform Line search to find a alpha, so that
    % setting x to x_opt + alpha*dx leads to better results
    alpha = 1;
    e_prev = objective_value(X_opt, Y, h);
    %
    success = 0;
    for i = 1:10
        X_cur = X_opt + dx*alpha;
        e_cur = objective_value(X_cur, Y, h);
        if e_cur < e_prev
            X_opt = X_cur;
            success = 1;
            break;
        end
        alpha = alpha/2;
    end
    if success == 0
        break;
    end
    fprintf('iter = %d, e_prev = %f, e_cur = %f, norm(dx) = %f, alpha = %f.\n', iter, e_prev, e_cur, sqrt(sum(sum(dx.*dx))), alpha);
end
%
