function [X] = normal_distr(dim, numSamples)

X = zeros(dim, numSamples);
for i = 1 : dim
    X(i, :) = normrnd(0, 1, [1, numSamples]);
end
