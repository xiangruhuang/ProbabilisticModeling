function [] = draw_distrs2(X, Y)
figure(1);
hold on;
plot(X(1,:), X(2,:), 'r+');
plot(Y(1,:), Y(2,:), 'bd');
daspect([1,1,1]);
