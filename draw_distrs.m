function [] = draw_distrs(X, Y)
figure(1);
hold on;
subplot(1,2,1);
plot(X(1,:), X(2,:), 'r+');
daspect([1,1,1]);
subplot(1,2,2);
plot(Y(1,:), Y(2,:), 'r+');
daspect([1,1,1]);
