% Préparation de l'animation
figure;
hold on;
axis equal;
xlim([-2.5 2.5]);
ylim([-2 2]);

quadrotor_plot = plot(0, 0, 'bo-', 'LineWidth', 2);  % Représentation du quadrotor
trajectory_plot = plot(0, 0, 'r--', 'LineWidth', 2);  % Trajectoire de référence
force_left_plot = quiver(0, 0, 0, 0, 'g', 'LineWidth', 2); % Flèche de poussée gauche
force_right_plot = quiver(0, 0, 0, 0, 'g', 'LineWidth', 2); % Flèche de poussée droite