function plotResults(sysParams, time, x, u, xRef, nTraj)
    % trajIdxs = (1:10)*sysParams.nTrajs/10;
    trajIdxs = (1:nTraj);
    
    for i=trajIdxs
        xTraj = squeeze(x(i, :, :));
        xRefTraj = squeeze(xRef(i, :, :));

        figure;
        subplot(8,1,1);
        plot(time, xTraj(:, 1), time, xRefTraj(:, 1), '--');
        ylabel('$x$ [m]', Interpreter="latex");
        legend('$x$', '$x_{ref}$', Interpreter="latex");
        
        subplot(8,1,2);
        plot(time, xTraj(:, 2), time, xRefTraj(:, 2), '--');
        ylabel('$z$ [m]', Interpreter="latex");
        legend('$z$', '$z_{ref}$', Interpreter="latex");
        
        subplot(8,1,3);
        theta = rad2deg(xTraj(:, 3));
        thetaRef = rad2deg(xRefTraj(:, 3));
        plot(time, theta, time, thetaRef, '--');
        ylabel('$\theta$ [deg]', Interpreter="latex");
        legend('$\theta$', '$\theta_{ref}$', Interpreter="latex");
        
        subplot(8,1,4);
        plot(time, xTraj(:, 4));
        ylabel('$\dot{x}$ [m/s]', Interpreter="latex");
        
        subplot(8,1,5);
        plot(time, xTraj(:, 5));
        ylabel('$\dot{z}$ [m/s]', Interpreter="latex");
        
        subplot(8,1,6);
        plot(time, rad2deg(xTraj(:, 6)))
        ylabel('$\dot{\theta}$ [deg/s]', Interpreter="latex");
        
        subplot(8,1,7);
        plot(time(1:end-1), u(i, :, 1));
        ylabel('$F$ [N]', Interpreter="latex");
        
        subplot(8,1,8);
        plot(time(1:end-1), u(i, :, 2));
        xlabel('Time [s]');
        ylabel('$T$ [N.m]', Interpreter="latex");
    end
end