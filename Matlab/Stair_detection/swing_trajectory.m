function [x_traj, y_traj] = swing_trajectory(p0, p1, n)

    t = linspace(0, 1, n)';

    a1  =  0.0830;
    b1  =  1.0363;
    c1  =  0.0554;
    a2  =  0.9739;
    b2  =  0.9485;
    c2  =  0.2971;
    
    x_traj_norm = a1*exp(-((t-b1)/c1).^2) + a2*exp(-((t-b2)/c2).^2);
    x_traj = p0(1) + (p1(1) - p0(1)) * x_traj_norm;

    a1  =  0.0662;
    b1  =  0.6255;
    c1  =  0.1152;
    a2  =  0.0457;
    b2  =  0.5633;
    c2  =  0.2574;

    y_traj_norm = a1*exp(-((t-b1)/c1).^2) + a2*exp(-((t-b2)/c2).^2);
    if p1(2) > 0
        y_traj = (p1(1)-p0(1)) * y_traj_norm;
        y_traj = 5*y_traj + linspace(p0(2), p1(2), n)';
    else
        y_traj = p0(2) + (p1(1)-p0(1)) * y_traj_norm;
    end

end