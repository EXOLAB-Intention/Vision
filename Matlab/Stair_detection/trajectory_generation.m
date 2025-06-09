clear; clc; close all
data = readmatrix('foot_traj_example.csv');
data = data(570:760, :);
%%
L = length(data);
idx = (1:L)';
dt = 1/250;
t = idx*dt;

v = 1.35; % treadmill speed 1.25m/s

x = data(:,1)/1000;
y = data(:,2)/1000 + v*t;
z = data(:,3)/1000;

%% Scalable trajectory

y_norm = (y - min(y))/(max(y)-min(y));
z_norm = (z - min(z))/(max(y)-min(y));

vy_norm = gradient(y_norm)/dt;
vz_norm = gradient(z_norm)/dt;

figure();
plot(y_norm, z_norm, 'k.')


%%
t_traj = linspace(0, 1, 50)';
[x_traj, y_traj] = swing_trajectory([0.4 0.0], [0.8 0.4], 50);

figure(); 
subplot(1,2,1); hold on
plot(t, y_norm, 'k');
plot(t_traj, x_traj, 'r.');
subplot(1,2,2); hold on
plot(t, z_norm, 'k');
plot(t_traj, y_traj, 'r.');






