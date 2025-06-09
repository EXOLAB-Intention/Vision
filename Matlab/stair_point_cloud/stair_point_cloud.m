clc; close all; clear;

data = load('far_from_stair.mat');

pc = data.depth;

pc = permute(pc, [3, 2, 1]);

X = pc(:, :, 3);
Y = -pc(:, :, 1);
Z = -pc(:, :, 2);

x = X(:);
y = Y(:);
z = Z(:);

figure;
scatter3(x, y, z, 1, z, 'filled');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Point Cloud from Depth Map');
axis equal;
colormap('jet');
colorbar;
