clear; clc; close all;

% ===========Information==============
% - Adaptive walking pattern generation
% - Human swing trajectory
% - Stair feature extraction
% ====================================

%% Mode 
mode = 'adaptive'; 

%% smoother fuction
smoother = @(d, total_d) 0.5 * (1 - cos(pi * min(d/total_d, 1)));

%% Stair data processing / Stair feature extraction

data = readtable('stairs_modeling_data_3rd.csv');

invalid_mask = data.stairs_height == -1 | data.stairs_depth == -1 | ...
               data.stairs_height_curvfit == -1 | data.stairs_depth_curvfit == -1 |...
               data.stairs_distance_curvfit == -1;
data = data(~invalid_mask, :);
extraction = data.stairs_height == -2;
extraction_idx = find(extraction);
data_filtered = data(~extraction,:);

stair_feature = data(extraction_idx+1,:);

% extraction_idx = find(data.stairs_height == -2);
% data = data(extraction_idx+1:end,:); 
% invalid_mask = data.stairs_height == -1 | data.stairs_depth == -1 | ...
%                data.stairs_height_curvfit == -1 | data.stairs_depth_curvfit == -1 | ...
%                data.stairs_distance_curvfit == -1;
% data_filtered = data(~invalid_mask, :);

stairs_height = data_filtered.stairs_height;
stairs_depth = data_filtered.stairs_depth;
distance_to_stair = data_filtered.stairs_distance;

stairs_height_curvfit = data_filtered.stairs_height_curvfit;
stairs_depth_curvfit = data_filtered.stairs_depth_curvfit;
distance_to_stair_curvfit = data_filtered.stairs_distance_curvfit;

window = 7;
n = 3;
stairs_height_filt = sgolayfilt(stairs_height,n, window);
stairs_depth_filt = sgolayfilt(stairs_depth,n,window);
distance_to_stair_filt = sgolayfilt(distance_to_stair,n,window);

stairs_height_curvfit_filt = sgolayfilt(stairs_height_curvfit,n,window);
stairs_depth_curvfit_filt = sgolayfilt(stairs_depth_curvfit,n,window);
distance_to_stair_curvfit_filt = sgolayfilt(distance_to_stair_curvfit,n,window);
% 
% figure(1);
% hold on;
% plot(distance_to_stair_curvfit)
% plot(distance_to_stair_curvfit_filt)
% legend("depth", "depth_{curvfit}");
% hold off;
% grid on;


%% Parameter

stride_max = 0.25;
stride_min = 0.2;
default_stride = stride_max;
buffer = 0.15;

upper_limb_pos = 0.96;

thighL = 0.5;
shankL = 0.5;
torso_len = 0.8;

stair.height = stair_feature.stairs_height_curvfit;
stair.depth = stair_feature.stairs_depth_curvfit;
stair.n_step = 4;
stair.start = stair_feature.stairs_distance_curvfit;

swing_frames = 40;
swing_clearance = 0.2;
swing_clearance_stair = stair.height + 0.3;

first_step = true;

%% Getting Foot Position 

x = 0;
y = 0;
step_num = 0;
total_distance_to_stair = stair.start - buffer;

positions = [];
positions(end+1,:) = [0, 0];

stride = 0;
while x < total_distance_to_stair
    dist_left = total_distance_to_stair - x;

    switch mode
        case 'smooth'
            weight = smoother(dist_left, total_distance_to_stair);
            stride = stride_min + (stride_max - stride_min) * weight;
        case 'adaptive'
            stride = adaptive_stride(x, stair.start, default_stride, buffer);
        otherwise
            error('Unknown mode: choose "smooth" or "adaptive"');
    end
    step_num = step_num + 1;
    x = x + stride;
    positions(end+1,:) = [x, 0];
end

for i = 1:stair.n_step
    x = x + stair.depth;
    y = stair.height * i;
    positions(end+1,:) = [x - stair.depth/1.8 + buffer, y];
    step_num = step_num + 1;
end

%% Simulation

close all;

stairs_x = stair.start + (0:stair.n_step)*stair.depth;
stairs_y = (0:stair.n_step)*stair.height;

left_first = true;
foot_L = positions(1,:);
foot_R = positions(1,:);
draw_interval = 0.001;

% animation parameters
frames_swing  = swing_frames;               % swing 프레임 수
% overlap       = round(frames_swing * 0.45);  % 겹침 프레임 수 (0~frames_swing-1)
overlap       = round(frames_swing * 0.0);
step_interval = frames_swing - overlap;     % 다음 스텝 시작 간격
step_count    = length(positions) - 1;
totalFrames   = (step_count - 1)*step_interval + frames_swing;

% 궤적 초기화 (NaN)
footR_traj = nan(totalFrames,2);
footL_traj = nan(totalFrames,2);

% 초기 접지 위치
lastPos_R = positions(1,:);
lastPos_L = positions(1,:);

% 각 스텝의 swing 구간만 할당
for idx = 1:step_count
    startF = (idx-1)*step_interval + 1;
    endF   = startF + frames_swing - 1;
    target = positions(idx+1,:);
    
    if mod(idx,2)==1
        % Right foot swing
        [x_s, y_s] = swing_trajectory(lastPos_R, target, frames_swing);
        footR_traj(startF:endF, :) = [x_s, y_s];
        lastPos_R = target;
            
    else
        % Left foot swing
        [x_s, y_s] = swing_trajectory(lastPos_L, target, frames_swing);
        footL_traj(startF:endF, :) = [x_s, y_s];
        lastPos_L = target;
    end
end

% NaN 구간을 '이전 값'으로 채워서 static hold 효과
footR_traj = fillmissing(footR_traj, 'previous');
footL_traj = fillmissing(footL_traj, 'previous');

footR_traj(isnan(footR_traj(:,1)),1) = positions(1,1);
footL_traj(isnan(footL_traj(:,1)),1) = positions(1,1);
footR_traj(isnan(footR_traj(:,2)),2) = positions(1,2);
footL_traj(isnan(footL_traj(:,2)),2) = positions(1,2);

figure(5)
close all
hold on;
plot(footR_traj(:,2))
plot(footL_traj(:,2))
legend("right","left")


record = false;
v = VideoWriter('Stairs_ClimbUp_HumanSwingTraject_CameraData.avi');
v.FrameRate = 10;
if record
    open(v);
end

figure;
for f = 1:totalFrames + extraction_idx
    if f > extraction_idx
        if first_step
            fprintf("--------------------------------\n")
            fprintf("Staircase Step Height : %.2f\n m", stair.height)
            fprintf("Staircase Step Depth : %.2f\n m", stair.depth)
            fprintf("Distance to first step of stair : %.2f\n m", stair.start)
            fprintf("--------------------------------\n")
            first_step = false;
        end

        foot_R = footR_traj(f-extraction_idx,:);
        foot_L = footL_traj(f-extraction_idx,:);
        
        % hip 위치 계산
        support_y = min(foot_L(2), foot_R(2));
        hip_y     = support_y + upper_limb_pos;
        hip_x     = (foot_L(1) + foot_R(1)) / 2;
        
        clf; hold on; grid on; axis equal;
        xlim([-0.5, max(positions(:,1)) + 1]);
        ylim([-0.2, max(positions(:,2)) + 1]);
        xlabel('X [m]'); ylabel('Y [m]');
        
        % 계단 그리기
        for j = 1:stair.n_step
            fill([stairs_x(j) stairs_x(j+1) stairs_x(j+1) stairs_x(j)], ...
                 [stairs_y(j) stairs_y(j) stairs_y(j+1) stairs_y(j+1)], ...
                 [1 0 0], 'FaceAlpha', 0.3);
        end
        
        % 발과 몸체 그리기
        plot(foot_L(1), foot_L(2), 'bo', 'MarkerSize',5,'LineWidth',2);
        plot(foot_R(1), foot_R(2), 'ro', 'MarkerSize',5,'LineWidth',2);
        plot([hip_x, hip_x], [hip_y, hip_y + torso_len], 'k', 'LineWidth',3);
        draw_leg(hip_x, hip_y, foot_L(1), foot_L(2), thighL, shankL, 'b');
        draw_leg(hip_x, hip_y, foot_R(1), foot_R(2), thighL, shankL,'r');
    else
        clf; hold on; grid on; axis equal;
        stairs_x = distance_to_stair_curvfit_filt(f) + (0:stair.n_step)*stairs_depth_curvfit_filt(f);
    %     stairs_x = 1+(0:n_step)*stairs_depth(i);.
        stairs_y = (0:stair.n_step)*stairs_height_curvfit_filt(f);
    
        for j = 1:stair.n_step
            fill([stairs_x(j) stairs_x(j+1) stairs_x(j+1) stairs_x(j)], ...
                 [stairs_y(j) stairs_y(j) stairs_y(j+1) stairs_y(j+1)], ...
                 [1 0 0], 'FaceAlpha', 0.3);
        end
        xlim([-0.5, max(positions(:,1)) + 1]);
        ylim([-0.2, max(positions(:,2)) + 1.5]);

        support_y = min(foot_L(2), foot_R(2));
        hip_y     = support_y + upper_limb_pos;
        hip_x     = (foot_L(1) + foot_R(1)) / 2;
        plot(foot_L(1), foot_L(2), 'bo', 'MarkerSize',5,'LineWidth',2);
        plot(foot_R(1), foot_R(2), 'ro', 'MarkerSize',5,'LineWidth',2);
        plot([hip_x, hip_x], [hip_y, hip_y + torso_len], 'k', 'LineWidth',3);
        draw_leg(hip_x, hip_y, foot_L(1), foot_L(2), thighL, shankL, 'b');
        draw_leg(hip_x, hip_y, foot_R(1), foot_R(2), thighL, shankL,'r');

    end
    drawnow;

    if record
        frame = getframe(gcf);
        writeVideo(v, frame);
    end
end

if record
    close(v);
end

%% draw leg function
function draw_leg(hip_x, hip_y, foot_x, foot_y, thighL, shankL, color)

    dx = foot_x - hip_x;
    dy = foot_y - hip_y;

    D = (dx^2 + dy^2 - thighL^2 - shankL^2) / (2*thighL*shankL);


    if abs(D) > 1
        warning('Unreachable leg configuration');
        return;
    end

    theta2 = atan2(-sqrt(1 - D^2), D);
    theta1 = atan2(dy, dx) - atan2(shankL*sin(theta2), thighL + shankL*cos(theta2));

    knee_x = hip_x + thighL * cos(theta1);
    knee_y = hip_y + thighL * sin(theta1);

    plot([hip_x, knee_x], [hip_y, knee_y], color, 'LineWidth', 3);
    plot([knee_x, foot_x], [knee_y, foot_y], color, 'LineWidth', 3);
end

%% adaptive_stride function
function stride = adaptive_stride(hip_pos, stair_start, default_stride, buffer)
    remain_distance = stair_start - hip_pos - buffer;

    stride = (remain_distance <= 0) * default_stride + ...
             ((remain_distance > 0) && (remain_distance < 2 * default_stride)) *...
             ((remain_distance) / ceil(remain_distance/default_stride)) + ...
             ((remain_distance >= 2 * default_stride)) * default_stride;
end


