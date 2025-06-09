import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from imu_calibrate import rotate_vector
import copy
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.linear_model import RANSACRegressor

"""

****************Plane Detection**************** 
실시간 처리, 속도 중요      --> Region Growing
정확한 평면 방정식 필요     --> RANSAC
노이즈가 많음             --> RANSAC
부드러운 표면 분할 위주     --> Region Growing


****************hyper parameter**************** 
distance_threshold : 평면으로 인식할 최소의 인접한 point 간 거리 [cm]
                     distance_threshold가 너무 작으면 노이즈에 민감해져 평면을 잘게 쪼갬.
ransac_n : 평면으로 인식할 이웃 point들의 수

"""

# ********************hyper parameter********************

angle_threshold=15              # Plane Merging Procedure : threshold when you compare two nomal vector of planes with degree angle
angle_threshold_normal = 0.9   # Plane Merging Procedure : threshold when you compare two nomal vector of planes with cosine value
d_threshold=0.05                # Plane Merging Procedure : if distance btw two planes is smaller than d_threshold, two planes will be merged

plane_threshold_normal = 0.95   # Dividing horizon, verical plane : if the y or z axis of normal vector is bigger than threshold, plane label will be divided into horizontal or vertical

angle_threshold_deg = 15

# *******************************************************

def fit_stair_model(sorted_centers, axis):
    coords = np.array(sorted_centers)
    step_indices = np.arange(len(coords)).reshape(-1, 1)
    basis_normal_vec = coords[:, axis].reshape(-1, 1) if axis == 1 else -coords[:, axis].reshape(-1, 1)

    model = RANSACRegressor(min_samples=2, residual_threshold=0.05)
    model.fit(step_indices, basis_normal_vec)

    step_size = model.estimator_.coef_[0][0]
    offset = model.estimator_.intercept_[0]

    predicted = model.predict(step_indices).flatten()
    residuals = basis_normal_vec.flatten() - predicted
    rmse = np.sqrt(np.mean(residuals**2))

    return step_size, offset, predicted, rmse


# def fit_stair_model(sorted_centers, axis):
#     coords = np.array(sorted_centers)
#     step_indices = np.arange(len(coords))
#     basis_normal_vec = coords[:, axis] if axis == 1 else -coords[:, axis] 
#     if axis ==1:
#         print(f"y_vector  : {basis_normal_vec}")
#     if axis ==2:
#         print(f"z_vector  : {basis_normal_vec}")

#     def stair_func(n, step_size, offset):
#         return step_size * n + offset
#     popt, _ = curve_fit(stair_func, step_indices, basis_normal_vec)
#     step_size, offset = popt

#     predicted = stair_func(step_indices, *popt)
#     print(f"predicted  : {predicted}")
#     residuals = basis_normal_vec - predicted
#     rmse = np.sqrt(np.mean(residuals**2))

#     return step_size, offset, predicted, rmse

def classify_planes(planes, cam_ori, stop_flag, angle_threshold=angle_threshold, d_threshold=d_threshold, max_area=3.5):

    cos_threshold = abs(np.cos(np.radians(angle_threshold_deg)))
    colored_planes = []
    horizontals = []
    verticals = []
    stair_steps = []
    merged_planes = []
    vertical_plane_distance = []
    horizontal_plane_distance = []
    distance = []

    used = [False] * len(planes)

    for i in range(len(planes)):
        if used[i]:
            continue
        used[i] = True
        merged = [planes[i][1]]
        normal_i = np.array(planes[i][0][:3])
        normal_i /= np.linalg.norm(normal_i)
        d_i = planes[i][0][3]

        for j in range(i + 1, len(planes)):
            if used[j]:
                continue
            normal_j = np.array(planes[j][0][:3])
            normal_j /= np.linalg.norm(normal_j)
            d_j = planes[j][0][3]

            angle = abs(np.dot(normal_i, normal_j)) 
            if angle > angle_threshold_normal and abs(d_i - d_j) < d_threshold:
            # angle = np.rad2deg(np.arccos(np.clip(abs(np.dot(normal_i, normal_j)), -1.0, 1.0)))
            # if angle < angle_threshold and abs(d_i - d_j) < d_threshold:
                merged.append(planes[j][1])
                used[j] = True

        merged_np = np.vstack(merged)

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_np)

        if hasattr(planes[i][1], 'colors') and len(planes[i][1].colors) == len(planes[i][1].points):
            merged_pcd.colors = planes[i][1].colors
        if hasattr(planes[i][1], 'normals') and len(planes[i][1].normals) == len(planes[i][1].points):
            merged_pcd.normals = planes[i][1].normals

        merged_planes.append((planes[i][0], merged_pcd))

    for plane_model, plane in merged_planes:
        
        plane_d = np.array(plane_model[3])
        normal = np.array(plane_model[:3])
        normal = rotate_vector(normal, cam_ori)
        normal /= np.linalg.norm(normal)
        # center = np.mean(np.asarray(plane.points), axis=0)
        center = plane.get_center()

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(center)

        center = rotate_vector(center, cam_ori)

        cos_normal_z = abs(np.dot(normal, [0,0,1])) 
        cos_normal_y = abs(np.dot(normal, [0,1,0]))
        cos_normal_x = abs(np.dot(normal, [1,0,0]))

        # horizontal : red
        # if abs(normal[1]) > plane_threshold_normal:
        # if cos_normal_y > cos_threshold and cos_normal_z < (1 - cos_threshold):
        if cos_normal_y > cos_threshold:
            plane.paint_uniform_color([1, 0, 0])
            sphere.paint_uniform_color([0, 0, 0])
            horizontals.append(center)
            horizontal_plane_distance.append(plane_d)

        # vertical : blue
        # elif abs(normal[2]) > plane_threshold_normal:
        # elif cos_normal_z > cos_threshold and cos_normal_y < (1 - cos_threshold)
        elif cos_normal_z > cos_threshold or cos_normal_x > cos_threshold:
            plane.paint_uniform_color([0, 0, 1])
            sphere.paint_uniform_color([0, 0, 0])
            verticals.append(center)
            vertical_plane_distance.append(plane_d)

        else:
            plane.paint_uniform_color([0, 1, 0])
            sphere.paint_uniform_color([1, 0.5, 0])

        colored_planes.append((plane, sphere))

    heights = []
    horizon_d = []
    horizontals_sorted = sorted(horizontals, key=lambda x: x[1])
    horizontal_plane_d_sorted = sorted(horizontal_plane_distance)
    for i in range(len(horizontals_sorted) - 1):
        center1 = horizontals_sorted[i]
        center2 = horizontals_sorted[i + 1]
        horizon_d.append(abs(horizontal_plane_d_sorted[i] - horizontal_plane_d_sorted[i+1]))
        height = abs(center2[1] - center1[1])
        if height > 0.01:
            heights.append(height)

    depths = []
    vertical_d = []
    verticals_sorted = sorted(verticals, key=lambda x: x[2])
    vertical_plane_d_sorted = sorted(vertical_plane_distance)
    for i in range(len(verticals_sorted) - 1):
        center1 = verticals_sorted[i]
        center2 = verticals_sorted[i + 1]
        vertical_d.append(abs(vertical_plane_d_sorted[i] - vertical_plane_d_sorted[i+1]))
        depth = abs(center2[2] - center1[2])
        if depth > 0.01:
            depths.append(depth)

    if len(horizontals_sorted) > 1:
        horizontals_sorted = sorted(horizontals, key=lambda x: x[1])
        height_step, height_offset, pred_heights, height_rmse = fit_stair_model(horizontals_sorted, axis=1)
        # print(f"Step height: {height_step:.3f}, RMSE: {height_rmse:.3f}")
    else:
        height_step = -1

    if len(verticals_sorted) > 1:
        verticals_sorted = sorted(verticals, key=lambda x: -x[2])
        depth_step, depth_offset, pred_depths, depth_rmse = fit_stair_model(verticals_sorted, axis=2)
        # print(f"Step depth: {depth_step:.3f}, RMSE: {depth_rmse:.3f}")
    else:
        depth_step = -1
        depth_offset = -1

    if len(verticals_sorted) >= 1:
        distance_to_stairs = (True, verticals_sorted[-1])
    else:
        distance_to_stairs = (False, 0)

    num_steps = min(len(heights), len(depths))
    for i in range(num_steps):
        stair_steps.append((heights[i], depths[i]))

    num_steps = min(len(heights), len(depths))
    for i in range(num_steps):
        distance.append((horizon_d[i], vertical_d[i]))

    return colored_planes, stair_steps, distance_to_stairs, distance, (height_step, depth_step, depth_offset)