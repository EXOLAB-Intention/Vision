import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from imu_calibration import rotate_vector
import copy

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

import open3d as o3d
import numpy as np

AngleThre = 10        # [deg]
DistanceThre = 0.05   # [m]

def segment_planes_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=100, min_ratio = 0.01):
    planes = []
    rest = pcd
    rest.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    total_points = np.asarray(rest.points).shape[0]

    while True:
        if len(rest.points) < 250 or len(planes) >= 8:
        # if len(rest.points) < 500:
            break
        plane_model, inliers = rest.segment_plane(distance_threshold=distance_threshold,
                                                  ransac_n=ransac_n,
                                                  num_iterations=num_iterations)
        
        if len(inliers) < total_points * min_ratio:    
            break

        inlier_cloud = rest.select_by_index(inliers)
        rest = rest.select_by_index(inliers, invert=True)
        planes.append((plane_model, inlier_cloud))

    return planes

# def classify_planes(planes, cam_ori):
#     colored_planes = []
#     horizontals = []
#     verticals = []
#     stair_steps = []
#     merged_planes = []

#     for plane_model, plane in planes:

#         normal = np.array(plane_model[:3])
#         normal = rotate_vector(normal, cam_ori)
#         normal /= np.linalg.norm(normal)
#         center = np.mean(np.asarray(plane.points), axis=0)

#         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
#         sphere.translate(center)

#         center = rotate_vector(center, cam_ori)
#         # print(center,"\n")

#         # horizontal : red
#         if abs(normal[1]) > 0.75 and abs(normal[2]) < 0.25:
#             plane.paint_uniform_color([1, 0, 0])
#             sphere.paint_uniform_color([0, 0, 0])
#             horizontals.append(center)

#         # vertical : blue
#         elif abs(normal[1]) < 0.25 and abs(normal[2]) > 0.75:
#             plane.paint_uniform_color([0, 0, 1])
#             sphere.paint_uniform_color([0, 0, 0])
#             verticals.append(center)

#         else:
#             plane.paint_uniform_color([0, 1, 0])

#         colored_planes.append((plane, sphere))

#     # for h_center in horizontals:
#     #     min_dist = float('inf')
#     #     closest_v = None
#     #     for v_center in verticals:
#     #         dist = np.linalg.norm(h_center - v_center)
#     #         if dist < min_dist:
#     #             min_dist = dist
#     #             closest_v = v_center
#     #     if closest_v is not None:
#     #         height = abs(h_center[1] - closest_v[1]) * 2
#     #         depth = abs(h_center[2] - closest_v[2]) * 2
#     #         stair_steps.append((height, depth))


#     horizontals_sorted = sorted(horizontals, key=lambda x: x[1])

#     for i in range(len(horizontals_sorted) - 1):
#         center1 = horizontals_sorted[i]
#         center2 = horizontals_sorted[i + 1]

#         height = abs(center2[1] - center1[1])
#         depth = abs(center2[2] - center1[2])
        
#         stair_steps.append((height, depth))
        
        
#     return colored_planes, stair_steps

def classify_planes(planes, cam_ori, angle_threshold = AngleThre, d_threshold = DistanceThre):
    colored_planes = []
    horizontals = []
    verticals = []
    stair_steps = []
    merged_planes = []

    used = [False] * len(planes)

    for i in range(len(planes)):
        if used[i]:
            continue
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

            angle = np.arccos(np.clip(np.dot(normal_i, normal_j), -1.0, 1.0)) * 180 / np.pi
            if angle < angle_threshold and abs(d_i - d_j) < d_threshold:
                merged.append(planes[j][1])
                used[j] = True

        merged_pcd = copy.deepcopy(merged[0])
        for m in merged[1:]:
            merged_pcd += m
        merged_planes.append((planes[i][0], merged_pcd))

    for plane_model, plane in merged_planes:
        normal = np.array(plane_model[:3])
        normal = rotate_vector(normal, cam_ori)
        normal /= np.linalg.norm(normal)
        center = np.mean(np.asarray(plane.points), axis=0)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(center)

        center_rotated = rotate_vector(center, cam_ori)

        # horizontal : red
        if abs(normal[1]) > 0.75 and abs(normal[2]) < 0.25:
            plane.paint_uniform_color([1, 0, 0])
            sphere.paint_uniform_color([0, 0, 0])
            horizontals.append(center_rotated)

        # vertical : blue
        elif abs(normal[1]) < 0.25 and abs(normal[2]) > 0.75:
            plane.paint_uniform_color([0, 0, 1])
            sphere.paint_uniform_color([0, 0, 0])
            verticals.append(center_rotated)

        else:
            plane.paint_uniform_color([0, 1, 0])

        colored_planes.append((plane, sphere))

    heights = []
    horizontals_sorted = sorted(horizontals, key=lambda x: x[1])
    for i in range(len(horizontals_sorted) - 1):
        center1 = horizontals_sorted[i]
        center2 = horizontals_sorted[i + 1]
        height = abs(center2[1] - center1[1])
        if height > 0.01:
            heights.append(height)

    depths = []
    verticals_sorted = sorted(verticals, key=lambda x: x[2])
    for i in range(len(verticals_sorted) - 1):
        center1 = verticals_sorted[i]
        center2 = verticals_sorted[i + 1]
        depth = abs(center2[2] - center1[2])
        if depth > 0.01:
            depths.append(depth)

    num_steps = min(len(heights), len(depths))
    for i in range(num_steps):
        stair_steps.append((heights[i], depths[i]))

    return colored_planes, stair_steps


def cluster_stairs(horizontals, verticals, eps, min_samples):
    if not horizontals or not verticals:
        return []

    z_coords = np.array([[center[2]] for center in horizontals])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(z_coords)
    labels = clustering.labels_

    cluster_map = {}
    for label, (h_center, h_plane) in zip(labels, horizontals):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(h_center)

    stair_steps = []

    for label, h_centers in cluster_map.items():
        cluster_center = np.mean(h_centers, axis=0)

        min_dist = float('inf')
        closest_v = None
        for v_center in verticals:
            dist = np.linalg.norm(cluster_center - v_center)
            if dist < min_dist:
                min_dist = dist
                closest_v = v_center

        if closest_v is not None:
            height = abs(cluster_center[1] - closest_v[1])
            depth = abs(cluster_center[2] - closest_v[2])
            stair_steps.append((height, depth))

    return stair_steps

def classify_planes_and_cluster_steps(planes, cam_ori, angle_threshold = AngleThre, d_threshold = AngleThre):
    colored_planes = []
    horizontals = []
    verticals = []

    merged_planes = []

    used = [False] * len(planes)

    for i in range(len(planes)):
        if used[i]:
            continue
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

            angle = np.arccos(np.clip(np.dot(normal_i, normal_j), -1.0, 1.0)) * 180 / np.pi
            if angle < angle_threshold and abs(d_i - d_j) < d_threshold:
                merged.append(planes[j][1])
                used[j] = True

        merged_pcd = copy.deepcopy(merged[0])
        for m in merged[1:]:
            merged_pcd += m
        merged_planes.append((planes[i][0], merged_pcd))

    for plane_model, plane in merged_planes:
        normal = np.array(plane_model[:3])
        normal = rotate_vector(normal, cam_ori)
        normal /= np.linalg.norm(normal)
        center = np.mean(np.asarray(plane.points), axis=0)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(center)
        center = rotate_vector(center, cam_ori)

        # horizontal : red
        if abs(normal[1]) > 0.75 and abs(normal[2]) < 0.25:
            plane.paint_uniform_color([1, 0, 0])
            sphere.paint_uniform_color([0, 0, 0])
            horizontals.append(center)

        # vertical : blue
        elif abs(normal[1]) < 0.25 and abs(normal[2]) > 0.75:
            plane.paint_uniform_color([0, 0, 1])
            sphere.paint_uniform_color([0, 0, 0])
            verticals.append(center)

        else:
            plane.paint_uniform_color([0, 1, 0])

        colored_planes.append((plane, sphere))

    stair_steps = cluster_stairs(horizontals, verticals, eps=0.07, min_samples=5)
    return colored_planes, stair_steps
