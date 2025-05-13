import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from imu_calibrate import rotate_vector
import copy
from scipy.spatial import cKDTree
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

DistanceThre = 0.05   # [m]
AngleThre = 15        # [deg]


def ransac_plane_fit_limited_radius(points,
                                    threshold=0.03,
                                    num_ransac_points=3,
                                    num_iterations=100,
                                    min_inliers=150,
                                    sample_radius=0.05):
    best_plane = []
    best_inliers = []

    num_points = points.shape[0]
    kdtree = cKDTree(points)

    for _ in range(num_iterations):
        var = [0,0,0]

        center_idx = np.random.randint(num_points)
        center = points[center_idx]

        # distances_from_center = np.linalg.norm(points - center, axis=1)
        # candidate_indices = np.where(distances_from_center < sample_radius)[0]

        candidate_indices = kdtree.query_ball_point(center, r=sample_radius)


        if len(candidate_indices) < num_ransac_points:
            continue

        sample_indices = np.random.choice(candidate_indices, num_ransac_points, replace=False)
        sample = points[sample_indices]
        mean = np.mean(sample, axis =0)

        var[0] = np.mean((sample[:, 0] - mean[0]) ** 2)
        var[1] = np.mean((sample[:, 1] - mean[1]) ** 2)
        var[2] = np.mean((sample[:, 2] - mean[2]) ** 2)
        var_sorted = np.sort(var)
        ratio  = var_sorted[0] / np.sum(var_sorted)

        # mean = np.mean(sample, axis=0)
        # sample_centered = sample - mean
        # cov_matrix = np.cov(sample_centered.T, bias=True)
        # eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        # eigvals = np.sort(eigvals)
        # print(eigvals)
        # ratio = eigvals[0] / (eigvals[0] + eigvals[1] + eigvals[2])
        # # ratio = eigvals[0] / np.linalg.norm(eigvals)
        
        if abs(ratio) > 0.08:
            continue
        print(ratio)

        p1, p2, p3 = sample
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)
        if np.linalg.norm(normal) == 0:
            continue

        a, b, c = normal
        d = -np.dot(normal, p1)

        distances = np.abs((points @ normal) + d) / np.linalg.norm(normal)
        inlier_indices = np.where(distances < threshold)[0]

        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_plane = (a, b, c, d)

    inlier_points = points[best_inliers]
    return best_plane, inlier_points, best_inliers

def segment_planes_ransac(points,
                          distance_threshold=0.01,
                          num_iterations=150,
                          min_ratio=0.01,
                          min_num_points=150,
                          max_planes=30,
                          sampling_radius = 0.05):

    planes = []
    rest_points = points.copy()
    total_points = points.shape[0]

    while len(rest_points) > min_num_points and len(planes) < max_planes:

        result = ransac_plane_fit_limited_radius(rest_points,
                                  threshold=distance_threshold,
                                  num_iterations=num_iterations,
                                  min_inliers=min(min_ratio * total_points, min_num_points),
                                  sample_radius = sampling_radius)

        if result is None:
            break

        plane_model, inlier_points, inlier_indices = result

        
        if len(inlier_points) < total_points * min_ratio:
            break

        planes.append((plane_model, inlier_points))

        # mask = np.ones(len(rest_points), dtype=bool)
        # mask[inlier_indices] = False
        # rest_points = rest_points[mask]
        rest_points = np.delete(rest_points, inlier_indices, axis=0)

    return planes


def classify_planes(planes, cam_ori, angle_threshold=AngleThre, d_threshold=DistanceThre, max_area=3.5):
    import copy

    colored_planes = []
    horizontals = []
    verticals = []
    stair_steps = []
    merged_planes = []

    used = [False] * len(planes)
    print("-----------------------------------")
    print(f"planes : {len(planes)}")

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

            angle = np.rad2deg(np.arccos(np.clip(np.dot(normal_i, normal_j), -1.0, 1.0)))
            if angle < angle_threshold and abs(d_i - d_j) < d_threshold:
                merged.append(planes[j][1])
                used[j] = True

        # merged_pcd = copy.deepcopy(merged[0])
        # for m in merged[1:]:
        #     merged_pcd += m
        
        # merged_pcd_legacy = o3d.geometry.PointCloud()
        # merged_pcd_legacy.points = merged_pcd.points

        # if merged_pcd.has_colors():
        #     merged_pcd_legacy.colors = merged_pcd.colors
        # if merged_pcd.has_normals():
        #     merged_pcd_legacy.normals = merged_pcd.normals

        merged_np = np.vstack(merged)

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_np)

        if hasattr(planes[i][1], 'colors') and len(planes[i][1].colors) == len(planes[i][1].points):
            merged_pcd.colors = planes[i][1].colors
        if hasattr(planes[i][1], 'normals') and len(planes[i][1].normals) == len(planes[i][1].points):
            merged_pcd.normals = planes[i][1].normals

        merged_planes.append((planes[i][0], merged_pcd))
    print(f"merged planes : {len(merged_planes)}")

    for plane_model, plane in merged_planes:
        
        # plane_o3d = o3d.geometry.PointCloud()
        # plane_o3d.points = o3d.utility.Vector3dVector(plane)

        normal = np.array(plane_model[:3])
        normal = rotate_vector(normal, cam_ori)
        normal /= np.linalg.norm(normal)
        # center = np.mean(np.asarray(plane.points), axis=0)
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
            sphere.paint_uniform_color([1, 0.5, 0])

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
    for label, h_center in zip(labels, horizontals):
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

def classify_planes_and_cluster_steps(planes, cam_ori, angle_threshold = AngleThre, d_threshold = DistanceThre):
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
            sphere.paint_uniform_color([1, 0.5, 0])

        colored_planes.append((plane, sphere))

    stair_steps = cluster_stairs(horizontals, verticals, eps=0.07, min_samples=5)
    return colored_planes, stair_steps