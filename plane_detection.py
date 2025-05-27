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

def ransac_plane_fit_limited_radius(points,
                                    cam_rpy,
                                    threshold=0.03,
                                    num_ransac_points=3,
                                    num_iterations=100,
                                    min_inliers=150,
                                    sample_radius=0.05,
                                    sample_points_var_ratio = 0.08):
    best_plane = []
    best_inliers = []

    num_points = points.shape[0]
    kdtree = cKDTree(points)

    for _ in range(num_iterations):

        sample_cam_coord = [0,0,0]

        center_idx = np.random.randint(num_points)
        center = points[center_idx]

        # -----------points sampling in circle------------
        # candidate_indices = kdtree.query_ball_point(center, r=sample_radius)
        # 시간복잡도 ≈ O(log N + M)
        # N: KDTree에 저장된 포인트 수
        # M: 반경 r 안에 존재하는 이웃의 수

        # # -----------points sampling in limited number of points------------
        _, candidate_indices = kdtree.query(center, k=50)
        # 시간복잡도 ≈ O(log N + M)
        # N: KDTree에 저장된 포인트 수
        # M: 찾을 최근접 이웃 수

        if len(candidate_indices) < num_ransac_points:
            continue

        sample_indices = np.random.choice(candidate_indices, num_ransac_points, replace=False)
        sample = points[sample_indices]

        sample_cam_coord[0] = rotate_vector(sample[0], cam_rpy)
        sample_cam_coord[1] = rotate_vector(sample[1], cam_rpy)
        sample_cam_coord[2] = rotate_vector(sample[2], cam_rpy)

        # -----------compare sampling points with variance of each axis------------
        var = np.var(sample_cam_coord, axis=0)
        ratio = np.min(var) / np.sum(var)

        # # -----------compare sampling points with eigen value of covariance matrix------------
        # mean = np.mean(sample_cam_coord, axis=0)
        # sample_centered = sample_cam_coord - mean
        # cov_matrix = np.cov(sample_centered.T, bias=True)
        # eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        # eigvals = np.sort(eigvals)
        # ratio = eigvals[0] / (eigvals[0] + eigvals[1] + eigvals[2])
        # print(ratio)
        
        if abs(ratio) > sample_points_var_ratio:
            continue
        # print(ratio)

        v1, v2 = sample[1] - sample[0], sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm_val = np.linalg.norm(normal)
        if norm_val == 0:
            continue
        normal /= norm_val
        normal_rot = rotate_vector(normal, cam_rpy)

        # a, b, c = normal
        # d = -np.dot(normal, sample[0])

        # distances = np.abs((points @ normal) + d) / np.linalg.norm(normal)
        # inlier_indices = np.where(distances < threshold)[0]

        # # plane_point_center = np.mean(np.asarray(points)[inlier_indices], axis=0)
        # # _, plane_point_indices = kdtree.query(plane_point_center, k=4)

        # # neighbor_points = np.asarray(points)[plane_point_indices]
        # # distances = np.linalg.norm(neighbor_points - plane_point_center, axis=1)
        # # mean_distance = np.mean(distances)

        # # if mean_distance > 0.05:
        # #     continue

        # if len(inlier_indices) > len(best_inliers):
        #     best_inliers = inlier_indices
        #     best_plane = (a, b, c, d)   

        normal_dot_product_horizon  = abs(np.dot([0,1,0], normal_rot))
        normal_dot_product_vertical = abs(np.dot([0,0,1], normal_rot))

        if normal_dot_product_horizon > 0.9 or normal_dot_product_vertical > 0.9:
            a, b, c = normal
            d = -np.dot(normal, sample[0])

            distances = np.abs((points @ normal) + d) / np.linalg.norm(normal)
            inlier_indices = np.where(distances < threshold)[0]
            
            kdtree2 = cKDTree(points[inlier_indices])

            plane_point_center = np.mean(np.asarray(points)[inlier_indices], axis=0)
            _, plane_point_indices = kdtree2.query(plane_point_center, k=4)

            neighbor_points = np.asarray(points)[plane_point_indices]
            distances = np.linalg.norm(neighbor_points - plane_point_center, axis=1)
            mean_distance = np.mean(distances)
            print(mean_distance)

            if mean_distance > 3.0:
                continue

            if len(inlier_indices) > len(best_inliers):
                best_inliers = inlier_indices
                best_plane = (a, b, c, d)   
        else:
            continue

        

    inlier_points = points[best_inliers]
    return best_plane, inlier_points, best_inliers

def segment_planes_ransac(points,
                          cam_rpy,
                          distance_threshold=0.02,
                          num_iterations=50,
                          min_ratio=0.01,
                          min_num_points=150,
                          max_planes=12,
                          sampling_radius = 0.05,
                          sample_points_var_ratio = 0.05):

    # points = rotate_vector(points, cam_rpy)
    planes = []
    rest_points = points.copy()
    total_points = points.shape[0]

    while len(rest_points) > min_num_points and len(planes) < max_planes:

        result = ransac_plane_fit_limited_radius(rest_points,
                                                 cam_rpy,
                                                 threshold=distance_threshold,
                                                 num_iterations=num_iterations,
                                                 min_inliers=min(min_ratio * total_points, min_num_points),
                                                 sample_radius = sampling_radius,
                                                 sample_points_var_ratio = sample_points_var_ratio)

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


def classify_planes(planes, cam_ori, stop_flag, angle_threshold=15, d_threshold=0.06, max_area=3.5):
    import copy

    colored_planes = []
    horizontals = []
    verticals = []
    stair_steps = []
    merged_planes = []
    vertical_plane_distance = []
    horizontal_plane_distance = []
    distance = []

    used = [False] * len(planes)
    # print("-----------------")
    # print(f"plane N : {len(planes)}")


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
            if angle > 0.9 and abs(d_i - d_j) < d_threshold:
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
    # print(f"plane N : {len(merged_planes)}")

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


        # horizontal : red
        if abs(normal[1]) > 0.95 :
            plane.paint_uniform_color([1, 0, 0])
            sphere.paint_uniform_color([0, 0, 0])
            horizontals.append(center)
            horizontal_plane_distance.append(plane_d)

        # vertical : blue
        elif abs(normal[2]) > 0.95:
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

    return colored_planes, stair_steps, distance_to_stairs, distance