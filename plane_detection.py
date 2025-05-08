import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from imu_calibrate import rotate_vector

def segment_planes_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=100):
    planes = []
    rest = pcd

    while True:
        if len(rest.points) < 50 or len(planes) >= 8:
        # if len(rest.points) < 50:
            break
        plane_model, inliers = rest.segment_plane(distance_threshold=distance_threshold,
                                                  ransac_n=ransac_n,
                                                  num_iterations=num_iterations)
        inlier_cloud = rest.select_by_index(inliers)
        rest = rest.select_by_index(inliers, invert=True)
        planes.append((plane_model, inlier_cloud))

    return planes

def classify_planes(planes, cam_ori):
    colored_planes = []
    horizontals = []
    verticals = []
    stair_steps = []

    for plane_model, plane in planes:

        normal = np.array(plane_model[:3])
        normal = rotate_vector(normal, cam_ori)
        normal /= np.linalg.norm(normal)
        center = np.mean(np.asarray(plane.points), axis=0)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(center)

        center = rotate_vector(center, cam_ori)
        # print(center,"\n")

        # horizontal : red
        if abs(normal[1]) > 0.9 and abs(normal[2]) < 0.1:
            plane.paint_uniform_color([1, 0, 0])
            sphere.paint_uniform_color([0, 0, 0])
            horizontals.append(center)

        # vertical : blue
        elif abs(normal[1]) < 0.1 and abs(normal[2]) > 0.9:
            plane.paint_uniform_color([0, 0, 1])
            sphere.paint_uniform_color([0, 0, 0])
            verticals.append(center)

        else:
            plane.paint_uniform_color([0, 1, 0])

        colored_planes.append((plane, sphere))

        # if abs(normal[1]) > 0.75 and abs(normal[2]) < 0.25:  # horizontal : red
        #     plane.paint_uniform_color([1, 0, 0])
        #     horizontals.append(center)
        # elif abs(normal[1]) < 0.25 and abs(normal[2]) > 0.75: # vertical : blue
        #     plane.paint_uniform_color([0, 0, 1])
        #     verticals.append(center)
        # else:
        #     plane.paint_uniform_color([0, 1, 0])

        # colored_planes.append(plane)

    # for h_center in horizontals:
    #     min_dist = float('inf')
    #     closest_v = None
    #     for v_center in verticals:
    #         dist = np.linalg.norm(h_center - v_center)
    #         if dist < min_dist:
    #             min_dist = dist
    #             closest_v = v_center
    #     if closest_v is not None:
    #         height = abs(h_center[1] - closest_v[1]) * 2
    #         depth = abs(h_center[2] - closest_v[2]) * 2
    #         stair_steps.append((height, depth))


    horizontals_sorted = sorted(horizontals, key=lambda x: x[1])

    for i in range(len(horizontals_sorted) - 1):
        center1 = horizontals_sorted[i]
        center2 = horizontals_sorted[i + 1]

        height = abs(center2[1] - center1[1])
        depth = abs(center2[2] - center1[2])
        
        stair_steps.append((height, depth))
        
        
    return colored_planes, stair_steps

def cluster_stairs(horizontals, verticals, eps, min_samples):
    if not horizontals or not verticals:
        return []

    z_coords = np.array([[center[2]] for center, _ in horizontals])
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
        for v_center, _ in verticals:
            dist = np.linalg.norm(cluster_center - v_center)
            if dist < min_dist:
                min_dist = dist
                closest_v = v_center

        if closest_v is not None:
            height = abs(cluster_center[1] - closest_v[1])
            depth = abs(cluster_center[2] - closest_v[2])
            stair_steps.append((height, depth))

    return stair_steps

def classify_planes_and_cluster_steps(planes, cam_ori):
    colored_planes = []
    horizontals = []
    verticals = []

    for plane_model, plane in planes:
        normal = np.array(plane_model[:3])
        normal = rotate_vector(normal, cam_ori)
        normal /= np.linalg.norm(normal)
        center = np.mean(np.asarray(plane.points), axis=0)

        if abs(normal[1]) > 0.9 and abs(normal[2]) < 0.1:
            plane.paint_uniform_color([1, 0, 0])
            horizontals.append((center, plane))

        elif abs(normal[1]) < 0.1 and abs(normal[2]) > 0.9:
            plane.paint_uniform_color([0, 0, 1])
            verticals.append((center, plane))
            
        else:
            plane.paint_uniform_color([0, 1, 0])

        colored_planes.append(plane)

    stair_steps = cluster_stairs(horizontals, verticals, eps=0.07, min_samples=5)
    return colored_planes, stair_steps
