import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
from ultralytics import YOLO
import os
import math
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import time

"""
****************information****************
Yolo : N
Voxel Downsampling : Y
plane segmentation : RANSAC, Normal Vector
Outlier Remove : Statical method
Considering Camera Angle


****************hyper parameter**************** 
voxel_size : voxel 사이즈 설정[cm]
             너무 작은 voxel size → 노이즈 유지됨.
             너무 큰 voxel size → 디테일 손실됨.
nb_neighbors : 	각 포인트에 대해 고려할 이웃의 수
                예: 20이면 각 포인트 주변의 20개 포인트 기준으로 거리 계산
std_ratio : outlier 판단 기준이 되는 표준편차 배수
            큰 값일수록 더 많은 포인트가 inlier로 간주
            작은 값일수록 더 민감하게 outlier를 제거

distance_threshold : 평면으로 인식할 최소의 인접한 point 간 거리 [cm]
                     distance_threshold가 너무 작으면 노이즈에 민감해져 평면을 잘게 쪼갬.
ransac_n : 평면으로 인식할 이웃 point들의 수
len(rest.points) : 인식할 평면의 개수

****************Plane Detection**************** 

실시간 처리, 속도 중요      --> Region Growing
정확한 평면 방정식 필요     --> RANSAC
노이즈가 많음             --> RANSAC
부드러운 표면 분할 위주     --> Region Growing

"""

global first, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro, angleX, angleY, angleZ, accel_x, accel_y, accel_z
first = True
totalgyroangleX = 0
totalgyroangleY = 0
totalgyroangleZ = 0
last_ts_gyro = 0
angleX = 0
angleY = 180
angleZ = -90
flag = True
save_start = False
W = 640
H = 480


def calibrate(f):
    accel_frame = f.first_or_default(rs.stream.accel)
    accel = accel_frame.as_motion_frame().get_motion_data()

    ts = f.get_timestamp()
    last_ts_gyro = ts

    # accelerometer calculation
    accel_angle_x = math.degrees(math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)))
    accel_angle_y = math.degrees(math.atan2(-accel.z, math.sqrt(accel.x**2 + accel.y**2)))
    accel_angle_z = 0

    return accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro

def calculate_rotation(f, accel_angle_x, accel_angle_y, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro):
    alpha = 0.98
    
    gyro_frame = f.first_or_default(rs.stream.gyro)
    accel_frame = f.first_or_default(rs.stream.accel)

    accel = accel_frame.as_motion_frame().get_motion_data()
    gyro = gyro_frame.as_motion_frame().get_motion_data()

    ts = f.get_timestamp()

    dt_gyro = (ts - last_ts_gyro) / 1000
    last_ts_gyro = ts

    gyro_angle_x = gyro.z * dt_gyro
    gyro_angle_y = -gyro.x * dt_gyro
    gyro_angle_z = -gyro.y * dt_gyro

    dangleX = gyro_angle_x * 57.295791433
    dangleY = gyro_angle_y * 57.295791433
    dangleZ = gyro_angle_z * 57.295791433

    totalgyroangleX = accel_angle_x + dangleX
    totalgyroangleY = accel_angle_y + dangleY
    totalgyroangleZ = angleZ + dangleZ

    accel_angle_x = math.degrees(math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)))
    accel_angle_y = math.degrees(math.atan2(-accel.z, math.sqrt(accel.x**2 + accel.y**2)))
    accel_angle_z = 0
    # print(accel.x,", ", accel.y,", ", accel.z,", ")
    # print(accel_angle_x,", ", accel_angle_y,", ",accel_angle_z,", ")

    combinedangleX = totalgyroangleX * alpha + accel_angle_x * (1-alpha)
    combinedangleY = totalgyroangleY * alpha + accel_angle_y * (1-alpha)
    combinedangleZ = totalgyroangleZ

    return combinedangleX,combinedangleY,combinedangleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro


def rotate_coordinates(pcd, cam_rpy):
    points = np.asarray(pcd.points)
    cam_coordinates = [np.zeros(points.shape[0]), 3]

    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [1,  0,  0]])
 
    rx_rad, ry_rad, rz_rad = np.radians(np.dot(cam2world_R, cam_rpy))

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad), np.cos(rx_rad)]
    ])

    Ry = np.array([
        [np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])

    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad), np.cos(rz_rad), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    rotated_point = points @ R.T

    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_point)

    return rotated_pcd

def rotate_vector(vec, cam_rpy):
    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [1,  0,  0]])
 
    rx_rad, ry_rad, rz_rad = np.radians(np.dot(cam2world_R, cam_rpy))

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx_rad), -np.sin(rx_rad)],
        [0, np.sin(rx_rad), np.cos(rx_rad)]
    ])

    Ry = np.array([
        [np.cos(ry_rad), 0, np.sin(ry_rad)],
        [0, 1, 0],
        [-np.sin(ry_rad), 0, np.cos(ry_rad)]
    ])

    Rz = np.array([
        [np.cos(rz_rad), -np.sin(rz_rad), 0],
        [np.sin(rz_rad), np.cos(rz_rad), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx

    rotated_vec = R @ vec

    return rotated_vec

def get_camera_angle(frames):
    global first, accel_angle_x, accel_angle_y, angleZ, last_ts_gyro, totalgyroangleX, totalgyroangleY, totalgyroangleZ

    if first:
        accel_angle_x, accel_angle_y, angleZ, last_ts_gyro = calibrate(frames)
        first = False

    angleX, angleY, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro = calculate_rotation(
        frames, accel_angle_x, accel_angle_y, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro
    )
    # print(angleX,", ", angleY,", ",angleZ,", ")

    rotation_angle = angleX, angleY, 0

    return rotation_angle

def get_aligned_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None
    
    return depth_frame, color_frame, aligned_frames

def get_camera_intrinsics(aligned_frames):
    intr = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    
    return intr, pinhole_camera_intrinsic

def filter_point_cloud_by_bbox(pcd, bbox, depth_image):
    x1, y1, x2, y2 = bbox
    mask = np.zeros(depth_image.shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    indices = np.where(mask.flatten())[0]
    pcd_filtered = pcd.select_by_index(indices)
    return pcd_filtered
















def create_point_cloud(depth_frame, color_frame, intrinsics, cam_rpy, window_size = 3, order = 2):

    depth_image = np.asanyarray(depth_frame.get_data()).copy()
    color_image = np.asanyarray(color_frame.get_data()).copy()
    # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    depth_o3d = o3d.geometry.Image(depth_image)
    color_o3d = o3d.geometry.Image(color_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics)
    
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    

    points = np.asarray(pcd.points)
    distances = -points[:,2]
    mask = (distances >= 0.35) & (distances <= 2.0)
    indices = np.where(mask)[0]
    pcd_ = pcd.select_by_index(indices)
    # print(np.asarray(pcd_.points))


    # filtered_points = points[mask]

    # if len(filtered_points) >= window_size:

    #     x_values = savgol_filter(filtered_points[:, 0], window_size, order)
    #     y_values = savgol_filter(filtered_points[:, 1], window_size, order)
    #     z_values = savgol_filter(filtered_points[:, 2], window_size, order)
    #     filtered_points = np.vstack([x_values, y_values, z_values]).T

    # pcd.points = o3d.utility.Vector3dVector(filtered_points)

    voxel_size = 0.03
    pcd_ = pcd_.voxel_down_sample(voxel_size=voxel_size)
    pcd_, ind = pcd_.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0) # std_ratio=2.0

    # pcd = rotate_coordinates(pcd, cam_rpy)

    # Add greyscale color based on depth (z-axis)
    points = np.asarray(pcd_.points)
    depths = -points[:, 2]  # flip z to make farther = brighter
    depths_normalized = (depths - depths.min()) / (depths.max() - depths.min() + 1e-6)
    greyscale_colors = np.tile(depths_normalized[:, np.newaxis], (1, 3))  # shape: (N, 3)
    pcd_.colors = o3d.utility.Vector3dVector(greyscale_colors)

    return pcd, pcd_, color_image, depth_image


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

def classify_planes(planes, cam_ori, pcd_):
    colored_planes = []
    horizontals = []
    verticals = []
    stair_steps = []

    override_indices = []
    override_colors = []

    for plane_model, plane in planes:
        normal = np.array(plane_model[:3])
        normal = rotate_vector(normal, cam_ori)
        normal /= np.linalg.norm(normal)
        center = np.mean(np.asarray(plane.points), axis=0)

        # Classify plane
        if abs(normal[1]) > 0.75 and abs(normal[2]) < 0.25:  # horizontal
            color = [1, 0, 0]
            horizontals.append(center)
        elif abs(normal[1]) < 0.25 and abs(normal[2]) > 0.75:  # vertical
            color = [0, 0, 1]
            verticals.append(center)
        else:
            continue
            # color = [0, 1, 0]

        plane.paint_uniform_color(color)

        # Find corresponding indices in pcd_ to override color
        pcd_points = np.asarray(pcd_.points)
        plane_points = np.asarray(plane.points)
        plane_set = set(map(tuple, plane_points))
        indices = [i for i, pt in enumerate(pcd_points) if tuple(pt) in plane_set]

        override_indices.extend(indices)
        override_colors.extend([color] * len(indices))

        # Sphere for visualization
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(center)
        colored_planes.append((plane, sphere))

    return colored_planes, stair_steps, override_indices, override_colors

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

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    align = rs.align(rs.stream.color)

    pipeline.start(config)
    vis = o3d.visualization.Visualizer()

    # *****************GUI Parameter*****************
    image_with_ocv = True
    rendering = False
    ZOOM =0.25

    if image_with_ocv:
        vis.create_window(window_name='Stair Perception', width=W, height=H, visible=False)
    else:
        vis.create_window(window_name='Stair Perception', width=1600, height=720, visible=True)
    view_ctl = vis.get_view_control()
    added_geometry = False


    while True:
        depth_frame, color_frame, aligned_frames = get_aligned_frames(pipeline, align)
        if depth_frame is None or color_frame is None:
            continue

        camera_rpy = get_camera_angle(aligned_frames)
        intr, pinhole_camera_intrinsic = get_camera_intrinsics(aligned_frames)

        pcd_origin, pcd_, color_image, depth_image = create_point_cloud(depth_frame, color_frame, pinhole_camera_intrinsic, camera_rpy)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Using Ransac
        planes = segment_planes_ransac(pcd_)

        # Getting stairs step information from the distance btw horizontal and vertical plane
        colored_planes, stair_steps, override_indices, override_colors = classify_planes(planes, camera_rpy, pcd_)

        # Override colors on the greyscale point cloud
        np_colors = np.asarray(pcd_.colors)
        for idx, col in zip(override_indices, override_colors):
            np_colors[idx] = col
        pcd_.colors = o3d.utility.Vector3dVector(np_colors)

        # # Gettign stairs step infromation from the clustering
        # colored_planes, stair_steps = classify_planes_and_cluster_steps(merged_planes, camera_rpy)


        stair_steps_np = np.array(stair_steps)  # shape: (N, 2)
        if stair_steps_np.ndim == 2 and stair_steps_np.shape[1] == 2:
            avg_height = np.mean(stair_steps_np[:, 0])
            avg_depth = np.mean(stair_steps_np[:, 1])
            print(f"Average height: {avg_height}, Average depth: {avg_depth}")


        if not added_geometry:
            vis.add_geometry(pcd_)
            for plane, sphere in colored_planes:
                plane = plane + pcd_origin if rendering else plane
                vis.add_geometry(plane)
                vis.add_geometry(sphere)
            added_geometry = True
        else:
            vis.clear_geometries()
            vis.add_geometry(pcd_)
            for plane, sphere in colored_planes:
                plane = plane + pcd_origin if rendering else plane
                vis.add_geometry(plane)
                vis.add_geometry(sphere)

        view_ctl.set_zoom(ZOOM)
        vis.poll_events()
        vis.update_renderer()

        # view_ctl.set_lookat(pcd.get_center())
        # view_ctl.set_front([0, 0, -1])
        # view_ctl.set_up([0, -1, 0])

        if image_with_ocv:
            pcd_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            pcd_image = (pcd_image * 255).astype(np.uint8)
            
            color_image_resized = cv2.resize(color_image, (pcd_image.shape[1], pcd_image.shape[0]))
            stacked = np.hstack((color_image_resized, pcd_image))

            cv2.imshow("stairs detection", stacked)
            key = cv2.waitKey(1)

if __name__ == "__main__":
    main()
    