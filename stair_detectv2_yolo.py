import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import os
import time
import csv
import matplotlib.pyplot as plt

from ultralytics import YOLO
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import savgol_filter
from imu_calibrate import get_camera_angle
from camera_calibration import get_aligned_frames, get_camera_intrinsics, create_point_cloud
from plane_detection import classify_planes
from plane_detection_histogram_ransac import segment_planes

"""
****************information****************
Yolo : Y
Voxel Downsampling : Y
plane segmentation : Histogram, RANSAC, Normal Vector
Outlier Remove : Statical method
Considering Camera Angle

"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Yolo
WEIGHTS_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')  # Yolo
model = YOLO(WEIGHTS_PATH)  # Yolo
model.to("cpu")

W = 640
H = 480

global stairs_height, stairs_depth, step_info_buffer, final_step_info, stop_flag, staircase_faeature
stairs_height = []
stairs_depth = []
stairs_distance = []
step_info_buffer = []
final_step_info = None
MAX_BUFFER = 25
stop_flag = False
staircase_faeature = None

base_dir = 'stairs_step_info'
video_dir = 'stairs_step_video'

timestamp = time.time()

os.makedirs(base_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

csv_file_path = os.path.join(base_dir, f'step_data_{timestamp}.csv')
video_file_path = os.path.join(video_dir, f'step_video_{timestamp}.mp4')

def SaveData(avg_height, avg_depth, distance_to_stairs, save_distance_only):
    if save_distance_only:
        stairs_distance.append(distance_to_stairs)
    else:
        stairs_height.append(avg_height)
        stairs_depth.append(avg_depth)
        stairs_distance.append(distance_to_stairs)

def GetData():
    with open(csv_file_path, mode='w', newline='') as csvfile:
        fieldnames = ['stairs_height', 'stairs_depth', 'distance_to_stair']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(stairs_distance)):
            if i < len(stairs_height):
                writer.writerow({'stairs_height': stairs_height[i],
                                'stairs_depth'  : stairs_depth[i],
                                'distance_to_stair'  : stairs_distance[i]
                                })
            elif i ==len(stairs_height)+1:
                writer.writerow({'stairs_height': -2,
                                'stairs_depth'  : -2,
                                'distance_to_stair'  : -2
                                })
            else:
                writer.writerow({'stairs_height': staircase_faeature[1],
                                'stairs_depth'  : staircase_faeature[0],
                                'distance_to_stair' : stairs_distance[i]
                                })
            
    print(f"==================")
    print(f"Data Saving Done")

def rpy_to_rotmat(cam_rpy):

    # cam2world_R = np.array([[0, -1,  0],
    #                         [0,  0, 1],
    #                         [-1, 0,  0]])
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
    return Rz @ Ry @ Rx

def filter_point_cloud(pcd, bbox, intrinsic):
    x1, y1, x2, y2 = [max(0, v) for v in bbox]

    fx = intrinsic.intrinsic_matrix[0, 0]
    fy = intrinsic.intrinsic_matrix[1, 1]
    cx = intrinsic.intrinsic_matrix[0, 2]
    cy = intrinsic.intrinsic_matrix[1, 2]

    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        return o3d.geometry.PointCloud()

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    u = (fx * x / z + cx).astype(int)
    v = (fy * y / z + cy).astype(int)

    mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)

    filtered_points = points[mask]
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    return filtered_pcd

def is_converged(step_info_buffer, std_threshold=0.01):
    arr = np.array(step_info_buffer)
    std = np.std(arr, axis=0)
    
    return np.all(std < std_threshold)

def average_step_info(step_info_buffer):
    arr = np.array(step_info_buffer)
    return np.mean(arr, axis=0)

def main():

    # *****************Realsense Setting***********************
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel)
    config.enable_stream(rs.stream.gyro)
    align = rs.align(rs.stream.color)

    pipeline.start(config)
    vis = o3d.visualization.Visualizer()
    # *********************************************************

    # *******************GUI Setting***************************
    image_with_ocv = True
    rendering = False
    ZOOM = 0.25
    save_start = False
    flag = True

    if image_with_ocv:
        vis.create_window(window_name='Stair Perception', width=W, height=H, visible=False)
    else:
        vis.create_window(window_name='Stair Perception', width=1600, height=720, visible=True)
    view_ctl = vis.get_view_control()
    added_geometry = False

    # Staircas step height, depth
    fig, ax = plt.subplots(figsize=(16, 4))
    canvas = FigureCanvas(fig)
    x_data = deque(maxlen=30)
    height_data = deque(maxlen=30)
    depth_data = deque(maxlen=30)
    line1, = ax.plot([], [], label='Avg Height')
    line2, = ax.plot([], [], label='Avg Depth')
    ax.set_ylim(0, 0.5)
    ax.grid(True)
    ax.legend()

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    canvas2 = FigureCanvas(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    canvas3 = FigureCanvas(fig3)
    # *********************************************************

    global stairs_height, stairs_depth, step_info_buffer, final_step_info, stop_flag, staircase_faeature
    save_distance_only = False
    distance_to_stairs_ = -1

    while True:

        avg_height = -1.0
        avg_depth = -1.0

        depth_frame, color_frame, aligned_frames = get_aligned_frames(pipeline, align)
        if depth_frame is None or color_frame is None:
            continue

        camera_rpy = get_camera_angle(aligned_frames)
        intr, pinhole_camera_intrinsic = get_camera_intrinsics(aligned_frames)

        pcd_origin, pcd, color_image, depth_image = create_point_cloud(depth_frame, color_frame, pinhole_camera_intrinsic, camera_rpy)        
        results = model(color_image, verbose=False) # Yolo

        if results[0].boxes.data.shape[0] != 0:
            vis.clear_geometries()

            # for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = results[0].boxes.data[0].tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if int(cls) in [0, 1]:
                pcd_filtered = filter_point_cloud(pcd, [x1, y1, x2, y2], pinhole_camera_intrinsic)

                if len(pcd_filtered.points) != 0:
                    if np.asarray(pcd.points).shape[0] != 0:
                        planes, vertical_counts, horizon_counts, peak_mask = segment_planes(pcd_filtered, camera_rpy, stop_flag)
                    else:
                        planes = []

                    colored_planes, stair_steps, distance_to_stairs, distance = classify_planes(planes, camera_rpy, stop_flag)
                    if distance_to_stairs[0]:
                        distance_to_stairs_ = -distance_to_stairs[1][2]
                    else:
                        distance_to_stairs_ = -1

                    stair_steps_np = np.array(stair_steps)  # shape: (N, 2)
                    if stair_steps_np.ndim == 2 and stair_steps_np.shape[1] == 2 and staircase_faeature is None:
                        avg_height = np.mean(stair_steps_np[:, 0])
                        avg_depth = np.mean(stair_steps_np[:, 1])
                        # print(f"Average height: {avg_height}, Average depth: {avg_depth}")

                        if final_step_info is None:
                            if stair_steps_np is not None:
                                step_info_buffer.append([avg_depth, avg_height])
                            
                            if len(step_info_buffer) > MAX_BUFFER:
                                step_info_buffer.pop(0)

                            if len(step_info_buffer) == MAX_BUFFER and staircase_faeature is None:
                                if is_converged(step_info_buffer):
                                    final_step_info = average_step_info(step_info_buffer)
                                    print("Staircase Step features:", final_step_info)
                                    staircase_faeature = final_step_info

                                    final_step_info = None
                                    step_info_buffer.clear()
                            print(len(step_info_buffer))
                    elif staircase_faeature is not None and distance_to_stairs[0]:
                        print("Distance to Stair : ", distance_to_stairs_)

                    # GUI visualization
                    if not added_geometry:
                        for plane, sphere in colored_planes:
                            plane = plane + pcd_origin if rendering else plane
                            vis.add_geometry(plane)
                            vis.add_geometry(sphere)
                        added_geometry = True
                    else:
                        vis.clear_geometries()
                        for plane, sphere in colored_planes:
                            plane = plane + pcd_origin if rendering else plane
                            vis.add_geometry(plane)
                            vis.add_geometry(sphere)

        else: 
            vis.clear_geometries()
            x1 = x2 = y1 = y2 = 0

        R_cam = rpy_to_rotmat(camera_rpy)
        R_global = R_cam.T
        center = np.asarray(pcd_origin.get_center())

        axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        axis_frame.rotate(R_global, center=(0, 0, 0))
        axis_frame.translate(center, relative=False)
        
        vis.add_geometry(axis_frame)

        view_ctl.set_zoom(ZOOM)
        vis.poll_events()
        vis.update_renderer()

        if image_with_ocv:
            pcd_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            pcd_image = cv2.cvtColor(pcd_image, cv2.COLOR_RGB2BGR)
            pcd_image = (pcd_image * 255).astype(np.uint8)
            pcd_image = cv2.rectangle(pcd_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            color_image_resized = cv2.resize(color_image, (pcd_image.shape[1], pcd_image.shape[0]))
            stacked = np.hstack((color_image_resized, pcd_image))

            # 그래프 시각화
            t = time.time()
            x_data.append(t)
            height_data.append(avg_height)
            depth_data.append(avg_depth)

            line1.set_data(x_data, height_data)
            line2.set_data(x_data, depth_data)

            ax.relim()
            ax.autoscale_view()

            canvas.draw()
            buf = canvas.buffer_rgba()
            graph_image = np.asarray(buf)
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)

            graph_resized = cv2.resize(graph_image, (1280, pcd_image.shape[0]))
            combined_display = np.hstack((pcd_image, graph_resized))

            cv2.imshow("stairs detection with graph", combined_display)
            key = cv2.waitKey(1)


            if key == ord("s"):
                if flag:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_file_path, fourcc, 8.0, (W, H))

                    flag = False

                print(f"==================")
                print(f"Data Saving Start")
                save_start = True
            
            if save_start:
                if staircase_faeature is not None:
                    save_distance_only = True
                out.write(color_image)
                SaveData(avg_height, avg_depth, distance_to_stairs_, save_distance_only)
                
            if key == ord("d"):
                save_start = False
                out.release()
                GetData()

            if key in (27, ord("q")):
                break

if __name__ == "__main__":
    main()