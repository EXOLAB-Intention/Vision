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
Yolo : N
Voxel Downsampling : Y
plane segmentation : RANSAC, Normal Vector
Outlier Remove : Statical method
Considering Camera Angle

"""

# ********************hyper parameter********************

W = 640  # width of opence window size
H = 480  # height of opence window size

stair_feature_std_threshold = 0.01  # To estimate stair's features as high reliability
num_peak_calculated = 4             # The number of peak when calculating mean height, depth of stair

MAX_BUFFER = 50 # Buffer size when calculating standard deviation of stair's feautre

# *******************************************************


stairs_height = []
stairs_width = []

base_dir = 'stairs_step_info'
video_dir = 'stairs_step_video'

timestamp = time.time()

os.makedirs(base_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

csv_file_path = os.path.join(base_dir, f'step_data_{timestamp}.csv')
video_file_path = os.path.join(video_dir, f'step_video_{timestamp}.mp4')

def SaveData(avg_height, avg_depth):
    
    stairs_height.append(avg_height)
    stairs_width.append(avg_depth)


def GetData():
    with open(csv_file_path, mode='w', newline='') as csvfile:
        fieldnames = ['stairs_height', 'stairs_width']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(stairs_height)):
            writer.writerow({'stairs_height': stairs_height[i],
                             'stairs_width'  : stairs_width[i]
                             })
            
    print(f"==================")
    print(f"Data Saving Done")

def rpy_to_rotmat(cam_rpy):

    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [-1,  0,  0]])

    # cam2world_R = np.array([[0, -1,  0],
    #                         [0,  0, -1],
    #                         [1,  0,  0]])
    
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

def is_converged(step_info_buffer, std_threshold=stair_feature_std_threshold):
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
    
    step_info_buffer = []
    final_step_info = None
    stop_flag = False
    staircase_faeature = None

    while True:

        avg_height = 0.0
        avg_depth = 0.0
        avg_height_ = 0.0
        avg_depth_ = 0.0

        # *******************Depth / RGB image aligning************
        depth_frame, color_frame, aligned_frames = get_aligned_frames(pipeline, align)
        if depth_frame is None or color_frame is None:
            continue


        # *******************Camera information********************
        camera_rpy = get_camera_angle(aligned_frames)
        intr, pinhole_camera_intrinsic = get_camera_intrinsics(aligned_frames)


        # *******************Plane Detection***********************
        pcd_origin, pcd, color_image, depth_image = create_point_cloud(depth_frame, color_frame, pinhole_camera_intrinsic, camera_rpy)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)


        if final_step_info is None:
            stop_flag = True
            
        # Using Ransac
        points = np.asarray(pcd.points)  # PointCloud 객체 → NumPy 배열
        if points.shape[0] != 0:
            planes, vertical_counts, horizon_counts, peak_mask = segment_planes(points, camera_rpy, stop_flag)
        else:
            planes = []

        # Getting stairs step information from the distance btw horizontal and vertical plane
        colored_planes, stair_steps, distance_to_stairs, distance = classify_planes(planes, camera_rpy, stop_flag)














        # colored_planes, stair_steps, distance_to_stairs, distance, \
        # filtered_height, filtered_depth, smoothed_heights, smoothed_depths = classify_planes(
        #     planes, camera_rpy, stop_flag,
        #     angle_threshold=0.98, d_threshold=0.02, max_area=3.5
        # )

        # print(f"계단 단 높이 평균 (클러스터링): {filtered_height:.3f} m")
        # print(f"계단 단 깊이 평균 (클러스터링): {filtered_depth:.3f} m")
        # print(f"EMA 보정 높이 시계열: {np.round(smoothed_heights, 3)}")
        # print(f"EMA 보정 깊이 시계열: {np.round(smoothed_depths, 3)}")









        stair_steps_np = np.array(stair_steps)  # shape: (N, 2)
        distance_np = np.array(distance)

        if stair_steps_np.ndim == 2 and stair_steps_np.shape[1] == 2 and staircase_faeature is None:

            n = stair_steps_np.shape[0]
            step_num = n if n < num_peak_calculated else num_peak_calculated
            avg_height = np.mean(stair_steps_np[:step_num, 0])
            avg_depth = np.mean(stair_steps_np[:step_num, 1])
            avg_height_ = np.mean(distance_np[:step_num, 0])
            avg_depth_ = np.mean(distance_np[:step_num, 1])
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
        elif staircase_faeature is not None:
            print("Distance to Stair : ", -distance_to_stairs[1][2])


        R_cam = rpy_to_rotmat(camera_rpy)
        R_global = R_cam.T  # 지면 기준 절대 방향 (카메라 회전의 역행렬)
        center = np.asarray(pcd_origin.get_center())
        axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        axis_frame.rotate(R_global, center=(0, 0, 0))  # 절대 방향 적용
        axis_frame.translate(center, relative=False)

        # *******************GUI Visualization*********************
        if not added_geometry:
            for plane, sphere in colored_planes:
                plane = plane + pcd_origin if rendering else plane
                vis.add_geometry(plane)
                vis.add_geometry(sphere)
                vis.add_geometry(axis_frame)
            added_geometry = True

        else:
            vis.clear_geometries()
            for plane, sphere in colored_planes:
                plane = plane + pcd_origin if rendering else plane
                vis.add_geometry(plane)
                vis.add_geometry(sphere)
                vis.add_geometry(axis_frame)

        view_ctl.set_zoom(ZOOM)
        vis.poll_events()
        vis.update_renderer()

        if image_with_ocv:
            pcd_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            pcd_image = cv2.cvtColor(pcd_image, cv2.COLOR_RGB2BGR)
            pcd_image = (pcd_image * 255).astype(np.uint8)
            
            color_image_resized = cv2.resize(color_image, (pcd_image.shape[1], pcd_image.shape[0]))
            stacked = np.hstack((color_image_resized, pcd_image))


            # *******************Stair Step Information visualization*********************
            # ----------------Staircas Step Height, Depth-----------------
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

            # ----------------Histogram counts-----------------
            ax2.clear()
            x = np.arange(len(horizon_counts[0]))
            ax2.plot(x, horizon_counts[0], x, horizon_counts[1])
            ax2.scatter(x[peak_mask[1]], horizon_counts[1][peak_mask[1]], color='red', s=50, label="peaks")
            ax2.set_title("Horizon Histogram"); ax2.set_ylabel("Count")
            ax2.set_ylim(0, np.max(horizon_counts[0])); ax2.set_xlim(0, 140); ax2.grid(True)
            ax2.legend("histogram counts", "filtered counts")
            canvas2.draw()
            buf = np.asarray(canvas2.buffer_rgba())
            graph_horizon = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

            ax3.clear()
            x_ = np.arange(len(vertical_counts[0]))
            ax3.plot(x_, vertical_counts[0], x_, vertical_counts[1])
            ax3.scatter(x_[peak_mask[0]], vertical_counts[1][peak_mask[0]], color='red', s=50, label="peaks")
            ax3.set_title("Vertical Histogram"); ax3.set_ylabel("Count")
            ax3.set_ylim(0, np.max(vertical_counts[0])); ax3.set_xlim(0, 140); ax3.grid(True)
            ax3.legend("histogram counts", "filtered counts")
            canvas3.draw()
            buf_ = np.asarray(canvas3.buffer_rgba())
            graph_vertical = cv2.cvtColor(buf_, cv2.COLOR_RGBA2BGR)

            combined_display2 = np.hstack((graph_horizon, graph_vertical))
            graph_resized = cv2.resize(combined_display2, (combined_display.shape[1], 320))
            combined_display = np.vstack((combined_display, graph_resized))

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
                out.write(color_image)
                SaveData(avg_height, avg_depth)
                
            if key == ord("d"):
                save_start = False
                GetData()

            if key in (27, ord("q")):
                break

if __name__ == "__main__":
    main()