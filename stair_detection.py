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
from plane_detection import segment_planes_ransac, classify_planes, cluster_stairs, classify_planes_and_cluster_steps

"""
****************information****************
Yolo : N
Voxel Downsampling : Y
plane segmentation : RANSAC, Normal Vector
Outlier Remove : Statical method
Considering Camera Angle

"""

W = 640
H = 480

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

    fig, ax = plt.subplots(figsize=(16, 4))
    canvas = FigureCanvas(fig)
    x_data = deque(maxlen=30)
    height_data = deque(maxlen=30)
    depth_data = deque(maxlen=30)
    line1, = ax.plot([], [], label='Avg Height')
    line2, = ax.plot([], [], label='Avg Depth')
    ax.set_ylim(0, 0.5)
    ax.legend()
    # *********************************************************

    while True:

        avg_height = 0.0
        avg_depth = 0.0

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

        # Using Ransac
        planes = segment_planes_ransac(pcd)

        # Getting stairs step information from the distance btw horizontal and vertical plane
        colored_planes, stair_steps = classify_planes(planes, camera_rpy)

        # # Gettign stairs step infromation from the clustering
        # colored_planes, stair_steps = classify_planes_and_cluster_steps(planes, camera_rpy)


        stair_steps_np = np.array(stair_steps)  # shape: (N, 2)
        if stair_steps_np.ndim == 2 and stair_steps_np.shape[1] == 2:
            avg_height = np.mean(stair_steps_np[:, 0])
            avg_depth = np.mean(stair_steps_np[:, 1])
            # print(f"Average height: {avg_height}, Average depth: {avg_depth}")


        # *******************GUI Visualization*********************
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
            cv2.imshow("dd",color_image)
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
    