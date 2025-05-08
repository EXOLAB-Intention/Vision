import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
from ultralytics import YOLO
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import time
import csv
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
    save_start = False
    flag = True

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

        pcd_origin, pcd, color_image, depth_image = create_point_cloud(depth_frame, color_frame, pinhole_camera_intrinsic, camera_rpy)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Using Ransac
        planes = segment_planes_ransac(pcd)

        # Getting stairs step information from the distance btw horizontal and vertical plane
        colored_planes, stair_steps = classify_planes(planes, camera_rpy)

        # # Gettign stairs step infromation from the clustering
        # colored_planes, stair_steps = classify_planes_and_cluster_steps(merged_planes, camera_rpy)


        stair_steps_np = np.array(stair_steps)  # shape: (N, 2)
        if stair_steps_np.ndim == 2 and stair_steps_np.shape[1] == 2:
            avg_height = np.mean(stair_steps_np[:, 0])
            avg_depth = np.mean(stair_steps_np[:, 1])
            print(f"Average height: {avg_height}, Average depth: {avg_depth}")


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
            pcd_image = (pcd_image * 255).astype(np.uint8)
            
            color_image_resized = cv2.resize(color_image, (pcd_image.shape[1], pcd_image.shape[0]))
            stacked = np.hstack((color_image_resized, pcd_image))

            cv2.imshow("stairs detection", stacked)
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
    