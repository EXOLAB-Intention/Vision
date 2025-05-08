import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
from ultralytics import YOLO
import os

"""
Yolo : Y
Voxel Downsampling : Y
plane segmentation : RANSAC, Normal Vector
Outlier Remove : Statical method
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(WEIGHTS_PATH)
# model.to("cpu")

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
    
    return pinhole_camera_intrinsic

def create_point_cloud(depth_frame, color_frame, intrinsics):
    depth_image = np.asanyarray(depth_frame.get_data()).copy()
    color_image = np.asanyarray(color_frame.get_data()).copy()

    depth_o3d = o3d.geometry.Image(depth_image)
    color_o3d = o3d.geometry.Image(color_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsics)
    
    voxel_size = 0.03
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd

def filter_point_cloud_by_bbox(pcd, bbox, depth_image):
    x1, y1, x2, y2 = bbox
    mask = np.zeros(depth_image.shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    indices = np.where(mask.flatten())[0]
    pcd_filtered = pcd.select_by_index(indices)
    return pcd_filtered

def segment_planes(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    planes = []
    rest = pcd
    while True:
        if len(rest.points) < 50:
            break
        plane_model, inliers = rest.segment_plane(distance_threshold=distance_threshold,
                                                  ransac_n=ransac_n,
                                                  num_iterations=num_iterations)
        inlier_cloud = rest.select_by_index(inliers)
        rest = rest.select_by_index(inliers, invert=True)
        planes.append((plane_model, inlier_cloud))
    return planes

def classify_and_color_planes(planes):
    colored_planes = []

    for plane_model, plane in planes:

        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        print(normal)

        if abs(normal[1]) > 0.9:  # vertical : red
            plane.paint_uniform_color([1, 0, 0])
        elif abs(normal[1]) < 0.1: # vertical : blue
            plane.paint_uniform_color([0, 0, 1])
        else:
            plane.paint_uniform_color([0, 1, 0])

        colored_planes.append(plane)
    return colored_planes

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)

    pipeline.start(config)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Stair Perception', width=640, height=480)
    added_geometry = False

    while True:
        depth_frame, color_frame, aligned_frames = get_aligned_frames(pipeline, align)
        if depth_frame is None or color_frame is None:
            continue

        pinhole_camera_intrinsic = get_camera_intrinsics(aligned_frames)

        color_image = np.asanyarray(color_frame.get_data())
        results = model(color_image, verbose=False)

        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if int(cls) in [0, 1]:
                pcd = create_point_cloud(depth_frame, color_frame, pinhole_camera_intrinsic)
                depth_image = np.asanyarray(depth_frame.get_data())
                pcd_filtered = filter_point_cloud_by_bbox(pcd, (x1, y1, x2, y2), depth_image)
                planes = segment_planes(pcd_filtered)
                colored_planes = classify_and_color_planes(planes)

                if not added_geometry:
                    for plane in colored_planes:
                        vis.add_geometry(plane)
                    added_geometry = True
                else:
                    vis.clear_geometries()
                    for plane in colored_planes:
                        vis.add_geometry(plane)

                vis.poll_events()
                vis.update_renderer()

    # pipeline.stop()
    # vis.destroy_window()

if __name__ == "__main__":
    main()
