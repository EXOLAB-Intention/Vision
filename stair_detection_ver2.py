import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import copy
import time
import cv2

"""
Yolo : N
Voxel Downsampling : Y
plane segmentation : RANSAC
Outlier Remove : Statical method
"""

def get_point_cloud_from_realsense(pipeline, align, voxel_size=0.03):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None

    color = np.asarray(color_frame.get_data())
    depth = np.asarray(depth_frame.get_data())

    color_o3d = o3d.geometry.Image(color)
    depth_o3d = o3d.geometry.Image(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, convert_rgb_to_intensity=False, depth_scale=1000.0)

    intrinsics = aligned_frames.get_profile().as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height,
        intrinsics.fx, intrinsics.fy,
        intrinsics.ppx, intrinsics.ppy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)

    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

    return pcd, color

def region_growing_segmentation(pcd, min_points=5, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    planes = []
    rest = pcd
    count = 0

    while True:
        if len(rest.points) < min_points:
            break

        plane_model, inliers = rest.segment_plane(distance_threshold=distance_threshold,
                                                  ransac_n=ransac_n,
                                                  num_iterations=num_iterations)
        inlier_cloud = rest.select_by_index(inliers)
        outlier_cloud = rest.select_by_index(inliers, invert=True)

        if len(inlier_cloud.points) < min_points:
            break

        inlier_cloud.paint_uniform_color(np.random.rand(3)) 
        planes.append(inlier_cloud)
        rest = outlier_cloud

        count += 1
        if count > 10:
            break

    return planes

def estimate_step_geometry(planes):
    centers = []
    for plane in planes:
        center = np.mean(np.asarray(plane.points), axis=0)
        centers.append(center)

    # align depth 
    centers = sorted(centers, key=lambda x: x[2])

    step_heights = []
    step_depths = []
    for i in range(1, len(centers)):
        dy = abs(centers[i][1] - centers[i - 1][1])  # height
        dz = abs(centers[i][2] - centers[i - 1][2])  # depth
        step_heights.append(dy)
        step_depths.append(dz)

    return step_heights, step_depths

def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Stair Perception', width=640, height=480)
    added_geometry = False

    pipeline.start(config)

    try:
        while True:
            pcd, color_image = get_point_cloud_from_realsense(pipeline, align)
            if pcd is None:
                continue

            planes = region_growing_segmentation(pcd)

            heights, depths = estimate_step_geometry(planes)
            if heights and depths:
                print(f"[INFO] Estimated step heights: {np.round(heights, 3)} m")
                print(f"[INFO] Estimated step depths:  {np.round(depths, 3)} m")

            geometries = []
            for plane in planes:
                c = copy.deepcopy(plane)
                c.paint_uniform_color(np.random.rand(3))
                geometries.append(c)

            if not added_geometry:
                for g in geometries:
                    vis.add_geometry(g)
                added_geometry = True
            else:
                vis.clear_geometries()
                for g in geometries:
                    vis.add_geometry(g)

            vis.poll_events()
            vis.update_renderer()
            cv2.imshow("stair", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        vis.destroy_window()

if __name__ == "__main__":
    main()
