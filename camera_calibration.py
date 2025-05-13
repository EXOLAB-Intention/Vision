import open3d as o3d
import numpy as np

"""
****************hyper parameter**************** 
voxel_size : voxel 사이즈 설정[cm]
             너무 작은 voxel size → 노이즈 유지됨.
             너무 큰 voxel size → 디테일 손실됨.
nb_neighbors : 	각 포인트에 대해 고려할 이웃의 수
                예: 20이면 각 포인트 주변의 20개 포인트 기준으로 거리 계산
std_ratio : outlier 판단 기준이 되는 표준편차 배수
            큰 값일수록 더 많은 포인트가 inlier로 간주
            작은 값일수록 더 민감하게 outlier를 제거
"""

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
    mask = (distances >= 0.35) & (distances <= 3.5)
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

    voxel_size = 0.04
    pcd_ = pcd_.voxel_down_sample(voxel_size=voxel_size)
    pcd_, ind = pcd_.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0) # std_ratio=2.0

    # pcd = rotate_coordinates(pcd, cam_rpy)

    return pcd, pcd_, color_image, depth_image

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
