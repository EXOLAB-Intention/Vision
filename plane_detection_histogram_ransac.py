# plane_detection_histogram_ransac.py

import open3d as o3d
import numpy as np
from imu_calibrate import rotate_points, rotate_vector
from scipy.signal import savgol_filter, find_peaks

def segment_planes(pcd_or_pts, cam_rpy, stop_flag,
                                    # bin_width=0.02,
                                    # height_tol=0.03,
                                    # depth_tol=0.03,
                                    # min_slice_pts=200,
                                    # ransac_dist=0.005,
                                    # ransac_n=3,
                                    # ransac_iter=80):
                                    bin_width=0.02,
                                    height_tol=0.03,
                                    depth_tol=0.03,
                                    min_slice_pts=100,
                                    ransac_dist=0.005,
                                    ransac_n=3,
                                    ransac_iter=80):
    """
    히스토그램 슬라이싱 후 RANSAC으로 평면 모델을 뽑아내는 함수.
    • 입력
        pcd_or_pts : o3d.geometry.PointCloud 또는 (N,3) numpy array
        cam_rpy    : (roll, pitch, yaw) in degrees
    • 파라미터
        bin_width    : 히스토그램 bin 크기 (m)
        height_tol   : 수평 슬라이스 거리 허용 (m)
        depth_tol    : 수직 슬라이스 거리 허용 (m)
        min_slice_pts: 슬라이스 내 최소 포인트 개수
        ransac_dist  : RANSAC inlier 거리 임계 (m)
        ransac_n     : RANSAC 샘플 점 개수
        ransac_iter  : RANSAC 반복 횟수
    • 출력
        planes: List of (plane_model, inlier_pts)
            plane_model = (a, b, c, d) in 카메라 좌표계
            inlier_pts  = ndarray of shape (M,3) in 카메라 좌표계
    """
    # 1) 입력을 (N,3) ndarray로 통일
    if isinstance(pcd_or_pts, o3d.geometry.PointCloud):
        pts = np.asarray(pcd_or_pts.points)
    else:
        pts = np.asarray(pcd_or_pts, dtype=float)

    # 2) 카메라→World 회전
    world_pts = rotate_points(pts, cam_rpy)  # (N,3)

    planes = []

    # --- A. 수평면 검출 (y축 히스토그램) ---
    heights = world_pts[:,1]

    bins = np.arange(heights.min(), heights.max() + bin_width, bin_width)
    # bins = np.histogram_bin_edges(heights, bins = 'fd')

    counts_horizon, edges = np.histogram(heights, bins)
    centers = (edges[:-1] + edges[1:]) / 2   # counts와 길이 일치

    window_length = 7
    polyorder = 3
    prominence = 100
    if len(counts_horizon) >= window_length:
        smoothed_counts_horizon = savgol_filter(counts_horizon,
                                    window_length,
                                    polyorder=polyorder)
        peak_mask_horizon, _    = find_peaks(smoothed_counts_horizon, prominence=prominence)
        peaks_y   = centers[peak_mask_horizon]
    else:
        peak_mask_horizon = (counts_horizon[1:-1] > counts_horizon[:-2]) & (counts_horizon[1:-1] > counts_horizon[2:])
        peaks_y   = centers[1:-1][peak_mask_horizon]
        smoothed_counts_horizon = counts_horizon
        peak_mask_horizon = [np.where(peak_mask_horizon)[0]+1]

    for h0 in peaks_y:
        idx_slice = np.where(np.abs(heights - h0) < height_tol)[0]
        if idx_slice.size < min_slice_pts:
            continue

        # 3) RANSAC 평면 추정 (World 좌표계)
        slice_world = world_pts[idx_slice]
        slice_pcd   = o3d.geometry.PointCloud()
        slice_pcd.points = o3d.utility.Vector3dVector(slice_world)

        # plane_model_world: [a,b,c,d], inliers_slice: list of indices in slice_pcd
        plane_model_w, inliers_slice = slice_pcd.segment_plane(
            distance_threshold=ransac_dist,
            ransac_n=ransac_n,
            num_iterations=ransac_iter
        )
        if len(inliers_slice) < min_slice_pts:
            continue

        # 4) 글로벌 인덱스로 변환
        global_inliers = idx_slice[np.array(inliers_slice, dtype=int)]
        normal_w = np.array(plane_model_w[:3], float)
        d_w      = float(plane_model_w[3])

        # 5) World→카메라 좌표계 역회전
        inv_rpy   = [-ang for ang in cam_rpy]
        normal_c  = rotate_vector(normal_w, inv_rpy)
        # d_c 계산: -normal_c·(원본 카메라 pts 중 하나)
        point0_c  = pts[global_inliers][0]
        d_c       = -float(np.dot(normal_c, point0_c))

        planes.append((
            (float(normal_c[0]), float(normal_c[1]), float(normal_c[2]), d_c),
            pts[global_inliers]
        ))

    # --- B. 수직면 검출 (z축 히스토그램) ---
    depths = world_pts[:,2]

    bins=np.arange(depths.min(), depths.max() + bin_width, bin_width)
    # bins = np.histogram_bin_edges(depths, bins = 'fd')

    counts_vertical, edges = np.histogram(depths, bins)
    centers = (edges[:-1] + edges[1:]) / 2

    if len(counts_vertical) >= window_length:
        smoothed_counts_verical = savgol_filter(counts_vertical,
                                    window_length,
                                    polyorder=polyorder)
        peak_mask_vertical, _    = find_peaks(smoothed_counts_verical, prominence=prominence)
        peaks_z   = centers[peak_mask_vertical]
    else:
        peak_mask_vertical = (counts_vertical[1:-1] > counts_vertical[:-2]) & (counts_vertical[1:-1] > counts_vertical[2:])
        peaks_z   = centers[1:-1][peak_mask_vertical]
        smoothed_counts_verical = counts_vertical
        peak_mask_vertical = [np.where(peak_mask_vertical)[0]+1]

    for z0 in peaks_z:
        idx_slice = np.where(np.abs(depths - z0) < depth_tol)[0]
        if idx_slice.size < min_slice_pts:
            continue

        slice_world = world_pts[idx_slice]
        slice_pcd   = o3d.geometry.PointCloud()
        slice_pcd.points = o3d.utility.Vector3dVector(slice_world)

        plane_model_w, inliers_slice = slice_pcd.segment_plane(
            distance_threshold=ransac_dist,
            ransac_n=ransac_n,
            num_iterations=ransac_iter
        )
        if len(inliers_slice) < min_slice_pts:
            continue

        global_inliers = idx_slice[np.array(inliers_slice, dtype=int)]
        normal_w = np.array(plane_model_w[:3], float)
        d_w      = float(plane_model_w[3])

        inv_rpy   = [-ang for ang in cam_rpy]
        normal_c  = rotate_vector(normal_w, inv_rpy)
        point0_c  = pts[global_inliers][0]
        d_c       = -float(np.dot(normal_c, point0_c))

        planes.append((
            (float(normal_c[0]), float(normal_c[1]), float(normal_c[2]), d_c),
            pts[global_inliers]
        ))

    return planes, (counts_vertical, smoothed_counts_verical), (counts_horizon, smoothed_counts_horizon), (peak_mask_vertical, peak_mask_horizon)