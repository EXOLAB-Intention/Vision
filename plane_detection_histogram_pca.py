import open3d as o3d
import numpy as np
from imu_calibrate import rotate_points, rotate_vector

def segment_planes(pcd_or_pts, cam_rpy,
                             bin_width=0.02,
                             height_tol=0.03,
                             depth_tol=0.03,
                             min_slice_pts=100):
    """
    수평면(y축)과 수직면(z축) 모두 검출하는 히스토그램 기반 평면 분할.
    입력:
        pcd_or_pts: o3d.geometry.PointCloud 또는 (N,3) ndarray
        cam_rpy    : (roll, pitch, yaw) in deg
    반환:
        planes: List of (plane_model, inlier_pts)
            plane_model = (a, b, c, d) in 카메라 좌표계
            inlier_pts  = ndarray of shape (M,3) in 카메라 좌표계
    """
    # 1) 입력 포인트 배열 확보
    if isinstance(pcd_or_pts, o3d.geometry.PointCloud):
        pts = np.asarray(pcd_or_pts.points)
    else:
        pts = np.asarray(pcd_or_pts, dtype=float)

    # 2) 카메라 -> World 좌표계 회전
    world_pts = rotate_points(pts, cam_rpy)

    planes = []

    # --- A. 수평면 검출 (y축 히스토그램) ---
    heights = world_pts[:, 1]
    counts, edges = np.histogram(
        heights,
        bins=np.arange(heights.min(), heights.max() + bin_width, bin_width)
    )
    centers = (edges[:-1] + edges[1:]) / 2             # counts와 길이 일치
    mask_peaks = (counts[1:-1] > counts[:-2]) & (counts[1:-1] > counts[2:])
    peaks_y = centers[1:-1][mask_peaks]

    for h0 in peaks_y:
        idx = np.where(np.abs(heights - h0) < height_tol)[0]
        if idx.size < min_slice_pts:
            continue

        slice_world = world_pts[idx]
        cov   = np.cov(slice_world.T)
        eigv, eigvecs = np.linalg.eigh(cov)
        normal_w = eigvecs[:, np.argmin(eigv)]
        d_w      = -normal_w.dot(slice_world[0])

        # inlier 전체 재검출
        dists = np.abs(world_pts.dot(normal_w) + d_w)
        inliers = np.where(dists < height_tol)[0]

        # World -> 카메라 좌표계 역회전
        inv_rpy   = [-ang for ang in cam_rpy]
        normal_c  = rotate_vector(normal_w, inv_rpy)
        d_c       = -normal_c.dot(pts[inliers][0])

        planes.append((
            (float(normal_c[0]), float(normal_c[1]), float(normal_c[2]), float(d_c)),
            pts[inliers]
        ))

    # --- B. 수직면 검출 (z축 히스토그램) ---
    depths = world_pts[:, 2]
    counts, edges = np.histogram(
        depths,
        bins=np.arange(depths.min(), depths.max() + bin_width, bin_width)
    )
    centers = (edges[:-1] + edges[1:]) / 2
    mask_peaks = (counts[1:-1] > counts[:-2]) & (counts[1:-1] > counts[2:])
    peaks_z = centers[1:-1][mask_peaks]

    for z0 in peaks_z:
        idx = np.where(np.abs(depths - z0) < depth_tol)[0]
        if idx.size < min_slice_pts:
            continue

        slice_world = world_pts[idx]
        cov   = np.cov(slice_world.T)
        eigv, eigvecs = np.linalg.eigh(cov)
        normal_w = eigvecs[:, np.argmin(eigv)]
        d_w      = -normal_w.dot(slice_world[0])

        dists = np.abs(world_pts.dot(normal_w) + d_w)
        inliers = np.where(dists < depth_tol)[0]

        inv_rpy  = [-ang for ang in cam_rpy]
        normal_c = rotate_vector(normal_w, inv_rpy)
        d_c      = -normal_c.dot(pts[inliers][0])

        planes.append((
            (float(normal_c[0]), float(normal_c[1]), float(normal_c[2]), float(d_c)),
            pts[inliers]
        ))

    return planes
