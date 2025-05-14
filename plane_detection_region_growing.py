# plane_detection_region.py

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def segment_planes(pcd_or_pts,
                          cam_rpy=None,
                          radius=0.05,
                          angle_threshold=10.0,
                          min_cluster_size=30):
    """
    Region Growing 기반 평면 분할 함수.

    입력:
      pcd_or_pts      : o3d.geometry.PointCloud 또는 (N,3) numpy array (카메라 좌표계)
      cam_rpy         : (roll, pitch, yaw) tuple/list (deg) — classify_planes 호출 시 필요
                        (이 함수 내부에선 normals를 카메라 좌표계로 취급)
      radius          : 이웃 검색 반경 (m)
      angle_threshold : 노멀 벡터 간 최대 각도 차이 (deg)
      min_cluster_size: 유효 평면으로 인정할 최소 포인트 수

    반환:
      planes: List of tuples (plane_model, inlier_points)
        plane_model   = (a,b,c,d)  평면 방정식 계수 in 카메라 좌표계
        inlier_points = np.ndarray(M,3) 포인트 좌표 in 카메라 좌표계
    """

    # 1) 포인트 배열 확보
    if isinstance(pcd_or_pts, o3d.geometry.PointCloud):
        pts = np.asarray(pcd_or_pts.points)
    else:
        pts = np.asarray(pcd_or_pts, dtype=float)

    N = pts.shape[0]
    if N == 0:
        return []

    # 2) Open3D PointCloud 생성 및 노멀 추정
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30
        )
    )
    normals = np.asarray(pcd.normals)  # (N,3)

    # 3) KD-Tree로 이웃 탐색 준비
    tree = cKDTree(pts)
    visited = np.zeros(N, dtype=bool)
    planes = []

    # 4) 모든 포인트 순회하며 Region Growing 수행
    for i in range(N):
        if visited[i]:
            continue

        # 새로운 클러스터 시작
        queue = [i]
        cluster_idx = []

        while queue:
            idx = queue.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            cluster_idx.append(idx)

            # 반경 내 이웃 탐색
            nbrs = tree.query_ball_point(pts[idx], r=radius)
            for j in nbrs:
                if not visited[j]:
                    # 노멀 벡터 각도 차이 계산
                    cosang = np.dot(normals[idx], normals[j])
                    cosang = np.clip(cosang, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cosang))
                    if angle < angle_threshold:
                        queue.append(j)

        # 충분한 크기의 평면인지 확인
        if len(cluster_idx) < min_cluster_size:
            continue

        # 5) 해당 클러스터에 대해 PCA로 법선 추정 (카메라 좌표계)
        cluster_pts = pts[cluster_idx]
        cov = np.cov(cluster_pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 가장 작은 고유값 방향이 법선
        normal_cam = eigvecs[:, np.argmin(eigvals)]
        normal_cam = normal_cam / np.linalg.norm(normal_cam)
        # 평면 방정식: a x + b y + c z + d = 0
        d_cam = -float(np.dot(normal_cam, cluster_pts[0]))

        # 6) 결과 저장
        planes.append((
            (float(normal_cam[0]),
             float(normal_cam[1]),
             float(normal_cam[2]),
             d_cam),
            cluster_pts
        ))

    return planes
