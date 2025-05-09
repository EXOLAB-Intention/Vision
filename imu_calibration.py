import math
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from ahrs.filters import Madgwick

# pip install ahrs

madgwick_filter = Madgwick(sampleperiod=0.01)
cam_quat  = np.array([1.0, 0.0, 0.0, 0.0])
last_ts_gyro = None

# ------- quaternion operations ------- #
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_to_rotation_matrix(q):
    """q = [w, x, y, z] → 3×3 회전 행렬"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


# ------- IMU attitude estimation (quaternion) ------- #

def calibrate_orientation(accel_vec):
    """
    accel_vec: 카메라 좌표계에서 측정된 가속도 (np.array([ax,ay,az]))
    이 벡터가 중력 방향을 가리키므로, 이를 월드 중력 축([0,0,-1])에 맞추는 회전을 quaternion 으로 계산.
    """
    # 1. 단위 벡터화
    g = accel_vec / np.linalg.norm(accel_vec)
    # 2. 월드 중력 벡터 (Open3D world 기준)
    gw = np.array([0.0, 0.0, -1.0])
    # 3. 회전축(axis) = g × gw
    axis = np.cross(g, gw)
    if np.linalg.norm(axis) < 1e-6:
        # 이미 정렬된 상태
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis /= np.linalg.norm(axis)
    # 4. 회전각(angle) = arccos(g·gw)
    angle = math.acos(np.clip(np.dot(g, gw), -1.0, 1.0))
    # 5. quaternion 생성
    w = math.cos(angle/2)
    x, y, z = axis * math.sin(angle/2)
    return np.array([w, x, y, z])


# ------- vector and point cloud rotation ------ #
def rotate_vector(vec, cam_quat):
    # vec: np.array([x, y, z])
    # cam_quat: [w, x, y, z] (from Madgwick)
    
    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [1,  0,  0]])
    vec_world = cam2world_R @ vec
    vec_quat = np.array([0.0, *vec_world])

    # Rotation : q * v * q_conj
    cam_quat_conj = quaternion_conjugate(cam_quat)
    rotated_vec_quat = quaternion_multiply(quaternion_multiply(cam_quat, vec_quat), cam_quat_conj)
    rotate_vec = rotated_vec_quat[1:] 

    return rotate_vec

# ---- PointCloud 회전 (성능 최적화, 법선/색상 보존) ----
def rotate_coordinates(pcd, cam_quat):
    # 원본 배열
    pts    = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None
    colors  = np.asarray(pcd.colors)  if pcd.has_colors()  else None

    # 좌표계 변환: RealSense → World
    cam2world_R = np.array([[ 0, -1,  0],
                            [ 0,  0, -1],
                            [ 1,  0,  0]])
    pts_world = (cam2world_R @ pts.T).T
    if normals is not None:
        normals_world = (cam2world_R @ normals.T).T

    # quaternion → 회전 행렬
    R_cam    = quaternion_to_rotation_matrix(cam_quat)
    # R_total  = R_cam @ cam2world_R  # world→camera or camera→world 순 확인하여 적절히 곱셈

    # 한 번에 벡터화 회전
    pts_rot = (pts_world @ R_cam.T)
    pcd2    = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pts_rot)

    # 법선 회전
    if normals is not None:
        normals_rot = (normals_world @ R_cam.T)
        pcd2.normals = o3d.utility.Vector3dVector(normals_rot)

    # 색상 유지
    if colors is not None:
        pcd2.colors = o3d.utility.Vector3dVector(colors)

    return pcd2


def get_camera_angle(frames):
    global cam_quat, last_ts_gyro

    af = frames.first_or_default(rs.stream.accel)
    gf = frames.first_or_default(rs.stream.gyro)
    if not af or not gf:
        return cam_quat

    accel = af.as_motion_frame().get_motion_data()
    gyro  = gf.as_motion_frame().get_motion_data()
    ts    = frames.get_timestamp()

    a = np.array([accel.x, accel.y, accel.z])
    g = np.array([gyro.x,   gyro.y,   gyro.z])

    # 1) 최초 호출 → 초기 quaternion을 가속도 기준으로 보정
    if last_ts_gyro is None:
        last_ts_gyro = ts
        cam_quat = calibrate_orientation(a)
        print(f"[Calibrate] Initial cam_quat: {cam_quat}")
        return cam_quat

    # 2) 이후 프레임마다 Madgwick 업데이트
    dt = (ts - last_ts_gyro) / 1000.0
    last_ts_gyro = ts
    madgwick_filter.sampleperiod = dt
    cam_quat = madgwick_filter.updateIMU(q=cam_quat, acc=a, gyr=g)

    return cam_quat

