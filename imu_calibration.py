import math
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from ahrs.filters import Madgwick

# pip install ahrs

madgwick_filter = Madgwick(sampleperiod=0.01, beta=0.05)
cam_quat  = np.array([1.0, 0.0, 0.0, 0.0])
cam_rpy = np.array([0.0, 0.0, 0.0])
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

def quaternion_to_euler(q):
    # 입력 q = [w, x, y, z]
    w, x, y, z = q
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.rad2deg([roll, pitch, yaw])


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
def rotate_vector(vec, cam_rpy):
    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [1,  0,  0]])
 
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
    R = Rz @ Ry @ Rx

    rotated_vec = R @ vec

    return rotated_vec

# ---- PointCloud 회전 (성능 최적화, 법선/색상 보존) ----

def rotate_coordinates(pcd, cam_rpy):
    points = np.asarray(pcd.points)
    cam_coordinates = [np.zeros(points.shape[0]), 3]

    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, -1],
                            [1,  0,  0]])
 
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

    R = Rz @ Ry @ Rx

    rotated_point = points @ R.T

    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_point)

    return rotated_pcd


def get_camera_angle(frames):
    global cam_quat, cam_rpy, last_ts_gyro

    af = frames.first_or_default(rs.stream.accel)
    gf = frames.first_or_default(rs.stream.gyro)
    if not af or not gf:
        return np.degrees([0.0, 0.0, 0.0])  # fallback

    accel = af.as_motion_frame().get_motion_data()
    gyro  = gf.as_motion_frame().get_motion_data()
    ts    = frames.get_timestamp()

    a = np.array([accel.x, accel.y, accel.z])
    g = np.array([gyro.x,   gyro.y,  gyro.z])

    # print(f"[IMU] a: {a}, g: {g}")

    if last_ts_gyro is None:
        last_ts_gyro = ts
        cam_quat = calibrate_orientation(a)
        print(f"[Calibrate] Initial cam_quat: {cam_quat}")
    else:
        dt = (ts - last_ts_gyro) / 1000.0
        last_ts_gyro = ts
        madgwick_filter.sampleperiod = dt
        cam_quat = madgwick_filter.updateIMU(q=cam_quat, acc=a, gyr=g)
        cam_rpy = quaternion_to_euler(cam_quat)

        print(f"Roll: {cam_rpy[0]:.2f}, Pitch: {cam_rpy[1]:.2f}, Yaw: {cam_rpy[2]:.2f}")

    # 최종 출력: RPY(deg)
    return cam_rpy

