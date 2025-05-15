import math
import pyrealsense2 as rs
import numpy as np

# — 전역 상태 초기값 —
first = True
totalgyroangleX = 0.0
totalgyroangleY = 0.0
totalgyroangleZ = 0.0
last_ts_gyro = 0.0
accel_angle_x = 0.0
accel_angle_y = 0.0
angleZ = -90.0

def calibrate(frames):
    """초기 가속도계를 이용한 카메라 Roll/Pitch 보정."""
    accel_frame = frames.first_or_default(rs.stream.accel)
    accel = accel_frame.as_motion_frame().get_motion_data()
    ts = frames.get_timestamp()
    # 가속도계 기반 각도 계산 (deg)
    ax = math.degrees(math.atan2(-accel.x, math.hypot(accel.y, accel.z)))
    ay = math.degrees(math.atan2(-accel.z, math.hypot(accel.x, accel.y)))
    az = 0.0
    return ax, ay, az, ts

def calculate_rotation(frames,
                       prev_ax, prev_ay, prev_az,
                       totalX, totalY, totalZ,
                       last_ts):
    """자이로+가속도기 보정된 Complementary filter."""
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    accel_frame = frames.first_or_default(rs.stream.accel)
    gyro  = gyro_frame.as_motion_frame().get_motion_data()
    accel = accel_frame.as_motion_frame().get_motion_data()
    ts    = frames.get_timestamp()

    dt = (ts - last_ts) / 1000.0
    last_ts = ts

    # 자이로 적분 (rad → deg)
    dX = gyro.z * dt * 57.2958
    dY = -gyro.x * dt * 57.2958
    dZ = -gyro.y * dt * 57.2958

    totalX = prev_ax + dX
    totalY = prev_ay + dY
    totalZ = prev_az + dZ

    # 가속도계 각도
    ax = math.degrees(math.atan2(-accel.x, math.hypot(accel.y, accel.z)))
    ay = math.degrees(math.atan2(-accel.z, math.hypot(accel.x, accel.y)))
    az = 0.0

    # Complementary filter 계수
    accel_norm = np.linalg.norm([accel.x, accel.y, accel.z])
    alpha = np.tanh(abs(accel_norm - 10.0)) / 1.2

    combinedX = totalX * alpha + ax * (1 - alpha)
    combinedY = totalY * alpha + ay * (1 - alpha)
    combinedZ = totalZ

    return (combinedX, combinedY, combinedZ,
            totalX, totalY, totalZ,
            combinedX, combinedY, combinedZ,
            last_ts)

def get_camera_angle(frames):
    """외부에서 호출할 카메라 RPY(roll, pitch, yaw)."""
    global first, accel_angle_x, accel_angle_y, angleZ
    global totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro

    if first:
        ax, ay, az, last_ts_gyro = calibrate(frames)
        accel_angle_x, accel_angle_y, angleZ = ax, ay, az
        first = False

    (angleX, angleY, angleZ,
     totalgyroangleX, totalgyroangleY, totalgyroangleZ,
     accel_angle_x, accel_angle_y, accel_angle_z,
     last_ts_gyro) = calculate_rotation(
        frames,
        accel_angle_x, accel_angle_y, angleZ,
        totalgyroangleX, totalgyroangleY, totalgyroangleZ,
        last_ts_gyro
    )

    # Yaw는 0 고정
    return (angleX, angleY, 0.0)

# — 회전 행렬 및 벡터화된 회전 함수 추가 —
def get_rotation_matrix(cam_rpy):
    """
    cam_rpy: (roll, pitch, yaw) in degrees
    리얼센스 좌표계→월드 좌표계 회전 행렬.
    """
    # 카메라→월드 기준 축 변환
    cam2world_R = np.array([[ 0, -1,  0],
                            [ 0,  0,  1],
                            [-1,  0,  0]])
    # rpy 순서: Rx(roll) → Ry(pitch) → Rz(yaw)
    rx, ry, rz = np.radians(cam2world_R.dot(cam_rpy))

    Rx = np.array([[1,           0,            0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [           0, 1,          0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [          0,            0, 1]])
    return Rz.dot(Ry).dot(Rx)

def rotate_vector(vec, cam_rpy):
    """(3,) 벡터 → 회전."""
    R = get_rotation_matrix(cam_rpy)
    return vec.dot(R.T)

def rotate_points(points, cam_rpy):
    """
    (N,3) 포인트 배열 → 한번에 회전.
    for문 없이 대량처리에 최적화.
    """
    R = get_rotation_matrix(cam_rpy)
    return points.dot(R.T)