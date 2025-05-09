import math
import pyrealsense2 as rs
import numpy as np
import open3d as o3d
from ahrs.filters import Madgwick

# pip install ahrs

madgwick_filter = Madgwick()
q = np.array([1.0, 0.0, 0.0, 0.0])
gyro_samples = []
gyro_bias = np.zeros(3)
initialized = False
frame_count = 0
INITIAL_SAMPLES = 50
GRAVITY_NORM_RANGE = (9.6, 10.0)
last_ts_gyro = None

# global first, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro, angleX, angleY, angleZ, accel_x, accel_y, accel_z
# first = True
# totalgyroangleX = 0
# totalgyroangleY = 0
# totalgyroangleZ = 0
# last_ts_gyro = 0
# angleX = 0
# angleY = 180
# angleZ = -90

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

def rotate_coordinates(pcd, cam_quat):
    points = np.asarray(pcd.points)
    rotated_points = np.array([rotate_vector(p, cam_quat) for p in points])
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
    return rotated_pcd


# ------- IMU attitude estimation (quaternion) ------- #
def get_camera_angle(frames):
    global cam_quat, initialized, gyro_bias, gyro_samples, frame_count, last_ts_gyro

    accel_frame = frames.first_or_default(rs.stream.accel)
    gyro_frame = frames.first_or_default(rs.stream.gyro)

    if not accel_frame or not gyro_frame:
        return None

    accel = accel_frame.as_motion_frame().get_motion_data()
    gyro = gyro_frame.as_motion_frame().get_motion_data()
    ts = frames.get_timestamp()

    accel_vec = np.array([accel.x, accel.y, accel.z])
    gyro_vec = np.array([gyro.x, gyro.y, gyro.z])

    if last_ts_gyro is None:
        last_ts_gyro = ts
        return None

    dt = (ts - last_ts_gyro) / 1000.0
    last_ts_gyro = ts

    # Initialize gyro bias
    if not initialized:
        accel_norm = np.linalg.norm(accel_vec)
        if GRAVITY_NORM_RANGE[0] <= accel_norm <= GRAVITY_NORM_RANGE[1]:
            gyro_samples.append(gyro_vec)
            frame_count += 1
        if frame_count >= INITIAL_SAMPLES:
            gyro_bias = np.mean(gyro_samples, axis=0)
            initialized = True
            print(f"[Madgwick Init] Gyro bias estimated: {gyro_bias}")
        return None

    gyro_corrected = gyro_vec - gyro_bias
    cam_quat = madgwick_filter.updateIMU(q=cam_quat, acc=accel_vec, gyr=gyro_corrected, dt=dt)

    return cam_quat  # [w, x, y, z]

# def calibrate(f):
#     accel_frame = f.first_or_default(rs.stream.accel)
#     accel = accel_frame.as_motion_frame().get_motion_data()

#     ts = f.get_timestamp()
#     last_ts_gyro = ts

#     # accelerometer calculation
#     accel_angle_x = math.degrees(math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)))
#     accel_angle_y = math.degrees(math.atan2(-accel.z, math.sqrt(accel.x**2 + accel.y**2)))
#     accel_angle_z = 0

#     return accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro

# def calculate_rotation(f, accel_angle_x, accel_angle_y, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro):
#     alpha = 0.98
    
#     gyro_frame = f.first_or_default(rs.stream.gyro)
#     accel_frame = f.first_or_default(rs.stream.accel)

#     accel = accel_frame.as_motion_frame().get_motion_data()
#     gyro = gyro_frame.as_motion_frame().get_motion_data()

#     ts = f.get_timestamp()

#     dt_gyro = (ts - last_ts_gyro) / 1000
#     last_ts_gyro = ts

#     gyro_angle_x = gyro.z * dt_gyro
#     gyro_angle_y = -gyro.x * dt_gyro
#     gyro_angle_z = -gyro.y * dt_gyro

#     dangleX = gyro_angle_x * 57.295791433
#     dangleY = gyro_angle_y * 57.295791433
#     dangleZ = gyro_angle_z * 57.295791433

#     totalgyroangleX = accel_angle_x + dangleX
#     totalgyroangleY = accel_angle_y + dangleY
#     totalgyroangleZ = angleZ + dangleZ

#     accel_angle_x = math.degrees(math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)))
#     accel_angle_y = math.degrees(math.atan2(-accel.z, math.sqrt(accel.x**2 + accel.y**2)))
#     accel_angle_z = 0
#     # print(accel.x,", ", accel.y,", ", accel.z,", ")
#     # print(accel_angle_x,", ", accel_angle_y,", ",accel_angle_z,", ")

#     combinedangleX = totalgyroangleX * alpha + accel_angle_x * (1-alpha)
#     combinedangleY = totalgyroangleY * alpha + accel_angle_y * (1-alpha)
#     combinedangleZ = totalgyroangleZ

#     return combinedangleX,combinedangleY,combinedangleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro

# def get_camera_angle(frames):
#     global first, accel_angle_x, accel_angle_y, angleZ, last_ts_gyro, totalgyroangleX, totalgyroangleY, totalgyroangleZ

#     if first:
#         accel_angle_x, accel_angle_y, angleZ, last_ts_gyro = calibrate(frames)
#         first = False

#     angleX, angleY, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro = calculate_rotation(
#         frames, accel_angle_x, accel_angle_y, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro
#     )
#     # print(angleX,", ", angleY,", ",angleZ,", ")

#     rotation_angle = angleX, angleY, 0

#     return rotation_angle