import math
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

global first, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro, angleX, angleY, angleZ, accel_x, accel_y, accel_z
first = True
totalgyroangleX = 0
totalgyroangleY = 0
totalgyroangleZ = 0
last_ts_gyro = 0
angleX = 0
angleY = 180
angleZ = -90

def calibrate(f):
    accel_frame = f.first_or_default(rs.stream.accel)
    accel = accel_frame.as_motion_frame().get_motion_data()

    ts = f.get_timestamp()
    last_ts_gyro = ts

    # accelerometer calculation
    accel_angle_x = math.degrees(math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)))
    accel_angle_y = math.degrees(math.atan2(-accel.z, math.sqrt(accel.x**2 + accel.y**2)))
    accel_angle_z = 0

    return accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro

def calculate_rotation(f, combinedangleX_prev, combinedangleY_prev, combinedangleZ_prev, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro):

    
    gyro_frame = f.first_or_default(rs.stream.gyro)
    accel_frame = f.first_or_default(rs.stream.accel)

    accel = accel_frame.as_motion_frame().get_motion_data()
    gyro = gyro_frame.as_motion_frame().get_motion_data()

    ts = f.get_timestamp()

    dt_gyro = (ts - last_ts_gyro) / 1000
    last_ts_gyro = ts

    gyro_angle_x = gyro.z * dt_gyro
    gyro_angle_y = -gyro.x * dt_gyro
    gyro_angle_z = -gyro.y * dt_gyro

    dangleX = gyro_angle_x * 57.295791433
    dangleY = gyro_angle_y * 57.295791433
    dangleZ = gyro_angle_z * 57.295791433

    totalgyroangleX = combinedangleX_prev + dangleX
    totalgyroangleY = combinedangleY_prev + dangleY
    totalgyroangleZ = combinedangleZ_prev + dangleZ

    accel_angle_x = math.degrees(math.atan2(-accel.x, math.sqrt(accel.y**2 + accel.z**2)))
    accel_angle_y = math.degrees(math.atan2(-accel.z, math.sqrt(accel.x**2 + accel.y**2)))
    accel_angle_z = 0
    # print(accel.x,", ", accel.y,", ", accel.z,", ")
    # print(accel_angle_x,", ", accel_angle_y,", ",accel_angle_z,", ")

    # alpha = 0.05
    # low alpha while stationary, high alpha while moving
    accel_norm = np.linalg.norm(np.array([accel.x, accel.y, accel.z]))
    alpha = np.tanh(abs(accel_norm - 10))/1.2  # 가속도 값이 10에 가까울수록 alpha는 작아짐

    combinedangleX = totalgyroangleX * alpha + accel_angle_x * (1-alpha)
    combinedangleY = totalgyroangleY * alpha + accel_angle_y * (1-alpha)
    combinedangleZ = totalgyroangleZ

    combinedangleX_prev = combinedangleX
    combinedangleY_prev = combinedangleY
    combinedangleZ_prev = combinedangleZ


    return combinedangleX,combinedangleY,combinedangleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, combinedangleX_prev, combinedangleY_prev, combinedangleZ_prev, last_ts_gyro


def rotate_coordinates(pcd, cam_rpy):
    points = np.asarray(pcd.points)

    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, 1],
                            [-1, 0,  0]])
 
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

    rotated_point = np.dot(points, R)

    # rotated_pcd = o3d.geometry.PointCloud()
    # rotated_pcd.points = o3d.utility.Vector3dVector(rotated_point)
    pcd.points = o3d.utility.Vector3dVector(rotated_point)
    


def rotate_vector(vec, cam_rpy):
    cam2world_R = np.array([[0, -1,  0],
                            [0,  0, 1],
                            [-1, 0,  0]])
 
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

    # rotated_vec = R @ vec
    rotated_vec = np.dot(vec, R.T)

    return rotated_vec

def get_camera_angle(frames):
    global first, accel_angle_x, accel_angle_y, angleZ, last_ts_gyro, totalgyroangleX, totalgyroangleY, totalgyroangleZ

    if first:
        accel_angle_x, accel_angle_y, angleZ, last_ts_gyro = calibrate(frames)
        first = False

    angleX, angleY, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, accel_angle_x, accel_angle_y, accel_angle_z, last_ts_gyro = calculate_rotation(
        frames, accel_angle_x, accel_angle_y, angleZ, totalgyroangleX, totalgyroangleY, totalgyroangleZ, last_ts_gyro
    )
    # print(angleX,", ", angleY,", ",angleZ,", ")

    rotation_angle = angleX, angleY, 0

    return rotation_angle