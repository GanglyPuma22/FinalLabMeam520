import numpy as np
from numpy import sin, cos, pi, linalg
import cv2

def H_w_cam(H_w_ee, H_ee_cam):
    """
    :param H_w_ee: [4x4] array of end-effector in world frame
    :param H_ee_cam: [4x4] array of camera in end-effector frame
    :return: [4x4] array of camera in world frame
    """
    return H_w_ee @ H_ee_cam

def objects_in_world(poses_cam_tag, H_w_cam):
    """
    :param poses_cam_tag: list of [4x4] array for object in camera frame
    :param H_w_cam: [4x4] array of camera in world frame
    :return: list of objects [4x4] transformation in world frame
    """
    return [H_w_cam @ p for p in poses_cam_tag]

def closest_object(tags):
    """
    :param tags: list of AprilTags in world frame
    :return: [4x4] array of the nearest object in world frame. This is the location of the center of the block
    """
    # only considering tags where the z-axis is pointing upwards within some tolerance
    z_axes = [tag[:3, 2] for tag in tags]
    dot_products = np.abs(np.dot(z_axes, np.array([0, 0, 1])))
    aligned_indices = dot_products > .99
    tags = np.array(tags)
    aligned_tags = tags[aligned_indices]
    return aligned_tags.tolist()

def add_noise(H, strength=.01):
    """
    :param H: homogeneous transformation
    :param strength: standard deviation of noise
    :return: homoegenous transformation with added noise
    """
    noise = np.random.normal(scale=strength, size=(4, 4))
    return H + noise

if __name__ == '__main__':
    H = np.array([[1, 0, 0, 1],
                  [0, 1, 0, 2],
                  [0, 0, 1, 1]])
    P = np.array([[1, 0, 1, 1],
                  [0, 1, 0, 2],
                  [0, 0, 1, 1]])    
    list = [H, P]
    print(closest_object(list))