import random

from cv2 import LineSegmentDetector

import rospy
import os
import math
import json
import errno
import shutil
import joblib
import random
import pyquaternion
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

def wrap_to_pi(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi-0.01:
        angle -= 2.0 * np.pi

    while angle < -np.pi+0.01:
        angle += 2.0 * np.pi

    return angle 


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])


def get_pose_euler(pose):
    q = pyquaternion.Quaternion(w=pose.orientation.w, x=pose.orientation.x, y=pose.orientation.y, z=pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def get_odom_euler(odom):    
    q = pyquaternion.Quaternion(w=odom.pose.pose.orientation.w, x=odom.pose.pose.orientation.x, y=odom.pose.pose.orientation.y, z=odom.pose.pose.orientation.z)
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]    
    rot_mat = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    return rot_mat


def get_local_vel(odom, is_odom_local_frame = True):
    local_vel = np.array([0.0, 0.0, 0.0])
    if is_odom_local_frame is False: 
        # convert from global to local 
        q_tmp = np.array([odom.pose.pose.orientation.w,odom.pose.pose.orientation.x,odom.pose.pose.orientation.y,odom.pose.pose.orientation.z])
        euler = get_odom_euler(odom)
        rot_mat_ = q_to_rot_mat(q_tmp)
        inv_rot_mat_ = np.linalg.inv(rot_mat_)
        global_vel = np.array([odom.twist.twist.linear.x,odom.twist.twist.linear.y,odom.twist.twist.linear.z])
        local_vel = inv_rot_mat_.dot(global_vel)        
    else:
        local_vel[0] = odom.twist.twist.linear.x
        local_vel[1] = odom.twist.twist.linear.y
        local_vel[2] = odom.twist.twist.linear.z
    return local_vel 



def dist2d(point1, point2):
    """
    Euclidean distance between two points
    :param point1:
    :param point2:
    :return:
    """

    x1, y1 = point1[0:2]
    x2, y2 = point2[0:2]

    dist2 = (x1 - x2)**2 + (y1 - y2)**2

    return math.sqrt(dist2)

def create_line_strip_marker(points):
    
    line = Marker()
    line.header.frame_id = "map"
    line.header.stamp = rospy.Time.now()
    line.type = 4
    line.scale.x = 0.05
    line.color.a = 1.0
    line.color.r = 0.0
    line.color.g = 1.0
    line.color.b = 0.0
    
    for i in range(len(points)):        
        p = Point()
        # p.header = line.header        
        # p.ns = "path_namespace"
        # p.id = i
        # p.type = 8       
        
        p.x = points[i][0]
        p.y = points[i][1]
        p.z = 5.0
        
        line.points.append(p)
    
    return line
        
        
def round_theta(theta, thetas):
    """ Round theta to closest discretized value. """

    return min(thetas, key=lambda x: abs(x-theta) % (2*np.pi))

def get_discretized_thetas(unit_theta):
    """ Get all discretized theta values by unit value. """

    thetas = [-np.pi+1e-2]

    while True:
        
        theta = thetas[-1] + unit_theta
        
        # if theta > (np.pi - unit_theta):
        if theta > (np.pi):
            break
        
        thetas.append(theta)
    
    return thetas



def create_arrow_markers(points):
    markers = MarkerArray()

    for i in range(len(points)):        
        arrow = Marker()        
        qat = euler_to_quaternion(0.0,0.0,points[i][2])
        arrow.pose.position.x = points[i][0]
        arrow.pose.position.y = points[i][1]
        arrow.id = i
        arrow.ns = "arrow"
        arrow.pose.position.z = 5.0
        arrow.pose.orientation.x = qat[1] 
        arrow.pose.orientation.y = qat[2]
        arrow.pose.orientation.z = qat[3]
        arrow.pose.orientation.w = qat[0]

        arrow.header.frame_id = "map"
        arrow.header.stamp = rospy.Time.now()
        arrow.type = 0
        arrow.scale.x = 0.5
        arrow.scale.y = 0.2
        arrow.scale.z = 0.1
        arrow.color.a = 1.0
        arrow.color.r = 0.0
        arrow.color.g = 1.0
        arrow.color.b = 0.0   
        markers.markers.append(arrow)
    
    return markers