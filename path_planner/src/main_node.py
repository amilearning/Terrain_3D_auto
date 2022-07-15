#!/usr/bin/env python
from cv2 import fastNlMeansDenoisingColored
import rospy

""" ROS node for path plan.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

import json
import time
import threading
import numpy as np
import pandas as pd
import math 
import sys
from std_msgs.msg import Bool, Empty, Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped 
from visualization_msgs.msg import MarkerArray, Marker
# from hmcl_msgs import Lane
from grid_map_msgs.msg import GridMap
from path_planner.utils import euler_to_quaternion, quaternion_to_euler, unit_quat, get_odom_euler, get_local_vel, wrap_to_pi

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class PathPlan:
    def __init__(self,planner="astar"):
        
        
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)           
        self.odom_available           = False 
        self.vehicle_status_available = False 
        self.waypoint_available = False         
        # Thread for planner
        self.mpc_thread = threading.Thread()        
        
        self.odom = Odometry()        
        self.debug_msg = PoseStamped()
        self.obs_pose = PoseWithCovarianceStamped()
        # Real world setup
        odom_topic = "/ground_truth/state"
        vehicle_status_topic = "/chassisState"                  
        goal_topic = "/move_base_simple/goal"                
        obs_topic = "/initialpose"       
        map_topic = "/traversability_estimation/terrain_map"
        status_topic = "/planner_status"
        
        # Publishers        
        self.mpc_predicted_trj_publisher = rospy.Publisher("/mpc_pred_trajectory", MarkerArray, queue_size=2)                
        self.debug_pub = rospy.Publisher("mpc_debug", PoseStamped, queue_size=2)    
        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                                               
        self.goal_sub = rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback)
        self.obs_sub = rospy.Subscriber(obs_topic, PoseWithCovarianceStamped, self.obs_callback)
        self.map_sub = rospy.Subscriber(map_topic,GridMap, self.mapCallback)
        # 20Hz planner callback 
        self.planner_timer = rospy.Timer(rospy.Duration(0.2), self.planner_callback)         
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    

        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():
            # Publish if MPC is busy with a current trajectory
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            rate.sleep()




    def obs_callback(self,msg):
        self.obs_pose = msg

    def vehicle_status_callback(self,data):
        if self.vehicle_status_available is False:
            self.vehicle_status_available = True
        self.chassisState = data
        
    def goal_callback(self, msg):
        if self.waypoint_available is False:
            self.waypoint_available = True
        self.waypoint = msg

    def run_planning(self):

        current_euler = get_odom_euler(self.odom)
        local_vel = get_local_vel(self.odom, is_odom_local_frame = True)
        self.debug_msg.header.stamp = rospy.Time.now()
        self.debug_msg.pose.position.x = local_vel[0]
        self.debug_msg.pose.position.y = local_vel[1]
        self.debug_msg.pose.position.z = local_vel[2]
        self.debug_pub.publish(self.debug_msg)
        
        # xinit = np.transpose(np.array([self.odom.pose.pose.position.x,self.odom.pose.pose.position.y, local_vel[0], current_euler[2], self.chassisState.steering]))
        

    def planner_callback(self,timer):
        if self.vehicle_status_available is False:
            rospy.loginfo("Vehicle status is not available yet")
            return
        elif self.odom_available is False:
            rospy.loginfo("Odom is not available yet")
            return
        elif self.waypoint_available is False:
            rospy.loginfo("Waypoints are not available yet")
            return
  

        def _thread_func():
            self.run_planning()            
        self.mpc_thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        self.mpc_thread.start()
        self.mpc_thread.join()
        

    

    def odom_callback(self, msg):
        """                
        :type msg: PoseStamped
        """              
        if self.odom_available is False:
            self.odom_available = True 
        self.odom = msg        
        
    def mapCallback(self,msg):
        self.grid_map = msg
        traversability_idx = self.grid_map.layers.index("terrain_traversability")
        self.trav_map = self.grid_map.data[traversability_idx]

        self.c_size  = self.trav_map.layout.dim[0].size
        self.r_size  = self.trav_map.layout.dim[1].size
        self.map_resolution = self.grid_map.info.resolution 
        
        
        for i in range(len(self.trav_map.data)):
            pose = self.idx2pose(i)
            idx = self.pose2idx(pose)
            assert idx ==i, "idx pose convert fail"
        

    def idx2pose(self,idx):        
        # top right is 0 - bottom left is last 
        assert idx < self.r_size*self.c_size, "idx is out of bound"                    
        grid_r = int(idx/(self.r_size))
        grid_c = (idx - grid_r*self.r_size)
        pose_x = self.grid_map.info.pose.position.x+self.c_size/2*self.map_resolution-grid_c*self.map_resolution
        pose_y = self.grid_map.info.pose.position.y+self.r_size/2*self.map_resolution-grid_r*self.map_resolution
        return [pose_x, pose_y]
        
        
    def pose2idx(self,pose):    
        right_corner_x = self.grid_map.info.pose.position.x + self.grid_map.info.length_x/2
        right_corner_y = self.grid_map.info.pose.position.y + self.grid_map.info.length_y/2

        grid_c_idx = (int)((right_corner_x - pose[0]) / self.map_resolution)
        grid_r_idx = (int)((right_corner_y - pose[1]) / self.map_resolution)
        assert grid_c_idx < self.c_size, "pose out of x map bound"
        assert grid_r_idx < self.r_size, "pose out of y map bound"        
        idx = grid_c_idx + grid_r_idx*self.r_size 
        assert idx < self.c_size*self.r_size, "computed idx is out of bound"
         
        return idx

        
        
        
        
        
    # def map_pose2idx(self,pose):
    #     x =pose.x
    #     y =pose.y
    #     z =pose.z
        
    #     self.grid_map.info.resolution
    #     self.grid_map.info.length_x
    #     self.grid_map.info.length_y

    

    # def map_idx2pose(self,idx):
    #     idx*
    #     self.grid_map.info.pose.position.x + 


    #     self.grid_map.info.pose.position.y
        
     
        
  
###################################################################################

def main():
    rospy.init_node("path_plan")
    planner_type = rospy.get_param('~planner', default='astar')
    PathPlan(planner_type)
  

if __name__ == "__main__":
    main()




 
    

        
        