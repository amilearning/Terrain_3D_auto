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
from path_planner.utils import euler_to_quaternion, quaternion_to_euler, unit_quat, get_odom_euler, get_local_vel, wrap_to_pi, create_line_strip_marker, create_arrow_markers
from path_planner.astar import Astar
from path_planner.hybrid_astar import HybridAstar
from path_planner.hybrid_gp_astar import HybridGPAstar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class PathPlan:
    def __init__(self,planner="astar"):
        self.planner_type = planner
        if self.planner_type == "astar":
            self.planner = Astar()
            self.computed_path_pub = rospy.Publisher("/astar_path", Marker, queue_size=2)             
        elif self.planner_type == "hybrid_astar":
            self.planner = HybridAstar()
            self.computed_path_pub_markerarray = rospy.Publisher("/hybrid_astar_path", MarkerArray, queue_size=2)                               
        elif self.planner_type == "hybrid_gp_astar":
            self.planner = HybridGPAstar()
            self.computed_path_pub_markerarray = rospy.Publisher("/hybrid_gp_astar__path", MarkerArray, queue_size=2)                               
        else:
            rospy.logerr("unkown planner")
            return 
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)           
        self.odom_available           = False 
        
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
        # map_topic = "/traversability_estimation/terrain_map"
        map_topic = "/traversability_estimation/global_map"        
        status_topic = "/planner_status"
        self.layer_name = "terrain_traversability"
        self.elevation_layer_name = "elevation"

        # Publishers        
        
        
        self.debug_pub = rospy.Publisher("mpc_debug", PoseStamped, queue_size=2)    
        
        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                                               
        self.goal_sub = rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback)
        self.obs_sub = rospy.Subscriber(obs_topic, PoseWithCovarianceStamped, self.obs_callback)
        self.map_sub = rospy.Subscriber(map_topic,GridMap, self.mapCallback)
        # 20Hz planner callback 
        self.planner_timer = rospy.Timer(rospy.Duration(0.1), self.planner_callback)         
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

    
        
    def goal_callback(self, msg):
        if self.waypoint_available is False:
            self.waypoint_available = True
        
        self.goal_point = msg
        

    def run_planning(self):
        
        self.planner.set_goal(self.goal_point)
        self.planner.set_pose(self.cur_pose)    
        self.planner.set_map(self.trav_map, self.grid_map.info, self.elev_map)
        start = time.time()                 
        path = self.planner.path_plan()
        end = time.time()       
        print("Planning time: {:.5f}".format( end-start))          
        if len(path)> 0:
            if self.planner_type == "astar":                
                path_marker = create_line_strip_marker(path)                
                self.computed_path_pub.publish(path_marker)
            else:                
                path_markerArray = create_arrow_markers(path)            
                self.computed_path_pub_markerarray.publish(path_markerArray)


    def planner_callback(self,timer):
        
        if self.odom_available is False:
            rospy.loginfo("Odom is not available yet")
            return
        elif self.waypoint_available is False:
            rospy.loginfo("Waypoints are not available yet")
            return
        

        def _thread_func():
            self.run_planning()            
        self.planner_thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        self.planner_thread.start()
        self.planner_thread.join()
        

    def odom_callback(self, msg):        
        if self.odom_available is False:
            self.odom_available = True         
        
        self.cur_pose = msg
        
        
    def mapCallback(self,msg):        
        self.grid_map = msg    
        traversability_idx = self.grid_map.layers.index(self.layer_name)            
        elevation_idx = self.grid_map.layers.index(self.elevation_layer_name)            
        self.elev_map = self.grid_map.data[elevation_idx]        
        self.trav_map = self.grid_map.data[traversability_idx]        
     
###################################################################################

def main():
    rospy.init_node("path_plan")
# astar, hybrid_astar, hybrid_gp_astar
    planner_type = rospy.get_param('~planner', default='astar')
    PathPlan(planner_type)
  

if __name__ == "__main__":
    main()




 
    

        
        