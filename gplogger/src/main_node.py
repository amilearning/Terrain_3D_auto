#!/usr/bin/env python
import rospy


""" ROS node for the MPC GP in 3d offroad environment, to use in the Gazebo simulator and real world experiments.
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
from hmcl_msgs.msg import Lane, Waypoint, vehicleCmd
from visualization_msgs.msg import MarkerArray, Marker
from autorally_msgs.msg import chassisState


from gplogger.vehicle_model import VehicleModel
from gplogger.utils import euler_to_quaternion, quaternion_to_euler, unit_quat, get_odom_euler, get_local_vel, traj_to_markerArray, predicted_trj_visualize, ref_to_markerArray, wrap_to_pi
from gplogger.dataloader import DataLoader
import rospkg
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('gplogger')



def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class GPLoggerWrapper:
    def __init__(self):        
        self.n_nodes = rospy.get_param('~n_nodes', default=40)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)   
        self.gp_enable = rospy.get_param('~gp_enable', default=False)             
        self.GP_build_flag = rospy.get_param('~gp_build_flat', default=False)   
        self.gp_train_data_file = rospy.get_param('~gp_train_data_file', default="test_data.npz")             
        self.dt = self.t_horizon / self.n_nodes*1.0        
         # x, y, psi, vx, vy, wz, z , delta,  accelx,  roll, pitch 
         # 0  1  2     3  4   5   6   7,      8,        9,    10   
        self.cur_x = np.transpose(np.zeros([1,11]))
#################################################################        
        # Initialize GP         
#################################################################
        ##############################################
        # if self.gp_enable:
        #     self.GPmodel = GPModel(data_file_name = self.gp_train_data_file)                
        #     if self.GP_build_flag:
        #         self.GPmodel.all_model_build_and_save()            
        #     else:
        #         self.GPmodel.all_model_load()
        
        
        self.VehicleModel = VehicleModel(dt = self.dt)
        self.dataloader = DataLoader(state_dim = len(self.cur_x) )
        
        self.odom_available   = False 
        self.vehicle_status_available = False 
        self.waypoint_available = False 
        
        # Thread for optimization
        self.vehicleCmd = vehicleCmd()
        self._thread = threading.Thread()        
        self.chassisState = chassisState()
        self.odom = Odometry()
        self.waypoint = PoseStamped()
        self.debug_msg = PoseStamped()
        
        # Real world setup
        odom_topic = "/ground_truth/state"
        vehicle_status_topic = "/chassisState"
        control_topic = "/acc_cmd"                    
        status_topic = "/is_data_busy"
        self.file_name = "test_data.npz"
        self.logging = False
        # Publishers        
        self.predicted_trj_publisher = rospy.Publisher("/gplogger_pred_trajectory", MarkerArray, queue_size=2)        
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    
        self.debug_pub = rospy.Publisher("data_debug", PoseStamped, queue_size=2)    
        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                        
        self.vehicle_status_sub = rospy.Subscriber(vehicle_status_topic, chassisState, self.vehicle_status_callback)                
        self.ctrl_sub = rospy.Subscriber(control_topic, vehicleCmd, self.ctrl_callback)
        self.data_saver_sub = rospy.Subscriber("/data_save", Bool, self.data_saver_callback)
        self.data_logging_sub = rospy.Subscriber("/data_logging", Bool, self.data_logging_callback)
        
        
        # 20Hz control callback 
        self.cmd_timer = rospy.Timer(rospy.Duration(0.2), self.cmd_callback)         
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():
            # Publish if MPC is busy with a current trajectory
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            rate.sleep()

    def ctrl_callback(self,msg):
        self.vehicleCmd = msg

    def data_logging_callback(self,msg):
        if msg.data:
            self.logging = True
        else:
            self.logging = False

    def data_saver_callback(self,msg):
        if msg.data:
            save_path = pkg_dir + '/data/'+self.file_name            
            self.dataloader.file_save(save_path)
            messageToPring =  "file has been saved in " + str(save_path)
            rospy.loginfo(messageToPring)

    def vehicle_status_callback(self,data):
        if self.vehicle_status_available is False:
            self.vehicle_status_available = True
        self.chassisState = data
        self.steering = -data.steering*25*math.pi/180
        
    def odom_callback(self, msg):
        """                
        :type msg: PoseStamped
        """              
        if self.odom_available is False:
            self.odom_available = True 
        self.odom = msg        
        
    def waypoint_callback(self, msg):
        if self.waypoint_available is False:
            self.waypoint_available = True
        self.waypoint = msg

    

    def run_prediction(self):        
        xinit = self.cur_x.copy()
        start = time.time()                 
        x = self.cur_x.copy()
        u = self.cur_u.copy()        
        predictedStates = self.VehicleModel.predict_multistep(x,u,self.n_nodes)                     
        pred_traj_marker = predicted_trj_visualize(predictedStates)        
        self.predicted_trj_publisher.publish(pred_traj_marker)
        end = time.time()
        
        print("time: {:.5f}".format( end-start))
        
        # pred_maerker_refs = predicted_trj_visualize(x_pred)
        # self.mpc_predicted_trj_publisher.publish(pred_maerker_refs)
        
        ########  Log State Data ###################################################
        # if self.logging:
        #     add_state = np.array([u_pred[0,0],u_pred[1,0],xinit[0],xinit[1], xinit[2], xinit[3], xinit[4], xinit[5], xinit[6]])                    
        #     self.dataloader.append_state(add_state,temp[:,1])     
        #     rospy.loginfo("recording process %d",self.dataloader.n_data_set)
        ########  Log State Data END ###################################################

    def cmd_callback(self,timer):
        if self.vehicle_status_available is False:
            rospy.loginfo("Vehicle status is not available yet")
            return
        elif self.odom_available is False:
            rospy.loginfo("Odom is not available yet")
            return

        current_euler = get_odom_euler(self.odom)
        for i in range(3):
            current_euler[i] = wrap_to_pi(current_euler[i])
        if(abs(current_euler[0]) > 80*math.pi/180):
            return
        
        local_vel = get_local_vel(self.odom, is_odom_local_frame = False)
        cur_accel = self.vehicleCmd.acceleration

        self.debug_msg.header.stamp = rospy.Time.now()
        self.debug_msg.pose.position.x = local_vel[0]
        self.debug_msg.pose.position.y = current_euler[2]        
        self.debug_pub.publish(self.debug_msg)        
        # x, y, psi, vx, vy, wz, z, roll, pitch 
        # 0  1  2     3  4   5   6 7,    8    \
        self.cur_x = np.transpose(np.array([self.odom.pose.pose.position.x,
                                            self.odom.pose.pose.position.y,
                                            current_euler[2],
                                            local_vel[0],
                                            local_vel[1],
                                            self.odom.twist.twist.angular.z,
                                            self.odom.pose.pose.position.z,                                                   
                                            current_euler[0],
                                            current_euler[1]]))
        self.cur_u = np.transpose(np.array([ self.steering,cur_accel]))

        def _thread_func():
            self.run_prediction()            
        self._thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        self._thread.start()
        self._thread.join()
              
        

###################################################################################

def main():
    rospy.init_node("gplogger")
    
    GPLoggerWrapper()

if __name__ == "__main__":
    main()




 
    


