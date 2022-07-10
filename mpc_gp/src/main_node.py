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

from mpc_gp.traj_gen import TrajManager
from mpc_gp.mpc_model import GPMPCModel
from mpc_gp.mpc_utils import euler_to_quaternion, quaternion_to_euler, unit_quat, get_odom_euler, get_local_vel, traj_to_markerArray, predicted_trj_visualize, ref_to_markerArray, wrap_to_pi
from mpc_gp.dataloader import DataLoader

import rospkg
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('mpc_gp')



def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

class GPMPCWrapper:
    def __init__(self,environment="gazebo"):
        
        self.n_mpc_nodes = rospy.get_param('~n_nodes', default=40)
        self.t_horizon = rospy.get_param('~t_horizon', default=2.0)   
        self.model_build_flag = rospy.get_param('~build_flat', default=False)             
        self.dt = self.t_horizon / self.n_mpc_nodes*1.0
         # x, y, vx, psi
        self.cur_x = np.transpose(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
#################################################################        
        # Initialize GP MPC         
#################################################################
        self.MPCModel = GPMPCModel( model_build = self.model_build_flag,  N = self.n_mpc_nodes, dt = self.dt)
        self.TrajManager = TrajManager(MPCModel = self.MPCModel, dt = self.dt, n_sample = self.n_mpc_nodes)                    
        self.dataloader = DataLoader(input_dim = 2, state_dim = len(self.cur_x) )
        self.odom_available   = False 
        self.vehicle_status_available = False 
        self.waypoint_available = False 
        
        # Thread for MPC optimization
        self.mpc_thread = threading.Thread()        
        self.chassisState = chassisState()
        self.odom = Odometry()
        self.waypoint = PoseStamped()
        self.debug_msg = PoseStamped()
        self.obs_pose = PoseWithCovarianceStamped()
        # Real world setup
        odom_topic = "/ground_truth/state"
        vehicle_status_topic = "/chassisState"
        control_topic = "/acc_cmd"            
        waypoint_topic = "/move_base_simple/goal"                
        obs_topic = "/initialpose"       
        status_topic = "/is_mpc_busy"
        self.file_name = "test_data.npz"
        self.logging = False
        # Publishers
        self.control_pub = rospy.Publisher(control_topic, vehicleCmd, queue_size=1, tcp_nodelay=True)        
        self.mpc_predicted_trj_publisher = rospy.Publisher("/mpc_pred_trajectory", MarkerArray, queue_size=2)
        self.mpc_ref_traj_publisher = rospy.Publisher("/mpc_ref_traj", MarkerArray, queue_size=2)
        self.final_ref_publisher = rospy.Publisher("/final_trajectory", MarkerArray, queue_size=2)    
        self.xy_ref_publisher = rospy.Publisher("/mpc_xy_ref", MarkerArray, queue_size=2)    
        self.status_pub = rospy.Publisher(status_topic, Bool, queue_size=2)    
        self.debug_pub = rospy.Publisher("mpc_debug", PoseStamped, queue_size=2)    
        # Subscribers
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback)                        
        self.vehicle_status_sub = rospy.Subscriber(vehicle_status_topic, chassisState, self.vehicle_status_callback)                
        self.waypoint_sub = rospy.Subscriber(waypoint_topic, PoseStamped, self.waypoint_callback)
        self.obs_sub = rospy.Subscriber(obs_topic, PoseWithCovarianceStamped, self.obs_callback)
        self.data_saver_sub = rospy.Subscriber("/mpc_data_save", Bool, self.data_saver_callback)
        self.data_logging_sub = rospy.Subscriber("/mpc_data_logging", Bool, self.data_logging_callback)

        # Timers
        self.traj_dt = 0.1
        self.TrajGen_timer = rospy.Timer(rospy.Duration(self.traj_dt), self.trajGen_callback) 
        self.traj_random_count = 0
        self.local_traj = None
        self.sample_delta  = 0.0
        self.sample_velocity = 1.0
        # 20Hz control callback 
        self.cmd_timer = rospy.Timer(rospy.Duration(0.05), self.cmd_callback) 
        self.blend_min = 3
        self.blend_max = 5
        self.is_first_mpc = True
        self.init_traj = True
        rate = rospy.Rate(1)     
        while not rospy.is_shutdown():
            # Publish if MPC is busy with a current trajectory
            msg = Bool()
            msg.data = True
            self.status_pub.publish(msg)
            rate.sleep()

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

    def obs_callback(self,msg):
        self.obs_pose = msg

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

    def trajGen_callback(self,timer):
        if self.TrajManager is None:            
            return 
        if self.init_traj:
            self.TrajManager.setState(self.cur_x)
            self.init_traj = False
        path_duration = 2.5 # sec 
        if self.traj_random_count > path_duration/self.traj_dt:
            self.traj_random_count = 0
            self.TrajManager.setState(self.cur_x)
            self.sample_delta    = self.sample_delta + (np.random.rand(1)[0]-0.5)*0.025
            self.sample_delta = max(min(self.sample_delta,25*np.pi/180.0),-25*np.pi/180)
            self.sample_velocity = self.sample_velocity + (np.random.rand(1)[0]-0.5)*0.2
            self.sample_velocity = max(min(self.sample_velocity,0.0),1.5)

        self.ref_state = self.TrajManager.gen_traj(self.sample_delta, self.sample_velocity)        
        marker_refs = traj_to_markerArray(self.ref_state)        
        self.mpc_ref_traj_publisher.publish(marker_refs)
        self.traj_random_count+=1


    def run_mpc(self):
        if self.MPCModel is None:
            return        
        xinit = self.cur_x        
        if self.is_first_mpc:
            self.is_first_mpc = False
            x0i = np.array([0.,0.,xinit[0],xinit[1], xinit[2], xinit[3], xinit[4]])
            x0 = np.transpose(np.tile(x0i, (1, self.MPCModel.model.N)))
            problem = {"x0": x0,
                    "xinit": xinit}
        else:
            problem = {"xinit": xinit}                
        # obstacle_ = np.array([self.obs_pose.pose.pose.position.x, self.obs_pose.pose.pose.position.y])
        # goal_ = np.array([self.waypoint.pose.position.x, self.waypoint.pose.position.y])       

        # Compute Local Trajectory
        cur_xy =  np.array([xinit[0],xinit[1]])
        local_traj_points = self.TrajManager.extract_path_points(cur_xy,self.MPCModel.model.N)
        local_tarj_marker = ref_to_markerArray(local_traj_points)      
        self.xy_ref_publisher.publish(local_tarj_marker)  
        ## Set ref trajectories
        # problem["all_parameters"] = np.transpose(np.tile(goal_,(1,self.MPCModel.model.N)))        
        problem["all_parameters"] =np.reshape(local_traj_points,(3*self.MPCModel.model.N,1))                
        output, exitflag, info = self.MPCModel.solver.solve(problem)
        
        if exitflag != 1:             
            sys.stderr.write("exitflag = {}\n".format(exitflag))
            ctrl_cmd = vehicleCmd()
            ctrl_cmd.header.stamp = rospy.Time.now()
            ctrl_cmd.acceleration = -0.1
            target_steering = self.steering
            ctrl_cmd.steering = -1*target_steering  #-1*u_pred[1,0]*0.05+self.chassisState.steering
            self.control_pub.publish(ctrl_cmd)
            return
            
        temp = np.zeros((np.max(self.MPCModel.model.nvar), self.MPCModel.model.N))
        for i in range(0, self.MPCModel.model.N):
            temp[:, i] = output['x{0:02d}'.format(i+1)]
        u_pred = temp[0:2, :]
        x_pred = temp[2:7, :]
        # x_pred = temp[2:6, :]
        pred_maerker_refs = predicted_trj_visualize(x_pred)
        self.mpc_predicted_trj_publisher.publish(pred_maerker_refs)
        
        
        ctrl_cmd = vehicleCmd()
        ctrl_cmd.header.stamp = rospy.Time.now()
        ctrl_cmd.acceleration = u_pred[0,0]
        target_steering = (u_pred[1,0]*self.dt+self.steering)# temp[6, 3] 
        ctrl_cmd.steering = -target_steering        
        self.control_pub.publish(ctrl_cmd)

        ########  Log State Data ###################################################
        if self.logging:
            add_state = np.array([u_pred[0,0],u_pred[1,0],xinit[0],xinit[1], xinit[2], xinit[3], xinit[4]])                    
            self.dataloader.append_state(add_state,temp[:,1])     
            rospy.loginfo("recording process %d",self.dataloader.n_data_set)
        ########  Log State Data END ###################################################

    def cmd_callback(self,timer):
        
        current_euler = get_odom_euler(self.odom)
        current_euler[2] = wrap_to_pi(current_euler[2])
        
        local_vel = get_local_vel(self.odom, is_odom_local_frame = False)
        self.debug_msg.header.stamp = rospy.Time.now()
        self.debug_msg.pose.position.x = local_vel[0]
        self.debug_msg.pose.position.y = current_euler[2]        
        self.debug_pub.publish(self.debug_msg)        
        self.cur_x = np.transpose(np.array([self.odom.pose.pose.position.x,self.odom.pose.pose.position.y, local_vel[0], current_euler[2], self.steering]))
        

        if self.vehicle_status_available is False:
            rospy.loginfo("Vehicle status is not available yet")
            return
        elif self.odom_available is False:
            rospy.loginfo("Odom is not available yet")
            return
        elif self.TrajManager.is_path_computed() is False:
            rospy.loginfo("Waypoints are not available yet")
            return

        def _thread_func():
            self.run_mpc()            
        self.mpc_thread = threading.Thread(target=_thread_func(), args=(), daemon=True)
        self.mpc_thread.start()
        self.mpc_thread.join()
              
        

###################################################################################

def main():
    rospy.init_node("mpc_gp")
    env = rospy.get_param('~environment', default='gazebo')
    GPMPCWrapper(env)

if __name__ == "__main__":
    main()




 
    



        # msg.waypoints = msg.waypoints[1:-1]
        # if not self.waypoint_available:
        #     self.waypoint_available = True
        
        # self.x_ref = [msg.waypoints[i].pose.pose.position.x for i in range(len(msg.waypoints))]
        # self.y_ref = [msg.waypoints[i].pose.pose.position.y for i in range(len(msg.waypoints))]                        
        # quat_to_euler_lambda = lambda o: quaternion_to_euler([o[0], o[1], o[2], o[3]])            
        # self.psi_ref = [wrap_to_pi(quat_to_euler_lambda([msg.waypoints[i].pose.pose.orientation.w,msg.waypoints[i].pose.pose.orientation.x,msg.waypoints[i].pose.pose.orientation.y,msg.waypoints[i].pose.pose.orientation.z])[2]) for i in range(len(msg.waypoints))]                                    
        
        # self.vel_ref = [msg.waypoints[i].twist.twist.linear.x for i in range(len(msg.waypoints))]
 
        # while len(self.x_ref) < self.n_mpc_nodes:
        #     self.x_ref.insert(-1,self.x_ref[-1])
        #     self.y_ref.insert(-1,self.y_ref[-1])
        #     self.psi_ref.insert(-1,self.psi_ref[-1])
        #     self.vel_ref.insert(-1,self.vel_ref[-1])

        # self.ref_gen.set_traj(self.x_ref, self.y_ref, self.psi_ref, self.vel_ref)
        
        