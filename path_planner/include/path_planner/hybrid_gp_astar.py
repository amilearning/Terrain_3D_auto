import numpy as np
import sys
import os
import math
import heapq as hq
import time
from heapq import heappush, heappop
from path_planner.utils import dist2d
from path_planner.astar import Astar
from path_planner.utils import euler_to_quaternion, quaternion_to_euler, unit_quat, get_odom_euler, wrap_to_pi, get_pose_euler
from path_planner.utils import get_discretized_thetas, round_theta

class HybridGPAstar(Astar):
    def __init__(self):
        super().__init__()         
        self.l_r = 0.45
        self.l_f = 0.45
        self.vehicle_length = self.l_r + self.l_f
        self.dt = 0.5
        self.max_delta = 0.38
        self.thetas = get_discretized_thetas(np.pi/3)
        self.traversability_weight =1000.0
        
      

    def model_step(self,pose,delta,velocity,loop_count):        
        #  beta = arctan(l_r/(l_f + l_r)*tan(delta))
        #  dxPos/dt = v*cos(theta + beta)
        #  dyPos/dt = v*sin(theta + beta)
        #  dtheta/dt = v/l_r*sin(beta)
        tmp_pose = list(pose)
        for i in range(loop_count):
            beta = math.atan(self.l_r/(self.l_r+self.l_f)*math.tan(delta))
            dx = velocity*math.cos(tmp_pose[2]+beta)
            dy = velocity*math.sin(tmp_pose[2]+beta)
            dtheta = velocity/self.l_r*math.sin(beta)
            
            tmp_pose[0] += self.dt*dx
            tmp_pose[1] += self.dt*dy
            tmp_pose[2] += self.dt*dtheta
            tmp_pose[2] = wrap_to_pi(tmp_pose[2])
        
        return tuple(tmp_pose)


    def set_pose(self,msg):        
        euler = get_odom_euler(msg)        
        yaw = wrap_to_pi(euler[2])
        self.start_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]

    def set_goal(self,msg):      
        euler = get_pose_euler(msg.pose)  
        yaw = wrap_to_pi(euler[2])        
        self.goal_pose = [msg.pose.position.x, msg.pose.position.y, yaw]

    def euc_dist(self, position, target):
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2)+0.1*(((position[2]+4*np.pi) - (target[2]+4*np.pi)) ** 2))
        return float(output)

    def get_z_value_from_path(self,path):
        path3d_with_theta = []
        for i in range(len(path)):
            pose_tmp = (path[i][0], path[i][1])
            new_pose_idx = self.pose2idx(pose_tmp)
            z_tmp = self.elev_map.data[new_pose_idx]
            path3d_with_theta.append((path[i][0],path[i][1],z_tmp,path[i][2]))        
        return path3d_with_theta
    
    def path_plan(self):
        
        plan_start_time = time.time() 
        steering_inputs = np.linspace(-35, 35, 9)
        steering_inputs = steering_inputs*np.pi/180.0
        cost_steering_inputs= abs(np.linspace(-1,1,9))
        
        speed_inputs =  np.linspace(0.5,0.5,1)
        cost_speed_inputs =  np.linspace(0,0,1)

        # start and end are in discrete 
        start_grid_idx = self.pose2idx(self.start_pose) 
        if start_grid_idx < 0:
            print("Invalid Start point, outside of map area")
            return
        start_d = self.idx2pose(start_grid_idx) 
        start = (start_d[0],start_d[1],round_theta(self.start_pose[2] % (2*np.pi), self.thetas))
        # start = (round(self.start_pose[0]), round(self.start_pose[1]), round_theta(self.start_pose[2] % (2*np.pi), self.thetas))
        end_grid_idx = self.pose2idx(self.goal_pose) 
        if end_grid_idx < 0:
            print("Invalid Goal point, outside of map area")
            return 
        end_d = self.idx2pose(end_grid_idx) 
        end = (end_d[0],end_d[1],round_theta(self.goal_pose[2] % (2*np.pi), self.thetas))
        # end = (round(self.goal_pose[0]), round(self.goal_pose[1]), round_theta(self.goal_pose[2] % (2*np.pi), self.thetas))        
        open_heap = [] # element of this list is like (cost,node_d)
        open_diction={} #  element of this is like node_d:(cost,node_c,(parent_d,parent_c))        
        visited_diction={} #  element of this is like node_d:(cost,node_c,(parent_d,parent_c))                
        cost_to_neighbour_from_start = 0

        hq.heappush(open_heap,(cost_to_neighbour_from_start + self.euc_dist(start, end),start))
        
        open_diction[start]=(cost_to_neighbour_from_start + self.euc_dist(start, end), start,(start,start))

        while len(open_heap) > 0:
            accumulated_time = time.time() -plan_start_time
            if accumulated_time > 20:
                print("Planning takes too much time: {:.5f}".format( accumulated_time))          
                final_path=[]
                return final_path
            chosen_d_node =  open_heap[0][1]
            chosen_node_total_cost=open_heap[0][0]
            chosen_c_node=open_diction[chosen_d_node][1]
            visited_diction[chosen_d_node]=open_diction[chosen_d_node]            
            if self.euc_dist(chosen_d_node,end)<0.5:                
                rev_final_path=[end] # reverse of final path
                node=chosen_d_node
                m=1
                while m==1:
                    # visited_diction
                    open_node_contents=visited_diction[node] # (cost,node_c,(parent_d,parent_c))                   
                    parent_of_node=open_node_contents[2][1]
                    
                    rev_final_path.append(parent_of_node)
                    node=open_node_contents[2][0]
                    if node==start:
                        rev_final_path.append(start)
                        break
                    
                rev_final_path.pop()
                rev_final_path.pop()
                rev_final_path.append(self.start_pose)
                rev_final_path.reverse()
                rev_final_path.pop()
                rev_final_path.append(self.goal_pose)
                final_path = []
                for p in rev_final_path:
                    final_path.append(p)  
                final_path = self.get_z_value_from_path(final_path)
                return final_path
            
            hq.heappop(open_heap)
            for i in range(len(steering_inputs)):
                for j in range(len(speed_inputs)):                    
                    delta=steering_inputs[i]
                    velocity=speed_inputs[j]
                    cost_to_neighbour_from_start =  chosen_node_total_cost-self.euc_dist(chosen_d_node, end)
                    loop_count = 4
                    neighbour_cts = self.model_step(chosen_c_node,delta,velocity,loop_count)        
                    
                    neighbour_grid_idx = self.pose2idx(neighbour_cts) 
                    if neighbour_grid_idx > 0:                        
                        neighbour_grid_pose = self.idx2pose(neighbour_grid_idx)                     
                        neighbour_x_d = neighbour_grid_pose[0]
                        neighbour_y_d = neighbour_grid_pose[1]
                        # neighbour_x_d = round(neighbour_cts[0])
                        # neighbour_y_d = round(neighbour_cts[1])
                        neighbour_theta_d = wrap_to_pi(round_theta(neighbour_cts[2] % (2*np.pi), self.thetas))
                        
                        neighbour = ((neighbour_x_d,neighbour_y_d,neighbour_theta_d),(neighbour_cts[0],neighbour_cts[1],neighbour_cts[2]))
                                
                        heurestic = self.euc_dist((neighbour_x_d,neighbour_y_d,neighbour_theta_d),end)
                        cost_to_neighbour_from_start = abs(velocity)+ cost_to_neighbour_from_start +\
                                                                        cost_steering_inputs[i] + cost_speed_inputs[j]
                        
                        #print(heurestic,cost_to_neighbour_from_start)
                        new_pose_idx = self.pose2idx(neighbour_cts)
                        potential_function_cost =  (1/(self.trav_map.data[new_pose_idx]+1e-5))*self.traversability_weight
                        total_cost = heurestic+cost_to_neighbour_from_start+potential_function_cost
                        
                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor
                                
                        skip=0
                        #print(open_set_sorted)
                        # If the cost of going to this successor happens to be more
                        # than an already existing path in the open list to this successor,
                        # skip this successor
                        found_lower_cost_path_in_open=0                     
                        if neighbour[0] in open_diction:
                            
                            if total_cost>open_diction[neighbour[0]][0]: 
                                skip=1
                                
                            elif neighbour[0] in visited_diction:
                                
                                if total_cost>visited_diction[neighbour[0]][0]:
                                    found_lower_cost_path_in_open=1
                                    
                            
                        if skip==0 and found_lower_cost_path_in_open==0:                        
                            hq.heappush(open_heap,(total_cost,neighbour[0]))
                            open_diction[neighbour[0]]=(total_cost,neighbour[1],(chosen_d_node,chosen_c_node))


    