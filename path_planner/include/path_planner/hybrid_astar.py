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

class Node:
    """ Hybrid A* tree node. """

    def __init__(self, grid_pos, pos):

        self.grid_pos = grid_pos
        self.pos = pos
        self.g = None
        self.g_ = None
        self.f = None
        self.parent = None
        self.phi = 0
        self.m = None
        self.branches = []

    def __eq__(self, other):

        return self.grid_pos == other.grid_pos
    
    def __hash__(self):

        return hash((self.grid_pos))


class HybridAstar(Astar):
    def __init__(self):
        super().__init__()         
        self.l_r = 0.45
        self.l_f = 0.45
        self.vehicle_length = self.l_r + self.l_f
        self.dt = 0.1
        self.max_delta = 0.38
        self.thetas = get_discretized_thetas(np.pi/6)
        
        
  

    def construct_node(self, pos):
        """ Create node for a pos. """

        theta = pos[2]
        pt = pos[:2]

        theta = round_theta(theta % (2*np.pi), self.thetas)
        cell_id = self.pose2idx(pt)
        
        grid_pos = [cell_id] + [theta]

        node = Node(grid_pos, pos)

        return node


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
        output = np.sqrt(((position[0] - target[0]) ** 2) + ((position[1] - target[1]) ** 2)+(math.radians(position[2]) - math.radians(target[2])) ** 2)
        return float(output)

    def path_plan(self):
        plan_start_time = time.time() 
        steering_inputs = np.linspace(-25,25,9)
        steering_inputs = steering_inputs*np.pi/180.0
        cost_steering_inputs= abs(np.linspace(-1,1,9))*0.1
        
        speed_inputs = np.linspace(0.5,1.5,5)
        cost_speed_inputs = np.linspace(1,0,5)

        # start and end are in discrete 
        start_grid_idx = self.pose2idx(self.start_pose) 
        start_d = self.idx2pose(start_grid_idx) 
        start = (start_d[0],start_d[1],round_theta(self.start_pose[2] % (2*np.pi), self.thetas))
        # start = (round(self.start_pose[0]), round(self.start_pose[1]), round_theta(self.start_pose[2] % (2*np.pi), self.thetas))
        end_grid_idx = self.pose2idx(self.goal_pose) 
        end_d = self.idx2pose(end_grid_idx) 
        end = (end_d[0],end_d[1],round_theta(self.goal_pose[2] % (2*np.pi), self.thetas))
        # end = (round(self.goal_pose[0]), round(self.goal_pose[1]), round_theta(self.goal_pose[2] % (2*np.pi), self.thetas))        
        open_heap = [] # element of this list is like (cost,node_d)
        open_diction={} #  element of this is like node_d:(cost,node_c,(parent_d,parent_c))        
        visited_diction={} #  element of this is like node_d:(cost,node_c,(parent_d,parent_c))                
        cost_to_neighbour_from_start = 0

        hq.heappush(open_heap,(cost_to_neighbour_from_start + dist2d(self.start_pose, self.goal_pose),start))
        
        open_diction[start]=(cost_to_neighbour_from_start + dist2d(self.start_pose, self.goal_pose), start,(start,start))

        while len(open_heap) > 0:
            accumulated_time = time.time() -plan_start_time
            if accumulated_time > 10:
                print("Planning takes too much time: {:.5f}".format( accumulated_time))          
                final_path=[]
                return final_path
            chosen_d_node =  open_heap[0][1]
            chosen_node_total_cost=open_heap[0][0]
            chosen_c_node=open_diction[chosen_d_node][1]
            visited_diction[chosen_d_node]=open_diction[chosen_d_node]

            if self.euc_dist(chosen_d_node,end)<1:
                
                rev_final_path=[end] # reverse of final path
                node=chosen_d_node
                m=1
                while m==1:
                    visited_diction
                    open_node_contents=visited_diction[node] # (cost,node_c,(parent_d,parent_c))                   
                    parent_of_node=open_node_contents[2][1]
                    
                    rev_final_path.append(parent_of_node)
                    node=open_node_contents[2][0]
                    if node==start:
                        rev_final_path.append(start)
                        break
                final_path=[]
                for p in rev_final_path:
                    final_path.append(p)
                return final_path
            
            hq.heappop(open_heap)
            for i in range(len(steering_inputs)):
                for j in range(len(speed_inputs)):                    
                    delta=steering_inputs[i]
                    velocity=speed_inputs[j]
                    cost_to_neighbour_from_start =  chosen_node_total_cost-self.euc_dist(chosen_d_node, end)
                    loop_count = 10
                    neighbour_cts = self.model_step(chosen_c_node,delta,velocity,loop_count)        
                    
                    neighbour_grid_idx = self.pose2idx(neighbour_cts) 
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
            #a=a+1
            #print(open_set_sorted)













# # get array indices of start and goal
#         start_idx = self.pose2idx(self.start_pose)
#         goal_idx = self.pose2idx(self.goal_pose)
        


#         # add start node to front
#         # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
#         start_node_cost = 0
#         start_node_estimated_cost_to_goal = dist2d(self.start_pose, self.goal_pose) + start_node_cost
#         front = [(start_node_estimated_cost_to_goal, start_node_cost, start_idx, None)]

#         # use a dictionary to remember where we came from in order to reconstruct the path later on
#         came_from = {}

#         # get possible movements
        
#         movements = self._get_movements_8n()
        
        

#         # while there are elements to investigate in our front.
#         while front:
#             # get smallest item and remove from front.
#             element = heappop(front)
#             total_cost, cost, pos_idx, previous_idx = element
#             pos = self.idx2pose(pos_idx)
#             # now it has been visited, mark with cost                        
#             if self.grid_visited[pos_idx] == 1.0:
#                 continue
#             # now it has been visited, mark with cost
#             self.grid_visited[pos_idx] = 1.0            
#             # set its previous node
#             came_from[pos_idx] = previous_idx
            
#             # if the goal has been reached, we are done!
#             if pos_idx == goal_idx:
#                 break
            
            
#             # check all neighbors
#             for dx, dy, deltacost in movements:                
#                 # determine new position
#                 new_x = pos[0] + dx
#                 new_y = pos[1] + dy
#                 new_pos = (new_x, new_y)

#                 # check whether new position is inside the map
#                 # if not, skip node
#                 new_pose_idx = self.pose2idx(new_pos)
#                 if  new_pose_idx < 0:
#                     continue

#                 # add node to front if it was not visited before 
#                 if self.grid_visited[new_pose_idx] < 1.0:
#                     # traversability cost 
#                     potential_function_cost =  (1/(self.trav_map.data[new_pose_idx]+1e-5))*self.traversability_weight                              
#                     new_cost = cost + deltacost + potential_function_cost
#                     new_total_cost_to_goal = new_cost + dist2d(new_pos, self.goal_pose) + potential_function_cost

#                     heappush(front, (new_total_cost_to_goal, new_cost, new_pose_idx, pos_idx))

#         # reconstruct path backwards (only if we reached the goal)
#         path = []
#         path_idx = []
#         if pos_idx == goal_idx:
#             while pos_idx:
#                 path_idx.append(pos_idx)
#                 # transform array indices to meters
#                 pos_m_x, pos_m_y = self.idx2pose(path_idx[-1])                
#                 path.append((pos_m_x, pos_m_y))
#                 pos_idx = came_from[pos_idx]

#             # reverse so that path is from start to goal.
#             path.reverse()
#             path_idx.reverse()

#         return path, path_idx