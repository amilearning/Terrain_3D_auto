import numpy as np
import sys
import os
import math
from heapq import heappush, heappop
from path_planner.utils import dist2d

class Astar:
    def __init__(self):
            # solver_dir = pkg_dir+"/FORCESNLPsolver"
        self.start_pose = None
        self.goal_pose = None
        self.trav_map = None
                
        self.grid_map = None
        self.c_size = None
        self.r_size = None
        self.map_resolution = None
        self.movement = '8N'
        self.grid_visited = None

    def set_map(self,map, map_info):
        self.trav_map = map
        self.trav_map.data = np.where(np.isnan(self.trav_map.data),0,self.trav_map.data)
        self.grid_visited = np.zeros([len(self.trav_map.data)])
        self.c_size  = self.trav_map.layout.dim[0].size
        self.r_size  = self.trav_map.layout.dim[1].size
        self.map_info = map_info
        self.map_resolution = self.map_info.resolution         
        

    def set_pose(self,pose):
        self.start_pose = pose

    def set_goal(self,goal):
        self.goal_pose = goal
    
    def _get_movements_4n(self):    
        return [(self.map_info.resolution, 0, self.map_info.resolution),
                (0, self.map_info.resolution, self.map_info.resolution),
                (-self.map_info.resolution, 0, self.map_info.resolution),
                (0, -self.map_info.resolution, self.map_info.resolution)]

    def _get_movements_8n(self):
        s2 = self.map_info.resolution*math.sqrt(2)
        return [(self.map_info.resolution, 0, self.map_info.resolution),
                (0, self.map_info.resolution, self.map_info.resolution),
                (-self.map_info.resolution, 0, self.map_info.resolution),
                (0, -self.map_info.resolution, self.map_info.resolution),
                (self.map_info.resolution, self.map_info.resolution, s2),
                (-self.map_info.resolution, self.map_info.resolution, s2),
                (-self.map_info.resolution, -self.map_info.resolution, s2),
                (self.map_info.resolution, -self.map_info.resolution, s2)]
        
    
    def idx2pose(self,idx):        
        # top right is 0 - bottom left is last            
        assert idx < self.r_size*self.c_size, "idx is out of bound"                    
        grid_r = int(idx/(self.r_size))
        grid_c = (idx - grid_r*self.r_size)
        pose_x = self.map_info.pose.position.x+self.c_size/2*self.map_resolution-grid_c*self.map_resolution
        pose_y = self.map_info.pose.position.y+self.r_size/2*self.map_resolution-grid_r*self.map_resolution
        return [pose_x, pose_y]
        
        
    def pose2idx(self,pose):    
        right_corner_x = self.map_info.pose.position.x + self.map_info.length_x/2
        right_corner_y = self.map_info.pose.position.y + self.map_info.length_y/2

        grid_c_idx = (int)((right_corner_x - pose[0]) / self.map_resolution)
        grid_r_idx = (int)((right_corner_y - pose[1]) / self.map_resolution)
        if grid_c_idx >= self.c_size:
            return -1
        if grid_r_idx >= self.r_size:
            return -1
        
        idx = grid_c_idx + grid_r_idx*self.r_size 
        if idx >= self.c_size*self.r_size:
            return -1        
         
        return idx
    
    

    def path_plan(self):
# get array indices of start and goal
        start_idx = self.pose2idx(self.start_pose)
        goal_idx = self.pose2idx(self.goal_pose)
        
        # add start node to front
        # front is a list of (total estimated cost to goal, total cost from start to node, node, previous node)
        start_node_cost = 0
        start_node_estimated_cost_to_goal = dist2d(self.start_pose, self.goal_pose) + start_node_cost
        front = [(start_node_estimated_cost_to_goal, start_node_cost, start_idx, None)]

        # use a dictionary to remember where we came from in order to reconstruct the path later on
        came_from = {}

        # get possible movements
        if self.movement == '4N':
            movements = self._get_movements_4n()
        elif self.movement == '8N':
            movements = self._get_movements_8n()
        else:
            raise ValueError('Unknown movement')

        # while there are elements to investigate in our front.
        while front:
            # get smallest item and remove from front.
            element = heappop(front)
            total_cost, cost, pos_idx, previous_idx = element
            pos = self.idx2pose(pos_idx)
            # now it has been visited, mark with cost                        
            if self.grid_visited[pos_idx] == 1.0:
                continue
            # now it has been visited, mark with cost
            self.grid_visited[pos_idx] = 1.0            
            # set its previous node
            came_from[pos_idx] = previous_idx
            
            # if the goal has been reached, we are done!
            if pos_idx == goal_idx:
                break
            
            
            # check all neighbors
            for dx, dy, deltacost in movements:                
                # determine new position
                new_x = pos[0] + dx
                new_y = pos[1] + dy
                new_pos = (new_x, new_y)

                # check whether new position is inside the map
                # if not, skip node
                new_pose_idx = self.pose2idx(new_pos)
                if  new_pose_idx < 0:
                    continue

                # add node to front if it was not visited before 
                if self.grid_visited[new_pose_idx] < 1.0:
                    # traversability cost 
                    potential_function_cost =  0.0 #self.trav_map.data[new_pose_idx]                              
                    new_cost = cost + deltacost + potential_function_cost
                    new_total_cost_to_goal = new_cost + dist2d(new_pos, self.goal_pose) + potential_function_cost

                    heappush(front, (new_total_cost_to_goal, new_cost, new_pose_idx, pos_idx))

        # reconstruct path backwards (only if we reached the goal)
        path = []
        path_idx = []
        if pos_idx == goal_idx:
            while pos_idx:
                path_idx.append(pos_idx)
                # transform array indices to meters
                pos_m_x, pos_m_y = self.idx2pose(path_idx[-1])                
                path.append((pos_m_x, pos_m_y))
                pos_idx = came_from[pos_idx]

            # reverse so that path is from start to goal.
            path.reverse()
            path_idx.reverse()

        return path, path_idx