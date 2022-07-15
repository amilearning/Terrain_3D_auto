import numpy as np
import sys
import os
import math
from heapq import heappush, heappop
from path_planner.utils import dist2d
from path_planner.astar import Astar

class HybridAstar(Astar):
    

    def set_map(self,map, map_info):
        self.trav_map = map
        self.trav_map.data = np.where(np.isnan(self.trav_map.data),0,self.trav_map.data)
        self.grid_visited = np.zeros([len(self.trav_map.data)])
        self.c_size  = self.trav_map.layout.dim[0].size
        self.r_size  = self.trav_map.layout.dim[1].size
        self.map_info = map_info
        self.map_resolution = self.map_info.resolution         
        

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
        
        movements = self._get_movements_8n()
        
        

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
                    potential_function_cost =  (1/(self.trav_map.data[new_pose_idx]+1e-5))*self.traversability_weight                              
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