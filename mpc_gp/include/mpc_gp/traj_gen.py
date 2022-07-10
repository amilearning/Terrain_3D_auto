import numpy as np
from regex import P
import forcespro
import forcespro.nlp
import casadi
import sys
import os
import math

class TrajManager:
    def __init__(self, MPCModel = None, dt = 0.05, n_sample = 20):                
        self.MPCModel = MPCModel
        self.xstate = None
        self.dt = dt     
        self.n_sample = n_sample
        self.traj = None
    
    def setState(self,state):
        self.xstate = state

    def gen_traj(self, delta, velocity):
        tmp_state = self.xstate 
        trajs= np.empty((0,len(self.xstate)))
        trajs = np.append(trajs,[self.xstate],axis=0)        
        # cum_dist = 0.0        
        for i in range(self.n_sample*10):
            tmp_state = np.copy(trajs[-1,:]) 
            beta = math.atan(self.MPCModel.lr/(self.MPCModel.lf + self.MPCModel.lr) * math.tan(delta))
            tmp_state[0] = trajs[-1,0] + self.dt*velocity*math.cos(tmp_state[3]+beta)
            tmp_state[1] = trajs[-1,1] + self.dt*velocity*math.sin(tmp_state[3]+beta)
            tmp_state[2] = velocity
            tmp_state[3] = trajs[-1,3] + self.dt*velocity/self.MPCModel.lr * math.sin(beta)
            
            # if i > 0:                
            #     cum_dist = cum_dist +math.sqrt((trajs[-1,0]-tmp_state[0])**2+(trajs[-1,1]-tmp_state[1])**2)
            trajs = np.append(trajs, [tmp_state],axis=0)
            # if cum_dist >  traj_length:
            #     break
        self.traj = trajs
        return trajs
    
    def find_closest_idx(self,ref_point):
        xypoints = self.traj[:,0:2]
        diff =  xypoints - ref_point
        diff =  np.transpose(diff)
        squared_diff = np.power(diff,2)
        squared_dist = squared_diff[0,:] + squared_diff[1,:]
        return np.argmin(squared_dist)

    def extract_path_points(self,ref_pose,number_of_sample):
        idx = self.find_closest_idx(ref_pose)        
        return self.traj[idx+1:idx+number_of_sample+1,[0,1,3]]

    def is_path_computed(self):
        if self.traj is None:
            return False
        else:
            return True
        