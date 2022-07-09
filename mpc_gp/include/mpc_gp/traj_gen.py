import numpy as np
from regex import P
import forcespro
import forcespro.nlp
import casadi
import sys
import os
import math

class TrajManager:
    def __init__(self, MPCModel = None, dt = 0.05):                
        self.MPCModel = MPCModel
        self.xstate = None
        self.dt = dt        
    
    def setState(self,state):
        self.xstate = state

    def randome_traj(self, delta, velocity,traj_length):
        tmp_state = self.xstate 
        trajs= np.array((0,len(self.xstate)))
        trajs = np.append(trajs, self.xstate,axis=0)
        cum_dist = 0.0
        for i in range(len(1e3)):
            tmp_state = np.copy(trajs[-1,:]) 
            beta = math.atan(self.MPCModel.l_r/(self.MPCModel.l_f + self.MPCModel.l_r) * math.tan(delta))
            tmp_state[0] = trajs[-1,0] + self.dt*velocity*math.cos(tmp_state[3]+beta)
            tmp_state[1] = trajs[-1,1] + self.dt*velocity*math.sin(tmp_state[3]+beta)
            tmp_state[2] = velocity
            tmp_state[3] = trajs[-1,3] + self.dt*velocity/self.MPCModel.l_r * math.sin(beta)
            
            if i > 0:                
                cum_dist = cum_dist +math.sqrt((trajs[i,0]-tmp_state[0])**2+(trajs[i,1]-tmp_state[1])**2)
            if cum_dist >  traj_length:
                return trajs
            else:
                trajs = np.append(trajs, tmp_state,axis=0)
        
        return trajs

    def disp_traj(self):
        print("disp")
