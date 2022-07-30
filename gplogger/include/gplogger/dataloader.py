import numpy as np
import sys
import os

class DataLoader:
    def __init__(self, state_dim = 11, dt = 0.05):                               
        self.Xstates = np.empty((0,state_dim))
        self.XpredStates = np.empty((0,state_dim))
        self.dt = dt
        self.n_data_set = 0
        
    def file_load(self,file_path):
        data = np.load(file_path)
        self.Xstates = data['xstate']
        self.XpredStates = data['xpredState']
        
    def file_save(self,fil_dir):   
        np.savez(fil_dir,xstate = self.Xstates, xpredState = self.XpredStates)

    def append_state(self,xstate_,XpredStates_):
        self.Xstates = np.append(self.Xstates,[xstate_],axis = 0)        
        self.XpredStates = np.append(self.XpredStates,[XpredStates_],axis = 0)       
        self.n_data_set= self.n_data_set+1 

    