import numpy as np
from regex import P
import sys
import os
import rospkg
import math
from gplogger.utils import b_to_g_rot
rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('mpc_gp')



class VehicleModel:
    def __init__(self, dt = 0.05):
        self.m = 25
        self.L = 0.9
        self.Lr = 0.45
        self.Lf = 0.45
        self.Caf = self.m * self.Lf/self.L * 0.5 * 0.35 * 180/math.pi
        self.Car = self.m * self.Lr/self.L * 0.5 * 0.35 * 180/math.pi
        self.Izz = self.Lf*self.Lr*self.m
        self.g= 9.814195
        self.h = 0.15
        self.dt = dt
        

    def compute_slip(self, x,u):
        clip_vx = max(1,x[3])
        alpha_f = u[0] - (x[4]+self.Lf*x[5])/clip_vx
        alpha_r = (-x[4]+self.Lr*x[5])/clip_vx
        return alpha_f, alpha_r
      
    def compute_normal_force(self,x,u,roll,pitch):        
        Fzf = self.Lr*self.m*self.g*math.cos(pitch)*math.cos(roll)/self.L + self.h*self.m/self.L*(u[1]+self.g*math.sin(pitch))
        Fzr = self.Lr*self.m*self.g*math.cos(pitch)*math.cos(roll)/self.L - self.h*self.m/self.L*(u[1]+self.g*math.sin(pitch))
        return Fzf, Fzr
    
    def predict_multistep(self,x,u,N):
        x_ = x.copy()
        XpredStates = np.empty((0,len(x_)))
        XpredStates = np.append(XpredStates,[x_],axis = 0)       
        for i in range(N):            
            x_ = self.dynamics_update(x_,u)
            x_[3] = max(0.0, x_[3])
            XpredStates = np.append(XpredStates,[x_],axis = 0)                   
        return XpredStates

    def dynamics_update(self,x,u):     
        # x(0), y(1), psi(2), vx(3), vy(4), wz(5) z(6) roll(7) pitch(8)                  
        # u(0) = delta, u(1) = ax 
        nx = x.copy()
        roll = x[7]
        pitch = x[8]
        delta = u[0]
        axb = u[1]
        rot_base_to_world = b_to_g_rot(roll,pitch,x[2])
        Fzf, Fzr = self.compute_normal_force(x,u,roll,pitch)
        alpha_f, alpha_r = self.compute_slip(x,u)
        Fyf = Fzf * alpha_f            
        Fyr =  Fzr * alpha_r
        # Fyf = self.Caf  * alpha_f            
        # Fyr = self.Car  * alpha_r
        vel_in_world = np.dot(rot_base_to_world,[x[3],x[4],0])
        vxw = vel_in_world[0]
        vyw = vel_in_world[1]
        vzw = vel_in_world[2]    
        
        nx[0] = nx[0]+self.dt*vxw
        nx[1] = nx[1]+self.dt*vyw
        nx[2] = nx[2]+self.dt*(math.cos(roll)/(math.cos(pitch)+1e-10)*x[5])
        nx[3] = nx[3]+self.dt*axb
        nx[4] = nx[4]+self.dt*((Fyf+Fyr+self.m*self.g*math.cos(pitch)*math.sin(roll))/self.m-x[3]*x[5])
        nx[5] = nx[5]+self.dt*((Fyf*self.Lf*math.cos(delta)-self.Lr*Fyr)/self.Izz)
        nx[6] = nx[6]+self.dt*vzw
    
        return nx


