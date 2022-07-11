import numpy as np

import casadi as cs
import sys
import os
import rospkg
import enum
import math 
import pickle
import torch 
import torch.utils.data
import random
import torch.optim.lr_scheduler as lr_scheduler
import gpytorch
from gpytorch.mlls import SumMarginalLogLikelihood
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

import _pickle as pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score


rospack = rospkg.RosPack()
# pkg_dir = rospack.get_path('mpc_gp')
pkg_dir = "/home/hjpc/research/offroad_ws/src/mpc_gp"


from scipy.linalg import solve_triangular
def CasadiRBF(X, Y, model):
    """ RBF kernel in CasADi
    """
    sX = X.shape[0]
    sY = Y.shape[0]    
    length_scale = model.kernel_.get_params()['k1__k2__length_scale'].reshape(1,-1)
    constant = model.kernel_.get_params()['k1__k1__constant_value']
    X = X / cs.repmat(length_scale, sX , 1)
    Y = Y / cs.repmat(length_scale, sY , 1)
    dist = cs.repmat(cs.sum1(X.T**2).T,1,sY) + cs.repmat(cs.sum1(Y.T**2),sX,1) - 2*cs.mtimes(X,Y.T)
    K = constant*cs.exp(-.5 * dist)
    return K

def CasadiConstant(X, Y, model):
    """ Constant kernel in CasADi
    """
    constant = model.kernel_.get_params()['k2__constant_value']
    sX = X.shape[0]
    sY = Y.shape[0]
    K = constant*cs.SX.ones((sX, sY))
    return K

def CasadiMatern(X, Y, model):
    """ Matern kernel in CasADi
    """
    length_scale = model.kernel_.get_params()['k2__length_scale'].reshape(1,-1)
    constant = model.kernel_.get_params()['k1__constant_value']
    nu = model.kernel_.get_params()['k2__nu']

    sX = X.shape[0]
    sY = Y.shape[0]
    X = X / cs.repmat(length_scale, sX , 1)
    Y = Y / cs.repmat(length_scale, sY , 1)
    dist = cs.repmat(cs.sum1(X.T**2).T,1,sY) + cs.repmat(cs.sum1(Y.T**2),sX,1) - 2*cs.mtimes(X,Y.T)

    if nu == 0.5:
        K = constant*cs.exp(-dist**0.5)
    elif nu == 1.5:
        K = np.sqrt(3)*dist**0.5
        K = constant*(1. + K) * cs.exp(-K)
    elif nu == 2.5:
        K = np.sqrt(5)*dist**0.5
        K = constant*(1. + K + 5/3*dist) * cs.exp(-K)
    else:
        raise NotImplementedError
    return K
    
def loadGPModel(name, model, xscaler, yscaler, kernel='RBF'):
    """ GP mean and variance as casadi.SX variable
    """
    X = model.X_train_
    x = cs.SX.sym('x', 1, X.shape[1])

    # mean
    if kernel == 'RBF':
        K1 = CasadiRBF(x, X, model)
        K2 = CasadiConstant(x, X, model)
        K = K1 + K2
    elif kernel == 'Matern':
        K = CasadiMatern(x, X, model)
    else:
        raise NotImplementedError

    y_mu = cs.mtimes(K, model.alpha_) + model._y_train_mean
    # y_mu = y_mu * yscaler.scale_ + yscaler.mean_
    y_mu = y_mu * 1.0 + 0.0

    # variance
    L_inv = solve_triangular(model.L_.T,np.eye(model.L_.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    if kernel == 'RBF':
        K1_ = CasadiRBF(x, x, model)
        K2_ = CasadiConstant(x, x, model)
        K_ = K1_ + K2_
    elif kernel == 'Matern':
        K_ = CasadiMatern(x, x, model)

    y_var = cs.diag(K_) - cs.sum2(cs.mtimes(K, K_inv)*K)
    y_var = cs.fmax(y_var, 0)
    y_std = cs.sqrt(y_var)
    y_std *= yscaler.scale_

    gpmodel = cs.Function(name, [x], [y_mu, y_std])
    return gpmodel


class GPModel:
    def __init__(self,dt = 0.05, data_file_name = "test_data.npz", model_file_name = "GP_.pth"):
        self.SAVE_MODELS = True
        self.dt = dt
        data_dir = pkg_dir+"/data/"+data_file_name
        self.model_file_name = model_file_name                
        data = np.load(data_dir)
        xstate = data['xstate']
        xpredState = data['xpredState']
        #                                        x(2),     y(3),     psi(4),    vx(5),   vy(6),  omega(7), delta(8)
# add_state = np.array([u_pred[0,0],u_pred[1,0],xinit[0],xinit[1], xinit[2], xinit[3], xinit[4], xinit[5], xinit[6]])                    
        pred_states = xpredState[0:-1,[5,6,7]]
        true_states = xstate[1:,[5,6,7]]
        self.err_states = true_states - pred_states 
        self.X_train = np.empty([len(self.err_states[:,0]), 6])                
        for i in range(len(self.err_states[:,0])):            
            self.X_train[i,:] = xstate[i,[0,1,5,6,7,8]]
        
        ############# for Vx
        
        
    def all_model_build_and_save(self):
        self.y_train = self.err_states[:,0].reshape(-1,1)  
        model_save_dir = pkg_dir+'/data/gp_data/GP_vx.pickle'      
        self.model_build_and_save(model_save_dir)
        print("Vx Model Saved")

        self.y_train = self.err_states[:,1].reshape(-1,1)  
        model_save_dir = pkg_dir+'/data/gp_data/GP_vy.pickle'      
        self.model_build_and_save(model_save_dir)
        print("Vy Model Saved")
        self.y_train = self.err_states[:,2].reshape(-1,1)  
        model_save_dir = pkg_dir+'/data/gp_data/GP_omega.pickle'      
        self.model_build_and_save(model_save_dir)
        print("Omega Model Saved")
        self.all_model_load()

    def all_model_load(self):
        vx_file_name = pkg_dir+'/data/gp_data/GP_vy.pickle'    
        vy_file_name = pkg_dir+'/data/gp_data/GP_vy.pickle'    
        omega_file_name = pkg_dir+'/data/gp_data/GP_omega.pickle'    
        self.vx_model, self.vx_xscalar, self.vx_yscalar = self.model_load(vx_file_name)
        print("Vx Model Loaded")
        self.vy_model, self.vy_xscalar, self.vy_yscalar = self.model_load(vy_file_name)
        print("Vy Model Loaded")
        self.omega_model, self.omega_xscalar, self.omega_yscalar = self.model_load(omega_file_name)
        print("Omega Model Loaded")


    def model_load(self,file_name):
        with open(file_name,'rb') as handle:
            (model, xscaler, yscaler) = pickle.load(handle)        
        return model, xscaler, yscaler

    def model_build_and_save(self,save_file_name):
        self.xscaler = StandardScaler()		
        self.yscaler = StandardScaler()
        self.xscaler.fit(self.X_train)
        self.yscaler.fit(self.y_train)

        x_train = self.X_train
        y_train = self.y_train
        k1 = 1.0*RBF(
            length_scale=np.ones(x_train.shape[1]),
            length_scale_bounds=(1e-5, 1e5),
            )
        k2 = ConstantKernel(0.1)
        kernel = k1 + k2
        self.model = GaussianProcessRegressor(
            alpha=1e-6, 
            kernel=kernel, 
            normalize_y=True,
            n_restarts_optimizer=10,
            )
        start = time.time()
        self.model.fit(x_train, y_train)
        end = time.time()
        # print('training time: %ss' %(end - start))        
        # print('final kernel: %s' %(self.model.kernel_))

        if self.SAVE_MODELS:
            with open(save_file_name, 'wb') as f:                
                pickle.dump((self.model, self.xscaler, self.yscaler), f)

    def get_casadi_gps(self):
        vxgp = loadGPModel('vx', self.vx_model, self.vx_xscalar, self.vx_yscalar)
        vygp = loadGPModel('vy', self.vy_model, self.vy_xscalar, self.vy_yscalar)
        omegagp = loadGPModel('omega', self.omega_model, self.omega_xscalar, self.omega_yscalar)
        return vxgp,vygp,omegagp

    def draw_output(self,y_predicted_mean,lower,upper,ground_truth= None):
        with torch.no_grad():
            Xaxis = np.linspace(0,0.05*len(y_predicted_mean[:,0]),len(y_predicted_mean[:,0]))
            # Initialize plot
            f, ax = plt.subplots(3)

            if torch.is_tensor(ground_truth):
                ground_truth = ground_truth.cpu()
            if torch.is_tensor(lower):
                lower = lower.cpu()
            if torch.is_tensor(upper):
                upper = upper.cpu()

            for i in range(len(ax)):
                if ground_truth is not None:
                    ax[i].plot(Xaxis, ground_truth[:,i], 'k*')
                # Plot predictive means as blue line
                ax[i].plot(Xaxis, y_predicted_mean[:,i], 'b')
                # Shade between the lower and upper confidence bounds
                ax[i].fill_between(Xaxis, lower[:,i], upper[:,i], alpha=0.5)
                # ax[i].set_ylim([-3, 3])
                if ground_truth is not None:
                    ax[i].legend(['true', 'Mean', 'Confidence'])
                else:
                    ax[i].legend(['Mean', 'Confidence'])

                
            plt.show()
  

# gpmodel.gp_eval(gpmodel.X_train)