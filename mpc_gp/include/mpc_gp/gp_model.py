import numpy as np

import casadi
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
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/gp')

rospack = rospkg.RosPack()
# pkg_dir = rospack.get_path('mpc_gp')
pkg_dir = "/home/hjpc/research/offroad_ws/src/mpc_gp"



class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,num_tasks):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class GPModel:
    def __init__(self,dt = 0.05, data_file_name = "test_data.npz", model_file_name = "GP_.pth"):
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
        err_states = true_states - pred_states 
        # self.X_train = xstate[0:-1,[0,1,5,6,7,8]]                
        # self.y_train = err_states
        
        self.X_train = np.empty([len(err_states[:,0]), 6])
        # self.y_train = np.empty([len(err_states[:,0]),1])
        self.y_train = err_states[:,0]
        for i in range(len(err_states[:,0])):            
            self.X_train[i,:] = xstate[i,[0,1,5,6,7,8]]
            
        self.X_train = torch.from_numpy(self.X_train).cuda()
        self.y_train = torch.from_numpy(self.y_train).cuda()
        # self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(self.y_train[0])).cuda().double()        
        self.likelihood =gpytorch.likelihoods.GaussianLikelihood().cuda().double()                
        self.training_iterations = 150
        model_save_dir = pkg_dir+"/data/gp_data"
        self.model_name = model_save_dir+'/GP_'+str(len(self.X_train[:,1]))+'.pth'
        # self.model_name = model_save_dir+'/GP_'+'.pth'
        self.model = ExactGPModel(self.X_train, self.y_train, self.likelihood).cuda().double()
        # self.model = MultitaskGPModel(self.X_train, self.y_train, self.likelihood, num_tasks = len(self.y_train[0])).cuda().double()

    def model_load(self):
        state_dict = torch.load(self.model_file_name)    
        self.model.load_state_dict(state_dict)
        print(f"Model has been loaded from : {self.model_file_name}")

    def model_train(self):     
        self.model.train()
        self.likelihood.train()
        # Use the Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model).cuda()
        step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        loss_set = []
        
        for i in range(self.training_iterations):
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = -mll(output, self.y_train)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, self.training_iterations, loss.item()))
            # writer.add_scalar('training loss', loss.item(), i)                        
            optimizer.step()
            step_lr_scheduler.step()
            loss_set.append(loss.item())

        self.model.eval()
        self.likelihood.eval()
        
        # train_rms = mean_squared_error(y_train, y_train_predicted, squared=False)
        # print(f'train_rms = {train_rms}')

        ## Save trained data    
        state_dict = self.model.state_dict()
        for param_name, param in self.model.named_parameters():
            param_items = param.cpu().detach().numpy()
            print(f'Parameter name: {param_name:42} value = {param_items}')
        torch.save(self.model.state_dict(), self.model_name)        

        print(f"Model has been saved as : {self.model_name}")



    def gp_eval(self,X_test):
        if not torch.is_tensor(X_test):
          X_test = torch.from_numpy(X_test).cuda()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():        
        # This contains predictions for both outcomes as a list        
            start_time = time.time()
            predictions = self.likelihood(self.model(X_test))
            end_time = time.time()
            print("--- %s seconds ---" % (end_time - start_time))
            mean = predictions.mean.cpu()
            lower, upper = predictions.confidence_region()
        y_predicted_mean = mean.numpy()        
        return y_predicted_mean
        # self.draw_output(y_predicted_mean,lower,upper,ground_truth=self.y_train)

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
  


gpmodel = GPModel()
gpmodel.model_train()
# gpmodel.gp_eval(gpmodel.X_train)