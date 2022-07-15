import numpy as np
from regex import P
import forcespro
import forcespro.nlp
import casadi
import sys
import os
import rospkg

rospack = rospkg.RosPack()
pkg_dir = rospack.get_path('mpc_gp')



class GPMPCModel:
    def __init__(self, model_build = False, N = 20, dt = 0.05, Q = None, R = None, solver_name = "MPCGPSOLVER", point_reference=False,gpmodel = None):
        self.gpmodel = gpmodel
        if self.gpmodel is not None:            
            self.vxgp, self.vygp, self.omegagp = self.gpmodel.get_casadi_gps()
            self.xscale = self.gpmodel.vx_xscalar
            self.xss = self.xscale.scale_
            self.xsm = self.xscale.mean_
        solver_dir = "/home/hjpc/.ros/FORCESNLPsolver"
        self.N = N
        self.dt = dt
        self.lr = 0.23
        self.lf = 0.34
        self.mass = 25.0
  
        try:
            # solver_dir = pkg_dir+"/FORCESNLPsolver"
            self.load_model()
            if(model_build):
                self.model_build()            
            else:
                self.solver = forcespro.nlp.Solver.from_directory(solver_dir)                        
        except:            
            print("prebuilt solver not found")            
            self.model_build()  
        
        # self.x0i = np.array([0.,0.,-1.5,1.5,1.,np.pi/4.])
        # self.x0 = np.transpose(np.tile(self.x0i, (1, self.model.N)))
        # self.xinit = np.transpose(np.array([-2.,0.,0.,np.deg2rad(90)]))
        # self.problem = {"x0": self.x0,
        #     "xinit": self.xinit}
    # def eval_gp(self,x_test):
    #     # x_test_numpy = 
    #     y_predicted_mean = self.gp_model.gp_eval(x_test)
    #     ##return  e_vx, e_vy, e_omega
    #     return y_predicted_mean[0][0], y_predicted_mean[0][1], y_predicted_mean[0][2]

    def traversability_cost(self,z):
        A = np.zeros(1e3)
        A[2] = 1e5        
        cost =0.0 
        if z[2] > 1.0:
            cost = 0.1        
        return cost 

    def running_obj(self,z,p):
        # z_states = x(2), y(3), psi(4), vx(5), vy(6), omega(7), delta(8), u = acc(0), delta_rate(1)        
        return 1 * casadi.fabs(z[2] -p[0]) + 1 * casadi.fabs(z[3] - p[1])  +10 * casadi.fabs(z[4] - p[2])+ 10* z[0]**2+ 100* z[1]**2
        
    def terminal_obj(self,z,p):
        return 2 * casadi.fabs(z[2] -p[0]) + 2 * casadi.fabs(z[3] - p[1]) +20 * casadi.fabs(z[4] - p[2])

    def fake_function(self,p):
        p = p-2+32-p**2
        return p
    def setState(self,x_np_array):
        self.xinit = np.transpose(x_np_array)
        self.problem["xinit"] = self.xinit

    def setParam(self,params_np_array):
        params_np_array = np.array([2.5, 2.5])        
        self.problem["all_parameters"] = np.transpose(np.tile(params_np_array,(1,self.model.N)))

    def load_model(self):
        self.model = forcespro.nlp.SymbolicModel()
        self.model.N = self.N # horizon length
        self.model.nvar = 9  # number of variables
        self.model.neq = 7  # number of equality constraints
        # self.model.nh = 1  # number of inequality constraint functions
        self.model.npar = 3 # number of runtime parameters    
        # Objective function        
        self.model.objective = self.running_obj 
        self.model.objective = self.terminal_obj
        
        # We use an explicit RK4 integrator here to discretize continuous dynamics
        integrator_stepsize = self.dt
        self.model.eq = lambda z,p: forcespro.nlp.integrate(self.continuous_dynamics, z[2:9], z[0:2],p,
                                                    integrator=forcespro.nlp.integrators.RK4,
                                                    stepsize=integrator_stepsize)
        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        self.model.E = np.concatenate([np.zeros((7,2)), np.eye(7)], axis=1)

        # Inequality constraints
        # Simple bounds
        #  upper/lower variable bounds lb <= z <= ub
        #                     inputs                 |  states        
        #                         accel,  delta_rate,   x(2), y(3), psi(4), vx(5), vy(6), omega(7), delta(8)
        self.model.lb = np.array([-4.,  np.deg2rad(-40.),  -np.inf,   -np.inf,   -np.inf,  -np.inf, -np.inf,-np.inf, -0.437])
        self.model.ub = np.array([+1.5,  np.deg2rad(+40.),   np.inf,   np.inf,    np.inf,    np.inf,  np.inf, np.inf, 0.437])
        # #                     a          delta                x            y     v             theta        
        # self.model.lb = np.array([-2.,  np.deg2rad(-25.),  -np.inf,   -np.inf,   -np.inf,  -np.inf])
        # self.model.ub = np.array([+1.5,  np.deg2rad(+25.),   np.inf,   np.inf,    np.inf,   np.inf])
        # General (differentiable) nonlinear inequalities hl <= h(z,p) <= hu
        # self.model.ineq = lambda z,p:  casadi.vertcat((z[2] -p[2]) ** 2 + (z[3] - p[3]) ** 2)
        # Upper/lower bounds for inequalities
        # self.model.hu = np.array([+np.inf])
        # self.model.hl = np.array([1.0**2])
        # Initial condition on vehicle states x
        self.model.xinitidx = range(2,9) # use this to specify on which variables initial conditions
       
        

    def model_build(self):
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 400     # Maximum number of iterations
        codeoptions.printlevel = 0  
        codeoptions.optlevel = 0    # 0 no optimization, 1 optimize for size, 
        #                             2 optimize for speed, 3 optimize for size & speed        
        codeoptions.cleanup = False
        codeoptions.timing = 1
        codeoptions.nlp.hessian_approximation = 'bfgs'
        codeoptions.solvemethod = 'SQP_NLP' # choose the solver method Sequential  
        codeoptions.nlp.bfgs_init = 1.5*np.identity(9) # initialization of the hessian
        #                             approximation
        # codeoptions.noVariableElimination = 1.       
        # codeoptions.sqp_nlp.reg_hessian = 100 # increase this if exitflag=-8
        codeoptions.sqp_nlp.reg_hessian = 5e-9 # increase this if exitflag=-8        
        # Creates code for symbolic model formulation given above, then contacts 
        # server to generate new solver
        self.solver = self.model.generate_solver(options=codeoptions)

    def continuous_dynamics(self,x, u,p):
        """Defines dynamics of the car, i.e. equality constraints.
        parameters:
        state x = [xPos,yPos,v,theta,delta]
        input u = [F,phi]
        """
        # set physical constants
        l_r = self.lr # distance rear wheels to center of gravitiy of the car
        l_f = self.lf # distance front wheels to center of gravitiy of the car
        m = self.mass   # mass of the car

        # set parameters
        # beta = casadi.arctan(l_r/(l_f + l_r) * casadi.tan(x[4]))        

#          u[0]accel,  u[1]delta_rate,   x[0]x(2), x[1]y(3), x[2]psi(4), x[3]vx(5), x[4]vy(6), x[5]omega(7), x[6]delta(8)
        # calculate dx/dt

        # gp_input = np.array([[1,2,3,4,5,6]])
        if self.gpmodel is not None:            
            gpinput = (casadi.vertcat(u[0],u[1],x[3],x[4],x[5],x[6]).T - self.xsm.reshape(1,-1)) / self.xss.reshape(1,-1)
        
            dxdt = casadi.vertcat(x[3]*casadi.cos(x[2])-x[4]*casadi.sin(x[2]),  # x
                                x[3]*casadi.sin(x[2])+x[4]*casadi.cos(x[2]),  # y 
                                x[5],                                         # psi       
                                u[0]+self.vxgp(gpinput)[0],                                         # vx 
                                (l_r/(l_f+l_r))*(u[1]*x[3]+x[6]*u[0])+self.vygp(gpinput)[0],        # vy  
                                (1.0/(l_r+l_f))*(u[1]*x[3]+x[6]*u[0])+self.omegagp(gpinput)[0],        # omega                   
                                u[1])                           # ddelta/dt = phi
        else:
            dxdt = casadi.vertcat(x[3]*casadi.cos(x[2])-x[4]*casadi.sin(x[2]),  # x
                                x[3]*casadi.sin(x[2])+x[4]*casadi.cos(x[2]),  # y 
                                x[5],                                         # psi       
                                u[0],                                         # vx 
                                (l_r/(l_f+l_r))*(u[1]*x[3]+x[6]*u[0]),        # vy  
                                (1.0/(l_r+l_f))*(u[1]*x[3]+x[6]*u[0]),        # omega                   
                                u[1])                       

        return dxdt