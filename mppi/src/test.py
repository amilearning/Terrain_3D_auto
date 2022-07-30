#!/usr/bin/env python
import rospy


""" ROS node for the MPC GP in 3d offroad environment, to use in the Gazebo simulator and real world experiments.
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""
# from cuda import cuda, nvrtc
import numpy as np
from numba import cuda  

import json
import time
import threading

import pandas as pd
import math 
import sys
from std_msgs.msg import Bool, Empty, Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped 
from hmcl_msgs.msg import Lane, Waypoint, vehicleCmd
from visualization_msgs.msg import MarkerArray, Marker
from autorally_msgs.msg import chassisState


map_data = np.ones([1,2])
###################################################################################
def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

@cuda.jit
def cudakernel0(array):
    thread_position = cuda.grid(1)
    array[thread_position]  +=0.5
    


def main():
    array = np.zeros(1024*1024, np.float32)
    print('Initial array:', array)
    gridsize = 1024
    blocksize = 1024
    print('Kernel launch: cudakernel0[1, 1](array)')
    cudakernel0[gridsize, blocksize](array)

    print('Updated array:',array)

if __name__ == "__main__":
    main()

