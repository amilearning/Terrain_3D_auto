import numpy
import ctypes

name = "FORCESNLPsolver"
requires_callback = True
lib = "lib/libFORCESNLPsolver.so"
lib_static = "lib/libFORCESNLPsolver.a"
c_header = "include/FORCESNLPsolver.h"
nstages = 40

# Parameter             | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
params = \
[("xinit"               , "dense" , ""               , ctypes.c_double, numpy.float64, (  4,   1),    4),
 ("x0"                  , "dense" , ""               , ctypes.c_double, numpy.float64, (240,   1),  240),
 ("all_parameters"      , "dense" , ""               , ctypes.c_double, numpy.float64, ( 80,   1),   80),
 ("reinitialize"        , ""      , "FORCESNLPsolver_int", ctypes.c_int   , numpy.int32  , (  0,   1),    1)]

# Output                | Type    | Scalar type      | Ctypes type    | Numpy type   | Shape     | Len
outputs = \
[("x01"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x02"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x03"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x04"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x05"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x06"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x07"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x08"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x09"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x10"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x11"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x12"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x13"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x14"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x15"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x16"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x17"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x18"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x19"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x20"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x21"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x22"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x23"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x24"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x25"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x26"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x27"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x28"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x29"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x30"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x31"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x32"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x33"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x34"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x35"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x36"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x37"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x38"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x39"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6),
 ("x40"                 , ""      , ""               , ctypes.c_double, numpy.float64,     (  6,),    6)]

# Info Struct Fields
info = \
[("it", ctypes.c_int),
 ("res_eq", ctypes.c_double),
 ("rsnorm", ctypes.c_double),
 ("pobj", ctypes.c_double),
 ("solvetime", ctypes.c_double),
 ("fevalstime", ctypes.c_double),
 ("QPtime", ctypes.c_double)]

# Dynamics dimensions
#   nvar    |   neq   |   dimh    |   dimp    |   diml    |   dimu    |   dimhl   |   dimhu    
dynamics_dims = [
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0), 
	(6, 4, 0, 2, 3, 4, 0, 0)
]