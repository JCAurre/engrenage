#rhodiagnostics.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.Grid import *
from source.tensoralgebra import *
from source.mymatter import *


def get_rho_diagnostic(solutions_over_time, t, my_grid) :

    start = time.time()
    
    # For readability
    r = my_grid.r_vector
    N = my_grid.num_points_r
    
    rho = []
    num_times = int(np.size(solutions_over_time) / (NUM_VARS * N))
    
    # unpack the vectors at each time
    for i in range(num_times) :
        
        t_i = t[i]
        
        if(num_times == 1):
            solution = solutions_over_time
        else :
            solution = solutions_over_time[i]

        # Assign the variables to parts of the solution
        (u, v , phi, hrr, htt, hpp, K, 
         arr, att, app, lambdar, shiftr, br, lapse) = np.array_split(solution, NUM_VARS)
              
        # Calculate some useful quantities
        ########################################################
        h = np.array([hrr, htt, hpp])
        em4phi = np.exp(-4.0*phi)  
        bar_gamma_UU = get_inverse_metric(r, h)
        dudx       = np.dot(my_grid.derivatives.d1_matrix, u      )      

        # Matter sources
        matter_rho            = get_rho(u, dudx, v, bar_gamma_UU, em4phi )
        rho_i = matter_rho
        
        # Add the rho value to the output
        rho.append(rho_i)
        
    # end of iteration over time  
    #########################################################################
    
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    return rho
