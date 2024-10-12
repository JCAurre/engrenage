#rhodiagnostics.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.grid import *
from source.tensoralgebra import *
from source.mymatter import *


def get_rho_diagnostic(solutions_over_time, t, grid: Grid) :

    start = time.time()
    
    # For readability
    r = grid.r
    N = grid.num_points
    
    rho = []
    num_times = int(np.size(solutions_over_time) / (NUM_VARS * N))
    
    # unpack the vectors at each time
    for i in range(num_times) :
        
        t_i = t[i]
        
        if(num_times == 1):
            solution = solutions_over_time
        else :
            solution = solutions_over_time[i]

        state = solution.reshape(NUM_VARS, -1)

        # Assign the variables to parts of the solution
        (
            u, v ,
            phi, hrr, htt, hpp,
            K, arr, att, app,
            lambdar, shiftr, br, lapse,
        ) = state
              
        # Calculate some useful quantities
        ########################################################
        h = np.array([hrr, htt, hpp])
        em4phi = np.exp(-4.0*phi)  
        bar_gamma_UU = get_inverse_metric(r, h)



        # First derivatives
        first_derivative_indices = [
            idx_u,
        ]
        dstate_dr = grid.get_first_derivative(state, first_derivative_indices)

        (
            du_dr
        ) = dstate_dr[first_derivative_indices]

        # Matter sources
        matter_rho            = get_rho(u, du_dr, v, bar_gamma_UU, em4phi )
        rho_i = matter_rho[0]
        
        # Add the rho value to the output
        rho.append(rho_i)
        
    # end of iteration over time  
    #########################################################################
    
    end = time.time()
    #print("time at t= ", t_i, " is, ", end-start)
    
    return rho
