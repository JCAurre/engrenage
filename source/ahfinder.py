#ahfinder.py

# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.grid import *
from source.tensoralgebra import *
from source.mymatter import *
from scipy import optimize
from scipy.interpolate import interp1d

# The diagnostic function returns the value of the expansion over the grid
# it takes in the solution of the evolution, which is the state vector at every
# time step, and returns the spatial profile theta(r) at each time step.
# In addition, it finds the zeros of such function, which define the apparent
# horizon, and are used to return the mass of such a horizon.
def find_massBH(solutions_over_time, t, grid: Grid) :

    start = time.time()
    
    # For readability
    r = grid.r
    N = grid.num_points
    
    theta = []
    ah_rad = []
    massBH = []
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

        # First derivatives
        first_derivative_indices = [
            idx_phi, idx_shiftr
        ]
        dstate_dr = grid.get_first_derivative(state, first_derivative_indices)

        (
            dphi_dr,
            dshiftr_dr,
        ) = dstate_dr[first_derivative_indices]

        # First derivatives - advec left and right along shiftr (left if shiftr < 0 and right if shiftr >= 0)
        advec_indices = [idx_phi]
        dstate_dr_advec = grid.get_advection(state, shiftr >= 0, advec_indices)

        (
            dphi_dr_advec,
        ) = dstate_dr_advec[advec_indices]

        # Calculate some useful quantities
        ########################################################

        # dphidx     = np.dot(my_grid.derivatives.d1_matrix, phi    )
        h = np.array([hrr, htt, hpp])
        a = np.array([arr, att, app])
        em4phi = np.exp(-4.0*phi)
        # dshiftrdx  = np.dot(my_grid.derivatives.d1_matrix, shiftr)
        # dphidx_advec_R     = np.dot(my_grid.derivatives.d_advec_matrix_right, phi    )
        bar_div_shift =  (dshiftr_dr + 2.0 * shiftr / r)

        dphidt = (- one_sixth * lapse * K + one_sixth * bar_div_shift) + shiftr * dphi_dr_advec

        # Expansion given by eqn. 7.22 in B&S
        theta_i = np.sqrt(2.0) / r * (2.0 * r / lapse * dphidt + (np.sqrt(em4phi) - shiftr / lapse) * (1.0 + 2.0 * r * dphi_dr))
        theta.append(theta_i)

        # find AH horizon and mass
        mintheta = min(theta[i][NUM_GHOSTS:])
        rmin = r[np.where(theta[i][:]==mintheta)][0]
        if mintheta > 0:
            ah_rad.append(0)
            massBH.append(0)
        else:
            th_interp = interp1d(r, r * theta[i])
            emphi_interp = interp1d(r, np.exp(-phi))
            ah_rad_i = optimize.brentq(th_interp, rmin, r[-1])
            ah_rad.append(ah_rad_i)
            massBH.append(0.5 * ah_rad[i] * emphi_interp(ah_rad[i])**-2)
        
    # end of iteration over time  
    #########################################################################
    
    end = time.time()

    return theta, massBH

