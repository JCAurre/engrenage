# oscillatoninitialconditions.py

# set the initial conditions for all the variables for an oscillaton
# see further details in https://github.com/GRChombo/engrenage/wiki/Running-the-oscillaton-example

from source.uservariables import *
from source.tensoralgebra import *
from source.Grid import *
import numpy as np
from scipy.interpolate import interp1d

def get_initial_state(a_grid, params) :
    
    # For readability
    r = a_grid.r_vector
    N = a_grid.num_points_r
    dx = a_grid.base_dx
    
    initial_state = np.zeros(NUM_VARS * N)
    [u,v,phi,hrr,htt,hpp,K,arr,att,app,lambdar,shiftr,br,lapse] = np.array_split(initial_state, NUM_VARS)
    
    
    a = params[0]
    b = params[1]
    c = params[2]
    def f_u(rad):
        return a * np.tanh((rad - b) / c)
    def df_u(rad):
        return a / np.cosh((rad - b) / c)**2.0
    def f_v(rad):
        return 0

    # set the (non zero) scalar field values
    u[:] = f_u(r)
    v[:] = f_v(r)
    
    mass = 0.0
    # solve Hamiltonian constraint, 2109.04896 eqn (14)
    def ham(r, y):
        psi, dpsi = y
        return [dpsi, -2.0 / r * dpsi - np.pi * psi * df_u(r)**2.0 
                                      - np.pi * psi**5.0 * f_v(r)**2.0
                                      - 2.0 * np.pi * psi**5.0 * mass * f_u(r)**2.0]

    y0 = [1, 0]
    from scipy.integrate import solve_ivp
    R = np.linspace(1e-10,r[-1],1000000)
    psi_0_sol = solve_ivp(ham, (R[0], R[-1]), y0, t_eval=R).y[0,:]
    psi_0_interp   = interp1d(R, psi_0_sol)
    psi_0 = np.concatenate((np.ones(3), psi_0_interp(r[3:])))

    # lapse and spatial metric
    # lapse[:] = f_lapse(r)
    grr = psi_0**4
    gtt_over_r2 = grr
    gpp_over_r2sintheta = gtt_over_r2
    phys_gamma_over_r4sin2theta = grr * gtt_over_r2 * gpp_over_r2sintheta

    # Work out the rescaled quantities
    # Note sign error in Baumgarte eqn (2), conformal factor
    phi[:] = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
    em4phi = np.exp(-4.0*phi)
    hrr[:] = em4phi * grr - 1.0
    htt[:] = em4phi * gtt_over_r2 - 1.0
    hpp[:] = em4phi * gpp_over_r2sintheta - 1.0
    
    # pre-collapsed lapse
    lapse[:] = em4phi/em4phi[-1]
    # overwrite inner cells using parity under r -> - r
    a_grid.fill_inner_boundary(initial_state)
    
    dhrrdx     = np.dot(a_grid.derivatives.d1_matrix, hrr)
    dhttdx     = np.dot(a_grid.derivatives.d1_matrix, htt)
    dhppdx     = np.dot(a_grid.derivatives.d1_matrix, hpp)

    # assign lambdar values
    h_tensor = np.array([hrr, htt, hpp])
    a_tensor = np.array([arr, att, app])
    dhdr   = np.array([dhrrdx, dhttdx, dhppdx])
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_metric(r, h_tensor)
    bar_gamma_UU = get_inverse_metric(r, h_tensor)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_connection(r, bar_gamma_UU, bar_gamma_LL, h_tensor, dhdr)
    lambdar[:]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    a_grid.fill_outer_boundary_ivar(initial_state, idx_lambdar)

    # overwrite inner cells using parity under r -> - r
    a_grid.fill_inner_boundary_ivar(initial_state, idx_lambdar)
            
    return initial_state
