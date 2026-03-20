"""
Set the initial conditions for all the variables for an isotropic Schwarzschild BH.

See further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example.
"""

import numpy as np

from core.grid import *
from bssn.bssnstatevariables import *
from bssn.tensoralgebra import *
from backgrounds.sphericalbackground import *
from matter.scalarmatter import *

from scipy.interpolate import splrep, splev
from scipy.integrate import solve_ivp

def get_initial_state(grid: Grid, background, params) :
    
    assert grid.NUM_VARS == 14, "NUM_VARS not correct for bssn + scalar field"
    
    # For readability
    r = grid.r
    N = grid.num_points
                     
    initial_state = np.zeros((grid.NUM_VARS, N))
    (
        phi,
        hrr,
        htt,
        hpp,
        K,
        arr,
        att,
        app,
        lambdar,
        shiftr,
        br,
        lapse,
        u, 
        v
    ) = initial_state

    # Set BH length scale
    GM = 0.0

    ampl = params[0]
    wid = params[1]
    r0 = params[2]
    Omega = params[3]
    def f_u(r):
        # return ampl * r**2 * np.exp(-((r - r0) / wid)**2) * np.cos(Omega * r)
        return ampl * np.exp(-0.5*((r-r0)/wid)**2) * np.cos(Omega * r)
    def df_u(r):
        gauss = np.exp(-((r - r0) / wid)**2)
        # return ampl * gauss * (2*r - 2*r**2 * (r - r0) / wid**2) * np.cos(Omega * r)  - Omega*np.sin(Omega*r) * ampl * r**2 * gauss
        return - ampl * np.exp(-0.5*((r - r0)/wid)**2) * ( Omega * np.sin(Omega * r) + ((r - r0) / (wid**2)) * np.cos(Omega * r) )

    u[:] = f_u(r)
    
    # solve Hamiltonian constraint, 2109.04896 eqn (14)
    def ham(r, y):
        psi, dpsi = y
        return [dpsi, -2.0 / r * dpsi - np.pi * psi * df_u(r)**2.0]

    y0 = [1, 0]
    R = np.linspace(1e-15, r[-1], 1000000)
    psi_0_sol = solve_ivp(ham, (R[0], R[-1]), y0, method='RK45', max_step=1e-1, t_eval=R).y[0,:]
    tck = splrep(R, psi_0_sol[:], k=4)  # 4th-order spline
    psi_0_interp = lambda r: splev(r, tck)
    psi_sol =  psi_0_interp(r[NUM_GHOSTS:])
    psi_bc = psi_sol[:NUM_GHOSTS][::-1]
    psi_0 = np.concatenate((psi_bc, psi_sol))
        
    # spatial metric
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
    lapse.fill(1.0)
    #lapse[:] = em4phi # optional, to pre collapse the lapse
    
    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state)
    
    # Set up matrices
    zeros = np.zeros_like(hrr)
    h_LL = np.array([[hrr, zeros, zeros],[zeros, htt, zeros],[zeros, zeros, hpp]])
    h_LL = np.moveaxis(h_LL, -1, 0) 
    first_derivative_indices = [idx_hrr, idx_htt, idx_hpp]
    dstate_dr = grid.get_first_derivative(initial_state, first_derivative_indices)
    (dhrr_dr, dhtt_dr, dhpp_dr) = dstate_dr[first_derivative_indices]
        
    # This is d h_ij / dx^k = dh_dx[x,i,j,k]
    d1_h_dx = np.zeros([N, SPACEDIM, SPACEDIM, SPACEDIM])
    d1_h_dx[:,i_r,i_r, i_r]  = dhrr_dr
    d1_h_dx[:,i_t,i_t, i_r]  = dhtt_dr
    d1_h_dx[:,i_p,i_p, i_r]  = dhpp_dr
        
    # (unscaled) \bar\gamma_ij and \bar\gamma^ij
    bar_gamma_LL = get_bar_gamma_LL(r, h_LL, background)
    bar_gamma_UU = get_bar_gamma_UU(r, h_LL, background)
        
    # The connections Delta^i, Delta^i_jk and Delta_ijk
    Delta_U, Delta_ULL, Delta_LLL  = get_tensor_connections(r, h_LL, d1_h_dx, background)
    lambdar[:]   = Delta_U[:,i_r]

    # Fill boundary cells for lambdar
    grid.fill_outer_boundary(initial_state, [idx_lambdar])

    # overwrite inner cells using parity under r -> - r
    grid.fill_inner_boundary(initial_state, [idx_lambdar])
            
    return initial_state.reshape(-1)
