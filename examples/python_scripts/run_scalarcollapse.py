#!/usr/bin/env python
# coding: utf-8

# ScalarCollapse example
# see further details in https://github.com/GRChombo/engrenage/wiki/Running-the-black-hole-example

# load the required python modules
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import time
import sys
import random
from tqdm import tqdm
from scipy.interpolate import splrep, splev
import h5py
import os
import re
import argparse

# homemade source code from source folder
sys.path.append('../../source')

from initialdata.scalarcollapseinitialconditions import *
from backgrounds.sphericalbackground import *

from bssn.constraintsdiagnostic import *
from bssn.ahfinder import *
from core.rhsevolution import *
from core.grid import Grid
from core.spacing import *
from core.display import *
from core.statevector import *
from matter.scalarmatter import *
from matter.nomatter import *


# --- MPI ---
from mpi4py import MPI



def run_example(params_grid, params_matter, output_vars, output_dir, output_name):
    
    scalar_mu = 0
    r_max, min_dr, max_dr, T = params_grid[0], params_grid[1], params_grid[2], params_grid[3]

    # Setup of the grid
    params = LinearSpacing.get_parameters(r_max, min_dr)
    spacing = LinearSpacing(**params)
    # params = SinhSpacing.get_parameters(r_max, min_dr, max_dr)
    # spacing = SinhSpacing(**params)
    my_matter = NoMatter()
    my_state_vector = StateVector(my_matter)
    grid = Grid(spacing, my_state_vector)
    r = grid.r
    num_points = r.size
    my_matter = ScalarMatter(scalar_mu)
    my_state_vector = StateVector(my_matter)
    grid = Grid(spacing, my_state_vector)
    background = FlatSphericalBackground(r)

    # Set initial conditions
    initial_state = get_initial_state(grid, background, params_matter)

    # Time evolution
    rtol = 1e-5
    atol = 1e-12
    dt = 0.5 * min_dr #* np.exp(t_end)
    with tqdm(total=int(T/dt), unit="‰") as progress_bar:
        dense_solution = solve_ivp(get_rhs, [0,T], initial_state, 
                                args=(grid, background, my_matter, progress_bar, [0, T/int(T/dt)]),
                            atol=atol, rtol=rtol,
                            first_step = dt,
                            method='DOP853', dense_output=True)
    num_points_t = 512
    t = np.linspace(0, T, num_points_t)
    solution = dense_solution.sol(t).T
    
    # Compute diagnostics
    Ham, Mom = get_constraints_diagnostic(solution, t, grid, background, my_matter)
    omega, ah_radius, bh_mass = get_horizon_diagnostics(solution, t, grid, background, my_matter)
        
    # Save variables to h5 files
    print("Saving vars to "+output_name+".h5 file...")
    
    os.makedirs(output_dir, exist_ok=True)

    outfile = os.path.join(output_dir, output_name+".h5")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("t", data=t)
        f.create_dataset("r", data=grid.r)
        f.create_dataset("Ham", data=Ham)
        f.create_dataset("Mom", data=Mom)
        f.create_dataset("bh_mass", data=bh_mass)
        f.create_dataset("ah_radius", data=ah_radius)
    
        for matter_var_name in output_vars[0]:
            idx = getattr(my_matter, f"idx_{matter_var_name}")
            f.create_dataset(
                matter_var_name,
                data=solution[:, idx * num_points : (idx + 1) * num_points]
            )
        for metric_var_name in output_vars[1]:
            idx = globals()[f"idx_{metric_var_name}"]
            f.create_dataset(
                metric_var_name,
                data=solution[:, idx * num_points : (idx + 1) * num_points]
            )

        f.attrs["ampl"] = params_matter[0]
        f.attrs["wid"] = params_matter[1]
        f.attrs["r0"] = params_matter[2]
        f.attrs["Omega"] = params_matter[3]


    print("Code finished running")



# -------------------------
# Main MPI driver
# -------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# -------------------------
# Argument parsing (rank 0 only logically matters)
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--T", type=float, default=1e1)
parser.add_argument("--ampl", type=float, default=1e-4)
parser.add_argument("--wid", type=float, default=1.0)
parser.add_argument("--r0", type=float, default=1.0)
parser.add_argument("--Omega", type=float, default=1.0)
parser.add_argument("--r_max", type=float, default=10.0)
parser.add_argument("--min_dr", type=float, default=0.1)

args = parser.parse_args()

# --- grid parameters ---
r_max = comm.bcast(args.r_max if rank == 0 else None, root=0)
min_dr = comm.bcast(args.min_dr if rank == 0 else None, root=0)
max_dr = 1e-1
T = comm.bcast(args.T if rank == 0 else None, root=0)

params_grid = [r_max, min_dr, max_dr, T]

# --- output ---
output_vars = [{"u", "v"}, {"lapse", "phi", "K"}]
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# --- fixed matter parameter ---
ampl = comm.bcast(args.ampl if rank == 0 else None, root=0)
wid = comm.bcast(args.wid if rank == 0 else None, root=0)
r0 = comm.bcast(args.r0 if rank == 0 else None, root=0)
Omega = comm.bcast(args.Omega if rank == 0 else None, root=0)    

# --- parameter sweep ---
param_sweep = ampl*np.linspace(0.1, 10, size) # This is useful for multiple CPUs


# -------------------------
# MPI-safe run numbering
# -------------------------
if rank == 0:
    msg = f"Running ampl = {ampl} |  T = {T}"
    line = "=" * (len(msg) + 6)

    print(f"\n{line}")
    print(f"== {msg} ==")
    print(f"{line}\n")   
    # Find existing run_XXXX.h5 files
    pattern = re.compile(r"run_(\d{4})\.h5")
    existing_ids = set()
    for f in os.listdir(output_folder):
        m = pattern.match(f)
        if m:
            existing_ids.add(int(m.group(1)))

    # Generate enough free IDs
    free_ids = []
    i = 0
    while len(free_ids) < len(param_sweep):
        if i not in existing_ids:
            free_ids.append(i)
        i += 1
else:
    free_ids = None

# Broadcast free IDs to all ranks
free_ids = comm.bcast(free_ids, root=0)

# -------------------------
# One parameter per rank
# -------------------------
if rank < len(param_sweep):

    # Assign MPI-safe run ID
    run_id = free_ids[rank]
    output_name = f"run_{run_id:04d}"

    # Set matter parameters for this run
    u_ampl = param_sweep[rank]
    params_matter = [u_ampl, wid, r0, Omega]

    print(f"Rank {rank}: starting ampl={u_ampl}, run_id={run_id}")

    run_example(
        params_grid,
        params_matter,
        output_vars,
        output_folder,
        output_name
    )

    print(f"Rank {rank}: finished run {run_id}")

else:
    print(f"Rank {rank}: idle")