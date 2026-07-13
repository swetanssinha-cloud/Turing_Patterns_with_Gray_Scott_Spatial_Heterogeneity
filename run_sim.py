"""
Run Gray-Scott simulations and save raw data to disk.
Does NOT perform any analysis or plotting.
"""

'''
This is paramterstudy.py but just the simulations part. This file will do like the 100 x parameter number simulaitons 
and save the data
'''

from multiprocessing import Pool, cpu_count
import numpy as np
import os
import GSw_tanhGraph as model

# Simulation parameters
N = model.N
n_steps = 400 
num_simulations = 100
seeds = np.arange(num_simulations)

def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1]

def solve_for_theta(dVx, dVy):
    a = [1, 0]
    dV = np.stack((dVx, dVy), axis=0)
    dV_new = np.stack((dVy, -1 * dVx), axis=0)
    resultant = dot_product(a, dV_new)
    cos_theta = resultant / (np.sqrt(dVx**2 + dVy**2) * np.sqrt(a[0]**2 + a[1]**2))
    cos_theta_squared = cos_theta**2
    cos_2_theta = 2 * cos_theta_squared - 1
    theta = 0.5 * np.arccos(cos_2_theta)
    theta = theta * (180/np.pi)  # convert to degrees
    mean_theta_column = np.mean(theta, axis=0)
    return mean_theta_column

def run_simulation(seed, Du, Dv, dx, dt, solver, first_derivative, F, k): 
    """Run a single Gray-Scott simulation and return both V and theta."""
    np.random.seed(int(seed))
    if seed % 5 == 0:
        print(f"Simulation {seed} started")
    
    U = np.ones((N, N))
    V = np.zeros((N, N))
    r = 20
    U[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.75
    V[N//2 - r:N//2 + r, N//2 - r:N//2 + r] = 0.50
    U += 0.05 * np.random.rand(N, N)
    V += 0.05 * np.random.rand(N, N)

    for i in range(n_steps):
        for _ in range(100):
            U, V = solver(U, V, Du, Dv, F, k, dx, dt)
    
    dVx = first_derivative(V, 1)
    dVy = first_derivative(V, 0)
    theta = solve_for_theta(dVx, dVy)
    
    # Return both V and theta
    return V, theta

if __name__ == '__main__':
    # Parameter values to sweep
    p_values = [30, 40, 50, 60, 70]
    seeds = np.arange(num_simulations)
    F = model.F
    k = model.k
    solver = model.solver
    first_derivative = model.first_derivative

    # Create main output directory
    os.makedirs('simulation_data', exist_ok=True)

    for p1 in p_values:
        # Initialize parameters
        params = model.init_params(p1)
        Du = params['Du']
        Dv = params['Dv']
        dx = params['dx']
        dt = params['dt']
        x = params['x']
        
        # Create subdirectory for this parameter value
        width = 200 - 2 * p1
        subdir = f'simulation_data/width_{width:.0f}'
        os.makedirs(subdir, exist_ok=True)
        
        # Save the x array once (common to all simulations)
        np.save(os.path.join(subdir, 'x.npy'), x)
        
        print(f"\nRunning simulations for p1={p1} (width={width:.0f})...")
        
        # Build list of argument tuples for starmap
        args = [(int(s), Du, Dv, dx, dt, solver, first_derivative, F, k) for s in seeds]
        
        # Run simulations in parallel
        with Pool(max(1, cpu_count()-1)) as pool:
            results = pool.starmap(run_simulation, args)
        
        # Save each simulation to its own .npz file
        for sim_idx, (V, theta) in enumerate(results):
            filename = os.path.join(subdir, f'simulation_{sim_idx:03d}.npz')
            np.savez(filename, V=V, theta=theta, seed=seeds[sim_idx])
        
        print(f"Saved {len(results)} simulations to {subdir}/")
    
    print("\nAll simulations complete!")