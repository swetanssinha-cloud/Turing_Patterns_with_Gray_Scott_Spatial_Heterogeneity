"""
Load saved simulation data, perform analysis, and generate plots.
Does NOT run any simulations.
"""

'''
This file is used to take simulation data and analyze. Goal of this is to type which parameters are going to be analyzed.

Note: In order to look at a NEW parameter - you will have to run_sim again. 

If you want to change what you are analyzing (mean, values at certain positions, etc) do that here.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_simulation_data(base_dir='simulation_data'):
    """
    Load all simulation data from disk.
    
    Returns:
        dict: Keys are width values (e.g., 140, 120), values are dicts containing:
              - 'x': x-axis array
              - 'theta_curves': array of shape (num_simulations, len(x))
              - 'V_arrays': list of V arrays (for future analysis)
    """
    data = {}
    
    # Find all width subdirectories
    subdirs = glob.glob(os.path.join(base_dir, 'width_*'))
    
    for subdir in sorted(subdirs):
        # Extract width from directory name
        width_str = os.path.basename(subdir).replace('width_', '')
        width = float(width_str)
        
        # Load x array
        x = np.load(os.path.join(subdir, 'x.npy'))
        
        # Load all simulation files
        sim_files = sorted(glob.glob(os.path.join(subdir, 'simulation_*.npz')))
        
        theta_curves = []
        V_arrays = []
        
        for sim_file in sim_files:
            sim_data = np.load(sim_file)
            theta_curves.append(sim_data['theta'])
            V_arrays.append(sim_data['V'])
        
        # Convert to numpy array for easier manipulation
        theta_curves = np.array(theta_curves)
        
        data[width] = {
            'x': x,
            'theta_curves': theta_curves,
            'V_arrays': V_arrays
        }
        
        print(f"Loaded {len(sim_files)} simulations for width={width}")
    
    return data

def compute_moving_window_average(results, window_size=20):
    """
    Apply moving window averaging algorithm (unchanged from original).
    
    Args:
        results: array of shape (num_simulations, len(x))
        window_size: size of moving window
    
    Returns:
        mean_theta: averaged curve of shape (len(x),)
    """
    if results.shape[0] >= window_size:
        n_windows = results.shape[0] - window_size + 1
        theta_to_plot = np.zeros((n_windows, results.shape[1]))
        for i in range(n_windows):
            theta_to_plot[i] = results[i:i+window_size].mean(axis=0)
        mean_theta = theta_to_plot.mean(axis=0)
    else:
        mean_theta = results.mean(axis=0)
    
    return mean_theta

def plot_mean_theta_curves(data, window_size=20):
    """
    Recreate the original plot: overlaid mean theta curves for all widths.
    """
    plt.figure(figsize=(8, 6))
    
    # Sort by width for consistent plotting order
    for width in sorted(data.keys(), reverse=True):
        x = data[width]['x']
        theta_curves = data[width]['theta_curves']
        
        # Apply the same moving-window averaging as original code
        mean_theta = compute_moving_window_average(theta_curves, window_size)
        
        plt.plot(x, mean_theta, label=f"width={width:.2f}")
    
    plt.xlabel('X')
    plt.ylabel('Mean Theta (degrees)')
    plt.title('Mean Theta vs X')
    plt.legend()
    plt.show()

def analyze_standard_deviation(data, window_size=20):
    """
    Example of additional analysis: compute and plot standard deviation.
    This can be called without re-running simulations.
    """
    plt.figure(figsize=(8, 6))
    
    for width in sorted(data.keys(), reverse=True):
        x = data[width]['x']
        theta_curves = data[width]['theta_curves']
        
        # Compute standard deviation across simulations
        std_theta = np.std(theta_curves, axis=0)
        
        plt.plot(x, std_theta, label=f"width={width:.2f}")
    
    plt.xlabel('X')
    plt.ylabel('Std Dev of Theta (degrees)')
    plt.title('Standard Deviation of Theta vs X')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load all simulation data
    print("Loading simulation data...")
    data = load_simulation_data('simulation_data')
    
    # Recreate the original plot
    print("\nGenerating mean theta plot...")
    plot_mean_theta_curves(data, window_size=20)
    
    # Example: Add additional analysis without rerunning simulations
    # Uncomment to see standard deviation plot
    # print("\nGenerating standard deviation plot...")
    # analyze_standard_deviation(data, window_size=20)
    
    # Future analysis ideas (examples - not implemented):
    # - Confidence intervals
    # - Alternative orientation metrics using stored V arrays
    # - Spatial correlation analysis
    # - etc.