'''Multiple simulations'''

from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
import GSw_tanhGraph as model


# =========================
# PARAMETERS
# =========================

N = model.N
n_steps = 400
num_simulations = 100

seeds = np.arange(num_simulations)


# =========================
# THETA CALCULATION
# =========================

def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1]


def solve_for_theta(dVx, dVy):

    a = [1, 0]

    dV_new = np.stack((dVy, -1 * dVx), axis=0)

    resultant = dot_product(a, dV_new)

    # ADD EPSILON TO PREVENT DIVIDE BY ZERO
    magnitude = np.sqrt(dVx**2 + dVy**2) + 1e-8

    cos_theta = resultant / magnitude

    cos_theta_squared = cos_theta**2

    cos_2_theta = 2 * cos_theta_squared - 1

    theta = 0.5 * np.arccos(cos_2_theta)

    theta = theta * (180 / np.pi)

    # Average over y-direction
    mean_theta_column = np.mean(theta, axis=0)

    return mean_theta_column


# =========================
# SINGLE SIMULATION
# =========================

def run_simulation(seed, Du, Dv, dx, dt, solver,
                   first_derivative, F, k):

    np.random.seed(int(seed))

    if seed % 10 == 0:
        print(f"Simulation {seed} started")

    # Initial conditions
    U = np.ones((N, N))
    V = np.zeros((N, N))

    r = 20

    U[N//2-r:N//2+r, N//2-r:N//2+r] = 0.75
    V[N//2-r:N//2+r, N//2-r:N//2+r] = 0.50

    U += 0.05 * np.random.rand(N, N)
    V += 0.05 * np.random.rand(N, N)

    # Run simulation
    for i in range(n_steps):

        for _ in range(100):
            U, V = solver(U, V, Du, Dv, F, k, dx, dt)

    # Compute derivatives
    dVx = first_derivative(V, 1)
    dVy = first_derivative(V, 0)

    # Return theta(x)
    return solve_for_theta(dVx, dVy)


# =========================
# MAIN
# =========================

if __name__ == '__main__':

    # Different sharpness values
    s_values = np.linspace(0, 1, 5)

    F = model.F
    k = model.k

    solver = model.solver
    first_derivative = model.first_derivative

    plt.figure(figsize=(8, 6))

    # Loop over sharpness values
    for s_val in s_values:

        print(f"\nRunning s = {s_val:.2f}")

        # Get model parameters
        params = model.init_params(50, s_value=s_val)

        Du = params['Du']
        Dv = params['Dv']
        dx = params['dx']
        dt = params['dt']
        x = params['x']

        # Build arguments for multiprocessing
        args = [(int(seed), Du, Dv, dx, dt, solver, first_derivative, F, k) for seed in seeds]

        # Run simulations in parallel
        with Pool(max(1, cpu_count()-1)) as pool:

            results = pool.starmap(run_simulation, args)

        # Convert to numpy array
        results = np.array(results)

        # =====================================
        # CORRECT ENSEMBLE AVERAGE
        # =====================================

        # Average across simulations
        mean_theta = np.mean(results, axis=0)

        # =====================================
        # OPTIONAL SPATIAL SMOOTHING
        # =====================================

        window_size = 20

        kernel = np.ones(window_size) / window_size

        mean_theta_smoothed = np.convolve(mean_theta, kernel, mode='same')

        # Plot
        plt.plot(x, mean_theta_smoothed, label=f"s={s_val:.2f}")

    # =========================
    # FINAL PLOT
    # =========================

    plt.xlabel('X')
    plt.ylabel('Mean Theta (degrees)')
    plt.title('Mean Theta vs X')

    plt.legend()

    plt.savefig(
        '/Users/Shared/Brandeis Coding/Functions/Final/Turring_Patterns/mean_theta_with_sharpness_change.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()