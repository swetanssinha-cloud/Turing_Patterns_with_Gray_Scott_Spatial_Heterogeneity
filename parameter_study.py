'''Multiple simulations'''




from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib.pyplot as plt
import GSw_tanhGraph as model


#N = 200
N = model.N
n_steps = 400 
num_simulations = 100 #run 500 simulations and average the results to get a smoother curve.
seeds = np.arange(num_simulations)

def dot_product(a, b):
    return a[0]*b[0] + a[1]*b[1]
def solve_for_theta(dVx, dVy):
    a = [1,0]

    dV = np.stack((dVx,dVy), axis = 0)

    dV_new = np.stack((dVy, -1 * dVx), axis = 0)

    resultant = dot_product(a, dV_new)


    cos_theta = resultant / (np.sqrt(dVx**2 + dVy**2) * np.sqrt(a[0]**2 + a[1]**2)) 

    # theta = np.arctan(dVy/dVx)

    #theta = np.arccos(cos_theta) * (180/np.pi) #convert to degrees


    '''this is where we transform cosine theta to cosine 2 theta and then to theta, which is the angle between the vector and the x-axis. '''

    cos_theta_squared = cos_theta**2

    cos_2_theta = 2 * cos_theta_squared - 1

    theta = 0.5 * np.arccos(cos_2_theta)

    theta = theta * (180/np.pi) #convert to degrees
    mean_theta_column = np.mean(theta, axis = 0)

    return mean_theta_column



def run_simulation(seed, Du, Dv, dx, dt, solver, first_derivative, F, k): 

    # np.random.seed(seed)

    # if seed % 5 == 0: #I want to keep the value at 10 - because I will be running 500 sims eventually
    #     print(f"Simulation {seed} finished")


    # U = np.ones((N,N))
    # V = np.zeros((N,N))
    # r = 20
    # U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.75 #orginally was 0.50
    # V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50 #orginally was 0.25
    # U += 0.05 * np.random.rand(N,N)
    # V += 0.05 * np.random.rand(N,N)


    # for i in range(n_steps):
    #     for _ in range(100):
    #         U, V = solver(U, V, Du, Dv, F, k, dx, dt)

    # dVx = first_derivative(V, 1)
    # dVy = first_derivative(V, 0)

    '''what chat'''
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
    return solve_for_theta(dVx, dVy)

    '''what chat said'''



if __name__=='__main__':

    p_values = [30, 40, 50, 60, 70]
    s_values = np.linspace(0, 1, 5) 
    seeds = np.arange(num_simulations)
    F = model.F
    k = model.k
    solver = model.solver
    first_derivative = model.first_derivative

    plt.figure(figsize=(8, 6))


    #for p1 in p_values:
    for s_val in s_values:
        #params = model.init_params(p1) for changing width of inner
        params = model.init_params(50, s_value=s_val) #for changing steepness of the tanh curve. I am keeping p1 constant at 50 for this.
        Du = params['Du']; Dv = params['Dv']; dx = params['dx']; dt = params['dt']; x = params['x']

        # build list of argument tuples for starmap (one tuple per simulation)
        args = [(int(s), Du, Dv, dx, dt, solver, first_derivative, F, k) for s in seeds]

        with Pool(max(1, cpu_count()-1)) as pool:
            # use starmap so each tuple is unpacked into run_simulation(...)
            results = pool.starmap(run_simulation, args)

        results = np.array(results)  # shape (num_simulations, len(x))

        # window smoothing / average
        window_size = 20
        if results.shape[0] >= window_size:
            n_windows = results.shape[0] - window_size + 1
            theta_to_plot = np.zeros((n_windows, results.shape[1]))
            for i in range(n_windows):
                theta_to_plot[i] = results[i:i+window_size].mean(axis=0)
            mean_theta = theta_to_plot.mean(axis=0)
        else:
            mean_theta = results.mean(axis=0)

        #plt.plot(x, mean_theta, label=f"p1={p1}")
        plt.plot(x, mean_theta, label=f"s={s_val:.2f}")

    plt.xlabel('X')
    plt.ylabel('Mean Theta (degrees)')
    plt.title('Mean Theta vs X')
    plt.legend()
    plt.savefig('/Users/Shared/Brandeis Coding/Functions/Final/Turring_Patterns/mean_theta_with_sharpness_change.png', dpi=300, bbox_inches='tight')
    plt.show()

    '''what I had before is below'''
    # '''changes'''
    # p_values = [30,40,50,60,70]
    # seeds = np.arange(num_simulations) #chat suggestion
    # F = model.F
    # k = model.k
    # solver = model.solver
    # first_derivative = model.first_derivative

    # plt.figure(figsize=(8,6))


    # for p1 in p_values:
    #     params = model.init_params(p1)
    #     Du = params['Du']; Dv = params['Dv']; dx = params['dx']; dt = params['dt']; x = params['x']
    # '''changes'''

    # args = [(int(s), Du, Dv, dx, dt, solver, first_derivative, F, k) for s in seeds] #chat suggestion


    # with Pool(cpu_count()-1) as pool:
    #     results = pool.map(run_simulation, args)

    #     results = np.array(results)

    #     window_size = 20
    #     n_runs = results.shape[0]
    #     if n_runs >= window_size:
    #         n_windows = n_runs - window_size + 1
    #         theta_to_plot = np.zeros((n_windows, results.shape[1]))
    #         for i in range(n_windows):
    #             theta_to_plot[i] = results[i:i+window_size].mean(axis=0)
    #         mean_theta = theta_to_plot.mean(axis=0)   # average the windowed curves
    #     else:
    #         # fewer runs than window -> just average all runs
    #         mean_theta = results.mean(axis=0)


    #     mean_theta = np.mean(theta_to_plot, axis=0)
    #     plt.plot(x, mean_theta, label=f"p1={p1}")


    # plt.xlabel('X')
    # plt.ylabel('Mean Theta (degrees)')
    # plt.title('Mean Theta vs X')
    # plt.savefig('/Users/Shared/Brandeis Coding/Functions/Final/Turring_Patterns/mean_theta.png', dpi=300, bbox_inches='tight')
    # plt.show()
    

        



