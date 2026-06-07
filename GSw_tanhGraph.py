import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

'''Trying out the way to make p1 changable'''

def init_params(p1_value, s_value):
    p1 = p1_value
    s_local = s_value if s_value is not None else s
    p2 = L - p1
    x = np.linspace(0, L, N, endpoint=False)
    input1 = s_local * (x - p1)
    input2 = s_local * -1 * (x - p2)
    tanfunc = scale * ((height - 1) / 2 * (2 - np.tanh(input1) - np.tanh(input2)) + 1)
    Du = np.full((N, N), tanfunc)
    Dv = (scale / 2) * np.ones_like(Du)
    dx = L / N
    dt = ((dx) ** 2) / (4 * Du.max()) - 0.1
    return dict(p1=p1, p2=p2, x=x, tanfunc=tanfunc, Du=Du, Dv=Dv, dx=dx, dt=dt, s=s_local)

F, k =  0.046, 0.064 #(Stripes) #0.028, 0.064 (Dots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200
dx = L / N
height = 3
dy = dx
n_steps =  400 #10000 #Using 300 for video. 
# p1 = 50. #normally what I use
# p2 = 150

p1 = 50
p2 = 200-p1
s = .1 #controls the shappness of the tanh function - keep s = 0.1  for normal use. 
x = np.linspace(0,L,N,endpoint=False)

scale = 0.16



input1 = s * (x - p1) #so, this creates a bunch of values between the two boundries and creates them into inputs for the tanh functions
input2 =  s * -1 * (x - p2)
tanfunc =   scale * ((height - 1)/2 * (2-np.tanh(input1) - np.tanh(input2)) +1) #now that we have a bunch of values between these two boundries, we can use the tanh function to create a bunch of diffusion constants between those two boundries
Du =  np.full((N,N), tanfunc) #Now a bunch of diffusion constants are created between those two boundries which all change. Each is slightly different, but the main change is at 25 and 75. So after those, the 
#diffusion constants are much lower. Currently, with sharpess = 0.1, and pi = 25 with p2 = 75, the diffusion constants are between 0 and 2. So we can define Dv later with the same process, maybe just saying that Dv = 2 * Du
Dv =  (scale/2) * np.ones_like(Du)  # Dv = 0.08 For now I have placed Dv = Du/2 because that is the ratio we have been using so far. I still don't know why works, but it does.

dt = ((dx)**2)/(4 * Du.max())-0.1 #This is the time step, which is based on the diffusion constant. This is a function for the time step. 

# print('Du max', Du.max())


# print('Du', Du)
# print('Dv', Dv)

def first_derivative(M, axis):
    firstD = (np.roll(M, 1, axis) - np.roll(M, -1, axis)) / (2 * dx)
    return firstD
        
def combined_derivative(D, Concentration):
    dD_dx = first_derivative(D, 0)
    dU_dx = first_derivative(Concentration, 0)
    dD_dy = first_derivative(D, 1)
    dU_dy = first_derivative(Concentration, 1)
    return dD_dx * dU_dx + dD_dy * dU_dy


def laplacian(Z, dx):
    d2Z_dx2 = np.roll(Z, 1, 0) - 2 * Z + np.roll(Z, -1, 0)
    d2Z_dy2 = np.roll(Z, 1, 1) - 2 * Z + np.roll(Z, -1, 1) 

    derivative = (d2Z_dx2 + d2Z_dy2)/(dx * dx) 
    # should this not be (d2Z_dx2/dx * dx) + (d2Z_dy2/dy * dy)? I think we are saying that 
    #dy = dx. So it doesn't matter and we combine like terms. 
    return derivative

def solver(U, V, Du, Dv, F, k, dx, dt):
    Lu = laplacian(U, dx)
    Lv = laplacian(V, dx)

    uvv = U * V * V

    U += (Du * Lu + combined_derivative(Du,U) - uvv + F * (1 - U)) * dt
    V += (Dv * Lv + combined_derivative(Dv,V) + uvv - (F+k) * V) * dt
    #Here is forward euler's method for the next time step. 
    return U, V
U = np.ones((N,N))
V = np.zeros((N,N))


r = 20
U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.75 #orginally was 0.50
V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50 #orginally was 0.25
U += 0.05 * np.random.rand(N,N)
V += 0.05 * np.random.rand(N,N)


if __name__ =='__main__':
    fig, ax = plt.subplots(figsize=(8,6)) #regular sizing in a cube

    im = ax.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=0.5)
    plt.axis('on')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('[V]')

    tanh_y = (tanfunc - tanfunc.min()) / (tanfunc.max() - tanfunc.min()) * (N - 1)

    # x-coordinates: pixel indices (0 to N-1)
    x_coords = np.arange(N)

    # Overlay line: static sine curve
    tanh_line, = ax.plot(x_coords, tanh_y, color='red', linewidth=3, alpha=0.6)

    # Set axes to avoid autoscaling that might hide the line
    ax.set_xlim(0, N-1)
    ax.set_ylim(0, N-1)

    ax.set_title('Du as a Function of Space (Hyperbolic Tangent)')

    def update(frame, U, V, Du, Dv, F, k, dx, dt, tanh_line):
        for _ in range(100):
            U, V = solver(U, V, Du, Dv, F, k, dx, dt)
        im.set_array(V)
        return [im, tanh_line]


    ani = animation.FuncAnimation(fig, update, frames=n_steps, interval = 0.1, blit=True, fargs=(U, V, Du, Dv, F, k, dx, dt, tanh_line))
    #ani.save('Gray_Scott_Animation.mp4', writer='ffmpeg', fps=30, dpi=300) #Save the animation as a video file
    plt.show()

