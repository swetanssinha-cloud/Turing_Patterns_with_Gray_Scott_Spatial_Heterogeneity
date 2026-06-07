import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


F, k =  0.046, 0.064 #(Stripes) #0.028, 0.064 (Dots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200 # 200 what? Im so confused what dimension this is in. 


dx = L / N # length of each distance step. 
# height = 5
dy = dx
n_steps =  200 #10000 #Using 300 for video. # number of time steps. 
# p1 = 50
# p2 = 150
# sharpness = 10 #this will control how sharp the chnage in concentration will be at the boundry p1 and p2
x = np.linspace(0,L,N,endpoint=False)
# print('x', x)
scale = 0.5 # keep scale = 0.16 nomrally
p = 3
freq = 1 * L

sinfunc =  scale * (np.sin((2 * np.pi * x)/freq)) + (p * scale)#This creates a sine wave that will be used to create diffusion constants that change over the x axis
#in last term change 2 to 1.5 and back and see differences. 


Du = np.full((N,N), sinfunc) #creates an array of diffusion constants that change over the x axis. 

Dv =  (scale/2) * np.ones_like(Du)  # Dv = 0.08 For now I have placed Dv = Du/2 because that is the ratio we have been using so far. I still don't know why works, but it does.

dt = ((dx)**2)/(4 * Du.max())-0.1 #This is the time step, which is based on the diffusion constant. This is a good time step for the diffusion constants we have created.

print('Du max', Du.max())


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
U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50 #orginally was 0.50
V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.25 #orginally was 0.25
U += 0.05 * np.random.rand(N,N)
V += 0.05 * np.random.rand(N,N)

fig, ax = plt.subplots(figsize=(8,6)) #regular sizing in a cube

im = ax.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=1)
plt.axis('on')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('[V]')

sine_y = (sinfunc - sinfunc.min()) / (sinfunc.max() - sinfunc.min()) * (N - 1)

# x-coordinates: pixel indices (0 to N-1)
x_coords = np.arange(N)

# Overlay line: static sine curve
sine_line, = ax.plot(x_coords, sine_y, color='white', linewidth=1.5, alpha=0.6)

# Set axes to avoid autoscaling that might hide the line
ax.set_xlim(0, N-1)
ax.set_ylim(0, N-1)

ax.set_title('Gray-Scott Model with Sine-Based Diffusion Overlay')

def update(frame, U, V, Du, Dv, F, k, dx, dt, sine_line):
    for _ in range(100):
        U, V = solver(U, V, Du, Dv, F, k, dx, dt)
    
    im.set_array(V)
    return [im, sine_line]
ani = animation.FuncAnimation(fig, update, frames=n_steps, interval = 0.1, blit=True, fargs=(U, V, Du, Dv, F, k, dx, dt, sine_line))

#plt.plot(x, 20* np.sin((np.pi * x)/100)-100, label='Sine Function', color='blue')
#ani.save('Gray_Scott_Animation.mp4', writer='ffmpeg', fps=30, dpi=300) #Save the animation as a video file
plt.show()

