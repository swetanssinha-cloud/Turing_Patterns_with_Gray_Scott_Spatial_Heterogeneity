import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Du, Dv = 0.16, 0.08
F, k = 0.028, 0.064#0.028, 0.064 #0.046, 0.064 #(Stripes) #0.028, 0.064 (Spots) #0.010, 0.046 (Dynamic) #0.037, 0.06 (Standard)
N = 200
L = 200
dx = L / N
dt = 0.1
n_steps =  300 #10000 #Using 300 for video. 

def laplacian(Z, dx):
    d2Z_dx2 = np.roll(Z, 1, 0) - 2 * Z + np.roll(Z, -1, 0)
    d2Z_dy2 = np.roll(Z, 1, 1) - 2 * Z + np.roll(Z, -1, 1) 

    derivative = (d2Z_dx2 + d2Z_dy2)/(dx * dx)# should this not be (d2Z_dx2/dx * dx) + (d2Z_dy2/dy * dy)? I think we are saying that 
    #dy = dx. 
    return derivative

def solver(U, V, Du, Dv, F, k, dx, dt):
    Lu = laplacian(U, dx)
    Lv = laplacian(V, dx)
    uvv = U * V * V

    U += (Du * Lu - uvv + F * (1 - U)) * dt  #Lowk which one is the inhibitor and activator?
    V += (Dv * Lv + uvv - (F+k) * V) * dt
    #Here is forward euler's method for the next time step. 
    return U, V
U = np.ones((N,N))
V = np.zeros((N,N))

r = 20
U[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.50
V[N//2 - r:N//2+r, N//2 - r: N//2 + r] = 0.25
U += 0.05 * np.random.rand(N,N)
V += 0.05 * np.random.rand(N,N)

fig, ax = plt.subplots(figsize=(8,6)) #regular sizing in a cube

im = ax.imshow(V, cmap='inferno', interpolation='bilinear', vmin=0, vmax=0.5)
plt.axis('off')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('[V]')

def update(frame, U, V, Du, Dv, F, k, dx, dt):
    for _ in range(100):
        U, V = solver(U, V, Du, Dv, F, k, dx, dt)
    im.set_array(V)
    return [im]
ani = animation.FuncAnimation(fig, update, frames=n_steps, interval = 0.1, blit=True, fargs=(U, V, Du, Dv, F, k, dx, dt))
# ani.save('Gray_Scott_Animation.mp4', writer='ffmpeg', fps=30, dpi=300) #Save the animation as a video file
plt.show()

