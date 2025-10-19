# GPU-adapted wave solver (uses CuPy when available, falls back to NumPy)
import os
from typing import Optional

try:
    import cupy as cp
    xp = cp
    using_cupy = True
except Exception:
    import numpy as np
    xp = np
    using_cupy = False

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter


class WaveSolver2D_GPU:
    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float, c: float):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.c = c

        # Stability check (CFL condition) - use CPU numpy for sqrt for portability
        import math
        cfl = c * dt * math.sqrt(1/dx**2 + 1/dy**2)
        if cfl > 1:
            raise ValueError(f"CFL condition violated: {cfl} > 1")

        # allocate arrays on chosen backend
        self.u_curr = xp.zeros((nx, ny))
        self.u_prev = xp.zeros((nx, ny))
        self.u_next = xp.zeros((nx, ny))

    def set_initial_conditions(self, u0, v0: Optional[object] = None):
        # convert to backend arrays
        self.u_prev = xp.asarray(u0)
        self.u_curr = xp.asarray(u0).copy()
        if v0 is not None:
            self.u_curr = xp.asarray(u0) + self.dt * xp.asarray(v0)

    def apply_boundary_conditions(self, u):
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def step(self):
        c2_dt2 = (self.c * self.dt) ** 2

        # second derivatives using central differences
        d2u_dx2 = (self.u_curr[2:, 1:-1] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[:-2, 1:-1]) / self.dx**2
        d2u_dy2 = (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[1:-1, :-2]) / self.dy**2

        self.u_next[1:-1, 1:-1] = (2*self.u_curr[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] +
                                   c2_dt2 * (d2u_dx2 + d2u_dy2))

        self.apply_boundary_conditions(self.u_next)

        # rotate buffers
        self.u_prev, self.u_curr, self.u_next = self.u_curr, self.u_next, self.u_prev

    def solve(self, n_steps: int):
        # allocate history on host (NumPy) to keep memory smaller on device; copy frames back when needed
        history = []
        for i in range(n_steps):
            # append a copy of current field to history (move to host if on GPU)
            if using_cupy:
                history.append(self.u_curr.get())
            else:
                history.append(self.u_curr.copy())
            self.step()
        return xp.asarray(history) if not using_cupy else xp.asarray(history)

    def add_source(self, x: int, y: int, amplitude: float, frequency: float, t: float):
        self.u_curr[x, y] += amplitude * xp.sin(2 * xp.pi * frequency * t)


# Utility to create initial gaussian pulse (works with xp)
def create_gaussian_pulse_gpu(nx: int, ny: int, x0: int, y0: int, sigma: float, amplitude: float = 1.0):
    x = xp.arange(nx)
    y = xp.arange(ny)
    X, Y = xp.meshgrid(x, y, indexing='ij')
    pulse = amplitude * xp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return pulse


# animation helpers: accept GPU arrays but convert to host for plotting

def animate_solution_2d_gpu(solution, interval: int = 50, cmap: str = 'viridis', save_path: Optional[str] = None, dpi: int = 150):
    # solution may be a cupy or numpy array; ensure host numpy arrays for plotting
    if using_cupy:
        solution_host = [frame.get() for frame in solution]
    else:
        solution_host = [frame for frame in solution]

    import numpy as _np
    solution_host = _np.asarray(solution_host)

    n_steps, nx, ny = solution_host.shape
    fig, ax = plt.subplots()
    im = ax.imshow(solution_host[0], cmap=cmap, origin='lower')
    ax.set_title('Timestep 0')
    fig.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(solution_host[frame])
        ax.set_title(f'Timestep {frame}')
        return (im,)

    anim = FuncAnimation(fig, update, frames=range(n_steps), interval=interval, blit=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fps = max(1, int(1000/interval))
        if save_path.lower().endswith('.gif'):
            try:
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=dpi)
                print(f"Animation saved to {save_path} (GIF)")
            except Exception as e:
                print(f"Failed to save GIF animation: {e}")
        else:
            try:
                FFMpegWriter = animation.writers['ffmpeg']
                writer = FFMpegWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=dpi)
                print(f"Animation saved to {save_path} (mp4)")
            except Exception as e:
                print(f"Failed to save mp4 animation (ffmpeg may be missing): {e}")

    return anim


def animate_cross_section_gpu(solution, mid_y: int, interval: int = 50, save_path: Optional[str] = None, dpi: int = 150):
    if using_cupy:
        solution_host = [frame.get() for frame in solution]
    else:
        solution_host = [frame for frame in solution]

    import numpy as _np
    solution_host = _np.asarray(solution_host)

    n_steps, nx, ny = solution_host.shape
    fig, ax = plt.subplots()
    x = _np.arange(nx)
    line, = ax.plot(x, solution_host[0][:, mid_y])
    ax.set_ylim(solution_host.min(), solution_host.max())
    ax.set_xlabel('X position')
    ax.set_ylabel('Displacement')

    def update(frame):
        line.set_ydata(solution_host[frame][:, mid_y])
        ax.set_title(f'Timestep {frame}')
        return (line,)

    anim = FuncAnimation(fig, update, frames=range(n_steps), interval=interval, blit=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fps = max(1, int(1000/interval))
        if save_path.lower().endswith('.gif'):
            try:
                writer = PillowWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=dpi)
                print(f"Cross-section animation saved to {save_path} (GIF)")
            except Exception as e:
                print(f"Failed to save GIF animation: {e}")
        else:
            try:
                FFMpegWriter = animation.writers['ffmpeg']
                writer = FFMpegWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=dpi)
                print(f"Cross-section animation saved to {save_path} (mp4)")
            except Exception as e:
                print(f"Failed to save mp4 animation (ffmpeg may be missing): {e}")

    return anim


if __name__ == "__main__":
    nx, ny = 200, 200
    dx = dy = 0.1
    dt = 0.01
    c = 1.0

    solver = WaveSolver2D_GPU(nx, ny, dx, dy, dt, c)

    initial = create_gaussian_pulse_gpu(nx, ny, nx//2, ny//2, sigma=5.0)
    solver.set_initial_conditions(initial)

    # run a short simulation
    solution = solver.solve(200)

    print("Simulation finished (GPU mode: {} )".format(using_cupy))
    # save a quick animation to results/
    animate_solution_2d_gpu(solution, interval=50, save_path='results/gpu_wave_animation.gif')
