import numpy as np
from typing import Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
import sys
import os

class WaveSolver2D:
    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float, c: float):
        """
        2D Wave equation solver using finite differences
        
        Args:
            nx, ny: Grid dimensions
            dx, dy: Spatial step sizes
            dt: Time step
            c: Wave speed
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.c = c
        
        # Stability check (CFL condition)
        cfl = c * dt * np.sqrt(1/dx**2 + 1/dy**2)
        if cfl > 1:
            raise ValueError(f"CFL condition violated: {cfl} > 1")
        
        # Initialize wave fields
        self.u_curr = np.zeros((nx, ny))  # Current time step
        self.u_prev = np.zeros((nx, ny))  # Previous time step
        self.u_next = np.zeros((nx, ny))  # Next time step
        
    def set_initial_conditions(self, u0: np.ndarray, v0: Optional[np.ndarray] = None):
        """Set initial displacement and velocity"""
        self.u_prev = u0.copy()
        self.u_curr = u0.copy()
        
        # If initial velocity is provided, estimate u_curr using v0
        if v0 is not None:
            self.u_curr = u0 + self.dt * v0
    
    def apply_boundary_conditions(self, u: np.ndarray):
        """Apply Dirichlet boundary conditions (fixed boundaries)"""
        u[0, :] = 0   # Left boundary
        u[-1, :] = 0  # Right boundary
        u[:, 0] = 0   # Bottom boundary
        u[:, -1] = 0  # Top boundary
    
    def step(self):
        """Perform one time step using finite difference scheme"""
        c2_dt2 = (self.c * self.dt) ** 2
        
        # Second derivatives using central differences
        d2u_dx2 = (self.u_curr[2:, 1:-1] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[:-2, 1:-1]) / self.dx**2
        d2u_dy2 = (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[1:-1, :-2]) / self.dy**2
        
        # Update interior points
        self.u_next[1:-1, 1:-1] = (2*self.u_curr[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] + 
                                   c2_dt2 * (d2u_dx2 + d2u_dy2))
        
        # Apply boundary conditions
        self.apply_boundary_conditions(self.u_next)
        
        # Shift arrays for next iteration
        self.u_prev, self.u_curr, self.u_next = self.u_curr, self.u_next, self.u_prev
    
    def solve(self, n_steps: int) -> np.ndarray:
        """Solve for n_steps and return solution history"""
        history = np.zeros((n_steps, self.nx, self.ny))
        
        for i in range(n_steps):
            history[i] = self.u_curr.copy()
            self.step()
            
        return history
    
    def add_source(self, x: int, y: int, amplitude: float, frequency: float, t: float):
        """Add a sinusoidal source at position (x, y)"""
        self.u_curr[x, y] += amplitude * np.sin(2 * np.pi * frequency * t)


def create_gaussian_pulse(nx: int, ny: int, x0: int, y0: int, sigma: float, amplitude: float = 1.0) -> np.ndarray:
    """Create a 2D Gaussian pulse for initial conditions"""
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    pulse = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return pulse


def animate_solution_2d(solution: np.ndarray, interval: int = 50, cmap: str = 'viridis',
                        save_path: Optional[str] = None, dpi: int = 150):
    """Create a 2D animation (imshow) of the solution history.

    Args:
        solution: array of shape (n_steps, nx, ny)
        interval: delay between frames in ms
        cmap: matplotlib colormap
        save_path: if provided, save animation to this path (gif/mp4)
        dpi: output dpi when saving
    """
    n_steps, nx, ny = solution.shape

    fig, ax = plt.subplots()
    im = ax.imshow(solution[0], cmap=cmap, origin='lower')
    ax.set_title('Timestep 0')
    fig.colorbar(im, ax=ax)

    def update(frame):
        im.set_data(solution[frame])
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
            # Try ffmpeg writer for mp4
            try:
                FFMpegWriter = animation.writers['ffmpeg']
                writer = FFMpegWriter(fps=fps)
                anim.save(save_path, writer=writer, dpi=dpi)
                print(f"Animation saved to {save_path} (mp4)")
            except Exception as e:
                print(f"Failed to save mp4 animation (ffmpeg may be missing): {e}")

    return anim


def animate_cross_section(solution: np.ndarray, mid_y: int, interval: int = 50,
                          save_path: Optional[str] = None, dpi: int = 150):
    """Animate the central cross-section (x vs displacement) through time.

    Args:
        solution: array of shape (n_steps, nx, ny)
        mid_y: index of the y-slice to plot
        interval: delay between frames in ms
        save_path: if provided, save animation (gif/mp4)
    """
    n_steps, nx, ny = solution.shape

    fig, ax = plt.subplots()
    x = np.arange(nx)
    line, = ax.plot(x, solution[0][:, mid_y])
    ax.set_ylim(solution.min(), solution.max())
    ax.set_xlabel('X position')
    ax.set_ylabel('Displacement')

    def update(frame):
        line.set_ydata(solution[frame][:, mid_y])
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
    # Example usage
    nx, ny = 100, 100
    dx, dy = 0.1, 0.1
    dt = 0.01
    c = 1.0
    
    # Create solver
    solver = WaveSolver2D(nx, ny, dx, dy, dt, c)
    
    # Set initial Gaussian pulse
    initial_pulse = create_gaussian_pulse(nx, ny, nx//2, ny//2, sigma=5.0)
    solver.set_initial_conditions(initial_pulse)

    # Solve for 100 time steps
    solution = solver.solve(100)
    
    print(f"Simulation completed. Solution shape: {solution.shape}")
    animate_solution_2d(solution, interval=50, save_path='results/cpu_wave_animation.gif')