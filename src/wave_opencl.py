"""
OpenCL-backed 2D wave equation solver.

Attempts to use PyOpenCL with a GPU device (suitable for AMD GPUs). If PyOpenCL
or an OpenCL device is not available, falls back to a NumPy CPU implementation.

The OpenCL kernel computes the standard 5-point finite-difference update for the
2D wave equation. The Python class provides a simple `solve` method that can
optionally return snapshots at a given interval as NumPy arrays (ready for
plotting).

Notes:
- Arrays use float32 on the device for portability and performance.
- This is a minimal implementation focused on correctness and portability; for
  production you may want to tune work-group sizes, memory transfers, and
  avoid copying every timestep.
"""

import os
from typing import Optional

import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import pyplot as plt
import cmasher as cmr
plt.rcParams['figure.dpi']=200
map='cmr.bubblegum'
plt.rcParams['image.cmap']=map
try:
    import pyopencl as cl
    USING_OPENCL = True
except Exception:
    cl = None
    USING_OPENCL = False


_OPENCL_KERNEL = r"""
__kernel void step(
    const int nx, const int ny,
    const float dx2, const float dy2, const float c2_dt2,
    __global const float* u_curr,
    __global const float* u_prev,
    __global float* u_next)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < 0 || i >= nx || j < 0 || j >= ny) return;

    int idx = i * ny + j;

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
        int idx_ip = (i+1) * ny + j;
        int idx_im = (i-1) * ny + j;
        int idx_jp = i * ny + (j+1);
        int idx_jm = i * ny + (j-1);

        float center = u_curr[idx];
        float d2x = (u_curr[idx_ip] - 2.0f*center + u_curr[idx_im]) / dx2;
        float d2y = (u_curr[idx_jp] - 2.0f*center + u_curr[idx_jm]) / dy2;

        u_next[idx] = 2.0f*center - u_prev[idx] + c2_dt2 * (d2x + d2y);
    } else {
        // Dirichlet BC: fixed zero
        u_next[idx] = 0.0f;
    }
}
"""


class WaveSolver2D_OpenCL:
    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float, c: float):
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        self.c = float(c)

        # CFL check (use host math)
        import math
        cfl = c * dt * math.sqrt(1/dx**2 + 1/dy**2)
        if cfl > 1:
            raise ValueError(f"CFL condition violated: {cfl} > 1")

        # Decide backend
        self.use_opencl = USING_OPENCL
        self.ctx = None
        self.queue = None
        self.prg = None

        if self.use_opencl:
            # try to create a context on a GPU device (prefer GPU)
            try:
                platforms = cl.get_platforms()
                # pick first platform with GPU devices if possible
                dev = None
                for p in platforms:
                    devs = p.get_devices(device_type=cl.device_type.GPU)
                    if devs:
                        dev = devs[0]
                        break
                if dev is None:
                    # fallback to any device
                    dev = platforms[0].get_devices()[0]
                self.ctx = cl.Context(devices=[dev])
                self.queue = cl.CommandQueue(self.ctx)
                self.prg = cl.Program(self.ctx, _OPENCL_KERNEL).build()
            except Exception as e:
                # fallback to CPU numpy implementation
                print(f"OpenCL initialization failed, falling back to NumPy: {e}")
                self.use_opencl = False

        # device buffers will be created when initial conditions are set
        self.u_curr_buf = None
        self.u_prev_buf = None
        self.u_next_buf = None

    def set_initial_conditions(self, u0: np.ndarray, v0: Optional[np.ndarray] = None):
        # ensure float32 and correct shape
        u0f = np.asarray(u0, dtype=np.float32).reshape((self.nx, self.ny))
        if v0 is not None:
            v0f = np.asarray(v0, dtype=np.float32).reshape((self.nx, self.ny))
        else:
            v0f = None

        if self.use_opencl:
            mf = cl.mem_flags
            # flatten in row-major order (C) as kernel expects i*ny + j indexing
            host_curr = u0f.ravel().astype(np.float32)
            host_prev = host_curr.copy()
            # create device buffers
            self.u_curr_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_curr)
            self.u_prev_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host_prev)
            self.u_next_buf = cl.Buffer(self.ctx, mf.READ_WRITE, host_curr.nbytes)
            # if initial velocity provided, quick first-step estimate (optional)
            if v0f is not None:
                # u_curr = u0 + dt*v0
                tmp = (u0f + self.dt * v0f).ravel().astype(np.float32)
                cl.enqueue_copy(self.queue, self.u_curr_buf, tmp)
                cl.enqueue_copy(self.queue, self.u_prev_buf, host_curr)  # previous is u0
        else:
            # fallback to numpy arrays
            self.u_curr = u0f.copy()
            self.u_prev = u0f.copy()
            self.u_next = np.zeros_like(self.u_curr, dtype=np.float32)
            if v0f is not None:
                self.u_curr = u0f + self.dt * v0f

    def step(self):
        if self.use_opencl:
            nx = np.int32(self.nx)
            ny = np.int32(self.ny)
            dx2 = np.float32(self.dx**2)
            dy2 = np.float32(self.dy**2)
            c2_dt2 = np.float32((self.c * self.dt)**2)

            # launch kernel with global size (nx, ny)
            try:
                self.prg.step(self.queue, (self.nx, self.ny), None,
                              nx, ny, dx2, dy2, c2_dt2,
                              self.u_curr_buf, self.u_prev_buf, self.u_next_buf)
                # rotate buffers
                self.u_prev_buf, self.u_curr_buf, self.u_next_buf = self.u_curr_buf, self.u_next_buf, self.u_prev_buf
            except Exception as e:
                raise RuntimeError(f"OpenCL kernel execution failed: {e}")
        else:
            c2_dt2 = (self.c * self.dt) ** 2
            # second derivatives (interior)
            d2x = (self.u_curr[2:, 1:-1] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[:-2, 1:-1]) / (self.dx**2)
            d2y = (self.u_curr[1:-1, 2:] - 2*self.u_curr[1:-1, 1:-1] + self.u_curr[1:-1, :-2]) / (self.dy**2)
            self.u_next[1:-1, 1:-1] = 2*self.u_curr[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] + c2_dt2 * (d2x + d2y)
            # boundaries zero
            self.u_next[0, :] = 0
            self.u_next[-1, :] = 0
            self.u_next[:, 0] = 0
            self.u_next[:, -1] = 0
            # rotate
            self.u_prev, self.u_curr, self.u_next = self.u_curr, self.u_next, self.u_prev

    def solve(self, n_steps: int, snapshot_interval: int = 1):
        """Run n_steps and return a NumPy array of snapshots collected every snapshot_interval steps.

        Returns an array of shape (n_snapshots, nx, ny) dtype float32 on host.
        """
        snapshots = []
        if self.use_opencl:
            # preallocate a host buffer for copying
            host_buf = np.empty(self.nx * self.ny, dtype=np.float32)
            for i in range(n_steps):
                if (i % snapshot_interval) == 0:
                    # copy device u_curr to host
                    cl.enqueue_copy(self.queue, host_buf, self.u_curr_buf)
                    self.queue.finish()
                    snapshots.append(host_buf.reshape((self.nx, self.ny)).copy())
                self.step()
        else:
            for i in range(n_steps):
                if (i % snapshot_interval) == 0:
                    snapshots.append(self.u_curr.copy())
                self.step()
        if snapshots:
            return np.asarray(snapshots, dtype=np.float32)
        else:
            return np.empty((0, self.nx, self.ny), dtype=np.float32)

    def add_source(self, x: int, y: int, amplitude: float, frequency: float, t: float):
        if self.use_opencl:
            # read current frame, modify on host, write back (simple approach)
            host = np.empty(self.nx * self.ny, dtype=np.float32)
            cl.enqueue_copy(self.queue, host, self.u_curr_buf)
            self.queue.finish()
            host = host.reshape((self.nx, self.ny))
            host[x, y] += amplitude * np.sin(2*np.pi*frequency*t)
            cl.enqueue_copy(self.queue, self.u_curr_buf, host.ravel())
            self.queue.finish()
        else:
            self.u_curr[x, y] += amplitude * np.sin(2*np.pi*frequency*t)


# Utility: host-side gaussian pulse
def create_gaussian_pulse(nx: int, ny: int, x0: int, y0: int, sigma: float, amplitude: float = 1.0):
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    pulse = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return pulse.astype(np.float32)
def animate_solution_2d(solution: np.ndarray, interval: int = 50, cmap: str = map,
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

# Simple example to run when executed directly
if __name__ == "__main__":
    nx, ny = 200, 200
    dx = dy = 0.1
    dt = 0.01
    c = 1.0

    solver = WaveSolver2D_OpenCL(nx, ny, dx, dy, dt, c)
    initial = create_gaussian_pulse(nx, ny, nx//2, ny//2, sigma=5.0)
    solver.set_initial_conditions(initial)

    # collect snapshots every 2 steps to reduce transfer cost
    solution = solver.solve(200, snapshot_interval=2)
    print(f"Snapshots shape: {solution.shape}, OpenCL used: {solver.use_opencl}")
    animate_cross_section(solution, mid_y=ny//2, interval=50, save_path='results/opencl_cross_section.gif')
    animate_solution_2d(solution, interval=50, save_path='results/opencl_wave_animation.gif')
    # quick save of a single snapshot image
    try:
        import matplotlib.pyplot as plt
        os.makedirs('results', exist_ok=True)
        plt.imshow(solution[0], origin='lower', cmap=map)
        plt.colorbar()
        plt.title('OpenCL solver snapshot (t=0)')
        plt.savefig('results/opencl_snapshot.png', dpi=150)
        print('Saved results/opencl_snapshot.png')
    except Exception as e:
        print('Could not save image:', e)
