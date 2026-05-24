"""
MPI-parallel 2D wave equation solver — drop-in replacement for wave_opencl.py.

The class is named WaveSolver2D_MPI but is otherwise called identically to
WaveSolver2D_OpenCL.  The helper functions (create_gaussian_pulse,
animate_solution_2d, animate_cross_section) have the same signatures.

Parallelisation
---------------
Domain is split along the X axis — one contiguous slab of rows per MPI rank.
Each rank keeps one ghost row on each side and exchanges them with its
neighbours before every time step (blocking MPI Send/Recv).

Only rank 0 ever touches matplotlib or writes files; other ranks do the
compute and return None from solve().

Usage
-----
    mpirun -n 4 python wave_mpi.py        # same __main__ block as original
    srun        python wave_mpi.py        # inside a Slurm batch script
"""

import os
from typing import Optional

import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import pyplot as plt
import cmasher as cmr

plt.rcParams['figure.dpi'] = 200
map = 'cmr.bubblegum'
plt.rcParams['image.cmap'] = map

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    USING_MPI = True
except ImportError:
    _comm = None
    _rank = 0
    _size = 1
    USING_MPI = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decompose(nx, size, rank):
    """Return (i_start, i_end) — exclusive end, global row indices."""
    base, rem = divmod(nx, size)
    i_start = rank * base + min(rank, rem)
    i_end   = i_start + base + (1 if rank < rem else 0)
    return i_start, i_end


def _halo_exchange(comm, rank, size, u_curr):
    """Send/recv ghost rows with neighbouring ranks (in-place)."""
    if size == 1:
        return
    tag_dn, tag_up = 0, 1
    send_top    = np.ascontiguousarray(u_curr[1,  :])
    send_bottom = np.ascontiguousarray(u_curr[-2, :])
    ny = u_curr.shape[1]
    recv_top    = np.empty(ny, dtype=np.float32)
    recv_bottom = np.empty(ny, dtype=np.float32)

    if rank % 2 == 0:
        if rank + 1 < size:
            comm.Send(send_bottom, dest=rank + 1, tag=tag_dn)
            comm.Recv(recv_bottom, source=rank + 1, tag=tag_up)
        if rank - 1 >= 0:
            comm.Send(send_top, dest=rank - 1, tag=tag_up)
            comm.Recv(recv_top,  source=rank - 1, tag=tag_dn)
    else:
        if rank - 1 >= 0:
            comm.Recv(recv_top,  source=rank - 1, tag=tag_dn)
            comm.Send(send_top,  dest=rank - 1,   tag=tag_up)
        if rank + 1 < size:
            comm.Recv(recv_bottom, source=rank + 1, tag=tag_up)
            comm.Send(send_bottom, dest=rank + 1,   tag=tag_dn)

    if rank - 1 >= 0:
        u_curr[0,  :] = recv_top
    if rank + 1 < size:
        u_curr[-1, :] = recv_bottom


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class WaveSolver2D_MPI:
    """
    Drop-in replacement for WaveSolver2D_OpenCL using MPI domain decomposition.

    Constructor and every method have exactly the same signature.
    `solve` returns a (n_snapshots, nx, ny) float32 array on rank 0,
    and None on all other ranks.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float, c: float):
        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = float(dx)
        self.dy = float(dy)
        self.dt = float(dt)
        self.c  = float(c)

        import math
        cfl = c * dt * math.sqrt(1 / dx**2 + 1 / dy**2)
        if cfl > 1:
            raise ValueError(f"CFL condition violated: {cfl:.4f} > 1")

        self.comm  = _comm
        self.rank  = _rank
        self.size  = _size

        self.i_start, self.i_end = _decompose(nx, self.size, self.rank)
        self.local_nx = self.i_end - self.i_start

        # local arrays: local_nx real rows + 2 ghost rows
        shape = (self.local_nx + 2, ny)
        self.u_curr = np.zeros(shape, dtype=np.float32)
        self.u_prev = np.zeros(shape, dtype=np.float32)
        self.u_next = np.zeros(shape, dtype=np.float32)

        # keep the attribute name so any code that checks it still works
        self.use_mpi = USING_MPI

    def set_initial_conditions(self, u0: np.ndarray, v0: Optional[np.ndarray] = None):
        # rank 0 has the full array; broadcast to everyone
        if USING_MPI:
            u0 = self.comm.bcast(np.asarray(u0, dtype=np.float32) if self.rank == 0 else None, root=0)
            v0 = self.comm.bcast(np.asarray(v0, dtype=np.float32) if (self.rank == 0 and v0 is not None) else None, root=0)
        
        u0 = np.asarray(u0, dtype=np.float32).reshape((self.nx, self.ny))
        slab = u0[self.i_start:self.i_end, :]

        self.u_curr[1:-1, :] = slab
        self.u_prev[1:-1, :] = slab

        if v0 is not None:
            v0 = np.asarray(v0, dtype=np.float32).reshape((self.nx, self.ny))
            self.u_curr[1:-1, :] = slab + self.dt * v0[self.i_start:self.i_end, :]

    def step(self):
        if USING_MPI:
            _halo_exchange(self.comm, self.rank, self.size, self.u_curr)

        c2_dt2 = (self.c * self.dt) ** 2
        uc = self.u_curr
        d2x = (uc[2:,  1:-1] - 2*uc[1:-1, 1:-1] + uc[:-2, 1:-1]) / self.dx**2
        d2y = (uc[1:-1, 2:] - 2*uc[1:-1, 1:-1] + uc[1:-1, :-2]) / self.dy**2
        self.u_next[1:-1, 1:-1] = 2*uc[1:-1, 1:-1] - self.u_prev[1:-1, 1:-1] + c2_dt2 * (d2x + d2y)

        # Dirichlet BCs on physical boundaries
        self.u_next[1:-1,  0] = 0
        self.u_next[1:-1, -1] = 0
        if self.i_start == 0:
            self.u_next[1, :] = 0
        if self.i_end == self.nx:
            self.u_next[-2, :] = 0

        self.u_prev, self.u_curr, self.u_next = self.u_curr, self.u_next, self.u_prev

    def solve(self, n_steps: int, snapshot_interval: int = 1):
        """
        Run n_steps and return snapshots every snapshot_interval steps.

        Returns
        -------
        rank 0 : np.ndarray shape (n_snapshots, nx, ny) float32
        others : None
        """
        local_snaps = []
        for i in range(n_steps):
            if i % snapshot_interval == 0:
                local_snaps.append(self.u_curr[1:-1, :].copy())
            self.step()

        return self._gather(local_snaps)

    def _gather(self, local_snaps):
        n_snaps = len(local_snaps)

        if self.size == 1:
            return np.stack(local_snaps, axis=0) if local_snaps else np.empty((0, self.nx, self.ny), dtype=np.float32)

        # Stack all local snapshots: shape (n_snaps, local_nx, ny)
        local_arr = np.ascontiguousarray(
            np.stack(local_snaps, axis=0) if local_snaps
            else np.empty((0, self.local_nx, self.ny), dtype=np.float32),
            dtype=np.float32
        )

        # Tell rank 0 how many floats each rank is sending
        local_count = np.array([local_arr.size], dtype=np.int32)
        all_counts = np.empty(self.size, dtype=np.int32) if self.rank == 0 else None
        self.comm.Gather(local_count, all_counts, root=0)

        recvbuf = np.empty(int(all_counts.sum()), dtype=np.float32) if self.rank == 0 else None
        self.comm.Gatherv(local_arr.ravel(), (recvbuf, all_counts) if self.rank == 0 else None, root=0)

        if self.rank == 0:
            local_nxs = [
                _decompose(self.nx, self.size, r)[1] - _decompose(self.nx, self.size, r)[0]
                for r in range(self.size)
            ]
            # Each rank sends (n_snaps, local_nx, ny) — reconstruct that
            result = np.empty((n_snaps, self.nx, self.ny), dtype=np.float32)
            offset = 0
            for r in range(self.size):
                lnx = local_nxs[r]
                i_start, _ = _decompose(self.nx, self.size, r)
                chunk = recvbuf[offset: offset + n_snaps * lnx * self.ny]
                result[:, i_start:i_start+lnx, :] = chunk.reshape(n_snaps, lnx, self.ny)
                offset += n_snaps * lnx * self.ny
            return result

        return None

    def add_source(self, x: int, y: int, amplitude: float, frequency: float, t: float):
        """Inject a point source at global row x (same signature as original)."""
        if self.i_start <= x < self.i_end:
            lx = x - self.i_start + 1   # +1 for ghost row
            self.u_curr[lx, y] += amplitude * np.sin(2 * np.pi * frequency * t)


# ---------------------------------------------------------------------------
# Utilities — identical signatures to the original
# ---------------------------------------------------------------------------

def create_gaussian_pulse(nx: int, ny: int, x0: int, y0: int, sigma: float, amplitude: float = 1.0):
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    pulse = amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    return pulse.astype(np.float32)


def animate_solution_2d(solution: np.ndarray, interval: int = 50, cmap: str = map,
                        save_path: Optional[str] = None, dpi: int = 150):
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
        fps = max(1, int(1000 / interval))
        if save_path.lower().endswith('.gif'):
            try:
                anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
                print(f"Animation saved to {save_path} (GIF)")
            except Exception as e:
                print(f"Failed to save GIF animation: {e}")
        else:
            try:
                anim.save(save_path, writer=animation.writers['ffmpeg'](fps=fps), dpi=dpi)
                print(f"Animation saved to {save_path} (mp4)")
            except Exception as e:
                print(f"Failed to save mp4 animation (ffmpeg may be missing): {e}")

    return anim


def animate_cross_section(solution: np.ndarray, mid_y: int, interval: int = 50,
                           save_path: Optional[str] = None, dpi: int = 150):
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
        fps = max(1, int(1000 / interval))
        if save_path.lower().endswith('.gif'):
            try:
                anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
                print(f"Cross-section animation saved to {save_path} (GIF)")
            except Exception as e:
                print(f"Failed to save GIF animation: {e}")
        else:
            try:
                anim.save(save_path, writer=animation.writers['ffmpeg'](fps=fps), dpi=dpi)
                print(f"Cross-section animation saved to {save_path} (mp4)")
            except Exception as e:
                print(f"Failed to save mp4 animation (ffmpeg may be missing): {e}")

    return anim


# ---------------------------------------------------------------------------
# __main__ — identical to wave_opencl.py, just class name changed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    nx, ny = 200, 200
    dx = dy = 0.1
    dt = 0.01
    c = 1.0

    solver = WaveSolver2D_MPI(nx, ny, dx, dy, dt, c)
    print(solver.size,solver.rank)
    initial = create_gaussian_pulse(nx, ny, nx//2, ny//2, sigma=5.0)
    solver.set_initial_conditions(initial)

    # collect snapshots every 2 steps to reduce transfer cost
    solution = solver.solve(200, snapshot_interval=2)
    #print(f"Snapshots shape: {solution.shape}, MPI used: {solver.use_mpi}", flush=True)

    # only rank 0 has solution data and should do any plotting
    if _rank == 0 and solution is not None:
        animate_cross_section(solution, mid_y=ny//2, interval=50, save_path='results/mpi_cross_section.gif')
        animate_solution_2d(solution, interval=50, save_path='results/mpi_wave_animation.gif')
        try:
            os.makedirs('results', exist_ok=True)
            plt.imshow(solution[0], origin='lower', cmap=map)
            plt.colorbar()
            plt.title('MPI solver snapshot (t=0)')
            plt.savefig('results/mpi_snapshot.png', dpi=150)
            print('Saved results/mpi_snapshot.png')
        except Exception as e:
            print('Could not save image:', e)