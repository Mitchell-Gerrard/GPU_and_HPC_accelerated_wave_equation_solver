
"""
MPI-only benchmark for WaveSolver2D_MPI.
 
Reads config from benchmark_mpi_config.txt (same directory as this script).
Writes results to benchmark_mpi.csv in the configured results directory,
matching the exact CSV format of benchmark.py so the combined plots work.
 
Usage
-----
    mpirun -n 4 python benchmark_mpi_only.py
    srun   -n 8 python benchmark_mpi_only.py   # Slurm / HPC
"""
 
import os
import sys
import csv
import time
import importlib.util
import numpy as np
from mpi4py import MPI
 
# ---------------------------------------------------------------------------
# MPI setup
# ---------------------------------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
 
# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(path):
    """Parse key=value config file, ignoring comments and blank lines."""
    cfg = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, val = line.partition('=')
            cfg[key.strip()] = val.strip()
    return cfg
 
 
def parse_config(cfg):
    # With this:
    raw = cfg.get('SIZES', '100:5001:100')
    if ':' in raw:
        parts = [int(x.strip()) for x in raw.split(':')]
        sizes = list(np.arange(*parts))
    else:
        sizes = [int(x.strip()) for x in raw.split(',')]
    n_steps = int(cfg.get('N_STEPS', 100))
    repeats = int(cfg.get('REPEATS', 3))
    dx      = float(cfg.get('DX', 0.1))
    dy      = float(cfg.get('DY', 0.1))
    dt      = float(cfg.get('DT', 0.01))
    c       = float(cfg.get('C',  1.0))
    sigma_f = float(cfg.get('PULSE_SIGMA_FRACTION', 0.1))
    results_dir = cfg.get('RESULTS_DIR', '../results')
    # Resolve relative to script location
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.normpath(os.path.join(script_dir, results_dir))
    return dict(sizes=sizes, n_steps=n_steps, repeats=repeats,
                dx=dx, dy=dy, dt=dt, c=c,
                sigma_fraction=sigma_f, results_dir=results_dir)
 
 
# ---------------------------------------------------------------------------
# Load wave_mpi module dynamically (same approach as benchmark.py)
# ---------------------------------------------------------------------------
def load_wave_mpi():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try same directory first, then src/ sibling
    candidates = [
        os.path.join(script_dir, 'wave_mpi.py'),
        os.path.join(script_dir, '..', 'src', 'wave_mpi.py'),
        os.path.join(script_dir, 'src', 'wave_mpi.py'),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location('wave_mpi', path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        f"wave_mpi.py not found. Searched:\n" + "\n".join(candidates)
    )
 
 
# ---------------------------------------------------------------------------
# Single benchmark run — all ranks participate
# ---------------------------------------------------------------------------
def benchmark_one(WaveSolver, make_pulse, nx, ny, dx, dy, dt, c, n_steps, sigma_fraction):
    """Returns wall time (float) on rank 0, 0.0 on others."""
    solver = WaveSolver(nx, ny, dx, dy, dt, c)
 
    if rank == 0:
        u0 = make_pulse(nx, ny, nx // 2, ny // 2, sigma=max(nx, ny) * sigma_fraction)
    else:
        u0 = None
 
    solver.set_initial_conditions(u0)
 
    # Warm-up: 1% of steps (min 1)
    warm = max(1, n_steps // 100)
    for _ in range(warm):
        solver.step()
 
    # Sync all ranks before timing
    comm.Barrier()
    t0 = MPI.Wtime()
 
    solver.solve(n_steps, snapshot_interval=n_steps + 1)  # no snapshots — pure compute
 
    comm.Barrier()
    t1 = MPI.Wtime()
 
    return t1 - t0 if rank == 0 else 0.0
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load config
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'benchmark_mpi_config.txt')
 
    if not os.path.exists(config_path):
        if rank == 0:
            print(f"Config file not found: {config_path}")
            print("Using built-in defaults.")
        cfg_raw = {}
    else:
        cfg_raw = load_config(config_path)
 
    cfg = parse_config(cfg_raw)
 
    if rank == 0:
        print(f"{'='*55}")
        print(f"  MPI Wave Benchmark — {size} rank(s)")
        print(f"  Sizes:   {cfg['sizes']}")
        print(f"  Steps:   {cfg['n_steps']}   Repeats: {cfg['repeats']}")
        print(f"  Results: {cfg['results_dir']}")
        print(f"{'='*55}\n")
 
    # Load solver
    try:
        mod         = load_wave_mpi()
        WaveSolver  = mod.WaveSolver2D_MPI
        make_pulse  = mod.create_gaussian_pulse
    except Exception as e:
        if rank == 0:
            print(f"Failed to load wave_mpi.py: {e}")
        comm.Abort(1)
        return
 
    # CSV setup (rank 0 only)
    if rank == 0:
        os.makedirs(cfg['results_dir'], exist_ok=True)
        out_path = os.path.join(cfg['results_dir'], 'benchmark_mpi.csv')
        csvfile  = open(out_path, 'w', newline='')
        writer   = csv.writer(csvfile)
        header   = ['backend', 'nx', 'ny', 'n_steps', 'repeats',
                    'total_time_s', 'time_per_step_s', 'gridpoints_per_sec',
                    'mpi_ranks']
        writer.writerow(header)
    else:
        csvfile = writer = out_path = None
 
    # Benchmark loop
    for nx in cfg['sizes']:
        ny = nx
        times = []
 
        for r in range(cfg['repeats']):
            if rank == 0:
                print(f"  nx={nx:4d}  repeat {r+1}/{cfg['repeats']} ...", end='', flush=True)
 
            t = benchmark_one(
                WaveSolver, make_pulse,
                nx, ny,
                cfg['dx'], cfg['dy'], cfg['dt'], cfg['c'],
                cfg['n_steps'], cfg['sigma_fraction']
            )
 
            if rank == 0:
                times.append(t)
                print(f" {t:.4f}s")
 
        if rank == 0:
            avg   = sum(times) / len(times)
            tps   = avg / cfg['n_steps']
            gps   = (nx * ny * cfg['n_steps']) / avg if avg > 0 else float('inf')
            row   = [
                'mpi', nx, ny,
                cfg['n_steps'], cfg['repeats'],
                f"{avg:.6f}", f"{tps:.9f}", f"{gps:.3f}",
                size
            ]
            writer.writerow(row)
            csvfile.flush()
            print(f"  → avg {avg:.4f}s | {gps/1e6:.2f}M gridpoints/s\n")
 
    if rank == 0:
        csvfile.close()
        print(f"\nDone. Results saved to {out_path}")
        print("Run benchmark.py plotting section to include MPI in combined plots.")
 
 
if __name__ == '__main__':
    main()
