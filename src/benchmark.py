"""Benchmark runner for CPU (wave_cpu), CuPy GPU (wave_gpu), and OpenCL (wave_opencl) solvers.

Usage (from repository root):

    python src\benchmark.py --sizes 100,200 --n_steps 100 --repeats 3 --out results/benchmark.csv

The script attempts to load the solver modules from the `src/` directory and will gracefully skip backends that fail to import.
"""

import os
import sys
import time
import csv
import argparse
import importlib.util
from statistics import mean
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import glob
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, 'src')

# Helper to load a module from a file path
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise ImportError(f"No loader for spec {spec}")
    loader.exec_module(mod)
    return mod


def try_load_solvers():
    solvers = {}

    # CPU
    cpu_path = os.path.join(SRC_DIR, 'wave_cpu.py')
    try:
        cpu_mod = load_module_from_path('wave_cpu', cpu_path)
        WaveCPU = getattr(cpu_mod, 'WaveSolver2D', None)
        make_pulse_cpu = getattr(cpu_mod, 'create_gaussian_pulse', None)
        if WaveCPU and make_pulse_cpu:
            solvers['cpu'] = (WaveCPU, make_pulse_cpu)
            print('Loaded CPU solver')
    except Exception as e:
        print('CPU solver not available:', e)

    # CuPy GPU
    platforms = cl.get_platforms()
    if platforms[0].name.lower().find('nvidia') != -1:

        gpu_path = os.path.join(SRC_DIR, 'wave_gpu.py')
        try:
            gpu_mod = load_module_from_path('wave_gpu', gpu_path)
            WaveGPU = getattr(gpu_mod, 'WaveSolver2D_GPU', None)
            make_pulse_gpu = getattr(gpu_mod, 'create_gaussian_pulse_gpu', None)
            if WaveGPU and make_pulse_gpu:
                solvers['gpu'] = (WaveGPU, make_pulse_gpu)
                print('Loaded CuPy GPU solver (wave_gpu)')
        except Exception as e:
            print('GPU solver not available:', e)

    # OpenCL
    if platforms[0].name.lower().find('amd') != -1:
        ocl_path = os.path.join(SRC_DIR, 'wave_opencl.py')
        try:
            ocl_mod = load_module_from_path('wave_opencl', ocl_path)
            WaveOCL = getattr(ocl_mod, 'WaveSolver2D_OpenCL', None)
            make_pulse_ocl = getattr(ocl_mod, 'create_gaussian_pulse', None)
            if WaveOCL and make_pulse_ocl:
                solvers['opencl'] = (WaveOCL, make_pulse_ocl)
                print('Loaded OpenCL solver')
        except Exception as e:
            print('OpenCL solver not available:', e)

    return solvers


def benchmark_solver(constructor, make_pulse, nx, ny, dt, dx, dy, c, n_steps, repeats=3):
    times = []
    for r in range(repeats):
        # construct solver
        solver = constructor(nx, ny, dx, dy, dt, c)
        # initial pulse
        u0 = make_pulse(nx, ny, nx//2, ny//2, sigma=max(nx,ny)/10)
        solver.set_initial_conditions(u0)
        # warm-up
        warm = max(1, int(0.01 * n_steps))
        for _ in range(warm):
            solver.step()
        # timed run
        t0 = time.perf_counter()
        # If solve exists with n_steps parameter, call it to capture backend behavior
        if hasattr(solver, 'solve'):
            # some implementations return snapshots; we ignore returned data
            solver.solve(n_steps)
        else:
            for _ in range(n_steps):
                solver.step()
        t1 = time.perf_counter()
        total = t1 - t0
        times.append(total)
        print(f'  repeat {r+1}/{repeats}: {total:.4f}s')
    return mean(times)


def main():
    # Configuration (set values here instead of using command-line arguments)
    # Edit these variables when running from the IDE / interactive session.
    sizes = np.arange(100, 2001, 100)  # list of nx (and ny) grid sizes to benchmark
    n_steps = 100               # number of timesteps per benchmark
    repeats = 3                 # repeats per (backend, size)
    results_dir = os.path.join(REPO_ROOT, 'results')
 
    solvers = try_load_solvers()
    if not solvers:
        print('No solvers available to benchmark. Exiting.')
        return
 
    os.makedirs(results_dir, exist_ok=True)
    header = ['backend', 'nx', 'ny', 'n_steps', 'repeats', 'total_time_s', 'time_per_step_s', 'gridpoints_per_sec']
 
    # Write a separate CSV file per backend
    for backend, (constructor, make_pulse) in solvers.items():
        out_path = os.path.join(results_dir, f'benchmark_{backend}.csv')
        with open(out_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            for nx in sizes:
                ny = nx
                dx = dy = 0.1
                dt = 0.01
                c = 1.0
                print(f'Benchmarking {backend} nx={nx} n_steps={n_steps}...')
                try:
                    avg_total = benchmark_solver(constructor, make_pulse, nx, ny, dt, dx, dy, c, n_steps, repeats=repeats)
                    time_per_step = avg_total / n_steps
                    gridpts = nx * ny * n_steps
                    gps = gridpts / avg_total if avg_total > 0 else float('inf')
                    row = [backend, nx, ny, n_steps, repeats, f'{avg_total:.6f}', f'{time_per_step:.9f}', f'{gps:.3f}']
                    writer.writerow(row)
                    csvfile.flush()
                except Exception as e:
                    print(f'Failed benchmarking {backend} nx={nx}: {e}')
                    writer.writerow([backend, nx, ny, n_steps, repeats, 'error', str(e)])
        print(f'Benchmark finished for {backend}. Results saved to {out_path}')
    
    # Plotting: read per-backend CSVs and create per-backend and combined plots
    def plot_benchmark_results(results_dir: str):
        files = sorted(glob.glob(os.path.join(results_dir, 'benchmark_*.csv')))
        if not files:
            print('No benchmark CSV files found for plotting.')
            return

        # read all data into dict: backend -> list of (nx, gps, time_per_step)
        data = {}
        for f in files:
            backend = os.path.splitext(os.path.basename(f))[0].replace('benchmark_', '')
            nx_vals, gps_vals, tps_vals = [], [], []
            with open(f, 'r', newline='') as cf:
                reader = csv.reader(cf)
                header_row = next(reader, None)
                if header_row is None:
                    continue
                try:
                    idx_nx = header_row.index('nx')
                    idx_gps = header_row.index('gridpoints_per_sec')
                    idx_tps = header_row.index('time_per_step_s')
                except ValueError:
                    idx_nx, idx_gps, idx_tps = 1, 7, 6
                for row in reader:
                    if len(row) <= max(idx_nx, idx_gps, idx_tps):
                        continue
                    try:
                        nxv = int(row[idx_nx])
                        gps = float(row[idx_gps])
                        tps = float(row[idx_tps])
                    except Exception:
                        continue
                    nx_vals.append(nxv)
                    gps_vals.append(gps)
                    tps_vals.append(tps)
            if nx_vals:
                # sort by nx to ensure monotonic x-axis
                order = sorted(range(len(nx_vals)), key=lambda i: nx_vals[i])
                data[backend] = {
                    'nx': [nx_vals[i] for i in order],
                    'gps': [gps_vals[i] for i in order],
                    'tps': [tps_vals[i] for i in order],
                }

        if not data:
            print('No valid benchmark data found.')
            return

        # Combined plot: gridpoints/sec vs nx
        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = plt.get_cmap('tab10')
        for i, (backend, vals) in enumerate(sorted(data.items())):
            ax.plot(vals['nx'], vals['gps'], marker='o', linestyle='-', label=backend, color=cmap(i))
        ax.set_xlabel('Grid size (nx = ny)')
        ax.set_ylabel('Gridpoints / sec')
        ax.set_title('Benchmark comparison (gridpoints/sec)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        combined_path = os.path.join(results_dir, 'benchmark_combined.png')
        fig.tight_layout()
        fig.savefig(combined_path, dpi=150)
        fig.savefig(os.path.join(results_dir, 'benchmark_combined.pdf'))
        plt.close(fig)
        print(f'Saved combined benchmark plot: {combined_path} and PDF')

        # Optional: also plot time per step on same figure (secondary y)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for i, (backend, vals) in enumerate(sorted(data.items())):
            ax2.plot(vals['nx'], vals['tps'], marker='o', linestyle='-', label=backend, color=cmap(i))
        ax2.set_xlabel('Grid size (nx = ny)')
        ax2.set_ylabel('Time per step (s)')
        ax2.set_title('Benchmark comparison (time per step)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
        tps_path = os.path.join(results_dir, 'benchmark_time_per_step.png')
        fig2.tight_layout()
        fig2.savefig(tps_path, dpi=150)
        fig2.savefig(os.path.join(results_dir, 'benchmark_time_per_step.pdf'))
        plt.close(fig2)
        print(f'Saved time-per-step plot: {tps_path} and PDF')

        # Save per-backend PNGs (keep previous behavior)
        for backend, vals in data.items():
            fig3, ax3 = plt.subplots()
            ax3.plot(vals['nx'], vals['gps'], marker='o', linestyle='-')
            ax3.set_xlabel('Grid size (nx = ny)')
            ax3.set_ylabel('Gridpoints / sec')
            ax3.set_title(f'Benchmark: {backend}')
            ax3.grid(True)
            per_path = os.path.join(results_dir, f'benchmark_{backend}.png')
            fig3.tight_layout()
            fig3.savefig(per_path, dpi=150)
            plt.close(fig3)
            print(f'Saved per-backend plot: {per_path}')

    plot_benchmark_results(results_dir)
if __name__ == '__main__':
    main()
