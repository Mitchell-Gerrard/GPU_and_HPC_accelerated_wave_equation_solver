"""
Generate training data for the ML surrogate model.

Sweeps over physics parameters (sigma, c, x0_frac, y0_frac) using the CPU
solver and extracts scalar summary statistics from each simulation run.

Scalar summaries chosen (all physically meaningful and cheap to compute):
  - peak_amplitude   : max |u| over all time and space
  - final_energy     : sum of u^2 at final timestep
  - energy_half_time : timestep at which total energy first drops below 50% of initial
  - spread_x / spread_y : RMS width of the field at final timestep (spatial spreading)
  - peak_time        : timestep at which global max amplitude occurs

These are cheap to store (one row per simulation) and give the surrogate
something physically interesting to learn.

Output
------
results/surrogate_data.csv   — parameter + scalar summary per run
results/surrogate_fields.npy — (n_runs, n_steps, nx, ny) float32, full history
                               WARNING: can be large. Set SAVE_FIELDS=False to skip.

Usage
-----
    python generate_surrogate_data.py

Edit the CONFIG section below to change the sweep.
"""

import os
import sys
import csv
import time
import importlib.util

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit these
# ---------------------------------------------------------------------------

NX = NY = 64          # grid size (keep modest for data gen speed)
DX = DY = 0.1
DT = 0.01
N_STEPS = 200         # timesteps per simulation

# Parameter ranges to sweep
SIGMA_VALUES   = np.linspace(2.0, 12.0, 8)       # pulse width
C_VALUES       = np.linspace(0.5, 0.9, 5)         # wave speed (stay CFL-safe)
X0_FRACS       = [0.3, 0.5, 0.7]                  # pulse centre as fraction of NX
Y0_FRACS       = [0.5]                             # pulse centre as fraction of NY

SAVE_FIELDS    = False   # set True to save full field history (large!)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, 'src')
RESULTS_DIR    = os.path.join(REPO_ROOT, 'results')

# ---------------------------------------------------------------------------
# Load CPU solver (same dynamic loading pattern as benchmark.py)
# ---------------------------------------------------------------------------

def load_cpu_solver():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'wave_cpu.py'),
        os.path.join(script_dir, 'src', 'wave_cpu.py'),
        os.path.join(script_dir, '..', 'src', 'wave_cpu.py'),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location('wave_cpu', path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"Loaded wave_cpu from {path}")
            return mod
    raise ImportError("wave_cpu.py not found. Searched:\n" + "\n".join(candidates))


# ---------------------------------------------------------------------------
# Scalar summary extraction
# ---------------------------------------------------------------------------

def extract_summaries(history: np.ndarray, dt: float) -> dict:
    """
    Extract scalar summary statistics from a full solution history.

    Parameters
    ----------
    history : np.ndarray, shape (n_steps, nx, ny)
    dt      : float, timestep size

    Returns
    -------
    dict of scalar floats
    """
    n_steps, nx, ny = history.shape

    # energy at each timestep: sum of u^2 over spatial grid
    energy = np.sum(history**2, axis=(1, 2))          # shape (n_steps,)
    initial_energy = energy[0] if energy[0] > 0 else 1.0

    # peak amplitude (global max |u| over all time and space)
    abs_history = np.abs(history)
    peak_amplitude = float(abs_history.max())

    # timestep at which peak amplitude occurs
    flat_idx = np.argmax(abs_history)
    peak_time_step = int(np.unravel_index(flat_idx, abs_history.shape)[0])
    peak_time = peak_time_step * dt

    # final energy
    final_energy = float(energy[-1])

    # energy half-time: first step where energy < 50% of initial
    half_energy = 0.5 * initial_energy
    below_half = np.where(energy < half_energy)[0]
    if len(below_half) > 0:
        energy_half_time = float(below_half[0] * dt)
    else:
        energy_half_time = float(n_steps * dt)   # never dropped below half

    # spatial spread at final timestep: RMS width in x and y
    final_field = np.abs(history[-1])             # shape (nx, ny)
    total_weight = final_field.sum() + 1e-12      # avoid division by zero

    x_coords = np.arange(nx).reshape(-1, 1)
    y_coords = np.arange(ny).reshape(1, -1)

    mean_x = float((final_field * x_coords).sum() / total_weight)
    mean_y = float((final_field * y_coords).sum() / total_weight)

    spread_x = float(np.sqrt((final_field * (x_coords - mean_x)**2).sum() / total_weight))
    spread_y = float(np.sqrt((final_field * (y_coords - mean_y)**2).sum() / total_weight))

    return {
        'peak_amplitude'   : peak_amplitude,
        'peak_time'        : peak_time,
        'final_energy'     : final_energy,
        'energy_half_time' : energy_half_time,
        'spread_x'         : spread_x,
        'spread_y'         : spread_y,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # load solver
    mod         = load_cpu_solver()
    WaveSolver  = mod.WaveSolver2D
    make_pulse  = mod.create_gaussian_pulse

    # build parameter grid
    param_grid = [
        (sigma, c, x0_frac, y0_frac)
        for sigma    in SIGMA_VALUES
        for c        in C_VALUES
        for x0_frac  in X0_FRACS
        for y0_frac  in Y0_FRACS
    ]

    n_runs = len(param_grid)
    print(f"\nGenerating {n_runs} simulations on a {NX}x{NY} grid, {N_STEPS} steps each.")
    print(f"Estimated time: rough guess {n_runs * 0.5:.0f}–{n_runs * 2:.0f}s on CPU\n")

    csv_path = os.path.join(RESULTS_DIR, 'surrogate_data.csv')
    fieldnames = ['run_id', 'sigma', 'c', 'x0_frac', 'y0_frac',
                  'peak_amplitude', 'peak_time', 'final_energy',
                  'energy_half_time', 'spread_x', 'spread_y',
                  'wall_time_s']

    all_histories = [] if SAVE_FIELDS else None

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for run_id, (sigma, c, x0_frac, y0_frac) in enumerate(param_grid):
            x0 = int(x0_frac * NX)
            y0 = int(y0_frac * NY)

            # CFL check — skip unsafe combinations silently
            import math
            cfl = c * DT * math.sqrt(1/DX**2 + 1/DY**2)
            if cfl >= 1.0:
                print(f"  run {run_id:04d} SKIP (CFL={cfl:.3f} >= 1, c={c:.3f})")
                continue

            t0 = time.perf_counter()

            solver = WaveSolver(NX, NY, DX, DY, DT, c)
            u0     = make_pulse(NX, NY, x0, y0, sigma=sigma)
            solver.set_initial_conditions(u0)
            history = solver.solve(N_STEPS)   # (N_STEPS, NX, NY)

            wall_time = time.perf_counter() - t0

            summaries = extract_summaries(history, DT)

            row = {
                'run_id'    : run_id,
                'sigma'     : f'{sigma:.4f}',
                'c'         : f'{c:.4f}',
                'x0_frac'   : f'{x0_frac:.4f}',
                'y0_frac'   : f'{y0_frac:.4f}',
                'wall_time_s': f'{wall_time:.4f}',
                **{k: f'{v:.6f}' for k, v in summaries.items()},
            }
            writer.writerow(row)
            csvfile.flush()

            if SAVE_FIELDS:
                all_histories.append(history.astype(np.float32))

            print(f"  run {run_id:04d}/{n_runs}  sigma={sigma:.1f}  c={c:.2f}  "
                  f"x0={x0_frac:.1f}  peak={summaries['peak_amplitude']:.4f}  "
                  f"t={wall_time:.2f}s")

    print(f"\nCSV saved to {csv_path}")

    if SAVE_FIELDS and all_histories:
        npy_path = os.path.join(RESULTS_DIR, 'surrogate_fields.npy')
        np.save(npy_path, np.stack(all_histories, axis=0))
        print(f"Full field history saved to {npy_path}  "
              f"shape={np.stack(all_histories,axis=0).shape}")

    print("\nDone. Next step: train_surrogate.py")


if __name__ == '__main__':
    main()