"""
Benchmark ML surrogate inference on the same grid sizes used by benchmark.py.

This compares:
- ML inference time per grid size
- implied gridpoints_per_sec for the surrogate
- optional accuracy against the stored target if available

The script assumes a surrogate trained on:
  Inputs  : nx, ny
  Output  : gridpoints_per_sec

Files expected:
  results/surrogate_model.pt
  results/surrogate_norm.npz
  results/benchmark_ml.csv
"""

import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')
MODEL_PATH = os.path.join(RESULTS_DIR, 'surrogate_model.pt')
NORM_PATH = os.path.join(RESULTS_DIR, 'surrogate_norm.npz')


class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list, activation):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    norm = np.load(NORM_PATH)

    model = SurrogateMLP(
        in_dim=int(ckpt['in_dim']),
        out_dim=int(ckpt['out_dim']),
        hidden_dims=ckpt['hidden_dims'].tolist(),
        activation=nn.SiLU,
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, norm, ckpt


def predict_gps(model, norm, nx, ny):
    x = np.array([[nx, ny]], dtype=np.float32)
    x_n = (x - norm['x_mean']) / norm['x_std']
    with torch.no_grad():
        y_n = model(torch.tensor(x_n, dtype=torch.float32)).numpy()
    y = y_n * norm['y_std'] + norm['y_mean']
    return float(y[0, 0])


def benchmark_ml(model, norm, nx, ny, repeats=1000):
    # Average pure inference latency over many calls
    x = np.array([[nx, ny]], dtype=np.float32)
    x_n = (x - norm['x_mean']) / norm['x_std']
    x_t = torch.tensor(x_n, dtype=torch.float32)

    with torch.no_grad():
        _ = model(x_t)  # warmup
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x_t)
        t1 = time.perf_counter()

    avg_infer_s = (t1 - t0) / repeats
    gps = predict_gps(model, norm, nx, ny)
    return avg_infer_s, gps


def plot_results(rows, out_path_png, out_path_pdf):
    sizes = [r['nx'] for r in rows]
    gps = [r['gridpoints_per_sec'] for r in rows]
    infer_ms = [r['infer_ms'] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(sizes, gps, marker='o', linestyle='-', label='ML gridpoints/sec')
    ax1.set_xlabel('Grid size (nx = ny)')
    ax1.set_ylabel('Gridpoints / sec')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(sizes, infer_ms, marker='s', linestyle='--', color='tab:red', label='ML inference ms')
    ax2.set_ylabel('Inference time per sample (ms)')

    fig.suptitle('ML surrogate benchmark')
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=150)
    fig.savefig(out_path_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', default='100,200,300,400,500,1000,2000,3000,4000,5000')
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--out', default=os.path.join(RESULTS_DIR, 'benchmark_ml.csv'))
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(',') if s.strip()]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model, norm, ckpt = load_model()

    header = [
        'backend',
        'nx',
        'ny',
        'n_steps',
        'repeats',
        'total_time_s',
        'time_per_step_s',
        'gridpoints_per_sec',
    ]

    rows = []
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for nx in sizes:
            ny = nx
            infer_s, gps = benchmark_ml(model, norm, nx, ny, repeats=args.repeats)
            row = [
                'ml',
                nx,
                ny,
                1,
                args.repeats,
                f'{infer_s:.9f}',
                f'{infer_s:.9f}',
                f'{gps:.3f}',
            ]
            writer.writerow(row)
            csvfile.flush()

            rows.append({
                'nx': nx,
                'ny': ny,
                'infer_ms': infer_s * 1000.0,
                'gridpoints_per_sec': gps,
            })

            print(f'ml nx={nx} ny={ny} infer={infer_s*1000:.4f} ms  gps={gps:.3f}')

    plot_results(
        rows,
        os.path.join(RESULTS_DIR, 'benchmark_ml.png'),
        os.path.join(RESULTS_DIR, 'benchmark_ml.pdf'),
    )

    print(f'\nSaved {args.out}')
    print(f"Saved {os.path.join(RESULTS_DIR, 'benchmark_ml.png')}")
    print(f"Saved {os.path.join(RESULTS_DIR, 'benchmark_ml.pdf')}")


if __name__ == '__main__':
    main()