import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path



if __name__ == "__main__":


    files = sorted(
        Path("results").glob("benchmark_mpi_*ranks.csv"),
        key=lambda p: int(p.stem.split("_")[-1].replace("ranks", ""))
    )
    print(files)
    dfs = {
        int(f.stem.split("_")[-1].replace("ranks", "")): pd.read_csv(f)
        for f in files
    }

    print(dfs.keys())
    for ranks, df in dfs.items():
        plt.plot(df['ny'], df['gridpoints_per_sec'], marker='o', label=f'{ranks} ranks')
    plt.xlabel("Grid Size")
    plt.ylabel("Grid Points per Second")
    plt.legend()
    plt.savefig("results/benchmark_mpi_comparison.png")
    plt.clf()