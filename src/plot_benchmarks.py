import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmasher as cmr
plt.rcParams['figure.dpi']=200
mapyboi='cmr.bubblegum'
plt.rcParams['image.cmap']=mapyboi
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=cmr.take_cmap_colors('cmr.bubblegum_r', 3, cmap_range=(0.15, 0.8), return_fmt='hex'))
plt.rcParams.update({'font.size': 12})













def main():

    csvs=["results/benchmark_mpi.csv","results/benchmark_cpu.csv","results/benchmark_opencl.csv"]
    labels=["MPI","CPU","OpenCL"]
    
    fig, ax = plt.subplots(figsize=(8,6))
    for csv,label in zip(csvs,labels):
        df=pd.read_csv(csv)
        ax.plot(df['ny'],df['gridpoints_per_sec'],marker='o',label=label)
    ax.set_yscale('log')
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Grid Points per Second")
 
    ax.legend()
    plt.savefig("results/benchmark_comparison.png")
    plt.clf()
    cpu_df=pd.read_csv("results/benchmark_cpu.csv")
    opencl_df=pd.read_csv("results/benchmark_opencl.csv")
    mpi_df=pd.read_csv("results/benchmark_mpi.csv")
    cpu_speed=cpu_df['gridpoints_per_sec'].to_numpy()
    opencl_speed=opencl_df['gridpoints_per_sec'].to_numpy()
    mpi_speed=mpi_df['gridpoints_per_sec'].to_numpy()
    speedup_mpi=mpi_speed/cpu_speed
    speedup_opencl=opencl_speed/cpu_speed
    plt.plot(cpu_df['ny'],speedup_mpi,marker='o',label='MPI')
    plt.plot(cpu_df['ny'],speedup_opencl,marker='o',label='OpenCL')
    plt.plot(cpu_df['ny'],np.ones_like(cpu_df['ny']),linestyle='--',label='CPU')
    plt.xlabel("Grid Size")
    plt.ylabel("Speedup relative to CPU")
  
    plt.legend()
    plt.savefig("results/benchmark_speedup_comparison.png")
    plt.clf()
    max_speedup_mpi=np.max(speedup_mpi)
    max_speedup_opencl=np.max(speedup_opencl)
    print(f"Max speedup MPI: {max_speedup_mpi:.2f}")
    print(f"Max speedup OpenCL: {max_speedup_opencl:.2f}")










if __name__ == '__main__':
    main()
