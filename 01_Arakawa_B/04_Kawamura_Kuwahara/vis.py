"""
********************************************************************************
visualization
********************************************************************************
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyevtk

sys.path.append(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
from reference import *

################################################################################

parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--dx", type=float, default=2e-2, help="grid spacing")
parser.add_argument("-r", "--Re", type=float, default=1e3, help="Reynolds number")
# parser.add_argument("-t", "--time", type=float, default=120., help="maximum simulation time")
# parser.add_argument("-u", "--u_tol", type=float, default=1e-6, help="convergence tolerance for velocity")
# parser.add_argument("-p", "--p_tol", type=float, default=1e-6, help="convergence tolerance for pressure")
# parser.add_argument("-i", "--it_max", type=int, default=int(5e3), help="maximum iteration for PPE")
# parser.add_argument("-s", "--Cs", type=float, default=.1, help="Smagorinsky constant")
args = parser.parse_args()

################################################################################

def main():
    # domain
    Lx, Ly, Lz = 1., 1., 1.
    dx, dy, dz = 1e-2, 1e-2, 1e-2
    x = np.arange(0. - dx, Lx + 2. * dx, dx)   # long stencil
    y = np.arange(0. - dy, Ly + 2. * dy, dy)
    z = np.arange(0. - dz, Lz + 2. * dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # arguments
    Re = args.Re
    dir_res = Path(f"Re{Re:.0f}")
    dir_vis = dir_res / "vis"
    dir_vis.mkdir(parents=True, exist_ok=True)
    print(f"Reynolds number: {Re:.0f}")
    path_data = Path(f"Re{Re:.0f}")
    path_data_npz = path_data / "npz"
    data = np.load(path_data_npz / "data_it001000.npz", allow_pickle=True)
    print(f"data: {data.files}")

    # get all npz files
    files = sorted(glob.glob(str(path_data_npz / "*.npz")))
    print(f"files: {files}")

    # reference solutions
    ref_Jiang = Jiang(Re)
    ref_Wong = Wong(Re)
    df_Jiang = pd.DataFrame(ref_Jiang)
    df_Wong = pd.DataFrame(ref_Wong)

    for idx, file in enumerate(files):
        print(f"Processing file: {idx+1}/{len(files)}: {file}")
        data = np.load(file, allow_pickle=True)
        x = data["x"]
        y = data["y"]
        z = data["z"]
        X = data["X"]
        Y = data["Y"]
        Z = data["Z"]
        Xp = X[:-1, :-1, :-1] + dx / 2.
        Yp = Y[:-1, :-1, :-1] + dy / 2.
        Zp = Z[:-1, :-1, :-1] + dz / 2.

        u = data["u"]
        v = data["v"]
        w = data["w"]
        p = data["p"]
        p_bar = data["p_bar"]
        dt = data["dt"]
        t = data["t"]

        if Re in [100., 400., 1000.]:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(df_Jiang["u"], df_Jiang["z"], marker="1", label="Jiang et al., 1994")
            ax.scatter(df_Wong["u"],  df_Wong["z"],  marker="2", label="Wong & Baker, 2002")
            ax.plot(u[Nx//2, Ny//2, 1:-1], z[1:-1], c="k", ls="--", label="Present")
            ax.legend()
            ax.set(
                xticks=([-.2, 0., .2, .4, .6, .8, 1.]),
                yticks=([     0., .2, .4, .6, .8, 1.]),
                xlim=(-.35, 1.15),
                ylim=(-.15, 1.15),
                xlabel=r"$u$",
                ylabel=r"$z$",
                # title=rf"$t = {t:.3f} \ [\text{{s}}]$",
                aspect="equal",
            )
            fig.tight_layout()
            fig.savefig(dir_res / f"vel_comparison.png")
            plt.close(fig)


        # visualized object
        visualize = "velocity"  # "velocity", "pressure", "vorticity"
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={"projection": "3d"})

        if visualize == "velocity":
            vel_norm = np.sqrt(u**2 + v**2 + w**2)
            vmin, vmax = 0., 1.
            cf = ax[0].scatter(
                X[2:-2, 2:-2, 1:Nz//2], Y[2:-2, 2:-2, 1:Nz//2], Z[2:-2, 2:-2, 1:Nz//2],
                c=vel_norm[2:-2, 2:-2, 1:Nz//2],
                vmin=vmin, vmax=vmax, cmap="turbo", marker=".", alpha=.4
            )
            ax[0].set_title(rf"$z={Lz/2:.2f}$",)
            cf = ax[1].scatter(
                X[2:-2, Ny//2:-1, 2:-2], Y[2:-2, Ny//2:-1, 2:-2], Z[2:-2, Ny//2:-1, 2:-2],
                c=vel_norm[2:-2, Ny//2:-1, 2:-2],
                vmin=vmin, vmax=vmax, cmap="turbo", marker=".", alpha=.4
            )
            ax[1].set_title(rf"$y={Ly/2:.2f}$",)
            cf = ax[2].scatter(
                X[1:Nx//2, 2:-2, 2:-2], Y[1:Nx//2, 2:-2, 2:-2], Z[1:Nx//2, 2:-2, 2:-2],
                c=vel_norm[1:Nx//2, 2:-2, 2:-2],
                vmin=vmin, vmax=vmax, cmap="turbo", marker=".", alpha=.4
            )
            ax[2].set_title(rf"$x={Lx/2:.2f}$",)

        elif visualize == "pressure":
            vmin, vmax = -.02, .02
            cf = ax[0].scatter(
                Xp[1:-1, 1:-1, 1:Nz//2], Yp[1:-1, 1:-1, 1:Nz//2], Zp[1:-1, 1:-1, 1:Nz//2],
                c=p_bar[1:-1, 1:-1, 1:Nz//2],
                vmin=vmin, vmax=vmax, cmap="turbo", marker=".", alpha=.4
            )
            ax[0].set_title(rf"$z={Lz/2:.2f}$",)
            cf = ax[1].scatter(
                Xp[1:-1, Ny//2:-1, 1:-1], Yp[1:-1, Ny//2:-1, 1:-1], Zp[1:-1, Ny//2:-1, 1:-1],
                c=p_bar[1:-1, Ny//2:-1, 1:-1],
                vmin=vmin, vmax=vmax, cmap="turbo", marker=".", alpha=.4
            )
            ax[1].set_title(rf"$y={Ly/2:.2f}$",)
            cf = ax[2].scatter(
                Xp[1:Nx//2, 1:-1, 1:-1], Yp[1:Nx//2, 1:-1, 1:-1], Zp[1:Nx//2, 1:-1, 1:-1],
                c=p_bar[1:Nx//2, 1:-1, 1:-1],
                vmin=vmin, vmax=vmax, cmap="turbo", marker=".", alpha=.4
            )
            ax[2].set_title(rf"$x={Lx/2:.2f}$",)

        elif visualize == "vorticity":
            vmin, vmax = -5., 5.
            vor_x = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2. * dy) \
                    - (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2. * dz)
            vor_y = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2. * dz) \
                    - (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2. * dx)
            vor_z = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2. * dx) \
                    - (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2. * dy)
            cf = ax[0].scatter(
                X[2:-2, 2:-2, 1:Nz//2], Y[2:-2, 2:-2, 1:Nz//2], Z[2:-2, 2:-2, 1:Nz//2],
                c=vor_x[1:-1, 1:-1, 1:Nz//2],
                vmin=vmin, vmax=vmax, cmap="seismic", marker=".", alpha=.4
            )
            ax[0].set_title(rf"$z={Lz/2:.2f}$",)
            cf = ax[1].scatter(
                X[2:-2, Ny//2:-1, 2:-2], Y[2:-2, Ny//2:-1, 2:-2], Z[2:-2, Ny//2:-1, 2:-2],
                c=vor_y[1:-1, Ny//2-1:, 1:-1],
                vmin=vmin, vmax=vmax, cmap="seismic", marker=".", alpha=.4
            )
            ax[1].set_title(rf"$y={Ly/2:.2f}$",)
            cf = ax[2].scatter(
                X[1:Nx//2, 2:-2, 2:-2], Y[1:Nx//2, 2:-2, 2:-2], Z[1:Nx//2, 2:-2, 2:-2],
                c=vor_z[1:Nx//2, 1:-1, 1:-1],
                vmin=vmin, vmax=vmax, cmap="seismic", marker=".", alpha=.4
            )
            ax[2].set_title(rf"$x={Lx/2:.2f}$",)

        # Add colorbar
        # cbar = fig.colorbar(cf, ax=ax, orientation="horizontal")
        # cbar.set_label(rf"${visualize}$", fontsize=12)

        # Add cube wireframe
        cube_vertices = [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
        ]
        cube_edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
        ]
        for col in range(3):
            for edge in cube_edges:
                points = [cube_vertices[edge[0]], cube_vertices[edge[1]]]
                ax[col].plot3D(*zip(*points), 'k-', alpha=.3, linewidth=.5)

            ax[col].set(
                xlim=(0., 1.),
                ylim=(0., 1.),
                zlim=(0., 1.),
                # xlabel=r"$x$",
                # ylabel=r"$y$",
                # zlabel=r"$z$",
            )
            ax[col].set_axis_off()
        fig.suptitle(rf"$t = {t:.3f} \ [\text{{s}}]$",)
        fig.tight_layout()
        fig.savefig(dir_vis / f"vis_3D.png")
        fig.savefig(dir_vis / f"vis_3D_t{t:.3f}.png")
        plt.close(fig)



################################################################################

def plot_setting():
    plt.style.use("default")
    # plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("classic")
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True   # background grid
    plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300

################################################################################

if __name__ == "__main__":
    plot_setting()
    main()
