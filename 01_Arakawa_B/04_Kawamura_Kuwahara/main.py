"""
********************************************************************************
3D lid-driven cavity flow simulation with Finite Difference Method

space:
    advection:        Kawamura-Kuwahara (Kawamura&Kuwahara 1984)
    diffusion:         2nd order central
    pressure gradient: 2nd order central

time:
    projection method (Chorin 1968)

grid:
    staggered grid (Arakawa B-type grid, Arakawa&Lamb 1977)
********************************************************************************
"""

import os
import sys
import time
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


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dx", type=float, default=2e-2, help="grid spacing")
parser.add_argument("-r", "--Re", type=float, default=1e3, help="Reynolds number")
parser.add_argument("-t", "--time", type=float, default=120., help="maximum simulation time")
parser.add_argument("-u", "--u_tol", type=float, default=1e-6, help="convergence tolerance for velocity")
parser.add_argument("-p", "--p_tol", type=float, default=1e-6, help="convergence tolerance for pressure")
parser.add_argument("-i", "--it_max", type=int, default=int(5e3), help="maximum iteration for PPE")
parser.add_argument("-s", "--Cs", type=float, default=.1, help="Smagorinsky constant")
args = parser.parse_args()


def plot_setting():
    plt.style.use("default")
    # plt.style.use("seaborn-v0_8-deep")
    plt.style.use("seaborn-v0_8-poster")   # paper / notebook / talk / poster
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    # plt.rcParams["axes.grid"] = True
    # plt.rcParams["grid.alpha"] = .3
    plt.rcParams["legend.framealpha"] = 1.
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300


def get_advection(u, v, w, dx, dy, dz, beta=1./4.):
    # advection in x axis
    u_u_x = u[2:-2, 2:-2, 2:-2] * (- u[4:, 2:-2, 2:-2] + 8. * u[3:-1, 2:-2, 2:-2] - 8. * u[1:-3, 2:-2, 2:-2] + u[:-4, 2:-2, 2:-2]) / (12. * dx) \
            + beta * np.abs(u[2:-2, 2:-2, 2:-2]) * (u[4:, 2:-2, 2:-2] - 4. * u[3:-1, 2:-2, 2:-2] + 6. * u[2:-2, 2:-2, 2:-2] - 4. * u[1:-3, 2:-2, 2:-2] + u[:-4, 2:-2, 2:-2]) / dx
    v_u_y = v[2:-2, 2:-2, 2:-2] * (- u[2:-2, 4:, 2:-2] + 8. * u[2:-2, 3:-1, 2:-2] - 8. * u[2:-2, 1:-3, 2:-2] + u[2:-2, :-4, 2:-2]) / (12. * dy) \
            + beta * np.abs(v[2:-2, 2:-2, 2:-2]) * (u[2:-2, 4:, 2:-2] - 4. * u[2:-2, 3:-1, 2:-2] + 6. * u[2:-2, 2:-2, 2:-2] - 4. * u[2:-2, 1:-3, 2:-2] + u[2:-2, :-4, 2:-2]) / dy
    w_u_z = w[2:-2, 2:-2, 2:-2] * (- u[2:-2, 2:-2, 4:] + 8. * u[2:-2, 2:-2, 3:-1] - 8. * u[2:-2, 2:-2, 1:-3] + u[2:-2, 2:-2, :-4]) / (12. * dz) \
            + beta * np.abs(w[2:-2, 2:-2, 2:-2]) * (u[2:-2, 2:-2, 4:] - 4. * u[2:-2, 2:-2, 3:-1] + 6. * u[2:-2, 2:-2, 2:-2] - 4. * u[2:-2, 2:-2, 1:-3] + u[2:-2, 2:-2, :-4]) / dz
    advc_x = u_u_x + v_u_y + w_u_z

    # advection in y axis
    u_v_x = u[2:-2, 2:-2, 2:-2] * (- v[4:, 2:-2, 2:-2] + 8. * v[3:-1, 2:-2, 2:-2] - 8. * v[1:-3, 2:-2, 2:-2] + v[:-4, 2:-2, 2:-2]) / (12. * dx) \
            + beta * np.abs(u[2:-2, 2:-2, 2:-2]) * (v[4:, 2:-2, 2:-2] - 4. * v[3:-1, 2:-2, 2:-2] + 6. * v[2:-2, 2:-2, 2:-2] - 4. * v[1:-3, 2:-2, 2:-2] + v[:-4, 2:-2, 2:-2]) / dx
    v_v_y = v[2:-2, 2:-2, 2:-2] * (- v[2:-2, 4:, 2:-2] + 8. * v[2:-2, 3:-1, 2:-2] - 8. * v[2:-2, 1:-3, 2:-2] + v[2:-2, :-4, 2:-2]) / (12. * dy) \
            + beta * np.abs(v[2:-2, 2:-2, 2:-2]) * (v[2:-2, 4:, 2:-2] - 4. * v[2:-2, 3:-1, 2:-2] + 6. * v[2:-2, 2:-2, 2:-2] - 4. * v[2:-2, 1:-3, 2:-2] + v[2:-2, :-4, 2:-2]) / dy
    w_v_z = w[2:-2, 2:-2, 2:-2] * (- v[2:-2, 2:-2, 4:] + 8. * v[2:-2, 2:-2, 3:-1] - 8. * v[2:-2, 2:-2, 1:-3] + v[2:-2, 2:-2, :-4]) / (12. * dz) \
            + beta * np.abs(w[2:-2, 2:-2, 2:-2]) * (v[2:-2, 2:-2, 4:] - 4. * v[2:-2, 2:-2, 3:-1] + 6. * v[2:-2, 2:-2, 2:-2] - 4. * v[2:-2, 2:-2, 1:-3] + v[2:-2, 2:-2, :-4]) / dz
    advc_y = u_v_x + v_v_y + w_v_z

    # advection in z axis
    u_w_x = u[2:-2, 2:-2, 2:-2] * (- w[4:, 2:-2, 2:-2] + 8. * w[3:-1, 2:-2, 2:-2] - 8. * w[1:-3, 2:-2, 2:-2] + w[:-4, 2:-2, 2:-2]) / (12. * dx) \
            + beta * np.abs(u[2:-2, 2:-2, 2:-2]) * (w[4:, 2:-2, 2:-2] - 4. * w[3:-1, 2:-2, 2:-2] + 6. * w[2:-2, 2:-2, 2:-2] - 4. * w[1:-3, 2:-2, 2:-2] + w[:-4, 2:-2, 2:-2]) / dx
    v_w_y = v[2:-2, 2:-2, 2:-2] * (- w[2:-2, 4:, 2:-2] + 8. * w[2:-2, 3:-1, 2:-2] - 8. * w[2:-2, 1:-3, 2:-2] + w[2:-2, :-4, 2:-2]) / (12. * dy) \
            + beta * np.abs(v[2:-2, 2:-2, 2:-2]) * (w[2:-2, 4:, 2:-2] - 4. * w[2:-2, 3:-1, 2:-2] + 6. * w[2:-2, 2:-2, 2:-2] - 4. * w[2:-2, 1:-3, 2:-2] + w[2:-2, :-4, 2:-2]) / dy
    w_w_z = w[2:-2, 2:-2, 2:-2] * (- w[2:-2, 2:-2, 4:] + 8. * w[2:-2, 2:-2, 3:-1] - 8. * w[2:-2, 2:-2, 1:-3] + w[2:-2, 2:-2, :-4]) / (12. * dz) \
            + beta * np.abs(w[2:-2, 2:-2, 2:-2]) * (w[2:-2, 2:-2, 4:] - 4. * w[2:-2, 2:-2, 3:-1] + 6. * w[2:-2, 2:-2, 2:-2] - 4. * w[2:-2, 2:-2, 1:-3] + w[2:-2, 2:-2, :-4]) / dz
    advc_z = u_w_x + v_w_y + w_w_z

    return advc_x, advc_y, advc_z


def get_diffusion(u, v, w, dx, dy, dz, nu, Cs=.1):
    # Smagorinsky model
    # strain rate tensor
    u_x = (u[2:-2, 2:-2, 2:-2] - u[1:-3, 2:-2, 2:-2]) / dx
    u_y = (u[2:-2, 2:-2, 2:-2] - u[2:-2, 1:-3, 2:-2]) / dy
    u_z = (u[2:-2, 2:-2, 2:-2] - u[2:-2, 2:-2, 1:-3]) / dz
    v_x = (v[2:-2, 2:-2, 2:-2] - v[1:-3, 2:-2, 2:-2]) / dx
    v_y = (v[2:-2, 2:-2, 2:-2] - v[2:-2, 1:-3, 2:-2]) / dy
    v_z = (v[2:-2, 2:-2, 2:-2] - v[2:-2, 2:-2, 1:-3]) / dz
    w_x = (w[2:-2, 2:-2, 2:-2] - w[1:-3, 2:-2, 2:-2]) / dx
    w_y = (w[2:-2, 2:-2, 2:-2] - w[2:-2, 1:-3, 2:-2]) / dy
    w_z = (w[2:-2, 2:-2, 2:-2] - w[2:-2, 2:-2, 1:-3]) / dz
    S11, S12, S13 = u_x, .5 * (u_y + v_x), .5 * (u_z + w_x)
    S21, S22, S23 = S12, v_y, .5 * (v_z + w_y)
    S31, S32, S33 = S13, S23, w_z
    S = np.sqrt(
        2. * (
            S11**2 + S22**2 + S33**2
            + 2. * (S12**2 + S13**2 + S23**2)
        )
    )

    # # van Driest damping function
    # kappa = .41
    # A = 26.
    # B = 5.3
    # lm = kappa * 

    # Smagorinsky model
    ls = Cs * dx
    l0 = ls   # min(lm, ls)
    nu_t = l0**2 * S
    nu_eff = nu + nu_t
    print(f"   >>> nu_eff -> min: {nu_eff.min():.6e}, max: {nu_eff.max():.6e}, mean: {nu_eff.mean():.6e}")

    # diffusion of u
    u_xx = (u[3:-1, 2:-2, 2:-2] - 2. * u[2:-2, 2:-2, 2:-2] + u[1:-3, 2:-2, 2:-2]) / dx**2
    u_yy = (u[2:-2, 3:-1, 2:-2] - 2. * u[2:-2, 2:-2, 2:-2] + u[2:-2, 1:-3, 2:-2]) / dy**2
    u_zz = (u[2:-2, 2:-2, 3:-1] - 2. * u[2:-2, 2:-2, 2:-2] + u[2:-2, 2:-2, 1:-3]) / dz**2
    lap_u = u_xx + u_yy + u_zz
    diff_x = nu_eff * lap_u

    # diffusion of v
    v_xx = (v[3:-1, 2:-2, 2:-2] - 2. * v[2:-2, 2:-2, 2:-2] + v[1:-3, 2:-2, 2:-2]) / dx**2
    v_yy = (v[2:-2, 3:-1, 2:-2] - 2. * v[2:-2, 2:-2, 2:-2] + v[2:-2, 1:-3, 2:-2]) / dy**2
    v_zz = (v[2:-2, 2:-2, 3:-1] - 2. * v[2:-2, 2:-2, 2:-2] + v[2:-2, 2:-2, 1:-3]) / dz**2
    lap_v = v_xx + v_yy + v_zz
    diff_y = nu_eff * lap_v

    # diffusion of w
    w_xx = (w[3:-1, 2:-2, 2:-2] - 2. * w[2:-2, 2:-2, 2:-2] + w[1:-3, 2:-2, 2:-2]) / dx**2
    w_yy = (w[2:-2, 3:-1, 2:-2] - 2. * w[2:-2, 2:-2, 2:-2] + w[2:-2, 1:-3, 2:-2]) / dy**2
    w_zz = (w[2:-2, 2:-2, 3:-1] - 2. * w[2:-2, 2:-2, 2:-2] + w[2:-2, 2:-2, 1:-3]) / dz**2
    lap_w = w_xx + w_yy + w_zz
    diff_z = nu_eff * lap_w

    return diff_x, diff_y, diff_z


def get_source(u, v, w, dx, dy, dz, dt, b):
    # source term for PPE (for Arakawa B-type grid)
    # interpolate u_hat, v_hat, w_hat to cell center
    u_x = 1. / 4. * (
        (u[2:-1, 1:-2, 1:-2] - u[1:-2, 1:-2, 1:-2]) / dx \
        + (u[2:-1, 2:-1, 1:-2] - u[1:-2, 2:-1, 1:-2]) / dx \
        + (u[2:-1, 1:-2, 2:-1] - u[1:-2, 1:-2, 2:-1]) / dx \
        + (u[2:-1, 2:-1, 2:-1] - u[1:-2, 2:-1, 2:-1]) / dx
    )
    v_y = 1. / 4. * (
        (v[1:-2, 2:-1, 1:-2] - v[1:-2, 1:-2, 1:-2]) / dy \
        + (v[2:-1, 2:-1, 1:-2] - v[2:-1, 1:-2, 1:-2]) / dy \
        + (v[1:-2, 2:-1, 2:-1] - v[1:-2, 1:-2, 2:-1]) / dy \
        + (v[2:-1, 2:-1, 2:-1] - v[2:-1, 1:-2, 2:-1]) / dy
    )
    w_z = 1. / 4. * (
        (w[1:-2, 1:-2, 2:-1] - w[1:-2, 1:-2, 1:-2]) / dz \
        + (w[2:-1, 1:-2, 2:-1] - w[2:-1, 1:-2, 1:-2]) / dz \
        + (w[1:-2, 2:-1, 2:-1] - w[1:-2, 2:-1, 1:-2]) / dz \
        + (w[2:-1, 2:-1, 2:-1] - w[2:-1, 2:-1, 1:-2]) / dz
    )
    div_u = u_x + v_y + w_z   # divergence mapped to cell center
    b[1:-1, 1:-1, 1:-1] = div_u / dt
    return b, div_u


def Jacobi(p, b, dx, dy, dz, Nx, Ny, Nz, it_max, tol):
    # point Jacobi method
    for it in range(0, it_max+1):
        # previous pressure
        p_old = p.copy()

        # point Jacobi method (for Arakawa B-type grid)
        p[1:-1, 1:-1, 1:-1] = 1. / (2. * (dy**2 * dz**2 + dz**2 * dx**2 + dx**2 * dy**2)) \
                        * (
                            - b[1:-1, 1:-1, 1:-1] * dx**2 * dy**2 * dz**2 \
                            + (p_old[2:, 1:-1, 1:-1] + p_old[:-2, 1:-1, 1:-1]) * dy**2 * dz**2 \
                            + (p_old[1:-1, 2:, 1:-1] + p_old[1:-1, :-2, 1:-1]) * dz**2 * dx**2 \
                            + (p_old[1:-1, 1:-1, 2:] + p_old[1:-1, 1:-1, :-2]) * dx**2 * dy**2
                        )

        # boundary condition (for Arakawa B-type grid)
        p[0,  :, :] = p[1,  :, :]   # x = xmin plane
        p[-1, :, :] = p[-2, :, :]   # x = xmax plane
        p[:,  0, :] = p[:,  1, :]   # y = ymin plane
        p[:, -1, :] = p[:, -2, :]   # y = ymax plane
        p[:, :,  0] = p[:, :,  1]   # z = zmin plane
        p[:, :, -1] = p[:, :, -2]   # z = zmax plane
        p[1, Ny//2, 1] = 0.
        # p[1, 1, 1] = 0.   # (x, y, z) = (xmin, ymin, zmin) corner

        # converged?
        p_res = np.sqrt(np.sum((p - p_old)**2)) / np.sqrt(np.sum(p_old**2))
        if it % 1000 == 0:
            print(f"   >>> PPE -> it: {it}, p_res: {p_res:.6e}")
        if p_res < tol:
            print(f"   >>> PPE converged")
            break
    return p


def get_pressure_gradient(p, dx, dy, dz):
    # interpolate p to cell edge,
    # then apply average pressure gradient to correct velocity
    p_x = 1. / 4. * (
        (p[2:-1, 1:-2, 1:-2] - p[1:-2, 1:-2, 1:-2]) / dx \
        + (p[2:-1, 2:-1, 1:-2] - p[1:-2, 2:-1, 1:-2]) / dx \
        + (p[2:-1, 1:-2, 2:-1] - p[1:-2, 1:-2, 2:-1]) / dx \
        + (p[2:-1, 2:-1, 2:-1] - p[1:-2, 2:-1, 2:-1]) / dx
    )
    p_y = 1. / 4. * (
        (p[1:-2, 2:-1, 1:-2] - p[1:-2, 1:-2, 1:-2]) / dy \
        + (p[2:-1, 2:-1, 1:-2] - p[2:-1, 1:-2, 1:-2]) / dy \
        + (p[1:-2, 2:-1, 2:-1] - p[1:-2, 1:-2, 2:-1]) / dy \
        + (p[2:-1, 2:-1, 2:-1] - p[2:-1, 1:-2, 2:-1]) / dy
    )
    p_z = 1. / 4. * (
        (p[1:-2, 1:-2, 2:-1] - p[1:-2, 1:-2, 1:-2]) / dz \
        + (p[2:-1, 1:-2, 2:-1] - p[2:-1, 1:-2, 1:-2]) / dz \
        + (p[1:-2, 2:-1, 2:-1] - p[1:-2, 2:-1, 1:-2]) / dz \
        + (p[2:-1, 2:-1, 2:-1] - p[2:-1, 2:-1, 1:-2]) / dz
    )
    return p_x, p_y, p_z


def main():
    # plot setting
    plot_setting()

    # arguments
    dx = args.dx
    Re = args.Re

    dir_res = f"Re{Re:.0f}"
    os.makedirs(dir_res, exist_ok=True)

    # domain
    Lx, Ly, Lz = 1., 1., 1.
    dx, dy, dz = dx, dx, dx
    x = np.arange(0. - dx, Lx + 2. * dx, dx)   # long stencil
    y = np.arange(0. - dy, Ly + 2. * dy, dy)
    z = np.arange(0. - dz, Lz + 2. * dz, dz)
    # print(f"x: {x}")
    # print(f"y: {y}")
    # print(f"z: {z}")
    # print(f"z[1:-1]: {z[1:-1]}")
    # exit()
    Nx, Ny, Nz = len(x), len(y), len(z)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    # print(X[:,0,0] == x)
    # print(Y[0,:,0] == y)
    # print(Z[0,0,:] == z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z)
    ax.set(
        xlabel=r"$x$",
        ylabel=r"$y$",
        zlabel=r"$z$",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(dir_res, f"nodes.png"))
    plt.close()

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.scatter(X[0,:,:], Y[0,:,:])
    plt.title("X[0,:,:], Y[0,:,:]")
    plt.subplot(3, 3, 2)
    plt.scatter(X[:,0,:], Y[:,0,:])
    plt.title("X[:,0,:], Y[:,0,:]")
    plt.subplot(3, 3, 3)
    plt.scatter(X[:,:,0], Y[:,:,0])
    plt.title("X[:,:,0], Y[:,:,0]")

    plt.subplot(3, 3, 1+3)
    plt.scatter(Y[0,:,:], Z[0,:,:])
    plt.title("Y[0,:,:], Z[0,:,:]")
    plt.subplot(3, 3, 2+3)
    plt.scatter(Y[:,0,:], Z[:,0,:])
    plt.title("Y[:,0,:], Z[:,0,:]")
    plt.subplot(3, 3, 3+3)
    plt.scatter(Y[:,:,0], Z[:,:,0])
    plt.title("Y[:,:,0], Z[:,:,0]")

    plt.subplot(3, 3, 1+6)
    plt.scatter(Z[0,:,:], X[0,:,:])
    plt.title("Z[0,:,:], X[0,:,:]")
    plt.subplot(3, 3, 2+6)
    plt.scatter(Z[:,0,:], X[:,0,:])
    plt.title("Z[:,0,:], X[:,0,:]")
    plt.subplot(3, 3, 3+6)
    plt.scatter(Z[:,:,0], X[:,:,0])
    plt.title("Z[:,:,0], X[:,:,0]")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_res, f"table.png"))
    plt.close()

    # timestep
    M = 3                        # spatial dimension
    U = 1.                       # characteristic velocity
    h = min(dx, dy, dz)          # discretization parameter
    k = 1. / Re                  # diffusion rate
    dt_c = 1. * h**1 / (U * M)   # Courant number condition
    dt_d = .5 * h**2 / (k * M)   # diffusion number condition
    dt = min(dt_c, dt_d)         # critical timestep
    dt *= .4                     # safety

    # variables
    u = np.zeros(shape=(Nz, Ny, Nx)) + 1e-9
    v = np.zeros(shape=(Nz, Ny, Nx)) + 1e-9
    w = np.zeros(shape=(Nz, Ny, Nx)) + 1e-9
    p = np.zeros(shape=(Nz-1, Ny-1, Nx-1)) + 1e-9
    b = np.zeros(shape=(Nz-1, Ny-1, Nx-1)) + 1e-9

    u_tol = args.u_tol    # convergence tolerance for velocity
    p_tol = args.p_tol    # convergence tolerance for pressure
    it_max = args.it_max  # max iteration

    # reference solutions
    ref_Jiang = Jiang(Re)
    ref_Wong = Wong(Re)

    df_Jiang = pd.DataFrame(ref_Jiang)
    df_Wong = pd.DataFrame(ref_Wong)

    # figure directory
    dir_res = Path(f"Re{Re:.0f}")
    dir_vel = dir_res / "velocity"
    dir_prs = dir_res / "pressure"
    dir_vel_slc = dir_res / "velocity_slice"
    dir_prs_slc = dir_res / "pressure_slice"
    dir_vel_qvr = dir_res / "velocity_quiver"
    dir_vor_slc = dir_res / "vorticity_slice"

    dir_res.mkdir(exist_ok=True)
    dir_vel.mkdir(exist_ok=True)
    dir_prs.mkdir(exist_ok=True)
    dir_vel_slc.mkdir(exist_ok=True)
    dir_prs_slc.mkdir(exist_ok=True)
    dir_vel_qvr.mkdir(exist_ok=True)
    dir_vor_slc.mkdir(exist_ok=True)

    dir_npz = dir_res / "npz"
    dir_npz.mkdir(exist_ok=True)

    dir_vtk = dir_res / "vtk"
    dir_vtk.mkdir(exist_ok=True)

    # main loop
    n = 0
    t = 0.
    u_res = 9999.
    while u_res > u_tol:
        # previous velocity
        u_old = u.copy()
        v_old = v.copy()
        w_old = w.copy()

        # intermediate velocity
        u_hat = u.copy()
        v_hat = v.copy()
        w_hat = w.copy()

        # advection
        advc_x, advc_y, advc_z \
            = get_advection(u_old, v_old, w_old, dx, dy, dz)
        # print(advc_x, advc_y, advc_z)

        # diffusion
        diff_x, diff_y, diff_z \
            = get_diffusion(u_old, v_old, w_old, dx, dy, dz, k, Cs=args.Cs)
        # print(diff_x, diff_y, diff_z)

        # intermediate velocity
        u_hat[2:-2, 2:-2, 2:-2] = u_old[2:-2, 2:-2, 2:-2] + dt * (- advc_x + diff_x)
        v_hat[2:-2, 2:-2, 2:-2] = v_old[2:-2, 2:-2, 2:-2] + dt * (- advc_y + diff_y)
        w_hat[2:-2, 2:-2, 2:-2] = w_old[2:-2, 2:-2, 2:-2] + dt * (- advc_z + diff_z)

        # source term for PPE (for Arakawa B-type grid)
        b, div_u_hat = get_source(u_hat, v_hat, w_hat, dx, dy, dz, dt, b)

        # solve PPE
        p = Jacobi(p, b, dx, dy, dz, Nx, Ny, Nz, it_max, p_tol)

        # pressure gradient (for Arakawa B-type grid)
        p_x, p_y, p_z = get_pressure_gradient(p, dx, dy, dz)
        # if n == 10:
        #     print(p)
        #     print(p_x, p_y, p_z)
        #     exit()

        # velocity correction
        u[2:-2, 2:-2, 2:-2] = u_hat[2:-2, 2:-2, 2:-2] + dt * (- p_x)
        v[2:-2, 2:-2, 2:-2] = v_hat[2:-2, 2:-2, 2:-2] + dt * (- p_y)
        w[2:-2, 2:-2, 2:-2] = w_hat[2:-2, 2:-2, 2:-2] + dt * (- p_z)

        # boundary condition
        u[-2:, :, :], v[-2:, :, :], w[-2:, :, :] = 0., 0., 0.   # x = xmax plane
        u[:2,  :, :], v[:2,  :, :], w[:2,  :, :] = 0., 0., 0.   # x = xmin plane
        u[:, -2:, :], v[:, -2:, :], w[:, -2:, :] = 0., 0., 0.   # y = ymax plane
        u[:,  :2, :], v[:,  :2, :], w[:,  :2, :] = 0., 0., 0.   # y = ymin plane
        u[:, :, -2:], v[:, :, -2:], w[:, :, -2:] = 1., 0., 0.   # z = zmax plane
        u[:, :,  :2], v[:, :,  :2], w[:, :,  :2] = 0., 0., 0.   # z = zmin plane

        # parabola
        U = (X - 0.) * (1. - X) * (Y - 0.) * (1. - Y)
        U /= U.max()
        # u[:, :, -2:], v[:, :, -2:], w[:, :, -2:] = U[:, :, -2:], 0., 0.   # z = zmax plane

        # converged?
        n += 1
        t += dt
        C = (u / dx**1 + v / dy**1 + w / dz**1) * dt
        D = (k / dx**2 + k / dy**2 + k / dz**2) * dt
        u_res = np.sqrt(np.sum((u - u_old)**2)) / np.sqrt(np.sum(u_old**2))
        v_res = np.sqrt(np.sum((v - v_old)**2)) / np.sqrt(np.sum(v_old**2))
        w_res = np.sqrt(np.sum((w - w_old)**2)) / np.sqrt(np.sum(w_old**2))
        print(f"\n****************************************************************")
        print(f">>> main")
        print(f">>> it: {n:d}, t: {t:.3f}/{args.time:.3f}, dx: {dx:.3e}, dt: {dt:.3e}, C: {np.max(C):.3f}, D: {np.max(D):.3f}")
        print(f">>> Reynolds number: {Re:.3e}, Cs: {args.Cs:.3f}")
        print(f">>> u_res: {u_res:.6e}, v_res: {v_res:.6e}, w_res: {w_res:.6e}")
        print(f"****************************************************************")
        u_res = max(u_res, v_res, w_res)
        if u_res < u_tol:
            print("   >>> main converged")
            break
        if t > args.time:
            print("   >>> taking too long, terminating now...")
            break

        # plot
        if n % 1000 == 0:
            # velocity
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            vel_norm = np.sqrt(u**2 + v**2 + w**2)
            cntr = ax.scatter(
                X[1:-1, Ny//2:-1, 1:-1], Y[1:-1, Ny//2:-1, 1:-1], Z[1:-1, Ny//2:-1, 1:-1], 
                c=vel_norm[1:-1, Ny//2:-1, 1:-1], 
                vmin=0., vmax=1., cmap="turbo", marker=".", alpha=.5
            )
            # cntr = ax.scatter(
            #     X[1:-1, 1:-1, 1:-1], Y[1:-1, 1:-1, 1:-1], Z[1:-1, 1:-1, 1:-1], 
            #     c=vel_norm[1:-1, 1:-1, 1:-1], 
            #     vmin=0., vmax=1., cmap="turbo", marker="."
            # )
            # fig.colorbar(cntr, shrink=.5, aspect=15)

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
            for edge in cube_edges:
                points = [cube_vertices[edge[0]], cube_vertices[edge[1]]]
                ax.plot3D(*zip(*points), 'k-', alpha=.3, linewidth=.5)

            ax.set(
                xticks=np.linspace(0., 1., 6),
                yticks=np.linspace(0., 1., 6),
                zticks=np.linspace(0., 1., 6),
                xlim=(0., 1.),
                ylim=(0., 1.),
                zlim=(0., 1.),
                # xlabel=r"$x$",
                # ylabel=r"$y$",
                # zlabel=r"$z$",
                title=rf"$t = {t:.3f} \ [\text{{s}}]$",
            )
            fig.tight_layout()
            fig.savefig(dir_res / f"vel_norm.png")
            fig.savefig(dir_vel / f"vel_norm_{n:06d}.png")
            plt.close(fig)


            fig, ax = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={"projection": "3d"})

            cntr = ax[0].scatter(
                X[1:-1, 1:-1, 1:Nz//2], Y[1:-1, 1:-1, 1:Nz//2], Z[1:-1, 1:-1, 1:Nz//2],
                c=vel_norm[1:-1, 1:-1, 1:Nz//2],
                vmin=0., vmax=1., cmap="turbo", marker=".", alpha=.5
            )
            ax[0].set_title(rf"$z={Lz/2:.2f}$",)
            cntr = ax[1].scatter(
                X[1:-1, Ny//2:-1, 1:-1], Y[1:-1, Ny//2:-1, 1:-1], Z[1:-1, Ny//2:-1, 1:-1],
                c=vel_norm[1:-1, Ny//2:-1, 1:-1],
                vmin=0., vmax=1., cmap="turbo", marker=".", alpha=.5
            )
            ax[1].set_title(rf"$y={Ly/2:.2f}$",)
            cntr = ax[2].scatter(
                X[1:Nx//2, 1:-1, 1:-1], Y[1:Nx//2, 1:-1, 1:-1], Z[1:Nx//2, 1:-1, 1:-1],
                c=vel_norm[1:Nx//2, 1:-1, 1:-1],
                vmin=0., vmax=1., cmap="turbo", marker=".", alpha=.5
            )
            ax[2].set_title(rf"$x={Lx/2:.2f}$",)

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
            fig.savefig(dir_res / f"vel_norm_3D_slice.png")
            fig.savefig(dir_vel / f"vel_norm_3D_slice_{n:06d}.png")
            plt.close(fig)


            # pressure
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            Xp, Yp, Zp = X[:-1, :-1, :-1] + dx / 2., Y[:-1, :-1, :-1] + dy / 2., Z[:-1, :-1, :-1] + dz / 2.
            p_bar = p - np.mean(p)
            cntr = ax.scatter(
                Xp[:,Ny//2:,:], Yp[:,Ny//2:,:], Zp[:,Ny//2:,:],
                c=p_bar[:,Ny//2:,:],
                vmin=-.02, vmax=.02, cmap="turbo", marker="."
            )
            # cntr = ax.scatter(
            #     Xp, Yp, Zp, 
            #     c=p_bar,
            #     vmin=-.1, vmax=.1, cmap="turbo", marker="."
            # )
            # fig.colorbar(cntr, shrink=.5, aspect=15)
            ax.set(
                xlim=(0., 1.),
                ylim=(0., 1.),
                zlim=(0., 1.),
                # xlabel=r"$x$",
                # ylabel=r"$y$",
                # zlabel=r"$z$",
                title=rf"$t = {t:.3f} \ [\text{{s}}]$",
            )
            plt.tight_layout()
            plt.savefig(dir_res / f"prs.png")
            plt.savefig(dir_prs / f"prs_{n:06d}.png")
            plt.close()


            # velocity at the geometric center
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))

            levels = np.linspace(0., 1., 64)
            ticks = np.linspace(0., 1., 5)
            cf = ax[0].contourf(
                X[1:-1, 1:-1, Nz//2], Y[1:-1, 1:-1, Nz//2], vel_norm[1:-1, 1:-1, Nz//2],
                cmap="turbo", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[0])
            ax[0].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Ly, 5),
                xlim=(0., Lx),
                ylim=(0., Ly),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"$z={Lz/2:.2f}$ plane",
                aspect="equal",
            )

            cf = ax[1].contourf(
                X[1:-1, Ny//2, 1:-1], Z[1:-1, Ny//2, 1:-1], vel_norm[1:-1, Ny//2, 1:-1],
                cmap="turbo", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[1])
            ax[1].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Lz, 5),
                xlabel=r"$x$",
                ylabel=r"$z$",
                xlim=(0., Lx),
                ylim=(0., Lz),
                title=rf"$y={Ly/2:.2f}$ plane",
                aspect="equal",
            )

            cf = ax[2].contourf(
                Y[Nx//2, 1:-1, 1:-1], Z[Nx//2, 1:-1, 1:-1], vel_norm[Nx//2, 1:-1, 1:-1],
                cmap="turbo", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[2])
            ax[2].set(
                xticks=np.linspace(0., Ly, 5),
                yticks=np.linspace(0., Lz, 5),
                xlim=(0., Ly),
                ylim=(0., Lz),
                xlabel=r"$y$",
                ylabel=r"$z$",
                title=rf"$x={Lx/2:.2f}$ plane",
                aspect="equal",
            )

            fig.suptitle(rf"$t = {t:.3f} \ [\text{{s}}]$",)
            fig.tight_layout()
            fig.savefig(dir_res / f"vel_norm_slice.png")
            fig.savefig(dir_vel_slc / f"vel_norm_slice_{n:06d}.png")
            plt.close(fig)


            # pressure at the geometric center
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))

            p_lim = .02
            levels = np.linspace(-p_lim, p_lim, 64)
            ticks = np.linspace(-p_lim, p_lim, 5)

            cf = ax[0].contourf(
                Xp[1:-1, 1:-1, Nz//2], Yp[1:-1, 1:-1, Nz//2], p[1:-1, 1:-1, Nz//2],
                cmap="turbo", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[0])
            ax[0].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Ly, 5),
                xlim=(0., Lx),
                ylim=(0., Ly),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"$z={Lz/2:.2f}$ plane",
                aspect="equal",
            )

            cf = ax[1].contourf(
                Xp[1:-1, Ny//2, 1:-1], Zp[1:-1, Ny//2, 1:-1], p[1:-1, Ny//2, 1:-1],
                cmap="turbo", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[1])
            ax[1].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Lz, 5),
                xlabel=r"$x$",
                ylabel=r"$z$",
                xlim=(0., Lx),
                ylim=(0., Lz),
                title=rf"$y={Ly/2:.2f}$ plane",
                aspect="equal",
            )

            cf = ax[2].contourf(
                Yp[Nx//2, 1:-1, 1:-1], Zp[Nx//2, 1:-1, 1:-1], p[Nx//2, 1:-1, 1:-1],
                cmap="turbo", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[2])
            ax[2].set(
                xticks=np.linspace(0., Ly, 5),
                yticks=np.linspace(0., Lz, 5),
                xlim=(0., Ly),
                ylim=(0., Lz),
                xlabel=r"$y$",
                ylabel=r"$z$",
                title=rf"$x={Lx/2:.2f}$ plane",
                aspect="equal",
            )

            fig.suptitle(rf"$t = {t:.3f} \ [\text{{s}}]$",)
            fig.tight_layout()
            fig.savefig(dir_res / f"prs_slice.png")
            fig.savefig(dir_prs_slc / f"prs_slice_{n:06d}.png")
            plt.close(fig)


            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ticks = np.linspace(0., 1., 5)
            step = Nx // 20
            u_norm = u / vel_norm
            v_norm = v / vel_norm
            w_norm = w / vel_norm
            qv = ax[0].quiver(
                X[1:-1:step, 1:-1:step, Nz//2], Y[1:-1:step, 1:-1:step, Nz//2],
                u_norm[1:-1:step, 1:-1:step, Nz//2], v_norm[1:-1:step, 1:-1:step, Nz//2],
                vel_norm[1:-1:step, 1:-1:step, Nz//2],
                cmap="turbo", pivot="tail", units="xy", clim=(0., 1.)
            )
            fig.colorbar(qv, ticks=ticks, extend="both", ax=ax[0])
            ax[0].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Ly, 5),
                xlim=(0., Lx),
                ylim=(0., Ly),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"$z={Lz/2:.2f}$ plane",
                aspect="equal",
            )

            qv = ax[1].quiver(
                X[1:-1:step, Ny//2, 1:-1:step], Z[1:-1:step, Ny//2, 1:-1:step],
                u_norm[1:-1:step, Ny//2, 1:-1:step], w_norm[1:-1:step, Ny//2, 1:-1:step],
                vel_norm[1:-1:step, Ny//2, 1:-1:step],
                cmap="turbo", pivot="tail", units="xy", clim=(0., 1.)
            )
            fig.colorbar(qv, ticks=ticks, extend="both", ax=ax[1])
            ax[1].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Lz, 5),
                xlim=(0., Lx),
                ylim=(0., Lz),
                xlabel=r"$x$",
                ylabel=r"$z$",
                title=rf"$y={Ly/2:.2f}$ plane",
                aspect="equal",
            )

            qv = ax[2].quiver(
                Y[Nx//2, 1:-1:step, 1:-1:step], Z[Nx//2, 1:-1:step, 1:-1:step],
                v_norm[Nx//2, 1:-1:step, 1:-1:step], w_norm[Nx//2, 1:-1:step, 1:-1:step],
                vel_norm[Nx//2, 1:-1:step, 1:-1:step],
                cmap="turbo", pivot="tail", units="xy", clim=(0., 1.)
            )
            fig.colorbar(qv, ticks=ticks, extend="both", ax=ax[2])
            ax[2].set(
                xticks=np.linspace(0., Ly, 5),
                yticks=np.linspace(0., Lz, 5),
                xlim=(0., Ly),
                ylim=(0., Lz),
                xlabel=r"$y$",
                ylabel=r"$z$",
                title=rf"$x={Lx/2:.2f}$ plane",
                aspect="equal",
            )

            fig.suptitle(rf"$t = {t:.3f} \ [\text{{s}}]$",)
            fig.tight_layout()
            fig.savefig(dir_res / f"vel_quiver.png")
            fig.savefig(dir_vel_qvr / f"vel_quiver_{n:06d}.png")
            plt.close(fig)


            # vorticity at the geometric center
            vor_x = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2. * dy) \
                    - (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2. * dz)
            vor_y = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2. * dz) \
                    - (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2. * dx)
            vor_z = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2. * dx) \
                    - (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2. * dy)
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            levels = np.linspace(-5., 5., 32)
            ticks = np.linspace(-5., 5., 5)
            cf = ax[0].contourf(
                X[2:-2, 2:-2, Nz//2], Y[2:-2, 2:-2, Nz//2], vor_x[1:-1, 1:-1, Nz//2],
                cmap="seismic", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[0])
            ax[0].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Ly, 5),
                xlim=(0., Lx),
                ylim=(0., Ly),
                xlabel=r"$x$",
                ylabel=r"$y$",
                title=rf"$z={Lz/2:.2f}$ plane",
                aspect="equal",
            )

            cf = ax[1].contourf(
                X[2:-2, Ny//2, 2:-2], Z[2:-2, Ny//2, 2:-2], vor_y[1:-1, Ny//2, 1:-1],
                cmap="seismic", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[1])
            ax[1].set(
                xticks=np.linspace(0., Lx, 5),
                yticks=np.linspace(0., Lz, 5),
                xlim=(0., Lx),
                ylim=(0., Lz),
                xlabel=r"$x$",
                ylabel=r"$z$",
                title=rf"$y={Ly/2:.2f}$ plane",
                aspect="equal",
            )

            cf = ax[2].contourf(
                Y[Nx//2, 2:-2, 2:-2], Z[Nx//2, 2:-2, 2:-2], vor_z[Nx//2, 1:-1, 1:-1],
                cmap="seismic", levels=levels, extend="both"
            )
            fig.colorbar(cf, ticks=ticks, ax=ax[2])
            ax[2].set(
                xticks=np.linspace(0., Ly, 5),
                yticks=np.linspace(0., Lz, 5),
                xlim=(0., Ly),
                ylim=(0., Lz),
                xlabel=r"$y$",
                ylabel=r"$z$",
                title=rf"$x={Lx/2:.2f}$ plane",
                aspect="equal",
            )

            fig.suptitle(rf"$t = {t:.3f} \ [\text{{s}}]$",)
            fig.tight_layout()
            fig.savefig(dir_res / f"vor_slice.png")
            fig.savefig(dir_vor_slc / f"vor_slice_{n:06d}.png")
            plt.close(fig)


            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            levels = np.linspace(0., 1., 64)
            ticks = np.linspace(0., 1., 5)
            # plot projections of the contours for each dimension
            # by choosing offsets that match the appropriate axes limits
            # ax.contourf(
            #     X[1:-1, 1:-1, Nz//2], Y[1:-1, 1:-1, Nz//2], vel_norm[1:-1, 1:-1, Nz//2],
            #     zdir="z", offset=Z[1, 1, Nz//2], cmap="turbo", levels=levels, extend="both"
            # )
            # ax.contourf(
            #     X[1:-1, 1:-1, -5], Y[1:-1, 1:-1, -5], vel_norm[1:-1, 1:-1, -5],
            #     zdir="z", offset=Z[1, 1, -5], cmap="turbo", levels=levels, extend="both"
            # )
            # ax.contourf(
            #     X[1:-1, 1:-1, 5], Y[1:-1, 1:-1, 5], vel_norm[1:-1, 1:-1, 5],
            #     zdir="z", offset=Z[1, 1, 5], cmap="turbo", levels=levels, extend="both"
            # )

            ax.contourf(
                X[1:-1, 1:-1, Nz//2], Y[1:-1, 1:-1, Nz//2], vel_norm[1:-1, 1:-1, Nz//2],
                zdir="z", offset=0., cmap="turbo", levels=levels, extend="both"
            )
            ax.contourf(
                X[1:-1, Ny//2, 1:-1], Z[1:-1, Ny//2, 1:-1], vel_norm[1:-1, Ny//2, 1:-1],
                zdir="y", offset=0., cmap="turbo", levels=levels, extend="both"
            )
            ax.set(
                xlabel=r"$x$",
                ylabel=r"$y$",
                zlabel=r"$z$",
            )
            plt.tight_layout()
            plt.savefig(dir_res / f"slice.png")
            plt.close()


            # velocity at the geometric center
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
                    title=rf"$t = {t:.3f} \ [\text{{s}}]$",
                    aspect="equal",
                )
                fig.tight_layout()
                fig.savefig(dir_res / f"vel_comparison.png")
                plt.close(fig)

            # save
            np.savez(
                dir_npz / f"data_it{n:06d}.npz",
                x=x, y=y, z=z,
                X=X, Y=Y, Z=Z,
                u=u, v=v, w=w,
                p=p, p_bar=p_bar,
                dt=dt, t=t
            )
            # # save as vtk
            # pyevtk.hl.gridToVTK(
            #     dir_vtk / f"data_it{n:06d}.vtk",
            #     X[1:-1, 1:-1, 1:-1],
            #     Y[1:-1, 1:-1, 1:-1],
            #     Z[1:-1, 1:-1, 1:-1],
            #     pointData={
            #         "u": u[1:-1, 1:-1, 1:-1],
            #         "v": v[1:-1, 1:-1, 1:-1],
            #         "w": w[1:-1, 1:-1, 1:-1],
            #         "p": p,
            #         "p_bar": p_bar,
            #     }
            # )


################################################################################

if __name__ == "__main__":
    main()
