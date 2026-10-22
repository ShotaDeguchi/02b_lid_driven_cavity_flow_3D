from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================

def set_matplotlib():
    plt.style.use("default")
    # plt.style.use("classic")
    plt.style.use("seaborn-v0_8-talk")   # paper / notebook / talk / poster
    # plt.style.use("seaborn-v0_8-deep")
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "cm"   # dejavusans / dejavuserif / cm / stix / stixsans
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.axisbelow"] = True   # background grid
    plt.rcParams["grid.alpha"] = .2
    plt.rcParams["legend.framealpha"] = .8
    plt.rcParams["legend.facecolor"] = "w"
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["xtick.direction"] = "in"   # "in" / "out" / "inout"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
set_matplotlib()

# ==============================================================================

npz_dir = Path("Re100/npz")

ts = []
us = []
vs = []
ws = []
for file in sorted(npz_dir.glob("data_it*.npz")):
    data = np.load(file)

    t = data["t"]
    u = data["u"]
    v = data["v"]
    w = data["w"]
    nx, ny, nz = u.shape
    u = u[nx//2, ny//2, nz//2]
    v = v[nx//2, ny//2, nz//2]
    w = w[nx//2, ny//2, nz//2]

    print(f"file: {file}, t: {t}, u[nx//2, ny//2, nz//2]: {u}, v[nx//2, ny//2, nz//2]: {v}, w[nx//2, ny//2, nz//2]: {w}")

    ts.append(t.item())
    us.append(u.item())
    vs.append(v.item())
    ws.append(w.item())

print("len(ts):", len(ts))
print("len(us):", len(us))


print("ts:", ts)
print("us:", us)



fig, ax = plt.subplots()
ax.plot(ts, us, label="u")
# ax.plot(ts, vs, label="v")
# ax.plot(ts, ws, label="w")
ax.set(
    xlabel="Time",
    ylabel="Velocity",
    yscale="log",
)
plt.show()


# print("ts:", ts)
# print("us:", us)

