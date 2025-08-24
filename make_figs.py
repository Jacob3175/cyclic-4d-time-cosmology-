
import numpy as np
import matplotlib.pyplot as plt
from math import isfinite
from pathlib import Path

# Parameters
C = 1.0
rho_m0 = 1.0
rho_r0 = 0.1
rho_L  = -0.02
rho_c  = 100.0
Lambda = 0.0
k_curv = 0.0

a_min_scan = 1e-3
a_max_scan = 1e4
N_scan     = 20000

out_dir = Path(__file__).resolve().parent / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

def rho_of_a(a):
    return rho_m0 * a**(-3.0) + rho_r0 * a**(-4.0) + rho_L

def H2_of_a(a):
    rho = rho_of_a(a)
    return C * rho * (1.0 - rho / rho_c) + (Lambda / 3.0) - (k_curv / a**2)

def find_turning_points():
    a_grid = np.logspace(np.log10(a_min_scan), np.log10(a_max_scan), N_scan)
    H2_vals = H2_of_a(a_grid)
    s = np.sign(H2_vals)
    idx = np.where(np.diff(s) != 0)[0]

    roots = []
    for i in idx:
        aL, aR = a_grid[i], a_grid[i+1]
        fL, fR = H2_of_a(aL), H2_of_a(aR)
        for _ in range(60):
            aM = 0.5*(aL + aR)
            fM = H2_of_a(aM)
            if fL == 0.0:
                aR = aL
                break
            if fR == 0.0:
                aL = aR
                break
            if fL * fM <= 0:
                aR, fR = aM, fM
            else:
                aL, fL = aM, fM
        root = 0.5*(aL + aR)
        roots.append(root)

    roots = sorted(roots)
    dedup = []
    for r in roots:
        if not dedup or abs(np.log(r) - np.log(dedup[-1])) > 1e-3:
            dedup.append(r)
    return dedup, a_grid, H2_vals

def compute_half_period(a_lo, a_hi, num=20000):
    a = np.geomspace(a_lo, a_hi, num)
    H2 = H2_of_a(a)
    H2 = np.where(H2 < 0, np.nan, H2)
    integrand = 1.0 / (a * np.sqrt(H2))
    integrand = np.where(np.isfinite(integrand), integrand, 0.0)
    integral = np.trapz(integrand, a)
    return integral

def integrate_a_of_tau(a_lo, a_hi, T_half, nsteps=4000):
    a_up = np.geomspace(a_lo, a_hi, nsteps)
    H2_up = H2_of_a(a_up)
    H2_up = np.where(H2_up < 0, np.nan, H2_up)
    dtauda = 1.0 / (a_up * np.sqrt(H2_up))
    dtauda = np.where(np.isfinite(dtauda), dtauda, 0.0)
    tau_up = np.cumsum((dtauda[1:] + dtauda[:-1]) * np.diff(a_up) / 2.0)
    tau_up = np.concatenate(([0.0], tau_up))
    scale = (T_half / tau_up[-1]) if tau_up[-1] != 0 else 1.0
    tau_up *= scale
    T_full = 2.0 * T_half
    tau_full = np.concatenate([tau_up, T_half + (T_half - tau_up)])
    a_full   = np.concatenate([a_up, a_up[::-1]])
    return tau_full, a_full, T_full

def plot_H2(a_grid, H2_vals, roots):
    plt.figure(figsize=(7,5))
    plt.plot(a_grid, H2_vals, label=r"$H^2(a)$")
    if roots:
        y0 = np.zeros_like(roots)
        plt.scatter(roots, y0, marker="o", label="Turning points")
    plt.xscale("log")
    plt.xlabel("Scale factor a (log)")
    plt.ylabel(r"$H(a)^2$ (toy units)")
    plt.title("Effective Friedmann with LQC correction")
    plt.grid(True, which="both")
    plt.legend()
    out = out_dir / "plot_H2_from_params.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def plot_a_tau(tau, a, a_lo, a_hi, T):
    plt.figure(figsize=(7,5))
    plt.plot(tau, a, label=r"$a(\tau)$")
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.axvline(T/2, linestyle="--", linewidth=1, label="Turnaround")
    plt.axvline(T, linestyle="--", linewidth=1)
    plt.scatter([0.0, T/2], [a_lo, a_hi], zorder=5, label="Bounce & Turnaround")
    plt.xlabel(r"Cyclic time $\tau$")
    plt.ylabel(r"Scale factor $a(\tau)$")
    plt.title("Scale factor from integral (no sinusoid)")
    plt.grid(True)
    plt.legend()
    out = out_dir / "plot_a_tau_from_integral.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    return out

def main():
    roots, a_grid, H2_vals = find_turning_points()
    if len(roots) < 2:
        print("Did not find two turning points. Try adjusting rho_L (more negative) or k_curv (>0).")
        return
    a_lo, a_hi = roots[0], roots[-1]
    T_half = compute_half_period(a_lo, a_hi, num=40000)
    tau, a_of_tau, T_full = integrate_a_of_tau(a_lo, a_hi, T_half, nsteps=6000)

    # Save turning points and period
    txt = out_dir / "turning_points.txt"
    with open(txt, "w") as f:
        f.write(f"a_min (bounce): {a_lo:.6g}\n")
        f.write(f"a_max (turnaround): {a_hi:.6g}\n")
        f.write(f"T_half: {T_half:.6g}\n")
        f.write(f"T_full: {T_full:.6g}\n")

    out1 = plot_H2(a_grid, H2_vals, roots)
    out2 = plot_a_tau(tau, a_of_tau, a_lo, a_hi, T_full)
    print("Saved:", out1)
    print("Saved:", out2)
    print("Saved:", txt)

if __name__ == "__main__":
    main()
