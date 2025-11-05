#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv, eigh
import argparse

OUTPNG = "skymap_overlay_ra_dec.png"
c = 299_792_458.0
RAD2ARCMIN = 60.0 * 180.0/np.pi

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    a = 6378137.0
    f = 1.0/298.257223563
    e2 = f*(2 - f)
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    sin_lat = np.sin(lat); cos_lat = np.cos(lat)
    sin_lon = np.sin(lon); cos_lon = np.cos(lon)
    N = a / np.sqrt(1.0 - e2 * sin_lat**2)
    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z])

def get_detector_positions():
    try:
        import lal
        dets = {}
        for prefix in ["H1", "L1", "V1"]:
            d = lal.cached_detector_by_prefix[prefix]
            dets[prefix] = np.array([d.location.x, d.location.y, d.location.z])
        return dets
    except Exception:
        pass
    H1_lat, H1_lon, H1_h = 46.455, -119.408, 142.0
    L1_lat, L1_lon, L1_h = 30.563,  -90.774, 0.0
    V1_lat, V1_lon, V1_h = 43.630,   10.500, 51.0
    return {
        "H1": geodetic_to_ecef(H1_lat, H1_lon, H1_h),
        "L1": geodetic_to_ecef(L1_lat, L1_lon, L1_h),
        "V1": geodetic_to_ecef(V1_lat, V1_lon, V1_h),
    }

def build_basis(ra, dec):
    e_alpha = np.array([-np.sin(ra), np.cos(ra), 0.0])
    e_delta = np.array([
        -np.cos(ra)*np.sin(dec),
        -np.sin(ra)*np.sin(dec),
         np.cos(dec)
    ])
    return e_alpha, e_delta

def J_row(ri, rj, e_alpha, e_delta):
    L = (ri - rj) / c
    return np.array([np.dot(L, e_alpha), np.dot(L, e_delta)])

def ellipse_points(C, nsig=1.0, npts=500):
    w, V = np.linalg.eigh(C)
    axes = nsig * np.sqrt(np.maximum(w, 1e-20))
    t = np.linspace(0, 2*np.pi, npts)
    circ = np.vstack([np.cos(t), np.sin(t)])
    return (V @ (axes[:,None] * circ))

def level_radius(p):
    return np.sqrt(-2.0*np.log(1.0 - p))

def main():
    ap = argparse.ArgumentParser(description="RA/Dec overlay with targeted centroid shifts")
    # Geometry / noise
    ap.add_argument("--sigma-t-us", type=float, default=80.0, help="Per-detector timing std dev (microseconds)")
    ap.add_argument("--ra",  type=float, default=160.0, help="Reference RA in degrees")
    ap.add_argument("--dec", type=float, default=20.0,  help="Reference Dec in degrees")
    # Two modes to set the shift:
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--bias-mode", action="store_true",
                       help="Use explicit per-site timing biases (default mode if none of the target options are set)")
    group.add_argument("--align-major", action="store_true",
                       help="Shift along the major axis of the sky covariance by --target-mahal sigma")
    group.add_argument("--align-minor", action="store_true",
                       help="Shift along the minor axis of the sky covariance by --target-mahal sigma")
    group.add_argument("--target-angle-deg", type=float, default=None,
                       help="Shift along a specified angle (deg) in RA/Dec (0° = +RA, 90° = +Dec) by --target-mahal sigma")
    ap.add_argument("--target-mahal", type=float, default=2.0,
                    help="Mahalanobis distance (sigma units) for targeted shift")
    # Explicit biases (bias-mode)
    ap.add_argument("--bias-scale", type=float, default=1.0, help="Scale factor applied to all site biases")
    ap.add_argument("--bias-h1-us", type=float, default=100.0, help="Bias at H1 in "
                                                                    "microseconds")
    ap.add_argument("--bias-l1-us", type=float, default=100.0, help="Bias at L1 in "
                                                                    "microseconds")
    ap.add_argument("--bias-v1-us", type=float, default=-100.0, help="Bias at V1 in "
                                                                   "microseconds")
    ap.add_argument("--dpi", type=int, default=240, help="Figure DPI")
    args = ap.parse_args()

    r = get_detector_positions()
    ra = np.deg2rad(args.ra); dec = np.deg2rad(args.dec)
    e_alpha, e_delta = build_basis(ra, dec)

    ref = "H1"
    others = [s for s in ["H1","L1","V1"] if s != ref]
    J = np.vstack([J_row(r[s], r[ref], e_alpha, e_delta) for s in others])

    sigma_t = args.sigma_t_us * 1e-6
    C_dt = (2.0 * sigma_t**2) * np.eye(2)
    C_th = inv(J.T @ inv(C_dt) @ J)  # radians^2

    mu0 = np.array([0.0, 0.0])

    # Determine dtheta (centroid shift) either from explicit biases or targeted design
    use_target = args.align_major or args.align_minor or (args.target_angle_deg is not None)

    if use_target:
        # Eigen-decomposition of the sky covariance
        w, V = eigh(C_th)  # ascending: w[0] <= w[1]; V[:,0] minor, V[:,1] major
        if args.align_major:
            vdir = V[:, 1]
            lam = w[1]
        elif args.align_minor:
            vdir = V[:, 0]
            lam = w[0]
        else:
            angle = np.deg2rad(args.target_angle_deg)
            vdir = np.array([np.cos(angle), np.sin(angle)])  # in RA/Dec basis
            # Effective variance along this direction:
            lam = vdir @ C_th @ vdir

        # For a Gaussian, Mahalanobis distance m along direction v has magnitude s = m * sqrt(lam)
        s = args.target_mahal * np.sqrt(lam)
        dtheta = s * vdir  # radians
        # Derive per-baseline delays and per-site biases (set ref site bias = 0 for convenience)
        eps = J @ dtheta  # Δt_{others,ref}
        delta_t = {ref: 0.0}
        for k, sname in enumerate(others):
            delta_t[sname] = float(eps[k])
    else:
        # Explicit per-site biases
        biases_us = {"H1": args.bias_h1_us, "L1": args.bias_l1_us, "V1": args.bias_v1_us}
        delta_t = {k: args.bias_scale * v * 1e-6 for k, v in biases_us.items()}
        eps = np.array([delta_t[s] - delta_t[ref] for s in others])
        dtheta = inv(J) @ eps  # radians

    mu1 = mu0 + dtheta

    # Convert to arcmin for plotting
    C_arc = (RAD2ARCMIN**2) * C_th
    mu0_arc = RAD2ARCMIN * mu0
    mu1_arc = RAD2ARCMIN * mu1
    d_arc = mu1_arc - mu0_arc

    r50, r90 = level_radius(0.5), level_radius(0.90)
    ell0_50 = RAD2ARCMIN * ellipse_points(C_th, nsig=r50)
    ell0_90 = RAD2ARCMIN * ellipse_points(C_th, nsig=r90)
    ell1_50 = mu1_arc[:,None] + RAD2ARCMIN * ellipse_points(C_th, nsig=r50)
    ell1_90 = mu1_arc[:,None] + RAD2ARCMIN * ellipse_points(C_th, nsig=r90)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))

    # Filled translucent regions
    ax.fill(ell0_90[0], ell0_90[1], alpha=0.20, color="#2a6fdb", linewidth=0)
    ax.fill(ell0_50[0], ell0_50[1], alpha=0.35, color="#2a6fdb", linewidth=0, label="Original 50%/90%")

    ax.fill(ell1_90[0], ell1_90[1], alpha=0.20, color="#e46e2e", linewidth=0)
    ax.fill(ell1_50[0], ell1_50[1], alpha=0.35, color="#e46e2e", linewidth=0, label="Biased 50%/90%")

    # Outlines
    ax.plot(ell0_50[0], ell0_50[1], color="#1f4fb8", lw=2.0)
    # ax.plot(ell0_90[0], ell0_90[1], color="#1f4fb8", lw=1.6, ls="--")
    ax.plot(ell1_50[0], ell1_50[1], color="#b8551b", lw=2.0)
    # ax.plot(ell1_90[0], ell1_90[1], color="#b8551b", lw=1.6, ls="--")

    # Means + arrow
    ax.plot(mu0_arc[0], mu0_arc[1], "o", ms=6, color="#1f4fb8")
    ax.plot(mu1_arc[0], mu1_arc[1], "o", ms=6, color="#b8551b")
    ax.annotate("", xy=(mu1_arc[0], mu1_arc[1]), xytext=(mu0_arc[0], mu0_arc[1]),
                arrowprops=dict(arrowstyle="->", lw=2.4, color="black"))
    shift_arcmin = float(np.hypot(*d_arc))
    Ci_arc = inv(C_arc)
    mahal = float(np.sqrt(d_arc.T @ Ci_arc @ d_arc))
    # Bias summary text (also show derived per-site biases if targeted mode)
    if use_target:
        summary = (f"Centroid shift: {shift_arcmin:.2f} arcmin\n"
                   f"Mahalanobis: {mahal:.2f} σ\n"
                   f"Derived biases (µs): H1=0.00, L1={delta_t['L1']*1e6:.2f}, V1={delta_t['V1']*1e6:.2f}")
    else:
        summary = (f"Centroid shift: {shift_arcmin:.2f} arcmin\n"
                   f"Mahalanobis: {mahal:.2f} σ")
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5"))

    ax.set_xlabel(r"$\Delta \mathrm{RA}$ [arcmin]")
    ax.set_ylabel(r"$\Delta \mathrm{Dec}$ [arcmin]")
    ax.set_title("Skymap (RA/Dec): perturbation shifts 50%/90% credible regions")

    # Limits + RA convention
    allx = np.concatenate([ell0_90[0], ell1_90[0], [mu0_arc[0], mu1_arc[0]]])
    ally = np.concatenate([ell0_90[1], ell1_90[1], [mu0_arc[1], mu1_arc[1]]])
    x_span = (allx.max() - allx.min())
    y_span = (ally.max() - ally.min())
    rx = x_span * 0.15
    ry = y_span * 0.15
    zoom = 4.0
    x_mid = (allx.min() + allx.max()) / 2
    y_mid = (ally.min() + ally.max()) / 2
    x_min = x_mid - (0.5 * (1 / zoom) * x_span)
    x_max = x_mid + (0.5 * (1 / zoom) * x_span)
    y_min = y_mid - (0.5 * (1 / zoom) * y_span)
    y_max = y_mid + (0.5 * (1 / zoom) * y_span)
    ax.set_xlim(x_min-rx, x_max+rx)
    ax.set_ylim(y_min-ry, y_max+ry)
    ax.invert_xaxis()
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(OUTPNG, dpi=args.dpi)
    print(f"Wrote {OUTPNG}")

if __name__ == "__main__":
    main()
