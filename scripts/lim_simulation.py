"""
Numerical implementation of the Lens-free Interferometric on-chip Microscope (LIM)
-----------------------------------------------------------------------------------
Reference: R. A. Terborg et al., "Ultrasensitive interferometric on-chip microscopy
of transparent objects", Sci. Adv. 2, e1600077 (2016).

Pipeline (mirrors the paper):

    1. Sample        : gray map (refractive-index variation) -> OPD map -> phase map
    2. Savart plates : two orthogonally polarised beams (EPx, EPy) sheared by +-s/2
    3. Diffraction   : each beam propagates (angular spectrum) up to the sensor
    4. SP1 tilt scan : extra phase alpha between EPx and EPy  (Eq. 1)
    5. Analyser P2   : I(alpha, phi) = A - B cos(alpha + phi)  (Eqs. 5-6)
    6. PSI           : weighted integrals over an integer number of periods
                       (Eqs. 8-13) -> phase and intensity maps, per colour
    7. Reference     : repeat without sample and subtract (static phase / illumination)
    8. Reconstruction: Wiener deconvolution of the double-image PSF (Eq. 14),
                       extended with the diffraction transfer function and combined
                       across wavelengths (multispectral surface reconstruction)

Conventions
    * `gray` in [0, 1]  -> refractive-index variation:  dn = gray * delta_n_max
    * OPD = dn * thickness      (metres)
    * phase = 2 * pi * OPD / wavelength
    * pixel size is given in metres
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2, next_fast_len
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class LIMConfig:
    # Light source: multicolour LED (465, 515, 635 nm), as in the paper
    wavelengths: tuple = (465e-9, 515e-9, 635e-9)
    # Savart-plate shear vector (sx, sy) in metres; |s| = 25 um, along the diagonal
    shear: tuple = (25e-6 / np.sqrt(2), 25e-6 / np.sqrt(2))
    # Distance between the sample and the image sensor (paper: up to 30 mm)
    sample_distance: float = 5e-3
    # Sample model: OPD = gray * delta_n_max * thickness
    delta_n_max: float = 5e-3
    thickness: float = 1e-6
    # SP1 tilt scan, expressed as the OPD introduced between EPx and EPy (Eq. 1)
    n_tilts: int = 20
    tilt_opd_range: tuple = (0.0, 1500e-9)
    # Detector / non-idealities
    noise: bool = True
    photons_per_pixel: float = 5e4  # mean photo-electrons at unit intensity
    read_noise: float = 5.0  # electrons rms
    illumination_nonuniformity: float = 0.2  # relative, smooth
    background_opd_rms: float = 0.3e-9  # static OPD of optics, metres
    # Reconstruction
    wiener_k: float = 1e-2  # Wiener regularisation, relative to max |T|^2
    seed: int = 0


@dataclass
class LIMResult:
    opd_true: np.ndarray  # metres, cropped to the input size
    opd_rec: np.ndarray  # metres, reconstructed
    psi: dict  # wavelength -> retrieved phase map (rad), cropped
    i0: dict  # wavelength -> retrieved intensity map, cropped
    frame: dict  # wavelength -> one raw interferogram near quadrature, cropped
    pixel_size: float
    config: LIMConfig


# --------------------------------------------------------------------------- #
# Optics helpers
# --------------------------------------------------------------------------- #


def _freq_grids(shape, dx):
    fy = fftfreq(shape[0], dx)[:, None]
    fx = fftfreq(shape[1], dx)[None, :]
    return fx, fy


def propagation_phase(fx, fy, wavelength, z):
    """Angular-spectrum phase (global carrier removed) and propagating-wave mask."""
    inv = 1.0 / wavelength
    arg = inv**2 - fx**2 - fy**2
    mask = arg > 0
    kz = np.sqrt(np.maximum(arg, 0.0))
    return 2 * np.pi * z * (kz - inv), mask


def shear_phase_ramp(fx, fy, shear):
    """Fourier-domain factor that moves a field by +s/2: f(r) -> f(r - s/2)."""
    sx, sy = shear
    return np.exp(-1j * np.pi * (fx * sx + fy * sy))


def wrap_phase(p):
    return np.angle(np.exp(1j * p))


# --------------------------------------------------------------------------- #
# Phase-shifting interferometry (Eqs. 8-13)
# --------------------------------------------------------------------------- #


def psi_weights(alpha, n_grid=4096):
    """
    Discrete weights implementing the paper's PSI integrals

        Ysin = int I(a) sin(a) / (pi N) da,   Ycos = int I(a) cos(a) / (pi N) da

    over an integer number N of periods inside the scanned range of `alpha`.
    The intensity is interpolated with a cubic spline onto a fine grid and
    integrated with the trapezoid rule, so any (non-uniform) tilt sampling works.
    """
    alpha = np.asarray(alpha, dtype=float)
    n_periods = int(np.floor((alpha[-1] - alpha[0]) / (2 * np.pi)))
    if n_periods < 1:
        raise ValueError("The tilt scan must cover at least one full 2*pi period.")
    grid = np.linspace(alpha[0], alpha[0] + 2 * np.pi * n_periods, n_grid)
    dg = grid[1] - grid[0]
    trap = np.full(n_grid, dg)
    trap[[0, -1]] *= 0.5
    interp = CubicSpline(alpha, np.eye(len(alpha)), axis=0)(grid)  # (n_grid, K)
    norm = np.pi * n_periods
    w_sin = (trap * np.sin(grid) / norm) @ interp
    w_cos = (trap * np.cos(grid) / norm) @ interp
    return w_sin, w_cos


def acquire_and_retrieve(ex, ey, illum, bg_phase, alpha, cfg, rng):
    """
    Simulate the SP1 tilt scan and run PSI on the fly (no stack kept in memory).

    I(alpha) = illum/4 * | e^{i alpha} Ex - Ey e^{-i bg} |^2
             = A - B cos(alpha + phi_d + bg),  phi_d = arg(Ex conj(Ey))
    Returns phase (rad), intensity I0 (Eq. 13) and the frame closest to quadrature.
    """
    w_sin, w_cos = psi_weights(alpha)
    ey_eff = ey * np.exp(-1j * bg_phase)
    y_sin = np.zeros(ex.shape)
    y_cos = np.zeros(ex.shape)
    quad_idx = int(np.argmin(np.abs(wrap_phase(alpha - np.pi / 2))))
    kept = None
    for k, a in enumerate(alpha):
        frame = 0.25 * illum * np.abs(np.exp(1j * a) * ex - ey_eff) ** 2
        if cfg.noise:
            frame = rng.poisson(cfg.photons_per_pixel * frame).astype(float)
            frame += rng.normal(0.0, cfg.read_noise, frame.shape)
            frame /= cfg.photons_per_pixel
        y_sin += w_sin[k] * frame
        y_cos += w_cos[k] * frame
        if k == quad_idx:
            kept = frame.copy()
    # Eqs. 10-13:  Ysin = B sin(phi),  Ycos = -B cos(phi)
    phase = np.arctan2(y_sin, -y_cos)
    i0 = 2.0 * np.hypot(y_sin, y_cos)
    return phase, i0, kept


# --------------------------------------------------------------------------- #
# Forward model + retrieval
# --------------------------------------------------------------------------- #


def _smooth_random(shape, sigma, rng):
    field = gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    return field / (field.std() + 1e-30)


def simulate_lim(opd, dx, cfg: LIMConfig):
    """Run the virtual LIM on an OPD map. Returns padded per-colour phase maps."""
    rng = np.random.default_rng(cfg.seed)
    h, w = opd.shape

    # Pad with vacuum so that shearing, diffraction and FFTs do not wrap around
    reach = np.hypot(*cfg.shear) + 4 * np.sqrt(max(cfg.wavelengths) * cfg.sample_distance)
    pad = max(int(np.ceil(reach / dx)), 16)
    hp, wp = next_fast_len(h + 2 * pad), next_fast_len(w + 2 * pad)
    top, left = (hp - h) // 2, (wp - w) // 2
    crop = (slice(top, top + h), slice(left, left + w))
    opd_p = np.zeros((hp, wp))
    opd_p[crop] = opd

    fx, fy = _freq_grids((hp, wp), dx)
    ramp = shear_phase_ramp(fx, fy, cfg.shear)

    # Static non-idealities (identical with and without the sample)
    smin = min(hp, wp)
    vignette = 1.0 - 0.5 * cfg.illumination_nonuniformity * (
        ((np.arange(hp)[:, None] - hp / 2) / (hp / 2)) ** 2
        + ((np.arange(wp)[None, :] - wp / 2) / (wp / 2)) ** 2
    )
    illum = np.clip(
        vignette + 0.3 * cfg.illumination_nonuniformity * _smooth_random((hp, wp), smin / 8, rng),
        0.2,
        None,
    )
    illum /= illum.max()
    bg_opd = cfg.background_opd_rms * _smooth_random((hp, wp), smin / 10, rng)

    tilts = np.linspace(*cfg.tilt_opd_range, cfg.n_tilts)
    ones = np.ones((hp, wp), dtype=complex)

    psi, i0, frames = {}, {}, {}
    for lam in cfg.wavelengths:
        alpha = 2 * np.pi * tilts / lam  # Eq. 1: Delta-OPD is linear in the tilt
        bg_phase = 2 * np.pi * bg_opd / lam

        # Sample beams: EPx -> U(r - s/2), EPy -> U(r + s/2), U = propagated field
        phase_obj = 2 * np.pi * opd_p / lam
        prop_ph, mask = propagation_phase(fx, fy, lam, cfg.sample_distance)
        u_hat = fft2(np.exp(1j * phase_obj)) * np.exp(1j * prop_ph) * mask
        ex = ifft2(u_hat * ramp)
        ey = ifft2(u_hat * np.conj(ramp))

        ph_s, i0_s, frame = acquire_and_retrieve(ex, ey, illum, bg_phase, alpha, cfg, rng)
        # Reference measurement without sample (Methods: "with and without a sample")
        ph_r, _, _ = acquire_and_retrieve(ones, ones, illum, bg_phase, alpha, cfg, rng)

        psi[lam] = wrap_phase(ph_s - ph_r)
        i0[lam] = i0_s
        frames[lam] = frame
    return psi, i0, frames, crop, (hp, wp)


# --------------------------------------------------------------------------- #
# Reconstruction (Eq. 14 + diffraction, multispectral)
# --------------------------------------------------------------------------- #


def reconstruct_opd(psi, dx, cfg: LIMConfig, padded_shape):
    """
    Wiener deconvolution of the retrieved phase maps.

    For weak phase objects the retrieved phase obeys the linear model

        psi(r) = c * phi (r - s/2) - c * phi (r + s/2)
        PSI(f) = -2i sin(pi f.s) cos(phase_z(f)) Phi(f),

    i.e. the two-delta PSF of Eq. 14 (A = -B = 1) times the real part of the
    free-space propagator between sample and sensor. The zeros of this transfer
    function are the lost frequencies mentioned in the paper; the Wiener
    regularisation prevents noise blow-up there, and combining several
    wavelengths (different cos zeros) recovers part of the missing information.
    """
    fx, fy = _freq_grids(padded_shape, dx)
    sx, sy = cfg.shear
    sin_term = -2j * np.sin(np.pi * (fx * sx + fy * sy))
    num = np.zeros(padded_shape, dtype=complex)
    den = np.zeros(padded_shape)
    for lam, p in psi.items():
        prop_ph, mask = propagation_phase(fx, fy, lam, cfg.sample_distance)
        t = (2 * np.pi / lam) * sin_term * np.cos(prop_ph) * mask  # rad per metre of OPD
        num += np.conj(t) * fft2(p)
        den += np.abs(t) ** 2
    k = cfg.wiener_k * den.max()
    return ifft2(num / (den + k)).real  # OPD in metres


# --------------------------------------------------------------------------- #
# End-to-end driver
# --------------------------------------------------------------------------- #


def run_lim(gray, pixel_size, cfg: LIMConfig) -> LIMResult:
    opd = gray * cfg.delta_n_max * cfg.thickness
    psi_p, i0_p, frames_p, crop, shape_p = simulate_lim(opd, pixel_size, cfg)
    opd_rec_p = reconstruct_opd(psi_p, pixel_size, cfg, shape_p)
    return LIMResult(
        opd_true=opd,
        opd_rec=opd_rec_p[crop],
        psi={k: v[crop] for k, v in psi_p.items()},
        i0={k: v[crop] for k, v in i0_p.items()},
        frame={k: v[crop] for k, v in frames_p.items()},
        pixel_size=pixel_size,
        config=cfg,
    )


def metrics(res: LIMResult):
    """RMSE / correlation after removing the mean (the DC OPD is invisible to a shear)."""
    a = res.opd_true - res.opd_true.mean()
    b = res.opd_rec - res.opd_rec.mean()
    rmse = np.sqrt(np.mean((a - b) ** 2))
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    return {"rmse_nm": rmse * 1e9, "corr": corr, "opd_peak_nm": np.ptp(res.opd_true) * 1e9}


# --------------------------------------------------------------------------- #
# Data loading (starting code from the user)
# --------------------------------------------------------------------------- #


def _parse_scale(scale):
    """Accept a bare number or a dict containing the pixel size in metres."""
    if isinstance(scale, (int, float)):
        return float(scale)
    if isinstance(scale, dict):
        for key in ("pixel_size", "scale", "size", "pixel"):
            if key in scale:
                return float(scale[key])
        for v in scale.values():
            if isinstance(v, (int, float)):
                return float(v)
    raise ValueError(f"Cannot interpret the scale file content: {scale!r}")


def load_sample(data_dir, image_name="USAF_1951.jpg", scale_name="USAF_1951_scale.json", downscale=1):
    from skimage import io
    from skimage.filters import threshold_sauvola
    from skimage.transform import downscale_local_mean

    data_dir = Path(data_dir)

    # Filter the background and inverse the distribution (dark->bright)
    gray = io.imread(data_dir / image_name).mean(axis=2).astype(float)
    gray = 1 - (gray / gray.max())
    threshold = threshold_sauvola(gray, window_size=801, k=0.2, r=0.5 * (gray.max() - gray.min()))
    gray = (gray > threshold) * (gray)

    with open(data_dir / scale_name, "r") as f:
        scale = json.load(f)
    pixel_size = _parse_scale(scale)  # metres per pixel

    gray = gray / gray.max()
    if downscale > 1:
        gray = downscale_local_mean(gray, (downscale, downscale))
        pixel_size *= downscale
    return gray, pixel_size


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_result(res: LIMResult, path=None, show=False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = res.config.wavelengths[len(res.config.wavelengths) // 2]
    dx_mm = res.pixel_size * 1e3
    h, w = res.opd_true.shape
    extent = [0, w * dx_mm, h * dx_mm, 0]
    nm = 1e9
    true = res.opd_true * nm
    rec = res.opd_rec * nm
    rec = rec - rec.mean() + true.mean()

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    panels = [
        (true, "Ground-truth OPD (nm)", "viridis"),
        (res.frame[lam], f"Raw interferogram near quadrature ({lam*1e9:.0f} nm)", "gray"),
        (res.psi[lam] * 1e3, f"PSI phase ({lam*1e9:.0f} nm) (mrad)", "RdBu_r"),
        (rec, "Reconstructed OPD (nm)", "viridis"),
        (rec - true, "Reconstruction error (nm)", "RdBu_r"),
    ]
    for a, (img, title, cmap) in zip(ax.ravel()[:5], panels):
        im = a.imshow(img, cmap=cmap, extent=extent)
        a.set_title(title)
        a.set_xlabel("x (mm)")
        a.set_ylabel("y (mm)")
        fig.colorbar(im, ax=a, shrink=0.8)
    row = h // 2
    x = np.arange(w) * dx_mm
    ax[1, 2].plot(x, true[row], label="ground truth")
    ax[1, 2].plot(x, rec[row], label="reconstruction")
    ax[1, 2].set_title("Central row profile")
    ax[1, 2].set_xlabel("x (mm)")
    ax[1, 2].set_ylabel("OPD (nm)")
    ax[1, 2].legend()
    if path:
        fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Self-test and CLI
# --------------------------------------------------------------------------- #


def selftest():
    """PSI must recover the exact phase difference when there is no diffraction/noise."""
    cfg = LIMConfig(sample_distance=0.0, noise=False, background_opd_rms=0.0,
                    illumination_nonuniformity=0.0)
    n, dx = 128, 5e-6
    yy, xx = np.mgrid[:n, :n]
    gray = np.exp(-(((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * 8.0**2)))
    res = run_lim(gray, dx, cfg)
    opd = res.opd_true
    fx, fy = _freq_grids(opd.shape, dx)
    ramp = shear_phase_ramp(fx, fy, cfg.shear)
    for lam, psi in res.psi.items():
        phi = 2 * np.pi * opd / lam
        expected = ifft2(fft2(phi) * ramp).real - ifft2(fft2(phi) * np.conj(ramp)).real
        err = np.abs(psi - expected)[16:-16, 16:-16].max()
        print(f"[selftest] {lam*1e9:.0f} nm: max |PSI - analytic| = {err:.2e} rad")
        assert err < 5e-3, "PSI retrieval deviates from the analytic phase difference"
    print("[selftest] OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--image", default="USAF_1951.jpg")
    p.add_argument("--scale", default="USAF_1951_scale.json")
    p.add_argument("--downscale", type=int, default=1, help="integer block-mean downscaling")
    p.add_argument("--z-mm", type=float, default=5.0, help="sample-sensor distance (mm)")
    p.add_argument("--shear-um", type=float, default=25.0, help="Savart shear magnitude (um)")
    p.add_argument("--peak-opd-nm", type=float, default=5.0, help="OPD of gray=1 (nm)")
    p.add_argument("--wiener-k", type=float, default=1e-2)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="lim_output")
    p.add_argument("--show", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest()
        return

    gray, pixel_size = load_sample(args.data_dir, args.image, args.scale, args.downscale)
    s = args.shear_um * 1e-6 / np.sqrt(2)
    cfg = LIMConfig(
        shear=(s, s),
        sample_distance=args.z_mm * 1e-3,
        delta_n_max=args.peak_opd_nm * 1e-9 / 1e-6,  # with thickness = 1 um
        thickness=1e-6,
        noise=not args.no_noise,
        wiener_k=args.wiener_k,
        seed=args.seed,
    )
    print(f"Image: {gray.shape[1]}x{gray.shape[0]} px, pixel = {pixel_size*1e6:.2f} um")
    res = run_lim(gray, pixel_size, cfg)

    m = metrics(res)
    print(f"Peak OPD            : {m['opd_peak_nm']:.2f} nm")
    print(f"RMSE (mean removed) : {m['rmse_nm']:.3f} nm")
    print(f"Correlation         : {m['corr']:.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "lim_result.npz",
        opd_true=res.opd_true,
        opd_rec=res.opd_rec,
        pixel_size=res.pixel_size,
        **{f"psi_{int(k*1e9)}nm": v for k, v in res.psi.items()},
    )
    plot_result(res, out / "lim_result.png", show=args.show)
    print(f"Saved results to {out.resolve()}")


if __name__ == "__main__":
    main()
