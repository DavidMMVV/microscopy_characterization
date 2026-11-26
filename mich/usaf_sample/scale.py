"""
Pixel -> mm calibration from an image of a USAF-1951 target.

Steps:
  0. Estimates the tilt (dominant edge orientation + refinement by
     sharpness of the projections) and straightens the image. A rigid rotation
     does not change the pixel size, so it does not affect the scale.
     Can be disabled with analyze(..., deskew=False).
  1. Binarizes (Otsu) and labels connected components.
  2. Keeps the bar-shaped components (rectangle ~5:1).
  3. Groups the bars into triplets (3 parallel bars, spacing = 2 x width)
     -> each triplet is one element (horizontal or vertical) of the target.
  4. Measures the bar width of each triplet in px (from the pitch between
     bars, much more precise than the width of a single bar).
  5. Assigns group/element: starts from the LARGEST detected triplet, whose
     (group, element) you provide, and the rest is deduced from the 2^(1/6)
     ratio between consecutive elements (skipped elements are allowed).
  6. Fits the mm/px scale by least squares.

USAF-1951 formulas:
  line pairs/mm = 2 ** (group + (element - 1) / 6)
  width of one bar = 1 / (2 * line_pairs_per_mm)  [mm]
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

STEP = 2 ** (1 / 6)

# ----------------------------------------------------------------- USAF
def lp_per_mm(group, element):
    return 2.0 ** (group + (element - 1) / 6)


def bar_width_mm(group, element):
    return 1.0 / (2.0 * lp_per_mm(group, element))


# ------------------------------------------------------------ binarization


def binarize(gray, bars_are_dark=True):
    gray = np.asarray(gray, dtype=float)
    mask = gray < threshold_sauvola(gray, window_size=801, k=0.2, r=0.5*(gray.max()-gray.min()))
    return mask if bars_are_dark else ~mask


# ------------------------------------------------------------ tilt
def _background(gray, bars_are_dark):
    return np.percentile(gray, 99 if bars_are_dark else 1)


def rotate_gray(gray, angle_deg, bars_are_dark=True, reshape=True):
    """Rotates the image, filling with the background value (without changing
    the pixel size, so the mm/px scale is not altered)."""
    return ndi.rotate(gray, angle_deg, reshape=reshape, order=1,
                      mode='constant', cval=_background(gray, bars_are_dark))


def _coarse_tilt(gray):
    """Dominant edge orientation, modulo 90 deg, in (-45, 45]."""
    s = ndi.gaussian_filter(gray, 1.0)
    gy, gx = np.gradient(s)
    mag = np.hypot(gx, gy)
    th = np.arctan2(gy, gx)
    w = mag * (mag > 0.2 * mag.max())
    return np.degrees(np.arctan2((w * np.sin(4 * th)).sum(),
                                 (w * np.cos(4 * th)).sum())) / 4


def _alignment_score(gray, angle, bars_are_dark):
    """Aligned => the row/column profiles of the mask have abrupt jumps
    (bar edges perfectly horizontal/vertical)."""
    r = rotate_gray(gray, angle, bars_are_dark, reshape=False)
    threshold = threshold_sauvola(r, window_size=801, k=0.2, r=0.5*(r.max()-r.min()))
    m = (r < threshold) if bars_are_dark else (r > threshold)
    m = m.astype(float)
    return (np.sum(np.diff(m.sum(axis=0)) ** 2) +
            np.sum(np.diff(m.sum(axis=1)) ** 2))


def estimate_tilt(gray, bars_are_dark=True, refine_span=1.0, refine_step=0.05):
    """Angle (degrees) by which the image MUST BE ROTATED to straighten it
    (scipy.ndimage.rotate convention). Result in (-45, 45]; a 90/180/270 deg
    rotation of the target is irrelevant for the scale."""
    gray = np.asarray(gray, dtype=float)
    a0 = _coarse_tilt(gray)
    # refinement is done on a downsampled version (fast and sufficient)
    f = max(1, int(round(max(gray.shape) / 600)))
    small = gray[::f, ::f]
    cands = a0 + np.arange(-refine_span, refine_span + 1e-9, refine_step)
    scores = [_alignment_score(small, a, bars_are_dark) for a in cands]
    return float(cands[int(np.argmax(scores))])


# ----------------------------------------------------------------- bars
def detect_bars(mask, aspect_range=(3.5, 7.0), min_fill=0.8, min_len=6):
    labels, n = ndi.label(mask)
    slices = ndi.find_objects(labels)
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    H, W = mask.shape
    bars = []
    for i, sl in enumerate(slices, start=1):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if y0 == 0 or x0 == 0 or y1 == H or x1 == W:  # touches the border
            continue
        h, w = y1 - y0, x1 - x0
        L, T = max(h, w), min(h, w)
        area = areas[i]
        if L < min_len or area / (h * w) < min_fill:
            continue
        if not (aspect_range[0] <= L / T <= aspect_range[1]):
            continue
        bars.append(dict(cx=(x0 + x1 - 1) / 2, cy=(y0 + y1 - 1) / 2, h=h, w=w,
                         area=area, box=(x0, y0, x1, y1)))
    return bars


@dataclass
class Triplet:
    orient: str          # 'v' = vertical bars, 'h' = horizontal bars
    width_px: float      # width of one bar (px)
    width_area_px: float  # alternative estimate (area / length)
    box: tuple           # (x0, y0, x1, y1)
    group: int = None
    element: int = None
    width_mm: float = None


def find_triplets(bars, orient, tol=0.15):
    """Looks for 3 equal, aligned bars spaced 2*width apart."""
    if orient == 'v':   # long axis = y; stacked along x
        sel = [b for b in bars if b['h'] > b['w']]
        along, across, L, T = 'cy', 'cx', 'h', 'w'
    else:
        sel = [b for b in bars if b['w'] > b['h']]
        along, across, L, T = 'cx', 'cy', 'w', 'h'
    sel = sorted(sel, key=lambda b: b[across])
    used, out = set(), []

    def next_bar(a):
        for j, b in enumerate(sel):
            if j in used or b is a:
                continue
            d = b[across] - a[across]
            if (abs(b[along] - a[along]) < tol * a[L]
                    and abs(b[L] - a[L]) < tol * a[L]
                    and abs(b[T] - a[T]) <= max(1.5, tol * a[T])
                    and 1.5 * a[T] <= d <= 2.6 * a[T]):
                return j, b
        return None, None

    for i, a in enumerate(sel):
        if i in used:
            continue
        j, b = next_bar(a)
        if b is None:
            continue
        k, c = next_bar(b)
        if c is None:
            continue
        used.update({i, j, k})
        trio = [a, b, c]
        w_pitch = (c[across] - a[across]) / 4.0
        w_area = np.mean([t['area'] / t[L] for t in trio])
        x0 = min(t['box'][0] for t in trio); y0 = min(t['box'][1] for t in trio)
        x1 = max(t['box'][2] for t in trio); y1 = max(t['box'][3] for t in trio)
        out.append(Triplet(orient, w_pitch, w_area, (x0, y0, x1, y1)))
    return out


# --------------------------------------------------------------- analysis
def assign_elements(triplets, first_group, first_element):
    """Labels each triplet relative to the largest one, whose (group, element) is known."""
    w0 = max(t.width_px for t in triplets)
    base = first_group * 6 + (first_element - 1)
    for t in triplets:
        steps = np.log(w0 / t.width_px) / np.log(STEP)
        n = int(round(steps))
        t.residual = steps - n
        idx = base + n
        t.group, t.element = idx // 6, idx % 6 + 1
        t.width_mm = bar_width_mm(t.group, t.element)
    return triplets


def analyze_scale(gray, first_group, first_element, bars_are_dark=True,
            min_width_px=5.0, deskew=True):
    """min_width_px: triplets with thinner bars are listed but do NOT enter
    the scale fit (at 2-3 px per bar the measurement is unreliable)."""
    gray = np.asarray(gray, dtype=float)
    angle = 0.0
    if deskew:
        angle = estimate_tilt(gray, bars_are_dark)
        gray = rotate_gray(gray, angle, bars_are_dark)
    mask = binarize(gray, bars_are_dark)
    bars = detect_bars(mask)
    triplets = find_triplets(bars, 'v') + find_triplets(bars, 'h')
    if not triplets:
        raise RuntimeError("No bar triplet was detected. "
                           "Check bars_are_dark or the detect_bars() thresholds.")
    triplets = assign_elements(triplets, first_group, first_element)
    triplets.sort(key=lambda t: (-t.width_px, t.orient))

    fit = [t for t in triplets if t.width_px >= min_width_px] or triplets
    wp = np.array([t.width_px for t in fit])
    wm = np.array([t.width_mm for t in fit])
    scale = np.sum(wp * wm) / np.sum(wp ** 2)            # mm/px (weighted LS)
    per_tri = wm / wp
    return dict(mask=mask, gray=gray, rotation_deg=angle,
                triplets=triplets, scale_mm_per_px=scale,
                scale_std=per_tri.std(ddof=1) if len(per_tri) > 1 else np.nan)


def plot_result(gray, res):
    """Plots the (already straightened) image in mm, with each triplet labeled.

    Everything is converted to mm with the SAME transformation: the origin is at
    the central pixel and t.box is in pixels of the straightened image (res['gray']).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = res.get('gray', gray)          # the one that was analyzed (may have a different size)
    s = res['scale_mm_per_px']
    Ny, Nx = img.shape
    cx, cy = Nx // 2, Ny // 2

    def px2mm_x(p):                      # p: coordinate in pixels (pixel center = integer)
        return (p - cx) * s

    def px2mm_y(p):
        return (p - cy) * s

    # extent = outer edges of the image (half a pixel beyond the centers)
    extent = (px2mm_x(-0.5), px2mm_x(Nx - 0.5), px2mm_y(Ny - 0.5), px2mm_y(-0.5))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap='gray', extent=extent)
    for t in res['triplets']:
        x0, y0, x1, y1 = t.box           # px, x1 and y1 exclusive
        col = 'tab:red' if t.orient == 'v' else 'tab:cyan'
        ax.add_patch(Rectangle((px2mm_x(x0 - 0.5), px2mm_y(y0 - 0.5)),
                               (x1 - x0) * s, (y1 - y0) * s,
                               fill=False, ec=col, lw=1.2))
        ax.text(px2mm_x(x0 - 0.5), px2mm_y(y0 - 0.5) - 3 * s,
                f"G{t.group}E{t.element}", color=col, fontsize=7)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title(f"{s:.5f} mm/px")
    return fig