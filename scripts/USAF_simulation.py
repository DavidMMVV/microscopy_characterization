"""
Calibración píxel -> mm a partir de una imagen de un target USAF-1951.

Pasos:
  0. Estima la inclinación (orientación dominante de bordes + refinado por
     nitidez de proyecciones) y endereza la imagen. Un giro rígido no cambia
     el tamaño del píxel, así que no afecta a la escala. Desactivable con
     analyze(..., deskew=False).
  1. Binariza (Otsu) y etiqueta componentes conexas.
  2. Se quedan las componentes con forma de barra (rectángulo ~5:1).
  3. Agrupa las barras en tripletes (3 barras paralelas, separación = 2 x anchura)
     -> cada triplete es un elemento (horizontal o vertical) del target.
  4. Mide la anchura de barra de cada triplete en px (a partir del paso entre
     barras, mucho más preciso que el ancho de una sola barra).
  5. Asigna grupo/elemento: se parte del triplete MÁS GRANDE detectado, cuyo
     (grupo, elemento) das tú, y el resto se deduce por la razón 2^(1/6) entre
     elementos consecutivos (admite elementos saltados).
  6. Ajusta la escala mm/px por mínimos cuadrados.

Fórmulas USAF-1951:
  pares de líneas/mm = 2 ** (grupo + (elemento - 1) / 6)
  anchura de una barra = 1 / (2 * pares_de_lineas_por_mm)  [mm]
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

STEP = 2 ** (1 / 6)

def FT(array: np.ndarray) -> np.ndarray:
    """Transformada de Fourier centrada (frecuencias bajas en el centro)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(array)))
def IFT(array: np.ndarray) -> np.ndarray:
    """Transformada inversa de Fourier centrada (frecuencias bajas en el centro)."""
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(array)))
# ----------------------------------------------------------------- USAF
def lp_per_mm(group, element):
    return 2.0 ** (group + (element - 1) / 6)


def bar_width_mm(group, element):
    return 1.0 / (2.0 * lp_per_mm(group, element))


# ------------------------------------------------------------ binarizado


def binarize(gray, bars_are_dark=True):
    gray = np.asarray(gray, dtype=float)
    mask = gray < threshold_sauvola(gray, window_size=801, k=0.2, r=0.5*(gray.max()-gray.min()))
    return mask if bars_are_dark else ~mask


# ------------------------------------------------------------ inclinación
def _background(gray, bars_are_dark):
    return np.percentile(gray, 99 if bars_are_dark else 1)


def rotate_gray(gray, angle_deg, bars_are_dark=True, reshape=True):
    """Gira la imagen rellenando con el valor de fondo (sin cambiar el tamaño
    del píxel, así que la escala mm/px no se altera)."""
    return ndi.rotate(gray, angle_deg, reshape=reshape, order=1,
                      mode='constant', cval=_background(gray, bars_are_dark))


def _coarse_tilt(gray):
    """Orientación dominante de los bordes, módulo 90 deg, en (-45, 45]."""
    s = ndi.gaussian_filter(gray, 1.0)
    gy, gx = np.gradient(s)
    mag = np.hypot(gx, gy)
    th = np.arctan2(gy, gx)
    w = mag * (mag > 0.2 * mag.max())
    return np.degrees(np.arctan2((w * np.sin(4 * th)).sum(),
                                 (w * np.cos(4 * th)).sum())) / 4


def _alignment_score(gray, angle, bars_are_dark):
    """Alineado => los perfiles de la máscara por filas/columnas tienen saltos
    abruptos (bordes de barra perfectamente horizontales/verticales)."""
    r = rotate_gray(gray, angle, bars_are_dark, reshape=False)
    threshold = threshold_sauvola(r, window_size=801, k=0.2, r=0.5*(r.max()-r.min()))
    m = (r < threshold) if bars_are_dark else (r > threshold)
    m = m.astype(float)
    return (np.sum(np.diff(m.sum(axis=0)) ** 2) +
            np.sum(np.diff(m.sum(axis=1)) ** 2))


def estimate_tilt(gray, bars_are_dark=True, refine_span=1.0, refine_step=0.05):
    """Ángulo (grados) con el que HAY QUE GIRAR la imagen para enderezarla
    (convención de scipy.ndimage.rotate). Resultado en (-45, 45]; un giro de
    90/180/270 deg del target es irrelevante para la escala."""
    gray = np.asarray(gray, dtype=float)
    a0 = _coarse_tilt(gray)
    # el refinado se hace sobre una versión reducida (rápido y suficiente)
    f = max(1, int(round(max(gray.shape) / 600)))
    small = gray[::f, ::f]
    cands = a0 + np.arange(-refine_span, refine_span + 1e-9, refine_step)
    scores = [_alignment_score(small, a, bars_are_dark) for a in cands]
    return float(cands[int(np.argmax(scores))])


# ----------------------------------------------------------------- barras
def detect_bars(mask, aspect_range=(3.5, 7.0), min_fill=0.8, min_len=6):
    labels, n = ndi.label(mask)
    slices = ndi.find_objects(labels)
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    H, W = mask.shape
    bars = []
    for i, sl in enumerate(slices, start=1):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if y0 == 0 or x0 == 0 or y1 == H or x1 == W:  # toca el borde
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
    orient: str          # 'v' = barras verticales, 'h' = horizontales
    width_px: float      # anchura de una barra (px)
    width_area_px: float  # estimación alternativa (área / longitud)
    box: tuple           # (x0, y0, x1, y1)
    group: int = None
    element: int = None
    width_mm: float = None


def find_triplets(bars, orient, tol=0.15):
    """Busca 3 barras iguales, alineadas y espaciadas 2*anchura."""
    if orient == 'v':   # eje largo = y; se apilan en x
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


# --------------------------------------------------------------- análisis
def assign_elements(triplets, first_group, first_element):
    """Etiqueta cada triplete respecto al mayor, cuyo (grupo, elemento) es conocido."""
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
    """min_width_px: los tripletes con barras más finas se listan pero NO entran
    en el ajuste de la escala (a 2-3 px por barra la medida es poco fiable)."""
    gray = np.asarray(gray, dtype=float)
    angle = 0.0
    if deskew:
        angle = estimate_tilt(gray, bars_are_dark)
        gray = rotate_gray(gray, angle, bars_are_dark)
    mask = binarize(gray, bars_are_dark)
    bars = detect_bars(mask)
    triplets = find_triplets(bars, 'v') + find_triplets(bars, 'h')
    if not triplets:
        raise RuntimeError("No se detectó ningún triplete de barras. "
                           "Revisa bars_are_dark o los umbrales de detect_bars().")
    triplets = assign_elements(triplets, first_group, first_element)
    triplets.sort(key=lambda t: (-t.width_px, t.orient))

    fit = [t for t in triplets if t.width_px >= min_width_px] or triplets
    wp = np.array([t.width_px for t in fit])
    wm = np.array([t.width_mm for t in fit])
    scale = np.sum(wp * wm) / np.sum(wp ** 2)            # mm/px (LS ponderado)
    per_tri = wm / wp
    return dict(mask=mask, gray=gray, rotation_deg=angle,
                triplets=triplets, scale_mm_per_px=scale,
                scale_std=per_tri.std(ddof=1) if len(per_tri) > 1 else np.nan)


def plot_result(gray, res):
    """Dibuja la imagen (ya enderezada) en mm, con cada triplete etiquetado.

    Todo se pasa a mm con la MISMA transformación: el origen está en el píxel
    central y t.box está en píxeles de la imagen enderezada (res['gray']).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img = res.get('gray', gray)          # la que se analizó (puede tener otro tamaño)
    s = res['scale_mm_per_px']
    Ny, Nx = img.shape
    cx, cy = Nx // 2, Ny // 2

    def px2mm_x(p):                      # p: coordenada en píxeles (centro del píxel = entero)
        return (p - cx) * s

    def px2mm_y(p):
        return (p - cy) * s

    # extent = bordes exteriores de la imagen (medio píxel más allá de los centros)
    extent = (px2mm_x(-0.5), px2mm_x(Nx - 0.5), px2mm_y(Ny - 0.5), px2mm_y(-0.5))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap='gray', extent=extent)
    for t in res['triplets']:
        x0, y0, x1, y1 = t.box           # px, x1 e y1 exclusivos
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

def shearing(
        U_in: np.ndarray,
        shear: np.ndarray,
        order: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function returns two fields sheared one with respect to the other
    according to the specified shearing.

    Args:
        U_in (np.ndarray): Input field.
        shear (np.ndarray): Shearing vector in pixels.

    Returns:
        U_out(tuple(np.ndarray, np.ndarray)): Output sheared fields.
    """
    half_shear = shear / 2
    U1 = ndi.shift(U_in, half_shear, order = order)
    U2 = ndi.shift(U_in, -half_shear, order = order)
    return U1, U2

if __name__ == '__main__':
    import imageio.v2 as io
    import matplotlib.pyplot as plt
    from config import DATA_DIR

    # >>> Grupo y elemento del triplete MÁS GRANDE que se vea en la imagen <<<
    FIRST_GROUP, FIRST_ELEMENT = -2, 1   # tu imagen: grupo -2, elemento 1
    BARS_ARE_DARK = False      # barras negras sobre fondo claro

    gray = io.imread(DATA_DIR / 'USAF_1951.jpg').mean(axis=2).astype(float)
    gray = 1 - (gray / gray.max())
    threshold = threshold_sauvola(gray, window_size=801, k=0.2, r=0.5*(gray.max()-gray.min()))
    gray = (gray > threshold) * (gray)

    res = analyze_scale(gray, FIRST_GROUP, FIRST_ELEMENT, BARS_ARE_DARK)
    plot_result(gray, res)
    print(f"Escala: {res['scale_mm_per_px']:.5f} mm/px")

    Ny, Nx = gray.shape
    x = (np.arange(Nx) - Nx // 2) * res['scale_mm_per_px']  # mm
    y = (np.arange(Ny) - Ny // 2) * res['scale_mm_per_px']  # mm
    extent = (x[0], x[-1], y[-1], y[0])  # mm

    Y, X = np.meshgrid(y, x, indexing='ij')
    shear = np.array([0., 10 * res['scale_mm_per_px']]) # mm
    d = 5 * 1e-6 # um
    l = 635 * 1e-9 # nm
    E_out = np.exp(1j * 2 * np.pi * d * gray / l)
    plt.figure("OPD")
    plt.imshow(1e9 * np.angle(E_out) * l / (2 * np.pi), extent=extent)
    c = plt.colorbar()
    c.set_label("OPD (nm)")

    shear_pix = shear / res['scale_mm_per_px']  # px
    U1, U2 = shearing(E_out, shear_pix, order=0)
    Ut = U1 - U2
    fig, sub = plt.subplots(1,2)
    sub[0].imshow(np.abs(Ut), extent=extent)
    sub[0].set_title('Sheared field')   
    sub[1].imshow(np.angle(Ut), extent=extent)
    sub[1].set_title('Phase of sheared field')

    fig1, sub1 = plt.subplots(1,2)
    sub1[0].imshow(np.log(np.abs(FT(Ut))), extent=extent)
    sub1[0].set_title('Sheared field')   
    sub1[1].imshow(np.angle((FT(Ut))), extent=extent)
    sub1[1].set_title('Phase of sheared field')

    plt.show()