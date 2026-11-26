import json
import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage as ndi  # type: ignore[import-untyped]
from skimage.filters import threshold_sauvola

from typing import Tuple, cast
import cmath

#from mich.usaf_sample.scale import analyze_scale, plot_result
from config import DATA_DIR

@jax.jit
def FT(array: jnp.ndarray) -> jnp.ndarray:
    """Centered Fourier transform (low frequencies at the center)."""
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(array, axes=(-2, -1))), axes=(-2, -1))
@jax.jit
def IFT(array: jnp.ndarray) -> jnp.ndarray:
    """Centered inverse Fourier transform (low frequencies at the center)."""
    return jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.ifftshift(array, axes=(-2, -1))), axes=(-2, -1))

def propagate(
        Ui: jnp.ndarray,
        ftcoord: jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray],
        z: float,
        l: float
) -> jnp.ndarray:
    """This function propagates the field Ui over a distance z
    using angular spectrum method.

    Args:
        Ui (jnp.ndarray): Input field.
        ftcoord (jnp.ndarray | Tuple[jnp.ndarray, jnp.ndarray]): 3D array 
        with the coordinates in Fourier space.
        z (float): Propagation distance.
        l (float): Wavelength.

    Returns:
        Uo (jnp.ndarray): Propagated field.
    """
    ui_ft = FT(Ui)
    mask = ((ftcoord[0] ** 2 + ftcoord[1] ** 2) < (1 / l) ** 2) * 1
    if Ui.ndim == 2:
        uo_ft = ui_ft * jnp.exp(1j * (1j**(1-mask)) * 2 * jnp.pi * z * 
                               jnp.sqrt((-1)**(mask+1) / l ** 2 + (-1)**mask * 
                                       (ftcoord[0] ** 2 + ftcoord[1] ** 2)))
    elif Ui.ndim == 3:
        uo_ft = ui_ft * jnp.exp(1j * (1j**(1-mask)) * 2 * jnp.pi * z * 
                               jnp.sqrt((-1)**(mask+1) / l ** 2 + (-1)**mask * 
                                       (ftcoord[0][None] ** 2 + ftcoord[1][None] ** 2)))
    else:
        raise ValueError("Input field must be 2D or 3D.")
    U0 = IFT(uo_ft)
    return U0
def shearing(
        U_in: jnp.ndarray,
        shear: jnp.ndarray,
        order: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function returns two fields sheared one with respect to the other
    according to the specified shearing.

    Args:
        U_in (jnp.ndarray): Input field.
        shear (jnp.ndarray): Shearing vector in pixels.

    Returns:
        U_out(tuple(jnp.ndarray, jnp.ndarray)): Output sheared fields.
    """
    half_shear = shear / 2
    U1 = jnp.asarray(ndi.shift(U_in, half_shear, order=order, mode='nearest'))
    U2 = jnp.asarray(ndi.shift(U_in, -half_shear, order=order, mode='nearest'))
    return U1, U2

class Grid:
    """
    Represents a 2D spatial grid and its corresponding Fourier-space grid.

    The grid is defined by its shape and pixel size. Spatial coordinates are
    centered around the origin, and the corresponding spatial frequency
    coordinates are computed assuming the sampling determined by the grid.

    Attributes:
        shape (Tuple[int, int]): Number of pixels along each spatial dimension (Ny, Nx). 

        pix_size (float): Spatial sampling interval in both dimensions.

        ftpix_size (jnp.ndarray): Sampling interval in the Fourier domain, 
        expressed in inverse length.

        coord (Tuple[jnp.ndarray, jnp.ndarray]): Spatial coordinate meshes (x, y), 
        centered around the origin.

        ftcoord (Tuple[jnp.ndarray, jnp.ndarray]): Fourier-space coordinate meshes
        (fx, fy), centered around zero.

        extent (jnp.ndarray): Spatial boundaries of the grid, formatted as
        [x_min, x_max, y_max, y_min], suitable for plotting.

        ftextent (jnp.ndarray): Fourier-space boundaries, formatted as
        [fx_min, fx_max, fy_max, fy_min], suitable for plotting.
    """
    def __init__(self, shape: Tuple[int, int], pix_size: float):
        """
        Initialize the spatial and Fourier-space grids.

        Args:
            shape (Tuple[int, int]): Number of pixels in each direction
            pix_size (float): Pixel size
        """
        self.shape = shape
        self.pix_size = pix_size        
        self.ftpix_size = jnp.array([1 / (shape[0] * pix_size),
                                    1 / (shape[1] * pix_size)])
                                    
        x = (jnp.arange(shape[1]) - shape[1] // 2) * pix_size
        y = (jnp.arange(shape[0]) - shape[0] // 2) * pix_size
        fx = (jnp.arange(shape[0]) - shape[0] // 2) * self.ftpix_size[0]
        fy = (jnp.arange(shape[1]) - shape[1] // 2) * self.ftpix_size[1]

        self.coord = jnp.meshgrid(x, y, indexing='ij')
        self.ftcoord = jnp.meshgrid(fx, fy, indexing='ij')

        self.extent = jnp.array([x[0], x[-1], y[-1], y[0]])
        self.ftextent = jnp.array([fx[0], fx[-1], fy[-1], fy[0]])

if __name__ == '__main__':
    import imageio.v2 as io
    import matplotlib.pyplot as plt
    from config import DATA_DIR

    # >>> Group and element of the LARGEST triplet visible in the image <<<
    FIRST_GROUP, FIRST_ELEMENT = -2, 1   # your image: group -2, element 1
    BARS_ARE_DARK = False      # dark bars on a light background

    # Filter the background and inverse the distribution (dark->bright)
    gray = io.imread(DATA_DIR / 'USAF_1951.jpg').mean(axis=2).astype(float)
    gray = 1 - (gray / gray.max())
    threshold = threshold_sauvola(gray, window_size=801, k=0.2, r=0.5*(gray.max()-gray.min()))
    gray = (gray > threshold) * (gray)

    # Extract the pixel size
    #res = analyze_scale(gray, FIRST_GROUP, FIRST_ELEMENT, BARS_ARE_DARK)
    #plot_result(gray, res)
    #print(f"Scale: {res['scale_mm_per_px']:.5f} mm/px")
    #with open(DATA_DIR / 'USAF_1951_scale.json', 'w') as f:
    #    json.dump(cast(float, res['scale_mm_per_px']) * 1e-3, f, indent=4)

    with open(DATA_DIR / 'USAF_1951_scale.json', 'r') as f:
        scale = json.load(f)

    # Define the grid we use and some parameters
    grid = Grid(shape=gray.shape, pix_size=scale)    
    shear = jnp.array([0., 5 * grid.pix_size]) # m
    d = 10 * 1e-9 # m
    l = 635 * 1e-9 # m
    z = 30 * 1e-3 # m
    m = 1
    n = 0
    alpha = jnp.linspace(2 * jnp.pi * n,2 * jnp.pi * m, 128)

    # Compute the detected fields
    E_sample = jnp.exp(1j * 2 * jnp.pi * d * gray / l)
    shear_pix = shear / grid.pix_size  # px
    U1, U2 = shearing(E_sample, shear_pix, order=0)
    U1 = U1[None] * jnp.exp(1j * alpha[:, None, None])
    Ut = U1 - U2[None]
    Uo = propagate(Ut, (grid.ftcoord[0], grid.ftcoord[1]), z, l)
    I_meas = jnp.abs(Uo) ** 2

    # Reconstruct the phase and amplitude of the sheared field
    gamma_sin = jnp.sin(alpha) / (jnp.pi * (n + m))
    gamma_cos = jnp.cos(alpha) / (jnp.pi * (n + m))
    integrand_sin = I_meas * gamma_sin[:, None, None]
    integrand_cos = I_meas * gamma_cos[:, None, None]

    # Prepare the integration for Simpson's rule
    coef_simpson_np = np.ones_like(alpha)
    coef_simpson_np[1::2] = 4
    coef_simpson_np[2:-1:2] = 2
    coef_simpson = jnp.array(coef_simpson_np)
    h = alpha[1] - alpha[0]

    psi_sin = (h / 3) * jnp.sum(integrand_sin * coef_simpson[:, None, None], axis=0)
    psi_cos = (h / 3) * jnp.sum(integrand_cos * coef_simpson[:, None, None], axis=0)

    phi = -jnp.atan(psi_sin / psi_cos)
    I_0 = 2 * jnp.sqrt(psi_sin**2 + psi_cos**2)

    fig,sub = plt.subplots(1,2)
    im1 = sub[0].imshow(I_0, extent=grid.extent * 1e3)
    plt.colorbar(im1, ax=sub[0])
    im2 = sub[1].imshow(phi, extent=grid.extent * 1e3)
    plt.colorbar(im2, ax=sub[1])

    # Compute and backpropagate the field
    A = 1
    B = 1
    H = (A * jnp.exp(-2j * jnp.pi * (grid.ftcoord[0] * shear[0] + grid.ftcoord[1] * shear[1])) +
         B * jnp.exp(2j * jnp.pi * (grid.ftcoord[0] * shear[0] + grid.ftcoord[1] * shear[1])))

    plt.figure("PSF")
    plt.imshow(jnp.abs(IFT(H)), extent=grid.extent * 1e3)
    plt.colorbar()

    

    fig, sub = plt.subplots(1,2)
    im1 = sub[0].imshow(jnp.abs(Ut[0])**2, extent=grid.extent * 1e3)
    plt.colorbar(im1, ax=sub[0])
    sub[0].set_title('Sheared field')   
    sub[0].set_xlabel('x (mm)')
    sub[0].set_ylabel('y (mm)')
    im2 = sub[1].imshow(jnp.angle(Ut[0]), extent=grid.extent * 1e3)
    plt.colorbar(im2, ax=sub[1])
    sub[1].set_title('Phase of sheared field')
    sub[1].set_xlabel('x (mm)')
    sub[1].set_ylabel('y (mm)')

    fig0, sub0 = plt.subplots(1,2)
    im1 = sub0[0].imshow(jnp.abs(Uo[0])**2, extent=grid.extent * 1e3)
    plt.colorbar(im1, ax=sub0[0])
    sub0[0].set_title('Propagated sheared field')   
    sub0[0].set_xlabel('x (mm)')
    sub0[0].set_ylabel('y (mm)')
    im2 = sub0[1].imshow(jnp.angle(Uo[0]), extent=grid.extent * 1e3)
    plt.colorbar(im2, ax=sub0[1])
    sub0[1].set_title('Propagated phase of sheared field')
    sub0[1].set_xlabel('x (mm)')
    sub0[1].set_ylabel('y (mm)')

    fig1, sub1 = plt.subplots(1,2)
    sub1[0].imshow(jnp.log(jnp.abs(FT(Ut[0]))), extent=grid.ftextent)
    sub1[0].set_title('Sheared field')   
    sub1[1].imshow(jnp.angle((FT(Ut[0]))), extent=grid.ftextent)
    sub1[1].set_title('Phase of sheared field')

    plt.show()