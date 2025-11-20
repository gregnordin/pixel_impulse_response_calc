# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///

# Here’s a compact NumPy/SciPy script that computes the 2D irradiance for a single pixel as
# 
# [
# I_i(x,y) = \big[\text{square top-hat of width } w\big] \ast \text{Airy PSF}
# ]
# 
# with:
# 
# * mirror width = 6.8 µm (used only to define magnification if you want),
# * image pixel size = 24.3 µm (used here directly as the square width),
# * λ = 365 nm,
# * NA = 0.10.

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import j1
import matplotlib.pyplot as plt

# --- user parameters ---
wavelength      = 0.365        # λ [µm]
NA              = 0.10         # numerical aperture
img_pixel_pitch = 24.3         # pixel pitch [µm]
img_pixel_fill  = 0.80         # fill factor (area)

# derived pixel width [µm]
w_pix = img_pixel_pitch * np.sqrt(img_pixel_fill)

# --- grid ---
nx = ny = 512
dx = dy = 0.1
x = (np.arange(nx) - nx//2) * dx
y = (np.arange(ny) - ny//2) * dy
X, Y = np.meshgrid(x, y, indexing='xy')
R = np.hypot(X, Y)

# --- wavenumber & PSF scale ---
k0 = 2*np.pi / wavelength
alpha = k0 * NA

# --- Airy PSF ---
z = alpha * R
psf = np.ones_like(R)
mask = z != 0
psf[mask] = (2*j1(z[mask]) / z[mask])**2
psf[~mask] = 1.0
psf /= psf.sum() * dx * dy

# --- object: 1 pixel ---
obj = ((np.abs(X) <= w_pix/2) & (np.abs(Y) <= w_pix/2)).astype(float)

# --- convolution via FFT ---
obj0 = ifftshift(obj)
psf0 = ifftshift(psf)
I = fftshift(np.real(ifft2(fft2(obj0) * fft2(psf0))))
I /= I.max()

# --- unique ASCII-only filename ---
def fmt_ascii(v):
    return str(v).replace('.', 'p')

filename = (
    f"irr_lam{fmt_ascii(wavelength)}_"
    f"NA{fmt_ascii(NA)}_"
    f"pitch{fmt_ascii(img_pixel_pitch)}_"
    f"fill{fmt_ascii(img_pixel_fill)}.npz"
)

# --- save irradiance data ---
np.savez(
    filename,
    x=x, y=y, I=I,
    wavelength=wavelength,
    NA=NA,
    img_pixel_pitch=img_pixel_pitch,
    img_pixel_fill=img_pixel_fill,
    w_pix=w_pix
)

# --- plots ---
fig1, ax1 = plt.subplots(figsize=(6,5))
im = ax1.imshow(I, extent=[x[0], x[-1], y[0], y[-1]],
                origin='lower', cmap='gray')
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_title('Single-pixel irradiance (normalized)')
cbar = fig1.colorbar(im, ax=ax1)
cbar.set_label('Normalized I')

fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.plot(x, I[ny//2, :], 'k')
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('Normalized irradiance')
ax2.set_title('Center-line cross-section')

half = img_pixel_pitch / 2
ax2.axvline(-half, color='r', linestyle='--')
ax2.axvline(+half, color='r', linestyle='--')

plt.show()

print("Saved:", filename)

