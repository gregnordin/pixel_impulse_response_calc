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

lam = 0.365                      # wavelength [µm]
NA = 0.10                        # numerical aperture
w_pix = 24.3                     # image pixel size [µm]

nx = ny = 512
dx = dy = 0.1                    # sampling [µm]

# coordinate grids
x = (np.arange(nx) - nx//2) * dx
y = (np.arange(ny) - ny//2) * dy
X, Y = np.meshgrid(x, y, indexing='xy')
R = np.hypot(X, Y)

# physical wavenumber and Airy scaling factor
k0 = 2*np.pi/lam
alpha = k0 * NA                  # argument = alpha * r

# Airy PSF
z = alpha * R
psf = np.ones_like(R)
mask = z != 0
psf[mask] = (2*j1(z[mask])/z[mask])**2
psf[~mask] = 1.0
psf /= psf.sum() * dx * dy       # normalize to unit integral

# square top-hat object representing a single pixel
obj = ((np.abs(X) <= w_pix/2) & (np.abs(Y) <= w_pix/2)).astype(float)

# convolution via FFT (with proper centering)
obj0 = ifftshift(obj)
psf0 = ifftshift(psf)
I = fftshift(np.real(ifft2(fft2(obj0) * fft2(psf0))))
I /= I.max()                     # normalize peak to 1

# --- grayscale irradiance image ---
fig1, ax1 = plt.subplots(figsize=(6,5))
im = ax1.imshow(I, extent=[x[0], x[-1], y[0], y[-1]],
                origin='lower', cmap='gray')
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_title('Single-pixel irradiance (normalized)')
cbar = fig1.colorbar(im, ax=ax1)
cbar.set_label('Normalized I')

# --- center-line cross-section ---
fig2, ax2 = plt.subplots(figsize=(7,4))
ax2.plot(x, I[ny//2, :], color='k')
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('Normalized irradiance')
ax2.set_title('Center-line cross-section')

# vertical lines at ±27/2 µm (nominal pixel boundaries)
half = 27/2
ax2.axvline(-half, color='r', linestyle='--')
ax2.axvline(+half, color='r', linestyle='--')

plt.show()


