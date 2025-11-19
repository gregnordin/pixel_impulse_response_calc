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

lam = 0.365      # wavelength [µm]
NA = 0.10
w_pix = 24.3     # image pixel size [µm]

nx = ny = 512
dx = dy = 0.1    # sampling [µm]

x = (np.arange(nx) - nx//2) * dx
y = (np.arange(ny) - ny//2) * dy
X, Y = np.meshgrid(x, y, indexing='xy')
R = np.hypot(X, Y)

k = 2*np.pi*NA/lam
z = k*R
psf = np.ones_like(R)
mask = z != 0
psf[mask] = (2*j1(z[mask])/z[mask])**2
psf[~mask] = 1.0
psf /= psf.sum() * dx * dy

obj = ((np.abs(X) <= w_pix/2) & (np.abs(Y) <= w_pix/2)).astype(float)

obj0 = ifftshift(obj)
psf0 = ifftshift(psf)

I = fftshift(np.real(ifft2(fft2(obj0) * fft2(psf0))))
I /= I.max()

# --- Plot irradiance image in grayscale ---
plt.figure(figsize=(6,5))
plt.imshow(I, extent=[x[0], x[-1], y[0], y[-1]],
           origin='lower', cmap='gray')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.title('Single-pixel irradiance (normalized)')
plt.colorbar(label='Normalized I')

# --- Center-line cross section with vertical lines ---
plt.figure(figsize=(7,4))
plt.plot(x, I[ny//2, :], 'k')
plt.xlabel('x (µm)')
plt.ylabel('Normalized irradiance')
plt.title('Center-line cross-section')

# vertical lines at ± 27/2 µm
half = 27/2
plt.axvline(-half, color='r', linestyle='--')
plt.axvline(+half, color='r', linestyle='--')

plt.show()

