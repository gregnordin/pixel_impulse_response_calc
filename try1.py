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

# --- user parameters (all lengths in microns) ---
wavelength       = 0.365       # wavelength [µm]
NA_image         = 0.10        # image-side numerical aperture
mirror_pitch     = 7.6         # micromirror pixel pitch [µm]   <-- UPDATED
img_pixel_pitch  = 27.0        # image pixel pitch [µm]         <-- UPDATED
pixel_fill       = 0.80        # pixel fill factor (area)

# derived: square pixel width in image plane from pitch and fill factor
w_pix = img_pixel_pitch * np.sqrt(pixel_fill)

# derived magnification (image / object)
magnification = img_pixel_pitch / mirror_pitch

# --- analysis function: is NA appropriate for the pixel pitch? ---
def analyze_pixel_sampling(wavelength_um, NA_img, mirror_pitch_um, img_pix_pitch_um, M):
    # Rayleigh resolution in image plane
    delta_x_img = 0.61 * wavelength_um / NA_img
    ratio = img_pix_pitch_um / delta_x_img
    NA_min = 0.61 * wavelength_um / img_pix_pitch_um

    # object-side NA
    NA_obj = NA_img / M
    delta_x_obj = 0.61 * wavelength_um / NA_obj

    # DMD grating-like behavior
    # approximate first-order diffraction angle (in radians, small-angle)
    lam = wavelength_um
    d = mirror_pitch_um
    theta_1 = np.arcsin(min(1.0, lam / d))  # protect against lam>d
    theta_lens = np.arcsin(min(1.0, NA_img))

    if ratio > 3.0:
        regime = "pixel-limited (optics comfortably resolve pixels)"
    elif ratio > 1.0:
        regime = "borderline (diffraction and pixel scale are comparable)"
    else:
        regime = "diffraction-limited at the pixel scale"

    print("=== Pixel / NA analysis ===")
    print(f"wavelength           = {wavelength_um:.4f} µm")
    print(f"NA_image             = {NA_img:.4f}")
    print(f"mirror_pitch         = {mirror_pitch_um:.4f} µm")
    print(f"image_pixel_pitch    = {img_pix_pitch_um:.4f} µm")
    print(f"magnification (M)    = {M:.4f}")
    print(f"Rayleigh delta_x_img = {delta_x_img:.4f} µm")
    print(f"image_pitch / delta  = {ratio:.2f}")
    print(f"NA_min for pitch     = {NA_min:.4f}")
    print(f"object-side NA       = {NA_obj:.4f}")
    print(f"Rayleigh delta_x_obj = {delta_x_obj:.4f} µm")
    print(f"Regime: {regime}")
    print()
    print("=== Grating / order sanity check ===")
    print(f"first-order angle (DMD)   ≈ {np.degrees(theta_1):.2f} deg")
    print(f"lens semi-angle (NA)      ≈ {np.degrees(theta_lens):.2f} deg")
    print("Interpretation: lens semi-angle sets the angular field it accepts;")
    print("compare to +/- first-order angle to reason about which DMD orders")
    print("could fall into the pupil, depending on your off-axis design.")
    print()

# run the analysis with current parameters
analyze_pixel_sampling(
    wavelength_um=wavelength,
    NA_img=NA_image,
    mirror_pitch_um=mirror_pitch,
    img_pix_pitch_um=img_pixel_pitch,
    M=magnification
)

# --- numerical grid for PSF and pixel spot ---
nx = ny = 512
dx = dy = 0.1                  # sampling [µm]

x = (np.arange(nx) - nx//2) * dx
y = (np.arange(ny) - ny//2) * dy
X, Y = np.meshgrid(x, y, indexing='xy')
R = np.hypot(X, Y)

# --- wavenumber and Airy scaling ---
k0 = 2 * np.pi / wavelength    # physical wavenumber [1/µm]
alpha = k0 * NA_image          # scaling for Airy argument: alpha * r

# --- Airy PSF (intensity) ---
z = alpha * R
psf = np.ones_like(R)
mask = z != 0
psf[mask] = (2 * j1(z[mask]) / z[mask])**2
psf[~mask] = 1.0
psf /= psf.sum() * dx * dy     # unit integral

# --- object: single square pixel in the image plane ---
obj = ((np.abs(X) <= w_pix/2) & (np.abs(Y) <= w_pix/2)).astype(float)

# --- convolution via FFT with proper centering ---
obj0 = ifftshift(obj)
psf0 = ifftshift(psf)
I = fftshift(np.real(ifft2(fft2(obj0) * fft2(psf0))))
I /= I.max()                   # normalized peak = 1

# --- unique ASCII-only filename based on parameters ---
def fmt_ascii(v):
    return str(v).replace('.', 'p')

filename = (
    "irr_"
    f"lam{fmt_ascii(wavelength)}_"
    f"NA{fmt_ascii(NA_image)}_"
    f"mirpitch{fmt_ascii(mirror_pitch)}_"
    f"imgpitch{fmt_ascii(img_pixel_pitch)}_"
    f"fill{fmt_ascii(pixel_fill)}.npz"
)

np.savez(
    filename,
    x=x, y=y, I=I,
    wavelength=wavelength,
    NA_image=NA_image,
    mirror_pitch=mirror_pitch,
    img_pixel_pitch=img_pixel_pitch,
    pixel_fill=pixel_fill,
    w_pix=w_pix,
    magnification=magnification
)

# --- plots ---
fig1, ax1 = plt.subplots(figsize=(6, 5))
im = ax1.imshow(
    I,
    extent=[x[0], x[-1], y[0], y[-1]],
    origin='lower',
    cmap='gray'
)
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_title('Single-pixel irradiance (normalized)')
cbar = fig1.colorbar(im, ax=ax1)
cbar.set_label('Normalized I')

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.plot(x, I[ny//2, :], 'k')
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('Normalized irradiance')
ax2.set_title('Center-line cross-section')

half_pitch = img_pixel_pitch / 2
ax2.axvline(-half_pitch, color='r', linestyle='--')
ax2.axvline(+half_pitch, color='r', linestyle='--')

plt.show()

print("Saved file:", filename)


