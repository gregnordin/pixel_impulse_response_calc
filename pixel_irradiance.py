# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///

import os
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.special import j1
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


class PixelIrradianceModel:
    def __init__(self,
                 wavelength,
                 NA_image,
                 mirror_pitch,
                 img_pixel_pitch,
                 pixel_fill,
                 nx=512,
                 dx=0.1,
                 auto_compute=True,
                 use_cache=True):
        # physical parameters (all in microns except NA)
        self.wavelength = float(wavelength)
        self.NA_image = float(NA_image)
        self.mirror_pitch = float(mirror_pitch)
        self.img_pixel_pitch = float(img_pixel_pitch)
        self.pixel_fill = float(pixel_fill)

        # grid parameters
        self.nx = int(nx)
        self.ny = int(nx)   # square grid
        self.dx = float(dx)
        self.dy = float(dx)

        # derived quantities
        self.magnification = self.img_pixel_pitch / self.mirror_pitch
        self.w_pix = self.img_pixel_pitch * np.sqrt(self.pixel_fill)

        # placeholders for computed data
        self.x = None
        self.y = None
        self.X = None
        self.Y = None
        self.R = None
        self.psf = None
        self.obj = None
        self.I = None
        self._interp = None  # RegularGridInterpolator, built lazily

        # filename based on parameters
        self.filename = self._make_filename()

        if auto_compute:
            self.compute(use_cache=use_cache)

    # ---------- public API ----------

    def compute(self, use_cache=True):
        """Compute (or load) PSF-convolved single-pixel irradiance."""
        if use_cache and os.path.exists(self.filename):
            self._load_npz(self.filename)
            self._analyze_pixel_sampling()
            return

        self._build_grid()
        self._compute_psf()
        self._compute_pixel_object()
        self._convolve()
        self._analyze_pixel_sampling()
        self._save_npz(self.filename)

    def plot_irradiance_2d(self):
        if self.I is None:
            raise RuntimeError("Irradiance not computed yet.")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            self.I,
            extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
            origin='lower',
            cmap='gray'
        )
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('y (µm)')
        ax.set_title('Single-pixel irradiance (normalized)')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalized I')
        # plt.show()

    def plot_centerline(self):
        if self.I is None:
            raise RuntimeError("Irradiance not computed yet.")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.x, self.I[self.ny // 2, :], 'k')
        ax.set_xlabel('x (µm)')
        ax.set_ylabel('Normalized irradiance')
        ax.set_title('Center-line cross-section')
        half_pitch = self.img_pixel_pitch / 2.0
        ax.axvline(-half_pitch, color='r', linestyle='--')
        ax.axvline(+half_pitch, color='r', linestyle='--')
        # plt.show()

    def build_interpolator(self, bounds_error=False, fill_value=0.0):
        """Build a 2D interpolator I(x, y) from the stored irradiance."""
        if self.I is None:
            raise RuntimeError("Irradiance not computed yet.")
        self._interp = RegularGridInterpolator(
            (self.y, self.x),  # note (y, x) ordering
            self.I,
            bounds_error=bounds_error,
            fill_value=fill_value
        )

    def irradiance(self, xq, yq):
        """Evaluate normalized irradiance at arbitrary (xq, yq)."""
        if self._interp is None:
            self.build_interpolator()
        xq = np.asarray(xq)
        yq = np.asarray(yq)
        pts = np.stack([np.ravel(yq), np.ravel(xq)], axis=-1)
        vals = self._interp(pts)
        return vals.reshape(np.broadcast(xq, yq).shape)

    # ---------- internal helpers ----------

    def _build_grid(self):
        self.x = (np.arange(self.nx) - self.nx // 2) * self.dx
        self.y = (np.arange(self.ny) - self.ny // 2) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.R = np.hypot(self.X, self.Y)

    def _compute_psf(self):
        k0 = 2 * np.pi / self.wavelength
        alpha = k0 * self.NA_image
        z = alpha * self.R

        psf = np.ones_like(z)
        mask = z != 0.0

        # For z = alpha*r: z = 0 only at the image center.
        # The Airy formula (2*J1(z)/z)^2 is 0/0 at z=0, but its limit as z→0 is exactly 1.
        # We therefore assign psf = 1.0 at that point to avoid division-by-zero.
        psf[mask] = (2 * j1(z[mask]) / z[mask])**2
        psf[~mask] = 1.0

        psf /= psf.sum() * self.dx * self.dy
        self.psf = psf

    def _compute_pixel_object(self):
        self.obj = (
            (np.abs(self.X) <= self.w_pix / 2.0) &
            (np.abs(self.Y) <= self.w_pix / 2.0)
        ).astype(float)

    def _convolve(self):
        obj0 = ifftshift(self.obj)
        psf0 = ifftshift(self.psf)
        I = fftshift(np.real(ifft2(fft2(obj0) * fft2(psf0))))
        I /= I.max()
        self.I = I

    def _analyze_pixel_sampling(self):
        lam = self.wavelength
        NA_img = self.NA_image
        p_mir = self.mirror_pitch
        p_img = self.img_pixel_pitch
        M = self.magnification

        delta_x_img = 0.61 * lam / NA_img
        ratio = p_img / delta_x_img
        NA_min = 0.61 * lam / p_img

        NA_obj = NA_img / M
        delta_x_obj = 0.61 * lam / NA_obj

        d = p_mir
        theta_1 = np.arcsin(min(1.0, lam / d))
        theta_lens = np.arcsin(min(1.0, NA_img))

        if ratio > 3.0:
            regime = "pixel-limited (optics comfortably resolve pixels)"
        elif ratio > 1.0:
            regime = "borderline (diffraction and pixel scale are comparable)"
        else:
            regime = "diffraction-limited at the pixel scale"

        print("=== Pixel / NA analysis ===")
        print(f"wavelength           = {lam:.4f} µm")
        print(f"NA_image             = {NA_img:.4f}")
        print(f"mirror_pitch         = {p_mir:.4f} µm")
        print(f"image_pixel_pitch    = {p_img:.4f} µm")
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

    def _make_filename(self):
        def fmt(v):
            return str(v).replace('.', 'p')
        return (
            "irr_"
            f"lam{fmt(self.wavelength)}_"
            f"NA{fmt(self.NA_image)}_"
            f"mirpitch{fmt(self.mirror_pitch)}_"
            f"imgpitch{fmt(self.img_pixel_pitch)}_"
            f"fill{fmt(self.pixel_fill)}.npz"
        )

    def _save_npz(self, fname):
        np.savez(
            fname,
            x=self.x,
            y=self.y,
            I=self.I,
            wavelength=self.wavelength,
            NA_image=self.NA_image,
            mirror_pitch=self.mirror_pitch,
            img_pixel_pitch=self.img_pixel_pitch,
            pixel_fill=self.pixel_fill,
            w_pix=self.w_pix,
            magnification=self.magnification,
            nx=self.nx,
            ny=self.ny,
            dx=self.dx,
            dy=self.dy
        )
        print("Saved file:", fname)

    def _load_npz(self, fname):
        data = np.load(fname)
        self.x = data["x"]
        self.y = data["y"]
        self.I = data["I"]
        self.wavelength = float(data["wavelength"])
        self.NA_image = float(data["NA_image"])
        self.mirror_pitch = float(data["mirror_pitch"])
        self.img_pixel_pitch = float(data["img_pixel_pitch"])
        self.pixel_fill = float(data["pixel_fill"])
        self.w_pix = float(data["w_pix"])
        self.magnification = float(data["magnification"])
        self.nx = int(data["nx"])
        self.ny = int(data["ny"])
        self.dx = float(data["dx"])
        self.dy = float(data["dy"])
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')
        self.R = np.hypot(self.X, self.Y)
        self.psf = None
        self.obj = None
        self._interp = None
        print("Loaded file:", fname)


# ---------- example usage ----------

if __name__ == "__main__":
    model = PixelIrradianceModel(
        wavelength=0.365,
        NA_image=0.10,
        mirror_pitch=7.6,
        img_pixel_pitch=27.0,
        pixel_fill=0.80,
        nx=512,
        dx=0.1,
        auto_compute=True,
        use_cache=True
    )

    model.plot_irradiance_2d()
    model.plot_centerline()
    plt.show()

    val = model.irradiance(3.0, 1.0)
    print("I(3 µm, 1 µm) =", val)
