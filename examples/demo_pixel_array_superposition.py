import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from pir_optics import PixelIrradianceModel


def build_single_pixel_model(
    wavelength=0.365,
    NA_image=0.10,
    mirror_pitch=7.6,
    img_pixel_pitch=27.0,
    pixel_fill=0.80,
    nx=512,
    dx=0.1,
):
    """
    Build a PixelIrradianceModel for a single pixel on a fixed 512x512 grid
    with 0.1 µm sampling, and return x, y, I_single.
    """
    model = PixelIrradianceModel(
        wavelength=wavelength,
        NA_image=NA_image,
        mirror_pitch=mirror_pitch,
        img_pixel_pitch=img_pixel_pitch,
        pixel_fill=pixel_fill,
        nx=nx,
        dx=dx,
        auto_compute=True,
        use_cache=False,
    )
    return model.x, model.y, model.I


def check_edge_small(I_single, threshold=1e-3):
    """
    Check that the normalized single-pixel irradiance is small at the grid edges.
    """
    edges = np.concatenate(
        [
            I_single[0, :],
            I_single[-1, :],
            I_single[:, 0],
            I_single[:, -1],
        ]
    )
    max_edge = float(np.max(np.abs(edges)))
    print(f"Max |I_single| on edges = {max_edge:.3g}")
    if max_edge > threshold:
        print(
            f"Warning: edge value {max_edge:.3g} > threshold {threshold:.3g}; "
            f"consider increasing the 512x512 window."
        )


def build_array_grid(pixel_pitch, dx_array=0.25, extra_pixels=1.0):
    """
    Build the (x, y) grid for the final array.

    For an N x N pixel image (N=5 here), we want to accommodate an (N+1) x (N+1)
    pixel FOV (i.e., 1/2 pixel buffer on each side) ⇒ 6 x 6 pixels total.

    Here we implement the 6x6 case directly: half-width = 3 * pixel_pitch.

    Parameters
    ----------
    pixel_pitch : float
        Image pixel pitch [µm].
    dx_array : float
        Sampling period for the final array grid [µm].
    extra_pixels : float
        Half-pixel buffer expressed in units of pixel_pitch (here 1.0 ⇒ 6x6).

    Returns
    -------
    x_arr, y_arr : 1D arrays [µm]
        Coordinates for the array grid.
    """
    half_width = 3.0 * pixel_pitch  # 6 pixels total width (±3P)
    # Ensure symmetric grid with center ~0
    L = 2.0 * half_width  # total FOV
    n = int(np.ceil(L / dx_array)) + 1
    if n % 2 == 0:
        n += 1  # make odd so we have a symmetric center sample
    x_arr = (np.arange(n) - n // 2) * dx_array
    y_arr = (np.arange(n) - n // 2) * dx_array
    return x_arr, y_arr


def build_pixel_centers(pixel_pitch, N=5):
    """
    Build pixel-center locations for an N x N pixel grid, centered at (0,0).

    Example for N=5:
        x_centers = [-2P, -P, 0, +P, +2P]
    """
    idx = np.arange(N) - (N - 1) / 2.0
    centers = idx * pixel_pitch
    return centers, centers  # (x_centers, y_centers)


def superpose_pixels(
    x_hi,
    y_hi,
    I_single,
    pixel_pitch,
    N_pix=5,
    dx_array=0.25,
):
    """
    Superpose the single-pixel irradiance at each pixel center to form an
    N_pix x N_pix array, sampled on a separate array grid.

    Parameters
    ----------
    x_hi, y_hi : 1D arrays [µm]
        High-resolution coordinates for the single-pixel irradiance.
    I_single : 2D array
        Normalized single-pixel irradiance on (y_hi, x_hi) grid.
    pixel_pitch : float [µm]
        Image pixel pitch.
    N_pix : int
        Number of pixels along x and y (assume square array).
    dx_array : float [µm]
        Sampling period for the final array grid.

    Returns
    -------
    x_arr, y_arr : 1D arrays [µm]
        Coordinates of the final array grid.
    I_array : 2D array
        Normalized irradiance on the final array grid from the N_pix x N_pix array.
    """
    # interpolator for single-pixel irradiance
    interp = RegularGridInterpolator(
        (y_hi, x_hi),  # note (y, x) ordering
        I_single,
        bounds_error=False,
        fill_value=0.0,
    )

    # build final array grid and pixel centers
    x_arr, y_arr = build_array_grid(pixel_pitch, dx_array=dx_array)
    X_arr, Y_arr = np.meshgrid(x_arr, y_arr, indexing="xy")
    Ny_arr, Nx_arr = X_arr.shape

    x_centers, y_centers = build_pixel_centers(pixel_pitch, N=N_pix)

    # flatten array grid once
    base_points = np.stack([Y_arr.ravel(), X_arr.ravel()], axis=-1)

    # accumulate contributions
    I_array = np.zeros_like(X_arr, dtype=float)

    for yc in y_centers:
        for xc in x_centers:
            # shift coordinates so single-pixel center is at (xc, yc)
            pts = base_points.copy()
            pts[:, 0] -= yc
            pts[:, 1] -= xc
            vals = interp(pts).reshape(Ny_arr, Nx_arr)
            I_array += vals

    # normalize to peak = 1
    max_val = I_array.max()
    if max_val != 0.0:
        I_array /= max_val

    return x_arr, y_arr, I_array


def main():
    # --- user parameters (for this example) ---
    wavelength = 0.365       # µm
    NA_image = 0.02
    mirror_pitch = 7.6       # µm
    img_pixel_pitch = 27.0   # µm (assumed < 40 µm)
    pixel_fill = 0.80

    # high-resolution single-pixel grid
    nx_hi = 512
    dx_hi = 0.1              # µm

    # final array grid sampling
    dx_array = 0.25          # µm (settable)

    # number of pixels in final array
    N_pix = 5

    # --- 1. Compute single-pixel irradiance on 512x512, 0.1 µm grid ---
    x_hi, y_hi, I_single = build_single_pixel_model(
        wavelength=wavelength,
        NA_image=NA_image,
        mirror_pitch=mirror_pitch,
        img_pixel_pitch=img_pixel_pitch,
        pixel_fill=pixel_fill,
        nx=nx_hi,
        dx=dx_hi,
    )

    # ensure normalized
    I_single = I_single / I_single.max()

    # --- 2. Check edges are small ---
    check_edge_small(I_single, threshold=1e-3)

    # --- 3. Superpose pixels on a separate array grid ---
    x_arr, y_arr, I_array = superpose_pixels(
        x_hi=x_hi,
        y_hi=y_hi,
        I_single=I_single,
        pixel_pitch=img_pixel_pitch,
        N_pix=N_pix,
        dx_array=dx_array,
    )

    X_arr, Y_arr = np.meshgrid(x_arr, y_arr, indexing="xy")

    # --- 4. Plot results ---

    # single-pixel irradiance
    fig_single, ax_single = plt.subplots(figsize=(5, 4))
    im1 = ax_single.imshow(
        I_single,
        extent=[x_hi[0], x_hi[-1], y_hi[0], y_hi[-1]],
        origin="lower",
        cmap="gray",
    )
    ax_single.set_xlabel("x (µm)")
    ax_single.set_ylabel("y (µm)")
    ax_single.set_title("Single-pixel irradiance (normalized)")
    fig_single.colorbar(im1, ax=ax_single, label="Normalized I")

    # array irradiance
    fig_arr, ax_arr = plt.subplots(figsize=(6, 5))
    im2 = ax_arr.imshow(
        I_array,
        extent=[x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]],
        origin="lower",
        cmap="gray",
    )
    ax_arr.set_xlabel("x (µm)")
    ax_arr.set_ylabel("y (µm)")
    ax_arr.set_title(f"{N_pix} x {N_pix} pixel array irradiance (normalized)")
    fig_arr.colorbar(im2, ax=ax_arr, label="Normalized I")

    # center-line across array
    fig_line, ax_line = plt.subplots(figsize=(6, 3))
    mid_y = len(y_arr) // 2
    ax_line.plot(x_arr, I_array[mid_y, :], "k")
    ax_line.set_xlabel("x (µm)")
    ax_line.set_ylabel("Normalized irradiance")
    ax_line.set_title("Center-line across array")
    ax_line.set_ylim(-0.05, 1.05)

    # show pixel-center positions on the centerline plot
    x_centers, _ = build_pixel_centers(img_pixel_pitch, N=N_pix)
    for xc in x_centers:
        ax_line.axvline(xc, color="r", linestyle="--", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    main()
