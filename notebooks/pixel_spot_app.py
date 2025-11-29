import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from pir_optics import PixelIrradianceModel
    return PixelIrradianceModel, mo, plt


@app.cell
def _(mo):
    wavelength = mo.ui.number(start=0.2, stop=1.0, step=0.005, value=0.365)
    NA = mo.ui.number(start=0.01, stop=0.5, step=0.01, value=0.10)
    mirror_pitch = mo.ui.number(start=2.0, stop=20.0, step=0.1, value=7.6)
    img_pixel_pitch = mo.ui.number(start=5.0, stop=100.0, step=0.1, value=27.0)
    pixel_fill = mo.ui.number(start=0.1, stop=1.0, step=0.01, value=0.80)

    # new: grid controls
    nx_ctrl = mo.ui.number(start=128, stop=4096, step=1, value=512)
    dx_ctrl = mo.ui.number(start=0.05, stop=1.0, step=0.01, value=0.10)

    indent_size = 6

    controls = mo.vstack(
        [
            mo.md("### Parameters"),
            mo.hstack([mo.md(f"{indent_size * '&nbsp;'}Wavelength (µm)"), wavelength]),
            mo.hstack([mo.md(f"{indent_size * '&nbsp;'}NA (image side)"), NA]),
            mo.hstack([mo.md(f"{indent_size * '&nbsp;'}Mirror pitch (µm)"), mirror_pitch]),
            mo.hstack([mo.md(f"{indent_size * '&nbsp;'}Image pixel pitch (µm)"), img_pixel_pitch]),
            mo.hstack([mo.md(f"{indent_size * '&nbsp;'}Pixel fill factor"), pixel_fill]),
            mo.md("### Grid settings"),
            mo.hstack([mo.md(f"{indent_size * '&nbsp;'}nx (samples per axis)"), nx_ctrl]),
            mo.hstack([mo.md(f"{indent_size * '&nbsp;'}dx (µm per sample)"), dx_ctrl]),
        ]
    )
    return (
        NA,
        controls,
        dx_ctrl,
        img_pixel_pitch,
        indent_size,
        mirror_pitch,
        nx_ctrl,
        pixel_fill,
        wavelength,
    )


@app.cell
def _(
    NA,
    PixelIrradianceModel,
    controls,
    dx_ctrl,
    img_pixel_pitch,
    indent_size,
    mirror_pitch,
    mo,
    nx_ctrl,
    pixel_fill,
    plt,
    wavelength,
):
    import time

    nx = int(nx_ctrl.value)
    dx = dx_ctrl.value
    pitch = img_pixel_pitch.value

    t0 = time.perf_counter()
    model = PixelIrradianceModel(
        wavelength=wavelength.value,
        NA_image=NA.value,
        mirror_pitch=mirror_pitch.value,
        img_pixel_pitch=pitch,
        pixel_fill=pixel_fill.value,
        nx=nx,
        dx=dx,
        auto_compute=True,
        use_cache=False,
    )
    elapsed = time.perf_counter() - t0


    # --- 2D irradiance plot ---
    fig2d, ax2d = plt.subplots(figsize=(6, 5))
    im = ax2d.imshow(
        model.I,
        extent=[model.x[0], model.x[-1], model.y[0], model.y[-1]],
        origin="lower",
        cmap="gray",
    )
    ax2d.set_xlabel("x (µm)")
    ax2d.set_ylabel("y (µm)")
    ax2d.set_title("Single-pixel irradiance (normalized)")
    fig2d.colorbar(im, ax=ax2d, label="Normalized I")

    # --- 2D PSF ---
    fig_psf, ax_psf = plt.subplots(figsize=(6, 5))
    im_psf = ax_psf.imshow(
        model.psf,
        extent=[model.x[0], model.x[-1], model.y[0], model.y[-1]],
        origin="lower",
        cmap="gray",
    )
    ax_psf.set_xlabel("x (µm)")
    ax_psf.set_ylabel("y (µm)")
    ax_psf.set_title("PSF (normalized)")
    fig_psf.colorbar(im_psf, ax=ax_psf, label="Normalized PSF")

    # --- centerline irradiance ---
    fig1d, ax1d = plt.subplots(figsize=(6, 3))
    ax1d.plot(model.x, model.I[model.ny // 2, :], "k")
    ax1d.plot(model.x, model.ideal_pixel[model.ny // 2, :], "r", linestyle="--")
    ax1d.set_xlabel("x (µm)")
    ax1d.set_ylabel("Normalized irradiance")
    ax1d.set_title("Center-line: irradiance")
    ax1d.set_ylim(-0.05, 1.05)
    # half_pitch = model.img_pixel_pitch / 2.0
    # ax1d.axvline(-half_pitch, color="r", linestyle="--")

    # ax1d.axvline(+half_pitch, color="r", linestyle="--")

    # --- centerline PSF ---
    fig1d_psf, ax1d_psf = plt.subplots(figsize=(6, 3))
    ax1d_psf.plot(model.x, model.psf[model.ny // 2, :], "k")
    ax1d_psf.set_xlabel("x (µm)")
    ax1d_psf.set_ylabel("Normalized PSF")
    ax1d_psf.set_title("Center-line: PSF")
    ax1d_psf.set_ylim(-0.05, 1.05)

    # --- layout: controls on top, plots below ---
    top_row = mo.vstack([
        controls,
        mo.md("### Diagnostics"),
        mo.md(f"{indent_size * '&nbsp;'}Computation time: {elapsed*1000:.3f} ms"),
        mo.md(f"{indent_size * '&nbsp;'}Max normalized edge irradiance: {model.max_edge_I:.3e}"),
    ])

    two_d_row = mo.hstack([fig2d, fig_psf])
    centerline_row = mo.hstack([fig1d, fig1d_psf])

    layout = mo.vstack([
        top_row,
        mo.md("### 2D plots"),
        two_d_row,
        mo.md("### Center-line plots"),
        centerline_row,
    ])

    layout
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
