import marimo

__generated_with = "0.18.0"
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
    img_pixel_pitch = mo.ui.number(start=5.0, stop=100.0, step=0.5, value=27.0)
    pixel_fill = mo.ui.number(start=0.1, stop=1.0, step=0.01, value=0.80)

    controls = mo.vstack(
        [
            mo.hstack([mo.md("**Wavelength (µm)**"), wavelength]),
            mo.hstack([mo.md("**NA (image side)**"), NA]),
            mo.hstack([mo.md("**Mirror pitch (µm)**"), mirror_pitch]),
            mo.hstack([mo.md("**Image pixel pitch (µm)**"), img_pixel_pitch]),
            mo.hstack([mo.md("**Pixel fill factor**"), pixel_fill]),
        ]
    )
    return NA, controls, img_pixel_pitch, mirror_pitch, pixel_fill, wavelength


@app.cell
def _(
    NA,
    PixelIrradianceModel,
    controls,
    img_pixel_pitch,
    mirror_pitch,
    mo,
    pixel_fill,
    plt,
    wavelength,
):
    # base sampling and FOV
    dx_base = 0.1         # [µm], fixed sampling period
    nx_base = 512
    base_FOV = nx_base * dx_base  # 51.2 µm

    pitch = img_pixel_pitch.value

    if pitch <= 50.0:
        dx = dx_base
        nx = nx_base
    else:
        # enlarge FOV for big pixels, keeping dx fixed
        fov_factor = 1.5
        required_FOV = fov_factor * pitch          # want FOV = f * pixel_pitch
        nx = int(required_FOV / dx_base)
        if nx < nx_base:
            nx = nx_base
        dx = dx_base

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

    # 2D irradiance plot
    fig2d, ax2d = plt.subplots(figsize=(5, 4))
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

    # center-line cross-section
    fig1d, ax1d = plt.subplots(figsize=(5, 3))
    ax1d.plot(model.x, model.I[model.ny // 2, :], "k")
    ax1d.set_xlabel("x (µm)")
    ax1d.set_ylabel("Normalized irradiance")
    ax1d.set_title("Center-line cross-section")
    ax1d.set_ylim(-0.05, 1.05)

    half_pitch = model.img_pixel_pitch / 2.0
    ax1d.axvline(-half_pitch, color="r", linestyle="--")
    ax1d.axvline(+half_pitch, color="r", linestyle="--")

    layout = mo.hstack(
        [
            mo.vstack([mo.md("### Parameters"), controls]),
            mo.vstack(
                [
                    mo.md("### Irradiance 2D"),
                    fig2d,
                    mo.md("### Center line"),
                    fig1d,
                ]
            ),
        ]
    )

    layout
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
