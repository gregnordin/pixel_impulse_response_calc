import marimo as mo

app = mo.App()


# --- Cell 0: imports and class ---
@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    from pir_optics import PixelIrradianceModel
    return mo, plt, PixelIrradianceModel


# --- Cell 1: UI controls (only start/stop/step/value) ---
@app.cell
def __(mo):
    wavelength = mo.ui.number(start=0.2, stop=1.0, step=0.005, value=0.365)
    NA = mo.ui.number(start=0.01, stop=0.5, step=0.01, value=0.10)
    mirror_pitch = mo.ui.number(start=2.0, stop=20.0, step=0.1, value=7.6)
    img_pixel_pitch = mo.ui.number(start=5.0, stop=100.0, step=0.5, value=27.0)
    pixel_fill = mo.ui.number(start=0.1, stop=1.0, step=0.05, value=0.80)

    controls = mo.vstack(
        [
            mo.hstack([mo.md("**Wavelength (µm)**"), wavelength]),
            mo.hstack([mo.md("**NA (image side)**"), NA]),
            mo.hstack([mo.md("**Mirror pitch (µm)**"), mirror_pitch]),
            mo.hstack([mo.md("**Image pixel pitch (µm)**"), img_pixel_pitch]),
            mo.hstack([mo.md("**Pixel fill factor**"), pixel_fill]),
        ]
    )

    return wavelength, NA, mirror_pitch, img_pixel_pitch, pixel_fill, controls


# --- Cell 2: model + plots + layout ---
@app.cell
def __(
    mo,
    plt,
    PixelIrradianceModel,
    wavelength,
    NA,
    mirror_pitch,
    img_pixel_pitch,
    pixel_fill,
    controls,
):
    model = PixelIrradianceModel(
        wavelength=wavelength.value,
        NA_image=NA.value,
        mirror_pitch=mirror_pitch.value,
        img_pixel_pitch=img_pixel_pitch.value,
        pixel_fill=pixel_fill.value,
        nx=512,
        dx=0.1,
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
    half_pitch = model.img_pixel_pitch / 2.0
    ax1d.axvline(-half_pitch, color="r", linestyle="--")
    ax1d.axvline(+half_pitch, color="r", linestyle="--")

    # layout: controls on left, raw figs on right
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

    layout  # rendered output
    return layout
