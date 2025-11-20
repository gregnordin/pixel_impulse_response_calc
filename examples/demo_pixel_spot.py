from pir_optics import PixelIrradianceModel
import matplotlib.pyplot as plt

def main():
    model = PixelIrradianceModel(
        wavelength=0.365,
        NA_image=0.10,
        mirror_pitch=7.6,
        img_pixel_pitch=27.0,
        pixel_fill=0.80,
        nx=512,
        dx=0.1,
        auto_compute=True,
        use_cache=True,
    )

    model.plot_irradiance_2d()
    model.plot_centerline()

    plt.show()

if __name__ == "__main__":
    main()
