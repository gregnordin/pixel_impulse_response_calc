



Next:

- Document assumptions and calculations for ChatGPT result.
- Save figures to files?
- Save 2D result to a file.
- Use 2D single pixel result to show what multiple pixels look like.
- Do line plot of multiple pixels and superimpose appropriate shifted single pixels.
- Show line plot of point spread function and pixel rect function and convolution of them (dashed rect on top of convolved result).
- Show animation of convolution of PSF and rect function (in 2D?)?
- Put in marimo notebook.
    - Add fill factor slider.
    - Add NA slider?
    - Add wavelength selector (365, 385, 405 nm).



# Log

## Thu, 11/20/25

- Generalize code to handle user-defined values for wavelength, NA, image pixel pitch, micromirror array pixel pitch, and pixel fill factor.
- Create function to analyze numerical aperture and print relevant values.
- Put code into a class in `src/pir_optics/pixel_irradiance.py`.
- Add marimo notebook example in `notebooks`.

## Fri, 11/22/25

- Play with xy range for different pixel sizes. Settle on 2 ranges: 0.1 &mu;m sampling for pixel sizes <= 40 &mu;m and 0.2 &mu;m sampling for larger pixel sizes.



PIR for [Asiga Max X27, 27 um pixel pitch](https://www.asiga.com/max-x/) ([Asiga Ultra](https://www.asiga.com/ultra/) is 32 um pixel pitch):

- DMD: [DLP651NE 0.65-Inch 1080p Digital Micromirror Device](https://www.ti.com/lit/ds/symlink/dlp651ne.pdf)
- [ChatGPT Irradiance Distribution Analysis](https://chatgpt.com/share/691e574e-ed30-800e-9d1d-997b1b67ae19)



Develop PIR for HR3.3u

- Jupyter notebook: `/Users/nordin/Documents/Projects/photopolymerization/development/2024-07-09_v0.2/2024-07-17_pixel_profile_from_images.ipynb`
- PIR image data: `/Users/nordin/Documents/Projects/photopolymerization/development/2024-07-09_v0.2/PIR_3_15_21-LED_powersetting_150`

