



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

## Thu, 11/27/25

### Reconstitute ChatGPT chat

The chat thread I was using finally choked as it got too big (). Learn how to reconsititute it--see [Upload project as ZIP](https://chatgpt.com/share/692887d6-3d98-800e-9dcf-33754179e013)--which involves creating a **theory summary message** and a ~~**code summary message**~~ `PIR.zip` file.

Create zip file with project directory structure:

```bash
rm PIR.zip
zip -r PIR.zip PIR \
  -x "*/.git/*" \
  -x "*/.venv/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "try1.py" \
  -x "*.DS_Store"
```

#### Theory summary message

See `PIR_theory_summary.md`.

How to use this in future chats

- This entire block is your **theory summary message**.
- In a new chat, you can paste this first so I have the physics/model context.
- Then you can either:
    - Re-upload the **same ZIP** as your “code context” for the current state, or
    - Paste specific code snippets you want to modify.

You don’t *have* to maintain a separate “code summary” text file if managing the actual project ZIP is easier. The ZIP effectively *is* your code snapshot; a short textual “code structure” summary can help you, but it’s optional from my side as long as you can upload the project when needed.

## Wed, 11/26/25

### `PixelIrradianceModel`

- Change `_make_filename` to include all independent parameters so `.npz` filename is unique to a specific case.

- Change `examples/demo_pixel_spot.py` so it calculates and prints the elapsed time to run `PixelIrradianceModel`. For the results below, the runs after the first one take less time because `use_cache=True` so `compute` is just loading an `.npz` file.

    - Results: `nx=512, dx=0.1`: **4.24 ms, 2.11 ms, 2.30 ms**
    - Results: `nx=800, dx=0.1`: **28.70 ms, 3.91 ms, 4.16 ms**
    - Results: `nx=1024, dx=0.1`: **54.98 ms, 9.53 ms, 6.03 ms**
    - Results: `nx=1500, dx=0.1`: **98.89 ms, 10.71 ms, 11.22 ms**

- Change marimo `notebooks/pixel_spot_app.py` to have `nx` and `dx` as inputs and display elapsed time to run `PixelIrradianceModel`.

    `uv run examples/demo_pixel_spot.py`

- Find a problem with how I was running the marimo notebook with `uv` where it was stuck on an old version of my project. This is how it should be run now after fixing `pyproject.toml`:

    `uv run marimo edit notebooks/pixel_spot_app.py`

## Fri, 11/22/25

- Play with xy range for different pixel sizes. Settle on 2 ranges: 0.1 &mu;m sampling for pixel sizes <= 40 &mu;m and 0.2 &mu;m sampling for larger pixel sizes.

#### Superposition to get 5x5 pixel array

`examples/demo_pixel_array_superposition.py`

**Problems:**

- Change PSF to 0.05 and see all of the artifacts. **These need fixed**



## Thu, 11/20/25

- Generalize code to handle user-defined values for wavelength, NA, image pixel pitch, micromirror array pixel pitch, and pixel fill factor.
- Create function to analyze numerical aperture and print relevant values.
- Put code into a class in `src/pir_optics/pixel_irradiance.py`.
- Add marimo notebook example in `notebooks`.



PIR for [Asiga Max X27, 27 um pixel pitch](https://www.asiga.com/max-x/) ([Asiga Ultra](https://www.asiga.com/ultra/) is 32 um pixel pitch):

- DMD: [DLP651NE 0.65-Inch 1080p Digital Micromirror Device](https://www.ti.com/lit/ds/symlink/dlp651ne.pdf)
- [ChatGPT Irradiance Distribution Analysis](https://chatgpt.com/share/691e574e-ed30-800e-9d1d-997b1b67ae19)



Develop PIR for HR3.3u

- Jupyter notebook: `/Users/nordin/Documents/Projects/photopolymerization/development/2024-07-09_v0.2/2024-07-17_pixel_profile_from_images.ipynb`
- PIR image data: `/Users/nordin/Documents/Projects/photopolymerization/development/2024-07-09_v0.2/PIR_3_15_21-LED_powersetting_150`

