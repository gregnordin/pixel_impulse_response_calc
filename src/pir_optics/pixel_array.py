import numpy as np
import matplotlib.pyplot as plt

class PixelArrayModel:
    def __init__(self,
                 PIR=None,
                 n_pixels_x=1,
                 n_pixels_y=1,
                 use_image_pixel_pitch=True,
                 pixel_separation_x=None,
                 pixel_separation_y=None,
                 nx=1024,
                 dx=0.25):
        
        if PIR is None:
            raise ValueError("PIR must be provided")
        self.PIR = PIR

        # pixel array parameters
        self.n_pixels_x = n_pixels_x
        self.n_pixels_y = n_pixels_y
        if use_image_pixel_pitch:
            if pixel_separation_x is not None or pixel_separation_y is not None:
                raise ValueError(
                    "pixel_separation_x and pixel_separation_y must be None when use_image_pixel_pitch=True"
                    )
            self.pixel_separation_x = self.PIR.img_pixel_pitch
            self.pixel_separation_y = self.PIR.img_pixel_pitch
        else:
            if pixel_separation_x is None or pixel_separation_y is None: 
                raise ValueError(
                    "pixel_separation_x and pixel_separation_y must be an integer when use_image_pixel_pitch=False"
                )
            if pixel_separation_x <= 0:
                raise ValueError(f"pixel_separation_x must be > 0 but is {pixel_separation_x}")
            if pixel_separation_y <= 0:
                raise ValueError(f"pixel_separation_y must be > 0 but is {pixel_separation_y}")
            self.pixel_separation_x = self.pixel_separation_x
            self.pixel_separation_y = self.pixel_separation_y
        self.pixel_centers_x, self.pixel_centers_y = self._calc_pixel_centers()
        
        # grid parameters
        self.nx = int(nx)
        self.ny = int(nx)   # square grid
        self.dx = float(dx)
        self.dy = float(dx)
        self.grid_size_x = (self.nx - 1) * self.dx
        self.grid_size_y = (self.ny - 1) * self.dy

        # placeholders for computed data
        self.x = None
        self.y = None
        self.X = None
        self.Y = None
        self.I = None

        self._compute()

    # ---------- internal helpers ----------
    def _compute(self):
        self._build_grid()
        self._fill_grid()
        
    def _calc_pixel_centers(self):
        """
        Create N×M grid of equally spaced points centered at (0,0) where
        N is self.n_pixels_x and M is self.n_pixels_y.
        
        Returns
        -------
        x : 1D array of shape (N,) x-coordinates
        y : 1D array of shape (M,) y-coordinates
    """
        x = (np.arange(self.n_pixels_x) - (self.n_pixels_x - 1) / 2) * self.pixel_separation_x
        y = (np.arange(self.n_pixels_y) - (self.n_pixels_y - 1) / 2) * self.pixel_separation_y
        return x, y
    
    def _build_grid(self):
        self.x = (np.arange(self.nx) - self.nx // 2) * self.dx
        self.y = (np.arange(self.ny) - self.ny // 2) * self.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

    def _fill_grid(self):
        ...