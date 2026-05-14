import numpy as np
from .light import DistantLightOrbit
from PIL import Image
from tqdm import tqdm

class KerrImage:
    """Class used to compute the image of a Kerr black hole as seen by a distant observer, using the escape coordinates of light rays.

    Parameters
    ----------
    a : float
        spin parameter of the black hole
    theta : float
        inclination angle of the observer in radians
    size : tuple(int, int)
        size of the image in pixels
    max_bardeen : float
        maximum Bardeen coordinate to consider for the image determining the field of view
    M : float, optional
        mass of the black hole. If not specified, units are in terms of M
    """
    
    def __init__(self, a, theta, size, max_bardeen, M = None):
        self.a = a
        self.theta = theta
        self.size = size
        self.max_bardeen = max_bardeen
        self.uvs = np.zeros((size[1], size[0], 2)) # normalized (x, y)
        self.M = M

    def compute(self):
        """Computes uvs for each pixel in the image."""
        self.uvs.fill(-1)
        x_lim = self.size[0] // 2
        y_lim = self.size[1] // 2
        with tqdm(total=self.size[0] * self.size[1], ncols=80) as pbar:
            for x in range(-x_lim, x_lim):
                for y in range(-y_lim, y_lim):
                    pbar.update(1)
                    orbit = DistantLightOrbit(self.a, self.theta, 0, x / x_lim * self.max_bardeen, y / y_lim * self.max_bardeen * y_lim / x_lim, self.M)
                    orbit.trajectory()

                    theta, phi = orbit.escape_coordinates
                    if orbit.escapes:
                        self.uvs[y + y_lim, x + x_lim] = ((phi % (2 * np.pi)) / (2 * np.pi), theta / np.pi)

    def image(self, bg = None):
        """Generates the image from the computed uvs.

        Parameters
        ----------
        bg : PIL.Image, optional
            background image to use for the pixels that escape. If None, the uvs will be used to determine the color of the pixels
        
        Returns
        -------
        PIL.Image
            the generated image
        """
        pixels = np.zeros((*self.size, 3), dtype=np.uint8)

        mask = (self.uvs[..., 0] == -1) & (self.uvs[..., 1] == -1)

        if bg is None:
            pixels[~mask, 0] = (self.uvs[~mask, 0] * 255).astype(np.uint8)
            pixels[~mask, 1] = (self.uvs[~mask, 1] * 255).astype(np.uint8)
        else:
            w, h = bg.size
            bg_pixels = np.array(bg)

            valid_u = self.uvs[~mask, 0]
            valid_v = self.uvs[~mask, 1]
            finite = np.isfinite(valid_u) & np.isfinite(valid_v)

            valid_u = valid_u[finite]
            valid_v = valid_v[finite]

            x = (valid_u * (w - 1)).astype(int)
            y = (valid_v * (h - 1)).astype(int)

            iy, ix = np.where(~mask)
            ix = ix[finite]
            iy = iy[finite]

            pixels[iy, ix] = bg_pixels[y, x]

        return Image.fromarray(pixels)
                