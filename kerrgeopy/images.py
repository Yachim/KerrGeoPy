"""Module containing the KerrImage class for computing the image of a Kerr black hole as seen by a distant observer, using the escape coordinates of light rays"""
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
        width and height of the image in pixels
    max_bardeen : float
        maximum Bardeen coordinate to consider for the image determining the horizontal field of view
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

    def compute(self, transform=True):
        r"""Computes uvs for each pixel in the image.
        
        Parameters
        ----------
        transform : bool, optional
            whether to apply the transformation to the uvs, defaults to True; the transformation makes v = 0.5 at :math:`\pi - \theta`
        """
        self.uvs.fill(np.nan)
        x_lim = self.size[0] // 2
        y_lim = self.size[1] // 2
        with tqdm(total=self.size[0] * self.size[1], ncols=80) as pbar:
            for x in range(-x_lim, x_lim):
                for y in range(-y_lim, y_lim):
                    pbar.update(1)
                    # minus because images have y axis downwards but beta goes upwards
                    orbit = DistantLightOrbit(self.a, self.theta, 0, x / x_lim * self.max_bardeen, -y / y_lim * self.max_bardeen * y_lim / x_lim, self.M)
                    orbit.trajectory()

                    theta, phi = orbit.escape_coordinates
                    if transform:
                        theta -= np.pi / 2 - self.theta
                        theta %= np.pi
                    if orbit.escapes and np.isfinite(theta) and np.isfinite(phi):
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
        pixels = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        mask = np.isfinite(self.uvs[..., 0]) & np.isfinite(self.uvs[..., 1])

        if bg is None:
            pixels[mask, 0] = (self.uvs[mask, 0] * 255).astype(np.uint8)
            pixels[mask, 1] = (self.uvs[mask, 1] * 255).astype(np.uint8)
        else:
            w, h = bg.size
            bg_pixels = np.array(bg)

            valid_u = self.uvs[mask, 0]
            valid_v = self.uvs[mask, 1]

            x = (valid_u * (w - 1)).astype(int)
            y = (valid_v * (h - 1)).astype(int)

            pixels[mask] = bg_pixels[y, x]

        return Image.fromarray(pixels)
                