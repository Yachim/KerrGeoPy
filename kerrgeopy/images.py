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
    
    Attributes
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
    escape_coordinates : np.ndarray
        array of shape (height, width, 2) containing the escape coordinates (theta, phi) for each pixel in the image; if a pixel does not escape, the coordinates are (nan, nan)
    """
    
    def __init__(self, a, theta, size, max_bardeen, M = None):
        self.a = a
        self.theta = theta
        self.size = size
        self.max_bardeen = max_bardeen
        self.escape_coordinates = np.zeros((size[1], size[0], 2)) # (\theta, \phi) for each pixels
        self.M = M

    def compute(self):
        """Computes uvs for each pixel in the image."""
        self.escape_coordinates.fill(np.nan)
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
                    if orbit.escapes and np.isfinite(theta) and np.isfinite(phi):
                        self.escape_coordinates[y + y_lim, x + x_lim] = (theta, phi % (2 * np.pi))

    def image(self, bg=None):
        r"""Generates the image from the computed uvs.

        Parameters
        ----------
        angle : float, optional
            field of view in radians, defaults to :math:`2\pi`
        bg : PIL.Image, optional
            background image to use for the pixels that escape. If None, the uvs will be used to determine the color of the pixels
        
        Returns
        -------
        PIL.Image
            the generated image
        """
        pixels = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        theta = self.escape_coordinates[..., 0]
        phi = self.escape_coordinates[..., 1]

        mask_finite = np.isfinite(theta) & np.isfinite(phi)

        theta = theta[mask_finite]
        phi = phi[mask_finite]

        s0 = np.sin(self.theta)
        c0 = np.cos(self.theta)
        st = np.sin(theta)
        ct = np.cos(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        x =  s0 * st * cp           + c0 * ct
        y =               - st * sp
        z = -c0 * st * cp           + s0 * ct

        phi_obs = np.atan2(y, x) % (2 * np.pi)
        theta_obs = np.acos(z)

        u = phi_obs / (2 * np.pi)
        v = theta_obs / np.pi

        if bg is None:
            pixels[mask_finite, 0] = (u * 255).astype(np.uint8)
            pixels[mask_finite, 1] = (v * 255).astype(np.uint8)
        else:
            w, h = bg.size
            bg_pixels = np.array(bg)

            x = (u * (w - 1)).astype(int)
            y = (v * (h - 1)).astype(int)

            pixels[mask_finite] = bg_pixels[y, x]

        return Image.fromarray(pixels)
                