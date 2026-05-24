"""Module containing the KerrImage class for computing the image of a Kerr black hole as seen by a distant observer, using the escape coordinates of light rays"""
import numpy as np
from .light import DistantLightOrbit
from PIL import Image
from tqdm import tqdm
import subprocess

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
    shell_radius : double, optional
        radius of the shell used for generating image of distorted background, defaults to 50 in c = G = M = 1 units
    
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
    shell_radius : double
        radius of the shell used for generating image of distorted background
    shell_intersection_coordinates : np.ndarray
        array of shape (height, width, 2) containing the shell intersection coordinates coordinates (theta, phi) for each pixel in the image; if a pixel does not escape, the coordinates are (nan, nan)
    computed : bool
        whether the image has been computed or not
    """
    
    def __init__(self, a, theta, size, max_bardeen, shell_radius=50, M = None):
        self.a = a
        self.theta = theta
        self.size = size
        self.max_bardeen = max_bardeen
        self.shell_intersection_coordinates = np.empty((size[1], size[0], 2)) # (\theta, \phi) for each pixels
        self.M = M
        self.shell_radius = shell_radius
        self.computed = False

    def compute(self):
        """Computes uvs for each pixel in the image."""
        self.shell_intersection_coordinates.fill(np.nan)
        x_lim = self.size[0] // 2
        y_lim = self.size[1] // 2
        with tqdm(total=self.size[0] * self.size[1], ncols=80) as pbar:
            for x in range(-x_lim, x_lim):
                for y in range(-y_lim, y_lim):
                    pbar.update(1)
                    # minus because images have y axis downwards but beta goes upwards
                    orbit = DistantLightOrbit(self.a, self.theta, 0, x / x_lim * self.max_bardeen, -y / y_lim * self.max_bardeen * y_lim / x_lim, self.shell_radius, self.M)
                    if not orbit.escapes: continue

                    orbit.trajectory()

                    theta, phi = orbit.shell_intersection_coordinates[1:]
                    if np.isfinite(theta) and np.isfinite(phi):
                        self.shell_intersection_coordinates[y + y_lim, x + x_lim] = (theta, phi % (2 * np.pi))
        self.computed = True

    def image(self, uv_offset=(0, 0), bg=None):
        r"""Generates the image from the computed uvs.

        Parameters
        ----------
        angle : float, optional
            field of view in radians, defaults to :math:`2\pi`
        uv_offset : tuple(float, float), optional
            offset to apply to the uvs, defaults to (0, 0)
        bg : PIL.Image, optional
            background image to use for the pixels that escape. If None, the uvs will be used to determine the color of the pixels
        
        Returns
        -------
        PIL.Image
            the generated image
        """
        pixels = np.zeros((self.size[1], self.size[0], 3), dtype=np.uint8)

        theta = self.shell_intersection_coordinates[..., 0]
        phi = self.shell_intersection_coordinates[..., 1]

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
        y =                 st * sp
        z = -c0 * st * cp           + s0 * ct

        phi_obs = np.atan2(y, x) % (2 * np.pi)
        theta_obs = np.acos(z)

        u = (1 - phi_obs / (2 * np.pi)) + uv_offset[0]
        v = theta_obs / np.pi + uv_offset[1]
        u %= 1
        v %= 1

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
    
    def orbit(self, output, length, fps=30, direction=np.array([1, 0]), initial_uv_offset=np.array([0, 0]), portion=1, bg=None):
        r"""Animated the background by generating a sequence of images with the background rotated by a certain angle.

        Parameters
        ----------
        output : str
            output file name, should end with .mp4
        length : int
            length of the animation in seconds
        fps : int, optional
            frames per second, defaults to 30
        direction : tuple(float, float), optional
            direction of the animation in the uv space, defaults to (1, 0)
        initial_uv_offset : tuple(float, float), optional
            initial offset to apply to the uvs, defaults to (0, 0)
        portion : double, optional
            portion of the full rotation to animate, defaults to 1 (full rotation)
        bg : PIL.Image, optional
            background image to use for the pixels that escape. If None, the uvs will be used to determine the color of the pixels
        """
        if not self.computed:
            print("Computing image")
            self.compute()

        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        n_frames = int(length * fps)
        w, h = self.size

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            output
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        with tqdm(total=n_frames, ncols=80) as pbar:
            for i in range(n_frames):
                pbar.update(1)
                offset = initial_uv_offset + direction * (i / n_frames) * portion
                img = self.image(offset, bg)
                img = img.convert("RGB")
                frame = np.array(img, dtype=np.uint8)
                proc.stdin.write(frame.tobytes())

        proc.stdin.close()
        proc.wait()
