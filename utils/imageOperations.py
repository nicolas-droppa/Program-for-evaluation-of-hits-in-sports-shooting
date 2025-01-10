import cv2
from screeninfo import get_monitors
from .constants import SCREEN_RESIZE_FACTOR

def readImage(imagePath):
    """
    Opens the image from path.

    Parameters:
    -----------
    imagePath : string
        Path to image

    Returns:
    --------
    numpy.ndarray
        The openned image
    """
    img = cv2.imread(imagePath)

    if img is None:
        raise FileNotFoundError(f"Image at path '{imagePath}' could not be opened.")

    return img


def resizeImage(img):
    """
    Resizes an image to fit within the screen dimensions, keeping the aspect ratio intact.

    Parameters:
    -----------
    img : numpy.ndarray
        The image to be resized.

    Returns:
    --------
    numpy.ndarray
        The resized image.
    """
    screen = get_monitors()[0]
    screen_width = screen.width
    screen_height = screen.height

    max_width = screen_width * SCREEN_RESIZE_FACTOR
    max_height = screen_height * SCREEN_RESIZE_FACTOR

    height, width = img.shape[:2]

    scale_factor_width = max_width / width
    scale_factor_height = max_height / height

    scale_factor = min(scale_factor_width, scale_factor_height)

    if scale_factor >= 1:
        return img

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    return cv2.resize(img, (new_width, new_height))