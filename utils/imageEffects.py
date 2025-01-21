import cv2


def convertToGrayScale(image):
    """
    Converts image to gray scale

    Parameters:
    -----------
    image : numpy.ndarray
        The openned image

    Returns:
    --------
    numpy.ndarray
        The gray scaled image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def blurImage(image):
    """
    Applies Gaussian blur to an image

    Parameters:
    -----------
    image : numpy.ndarray
        The openned image

    Returns:
    --------
    numpy.ndarray
        The image with applied Gaussian blur
    """
    return cv2.GaussianBlur(image, (5, 5), 1)


def medianBlurImage(image):
    """
    Applies median blur to an image

    Parameters:
    -----------
    image : numpy.ndarray
        The openned image

    Returns:
    --------
    numpy.ndarray
        The image with applied Gaussian blur
    """
    return cv2.medianBlur(image,5)