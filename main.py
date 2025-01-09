from scipy.ndimage import rotate
from colorama import Fore
import cv2
import numpy as np

def correctSkew(image, delta=1, limit=50):
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)  # Updated usage
        histogram = np.sum(data, axis=1, dtype=float)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # cv2.imshow(f'{image}-THRESH', thresh)

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)
    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected


def isImageFound(skewedImg, errMessage):
    if skewedImg is None:
        print(errMessage)
        return False

    return True


def printLine(message, color):
    print(color + message)


def testSkewedImages(path, steps):
    for i in steps:
        skewedImgPath = f"{path}/terc-{i}.png"
        skewedImg = cv2.imread(skewedImgPath)

        if not isImageFound(skewedImg, Fore.RED + f"\rError: Image not found at {skewedImgPath}..."):
            continue

        print(Fore.CYAN + f"\rCorrecting [terc-{i}.png]...", end="")

        skewedAng, skewedCorr = correctSkew(skewedImg)
        print(Fore.GREEN + f"\rCorrected  [terc-{i}.png] -> angle: {skewedAng}")

        # Resize both images to half their original size
        original_half = cv2.resize(skewedImg, (0, 0), fx=0.5, fy=0.5)
        corrected_half = cv2.resize(skewedCorr, (0, 0), fx=0.5, fy=0.5)

        # Create a combined image to show both before and after
        combined_image = cv2.hconcat([original_half, corrected_half])

        # Add text to each image
        cv2.putText(original_half, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(corrected_half, "Corrected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the combined image
        cv2.imshow(f'Before and after - terc-{i}.png', combined_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_coordinate_system(image):
    """
    Vykreslí súradnicovú sústavu na obrázok, pričom osi prechádzajú celým obrázkom.

    :param image: Vstupný obrázok, na ktorý sa má vykresliť súradnicová sústava
    :return: Obrázok s vykreslenou súradnicovou sústavou
    """
    # Prekopírovanie pôvodného obrázka, aby sme ho neprepisovali
    img = cv2.imread(image)

    # Získanie rozmerov obrázka (šírka, výška)
    height, width = img.shape[:2]

    # Stred obrázka
    origin = (width // 2, height // 2)

    # Nastavenie farby pre osy a text (čierna farba)
    axis_color = (240, 240, 240)  # Čierna
    text_color = (0, 0, 255)  # Červená

    # Vykreslenie osí X a Y (prejdú celým obrázkom)
    cv2.line(img, (0, origin[1]), (width, origin[1]), axis_color, 2)  # Osa X cez celý obrázok
    cv2.line(img, (origin[0], 0), (origin[0], height), axis_color, 2)  # Osa Y cez celý obrázok

    # Osa X: vykresliť značky (súradnice) pozdĺž celej osi
    for i in range(0, width, 50):  # Značky na osi X
        cv2.line(img, (i, origin[1] - 5), (i, origin[1] + 5), axis_color, 1)
        if i != origin[0]:
            cv2.putText(img, str(i - origin[0]), (i - 10, origin[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Osa Y: vykresliť značky (súradnice) pozdĺž celej osi
    for i in range(0, height, 50):  # Značky na osi Y
        cv2.line(img, (origin[0] - 5, i), (origin[0] + 5, i), axis_color, 1)
        if i != origin[1]:
            cv2.putText(img, str(origin[1] - i), (origin[0] + 10, i + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    return img


def readImage(imagePath):
    """
    The function takes an path of stored image as an argument and returns opened image

    imagePath - path to locally stored image
    """
    img = cv2.imread(imagePath)

    if img is None:
        raise FileNotFoundError(f"Image at path '{imagePath}' could not be opened.")

    return img


def getImageSize(img):
    """
    The function takes an image as an argument and returns its height and width

    img - image
    """
    return img.shape


def resizeImage(img):
    """
    The function takes an image as an argument and returns resized image that is at least 1000px wide and high, while
    maintaining the aspect ratio

    img - image to be resized
    """
    height, width = img.shape[:2]

    if width < 1000 or height < 1000:
        scale_factor_width = 1000 / width
        scale_factor_height = 1000 / height
        scale_factor = max(scale_factor_width, scale_factor_height)
    elif width > 1500 or height > 1500:
        scale_factor_width = 1500 / width
        scale_factor_height = 1500 / height
        scale_factor = min(scale_factor_width, scale_factor_height)
    else:
        return img

    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    print(new_height, new_width)

    return cv2.resize(img, (new_width, new_height))


def showImage(title, img):
    """
    The function takes an image and title as arguments and displays the image with its title, then waits for key to
    be pressed

    :param title: displayed title in window
    :param img: displayed image
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)


def convertToGrayScale(img):
    """
    The function takes an image as an argument and returns converted image in gray color

    img - image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blurImage(img):
    """
    The function takes an image as an argument and returns blurred image

    img - image
    """
    return cv2.GaussianBlur(img, (5, 5), 1)


def findImageEdges(img):
    """
    Detects edges in an image using the Canny edge detection algorithm.

    img - image
    """
    return cv2.Canny(img, 50, 50)


def makeCountours(img, originalImg):
    """
    Finds contours in and image and draws them

    img - image
    """
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contourFrame = originalImg.copy()
    contourFrame = cv2.drawContours(contourFrame, contours, -1, (255, 0, 0), 4)
    cornerFrame = originalImg.copy()

    maxArea = 0
    biggest = []

    for i in contours:
        area = cv2.contourArea(i)
        if area > 500:
            peri = cv2.arcLength(i, True)
            edges = cv2.approxPolyDP(i, 0.02*peri, True)
            if area > maxArea:
                biggest = edges
                maxArea = area

    print(biggest[0])

    if len(biggest) != 0:
        # drawRec(biggest, cornerFrame)
        cornerFrame = cv2.drawContours(cornerFrame, biggest, -1, (255, 0, 255), 5)

    return cornerFrame

def func(imgPath):
    image = readImage(imgPath)
    image = resizeImage(image)
    #showImage("resized", image)

    grayImage = convertToGrayScale(image)
    #showImage("gray", grayImage)

    bluredImage = blurImage(grayImage)
    #showImage("blur", bluredImage)

    cannyImage = findImageEdges(bluredImage)
    #showImage("canny", cannyImage)

    contourFrame = makeCountours(cannyImage, image)
    showImage("contourFrame", contourFrame)


def drawRec(biggestNew, mainFrame):
    cv2.line(mainFrame, (biggestNew[0][0][0], biggestNew[0][0][1]), (biggestNew[1][0][0], biggestNew[1][0][1]),
        (0, 255, 0), 20)
    cv2.line(mainFrame, (biggestNew[0][0][0], biggestNew[0][0][1]), (biggestNew[2][0][0], biggestNew[2][0][1]),
        (0, 255, 0), 20)
    cv2.line(mainFrame, (biggestNew[3][0][0], biggestNew[3][0][1]), (biggestNew[2][0][0], biggestNew[2][0][1]),
        (0, 255, 0), 20)
    cv2.line(mainFrame, (biggestNew[3][0][0], biggestNew[3][0][1]), (biggestNew[1][0][0], biggestNew[1][0][1]),
        (0, 255, 0), 20)


if __name__ == '__main__':
    # detectAndHighlightCircles("../../images/circles.png")
    # testSkewedImages("../../images/", [0, 5, 10, 15, 20])
    # testSkewedImages("../../images/", [35])
    # testSkewedImages("../../images/", [50])

    #imagePath = "../images/t2.jpg"
    imagePath = "images/targets/target_3.jpg"
    func(imagePath)
    # result_image = draw_coordinate_system(image)
    # cv2.imshow("Coordinate System", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


print("hello")