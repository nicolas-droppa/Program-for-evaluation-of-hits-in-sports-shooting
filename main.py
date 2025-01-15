from scipy.ndimage import rotate
from colorama import Fore
import cv2
import numpy as np
from utils.imageOperations import readImage, resizeImage


def showImage(title, img):
    """
    The function takes an image and title as arguments and displays the image with its title, then waits for key to
    be pressed

    :param title: displayed title in window
    :param img: displayed image
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)


def showImagesGrid(images, window_title="Image Grid", grid_shape=None, cell_size=(200, 200), bg_color=(255, 255, 255)):
    """
    Displays multiple images in a grid layout on a single canvas, preserving their aspect ratio.

    :param images: List of images to display
    :param window_title: Title of the window displaying the canvas
    :param grid_shape: Tuple (rows, cols) specifying the grid layout. If None, the layout is automatically calculated.
    :param cell_size: Tuple (width, height) specifying the size of each image cell
    :param bg_color: Tuple (B, G, R) specifying the background color of the canvas
    """
    num_images = len(images)

    # Automatically determine grid shape if not provided
    if grid_shape is None:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_shape

    # Determine canvas size
    canvas_height = rows * cell_size[1]
    canvas_width = cols * cell_size[0]
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = bg_color  # Fill the background

    # Place each image in the grid
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x_start, y_start = col * cell_size[0], row * cell_size[1]
        x_end, y_end = x_start + cell_size[0], y_start + cell_size[1]

        # Ensure the image has 3 channels (convert grayscale to BGR if needed)
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Calculate the aspect ratio
        h, w = img.shape[:2]
        aspect_ratio = w / h

        # Resize the image while preserving its aspect ratio
        if aspect_ratio > 1:  # Wider than tall
            new_w = cell_size[0]
            new_h = int(cell_size[0] / aspect_ratio)
        else:  # Taller than wide
            new_h = cell_size[1]
            new_w = int(cell_size[1] * aspect_ratio)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center the resized image within the cell
        x_offset = x_start + (cell_size[0] - new_w) // 2
        y_offset = y_start + (cell_size[1] - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    # Display the canvas
    cv2.imshow(window_title, canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

def medianBlurImage(img):
    """
    The function takes an image as an argument and returns blurred image

    img - image
    """
    return cv2.medianBlur(img,5)


def findImageEdges(img):
    """
    Detects edges in an image using the Canny edge detection algorithm.

    img - image
    """
    return cv2.Canny(img, 100, 200)


def makeCountoursBEST(img, originalImg):
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

    #print(biggest)

    if len(biggest) != 0:
        # drawRec(biggest, cornerFrame)
        cornerFrame = cv2.drawContours(cornerFrame, biggest, -1, (255, 0, 255), 5)

    return cornerFrame

def makeContours(img, originalImg):
    """
    Finds contours in an image and draws the largest quadrilateral (e.g., paper corners).

    Parameters:
    -----------
    img : numpy.ndarray
        Preprocessed image (e.g., thresholded or edge-detected) for contour detection.
    originalImg : numpy.ndarray
        Original image on which contours will be drawn.

    Returns:
    --------
    numpy.ndarray
        Image with the largest quadrilateral's corners highlighted.
    """
    # Find contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contourFrame = originalImg.copy()
    cv2.drawContours(contourFrame, contours, -1, (255, 0, 0), 2)  # Optional: visualize all contours
    
    maxArea = 0
    biggest = []

    # Iterate through contours to find the largest quadrilateral
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ignore small areas (e.g., noise)
            peri = cv2.arcLength(contour, True)  # Calculate the perimeter
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # Approximate the polygon
            if len(approx) == 4 and area > maxArea:  # Check if it's a quadrilateral
                biggest = approx  # Save the largest quadrilateral
                maxArea = area

    if len(biggest) == 4:
        # Draw the largest quadrilateral
        cornerFrame = cv2.drawContours(originalImg.copy(), [biggest], -1, (255, 0, 255), 5)
    else:
        # If no quadrilateral is found, return the original image
        print("NO QUADRITERAL!")
        cornerFrame = originalImg.copy()

    return cornerFrame

def makeContoursB(img, originalImg):
    """
    Finds contours in an image, sorts them by area (largest first), and draws the largest contour.

    img - image (binary or edge-detected image)
    originalImg - original image to draw contours on
    """
    # Find contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contourFrame = originalImg.copy()
    contourFrame = cv2.drawContours(contourFrame, contours, -1, (255, 0, 0), 4)  # Draw all contours for debugging

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    maxArea = 0
    biggest = []

    # Iterate through sorted contours
    for i in contours:
        area = cv2.contourArea(i)
        if area > 500:  # Threshold for filtering small contours
            peri = cv2.arcLength(i, True)
            edges = cv2.approxPolyDP(i, 0.02 * peri, True)  # Approximate the polygonal curve
            if area > maxArea:  # Update the largest contour if necessary
                biggest = edges
                maxArea = area

    if len(biggest) != 0:
        # Draw the largest contour
        cornerFrame = cv2.drawContours(originalImg.copy(), [biggest], -1, (255, 0, 255), 5)
    else:
        cornerFrame = originalImg.copy()

    return cornerFrame

def func(imgPath):
    image = readImage(imgPath)
    image = resizeImage(image)
    #showImage("resized", image)

    grayImage = convertToGrayScale(image)
    #showImage("gray", grayImage)

    bluredImage = medianBlurImage(grayImage)
    bluredImage = blurImage(bluredImage)
    bluredImage = cv2.blur(bluredImage,(5,5))
    #showImage("blur", bluredImage)

    cannyImage = findImageEdges(bluredImage)
    showImage("canny", cannyImage)

    contourFrame = makeContoursB(cannyImage, image)
    showImage("contourFrame", contourFrame)


def detect_target_with_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images = []

    # 2. Definovanie rozsahu farby papiera (napr. biela alebo svetlá farba)
    lower_color = np.array([20, 20, 150])  # Dolná hranica (svetlo žltá)
    upper_color = np.array([35, 100, 255])  # Horná hranica (tmavšia žltá)

    # 3. Vytvorenie masky pre farbu
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(image, image, mask=mask)

    # 4. Nájdenie kontúr na maske
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Terč nebol detegovaný!")
        return

    # 5. Výber najväčšieho kontúru (predpokladáme, že je to terč)
    largest_contour = max(contours, key=cv2.contourArea)

    # 6. Získanie ohraničujúceho obdĺžnika (bounding box) terča
    x, y, w, h = cv2.boundingRect(largest_contour)
    target_roi = image[y-150:y+h+150, x-150:x+w+150]
    print("x:" + str(x) + "\ny:" + str(y) + "\nw:" + str(w) + "\nh:" + str(h))

    # 7. Detekcia rohov v oblasti terča
    """
    gray_target = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_target, maxCorners=4, qualityLevel=0.01, minDistance=30)

    if corners is not None:
        corners = np.int8(corners)
        for corner in corners:
            cx, cy = corner.ravel()
            cv2.circle(target_roi, (cx, cy), 10, (0, 255, 0), -1)
    """
    # 8. Zobrazenie výsledkov
    #cv2.imshow("Original Image", image)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Detected Target", target_roi)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
    images.append(image)
    images.append(mask)

    bluredImage = blurImage(result)
    #showImage("blur", bluredImage)

    cannyImage = findImageEdges(bluredImage)
    #showImage("canny", cannyImage)
    images.append(cannyImage)
    images.append(target_roi)
    """
    lines = cv2.HoughLinesP(cannyImage, 1, np.pi/180, 200, minLineLength=200, maxLineGap=200)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2,y2), (0,255,0), 3)
    """
    # Show results
    h = cv2.cvtColor(target_roi, cv2.COLOR_BGR2HSV)
    ma = cv2.inRange(h, lower_color, upper_color)
    r = cv2.bitwise_and(target_roi, target_roi, mask=ma)
    bImage = medianBlurImage(r)
    bImage = cv2.blur(bImage,(5,5))
    cImage = findImageEdges(bImage)
    cFrame = makeCountoursBEST(cImage, target_roi)
    #showImage("contourFrame-T", cFrame)

    #contourFrame = makeCountours(cannyImage, image)
    #showImage("contourFrame", contourFrame)
    contourFrame = makeCountoursBEST(cannyImage, image)
    #showImage("contourFrame", contourFrame)
    showImagesGrid(images, grid_shape=(3, 3), cell_size=(600, 600))
    tara, _ = find_paper_corners(target_roi)
    showImage("a", tara)


def target_roi_mask_corner_detection(image):
    lower_color = np.array([20, 20, 150])
    upper_color = np.array([35, 100, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        # 6. If the approximation is a quadrilateral, use it as the paper corners
        for point in approx:
            x, y = point.ravel()
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

        corners = approx.reshape(4, 2)
        return image, corners
    
    return False, None

def target_roi_binary_corner_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found!")
        return image, None

    # 4. Select the largest contour (assumes paper is the largest object)
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        # 6. If the approximation is a quadrilateral, use it as the paper corners
        for point in approx:
            x, y = point.ravel()
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

        corners = approx.reshape(4, 2)
        return image, corners

    else:
        print(f"Could not find 4 corners, found {len(approx)} points instead!")
        # Fallback: Use bounding rectangle as approximation
        x, y, w, h = cv2.boundingRect(largest_contour)
        corners = np.array([
            [x, y],           # Top-left
            [x + w, y],       # Top-right
            [x + w, y + h],   # Bottom-right
            [x, y + h]        # Bottom-left
        ], dtype="float32")
        
        # Draw the rectangle as fallback
        for corner in corners:
            cx, cy = corner
            cv2.circle(image, (int(cx), int(cy)), 10, (0, 0, 255), -1)

        result_image, corners = image, corners

        if corners is not None:
            print("Detected corners:")
            print(corners)

            # Display the result
            cv2.imshow("Detected Paper Corners", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Corners could not be detected.")


def find_paper_corners(target_roi):
    image, corners = target_roi_mask_corner_detection(target_roi)
    if image == False:
        image, corners = target_roi_binary_corner_detection(target_roi)

    showImage(image)

def overlayGrid(img, step=20, line_color=(128, 128, 150), thickness=2, darker_line_color=(0, 0, 255), font_scale=0.8, font_color=(0, 0, 255)):
    """
    Overlays a grid on the given image with every 5th line darker and adds numbers on top and left side.

    Parameters:
    -----------
    img : numpy.ndarray
        The image to overlay the grid on.
    step : int, optional
        The spacing between grid lines in pixels. Default is 10.
    line_color : tuple, optional
        The color of the grid lines in BGR format. Default is green (0, 255, 0).
    thickness : int, optional
        The thickness of the grid lines. Default is 1.
    darker_line_color : tuple, optional
        The color of the darker grid lines in BGR format. Default is dark green (0, 128, 0).
    font_scale : float, optional
        The scale of the font for displaying numbers. Default is 0.5.
    font_color : tuple, optional
        The color of the numbers in BGR format. Default is red (0, 0, 255).

    Returns:
    --------
    numpy.ndarray
        The image with the grid overlayed and numbers displayed.
    """
    # Make a copy of the image to avoid modifying the original
    overlayed_img = img.copy()
    height, width = overlayed_img.shape[:2]

    # Define the font for displaying numbers
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw vertical grid lines and display numbers on the left
    for x in range(0, width, step):
        color = darker_line_color if (x // step) % 5 == 0 else line_color
        cv2.line(overlayed_img, (x, 0), (x, height), color=color, thickness=thickness)
        # Display number on the left side of the image
        if x % (step * 5) == 0:  # Only display numbers every 5 steps for readability
            cv2.putText(overlayed_img, str(x), (x + 5, 20), font, font_scale, font_color, thickness=1)

    # Draw horizontal grid lines and display numbers on the top
    for y in range(0, height, step):
        color = darker_line_color if (y // step) % 5 == 0 else line_color
        cv2.line(overlayed_img, (0, y), (width, y), color=color, thickness=thickness)
        # Display number on the top side of the image
        if y % (step * 5) == 0:  # Only display numbers every 5 steps for readability
            cv2.putText(overlayed_img, str(y), (5, y + 20), font, font_scale, font_color, thickness=1)

    return overlayed_img


if __name__ == '__main__':
    # detectAndHighlightCircles("../../images/circles.png")
    # testSkewedImages("../../images/", [0, 5, 10, 15, 20])
    # testSkewedImages("../../images/", [35])
    # testSkewedImages("../../images/", [50])

    #imagePath = "../images/t2.jpg"
    imagePath = "images/targets/target_15.jpg"
    # func(imagePath)

    #goodCornerDetection(imagePath)

    
    #image = readImage(imagePath)
    #image = resizeImage(image)
    #showImage("har", image)
    #detectHarrisCorners(image)
    #showImage("har", image)
    
    #detect_target_with_color(imagePath)

    # result_image = draw_coordinate_system(image)
    # cv2.imshow("Coordinate System", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image = readImage(imagePath)
    image = resizeImage(image)
    gridImage = overlayGrid(image)
    #showImage("Grid image", gridImage)

    detect_target_with_color(image)


print("hello")