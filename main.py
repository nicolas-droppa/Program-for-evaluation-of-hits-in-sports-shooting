from scipy.ndimage import rotate
from colorama import Fore
import cv2
import numpy as np
from utils.imageOperations import readImage, resizeImage
from utils.imageEffects import blurImage, medianBlurImage, convertToGrayScale
from utils.constants import TARGET_MARGIN, MANUAL_SELECTION, DEV_MODE


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


def target_roi_mask_corner_detection(image, images):
    print("ROI_MASK")
    lower_color = np.array([20, 20, 150])
    upper_color = np.array([35, 100, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    images.append(hsv)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    images.append(mask)
    mask = medianBlurImage(mask)
    mask = blurImage(mask)
    mask = cv2.blur(mask,(5,5))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        # 6. If the approximation is a quadrilateral, use it as the paper corners
        for point in approx:
            x, y = point.ravel()
            if (DEV_MODE):
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

        corners = approx.reshape(4, 2)
        return image, corners, True, images
    
    return False, [], False, images

def target_roi_binary_corner_detection(image, images):
    print("ROI_BINARY")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images.append(gray)
    bluredImage = medianBlurImage(gray)
    bluredImage = blurImage(bluredImage)
    bluredImage = cv2.blur(bluredImage,(5,5))
    images.append(bluredImage)
    _, binary = cv2.threshold(bluredImage, 150, 255, cv2.THRESH_BINARY)
    binary = cv2.blur(binary,(5,5))
    images.append(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found!")
        return image, [], False, images

    # 4. Select the largest contour (assumes paper is the largest object)
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Approximate contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        # 6. If the approximation is a quadrilateral, use it as the paper corners
        for point in approx:
            x, y = point.ravel()
            if (DEV_MODE):
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

        corners = approx.reshape(4, 2)
        return image, corners, True, images

    else:
        print(f"Could not find 4 corners, found {len(approx)} points instead!")
        x, y, w, h = cv2.boundingRect(largest_contour)
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype="float32")
        
        for corner in corners:
            cx, cy = corner
            if (DEV_MODE):
                cv2.circle(image, (int(cx), int(cy)), 10, (0, 0, 255), -1)

        if corners is not None:
            print("Detected corners:")
            print(corners)
            return image, corners, True, images
        else:
            print("Corners could not be detected.")


def find_paper_corners(image):
    images = []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([20, 20, 150])
    upper_color = np.array([35, 100, 255])

    mask = cv2.inRange(hsv, lower_color, upper_color)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Terč nebol detegovaný!")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    target_roi = image[y-TARGET_MARGIN:y+h+TARGET_MARGIN, x-TARGET_MARGIN:x+w+TARGET_MARGIN]

    image, corners, foundCorners, images = target_roi_mask_corner_detection(target_roi, images)
    if foundCorners == False:
        image, corners, foundCorners, images = target_roi_binary_corner_detection(target_roi, images)

    images.append(image)
    showImagesGrid(images, grid_shape=(3, 3), cell_size=(600, 600))
    return corners, image


points = np.zeros((4, 2), np.int16)
counter = 0

def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        points[counter] = x, y
        counter = counter + 1
        print(points)


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect


def calculate_width_height(corners):
    """
    Calculate the width and height of a quadrilateral given its four corner coordinates.

    :param corners: A 4x2 numpy array of (x, y) coordinates.
    :return: A tuple (width, height) representing the dimensions of the quadrilateral.
    """
    corners = np.array(corners)

    width1 = np.linalg.norm(corners[0] - corners[1])
    width2 = np.linalg.norm(corners[2] - corners[3])
    height1 = np.linalg.norm(corners[0] - corners[2])
    height2 = np.linalg.norm(corners[1] - corners[3])
    width = (width1 + width2) / 2
    height = (height1 + height2) / 2

    return int(width), int(height)


def manualSelection(image):
    while True:
        if counter == 4:
            corners = order_points(np.array(points))
            width, height = calculate_width_height(corners)
            pts1 = np.float32([corners[0], corners[1], corners[2], corners[3]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warpedImage = cv2.warpPerspective(image, matrix, (width, height))
            cv2.imshow("warpedTarget", warpedImage)
            break

        for i in range(0, 4):
            cv2.circle(image, (points[i][0], points[i][1]), 15, (0, 255, 0), cv2.FILLED)

        cv2.imshow("originalImage", image)
        cv2.setMouseCallback("originalImage", mousePoints)
        cv2.waitKey(1)


if __name__ == '__main__':
    imagePath = "images/targets/target_11.jpg"
    image = readImage(imagePath)
    image = resizeImage(image)
    originalImage = image

    if (MANUAL_SELECTION):
        manualSelection(image)
    
    else:
        corners, image = find_paper_corners(image)
        corners = order_points(np.array(corners))
        print(corners)
        width, height = calculate_width_height(corners)
        pts1 = np.float32([corners[0], corners[1], corners[2], corners[3]])
        pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warpedImage = cv2.warpPerspective(image, matrix, (width, height))
        cv2.imshow("warpedTarget", warpedImage)
        cv2.setMouseCallback("warpedTarget", mousePoints)
    
    cv2.waitKey(0)