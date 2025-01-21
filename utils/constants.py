"""
SCREEN_RESIZE_FACTOR

Description:
------------
Variable responsible for resizing window of an image.

Value:
------
<0; inf>
    value for image to be resized x % of window size (0.9 -> 90%),
"""
SCREEN_RESIZE_FACTOR = 0.9

"""
TARGET_MARGIN

Description:
------------
Variable to make sure that during extracting target from image, we take cauple of
pixels more just to be safe for next algorithm.

Value:
------
<0; inf>
    Number of pixels for extra safety for each side.
"""
TARGET_MARGIN = 150

"""
TARGET_MARGIN

Description:
------------
Variable to make sure that during extracting target from image, we take cauple of
pixels more just to be safe for next algorithm.

Value:
------
True
    User will be prompted to select corners of target manually.

False
    Program will select corners automaticaly.
"""
MANUAL_SELECTION = False