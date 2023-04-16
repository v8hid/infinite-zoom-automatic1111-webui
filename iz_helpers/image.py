import requests
from PIL import Image
import numpy as np


def shrink_and_paste_on_blank(current_image, mask_width, mask_height):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame, 
    so that the image the function returns is the same size as the input. 
    :param current_image: input image to transform
    :param mask_width: width in pixels to shrink from each side
    """

    height = current_image.height
    width = current_image.width

    # shrink down by mask_width
    prev_image = current_image.resize(
        (width-2*mask_width, height-2*mask_height))
    prev_image = prev_image.convert("RGBA")
    prev_image = np.array(prev_image)

    # create blank non-transparent image
    blank_image = np.array(current_image.convert("RGBA"))*0
    blank_image[:, :, 3] = 1

    # paste shrinked onto blank
    blank_image[mask_height:height-mask_height,mask_width:width-mask_width  :] = prev_image
    prev_image = Image.fromarray(blank_image)

    return prev_image
