from PIL import Image
import requests
import base64
from io import BytesIO


def shrink_and_paste_on_blank(current_image, mask_width, mask_height):
    """
    Decreases size of current_image by mask_width pixels from each side,
    then adds a mask_width width transparent frame,
    so that the image the function returns is the same size as the input.
    :param current_image: input image to transform
    :param mask_width: width in pixels to shrink from each side
    :param mask_height: height in pixels to shrink from each side
    """

    # calculate new dimensions
    width, height = current_image.size
    new_width = width - 2 * mask_width
    new_height = height - 2 * mask_height

    # resize and paste onto blank image
    prev_image = current_image.resize((new_width, new_height))
    blank_image = Image.new("RGBA", (width, height), (0, 0, 0, 1))
    blank_image.paste(prev_image, (mask_width, mask_height))

    return blank_image


def open_image(image_path):
    if image_path.startswith('http'):
        # If the image path is a URL, download the image using requests
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    elif image_path.startswith('data'):
        # If the image path is a DataURL, decode the base64 string
        encoded_data = image_path.split(',')[1]
        decoded_data = base64.b64decode(encoded_data)
        img = Image.open(BytesIO(decoded_data))
    else:
        # Assume that the image path is a file path
        img = Image.open(image_path)
    
    return img
