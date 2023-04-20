from PIL import Image

def shrink_rotate_and_paste_on_blank(current_image, mask_width, mask_height, rotate_angle=0):
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
    blank_image = blank_image.rotate(rotate_angle, resample=Image.BICUBIC)

    return blank_image


from PIL import Image

def zoom_image(img, zoom_factor, rotate_angle):
    # Get the original size of the image
    width, height = img.size

    # Calculate the new size of the image
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    img = img.convert("RGBA")
    img = img.rotate(rotate_angle)

    # Resize the image using the new size
    zoomed_image = img.resize((new_width, new_height))

    # Create a new image with the same dimensions as the original image
    output_image = Image.new("RGBA", (width, height), (1,0,0,0))

    # Paste the zoomed image onto the output image
    output_image.paste(zoomed_image, (int((width - new_width) / 2), int((height - new_height) / 2)))

    return output_image
