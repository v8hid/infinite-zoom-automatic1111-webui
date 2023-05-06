from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps
import requests
import base64
import numpy as np
import math
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
    blank_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    blank_image.paste(prev_image, (mask_width, mask_height))

    return blank_image


def open_image(image_path):
    """
    Opens an image from a file path or URL, or decodes a DataURL string into an image.

    Parameters:
        image_path (str): The file path, URL, or DataURL string of the image to open.

    Returns:
        Image: A PIL Image object of the opened image.
    """
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

def apply_alpha_mask(image, mask_image, invert = False):
    """
    Applies a mask image as the alpha channel of the input image.

    Parameters:
        image (Image): A PIL Image object of the image to apply the mask to.
        mask_image (Image): A PIL Image object of the alpha mask to apply.

    Returns:
        Image: A PIL Image object of the input image with the applied alpha mask.
    """
    # Resize the mask to match the current image size
    mask_image = resize_and_crop_image(mask_image, image.width, image.height).convert('L') # convert to grayscale
    if invert:
        ImageEnhance.Contrast(mask_image).enhance(-1.0)
    # Apply the mask as the alpha layer of the current image
    result_image = image.copy() 
    result_image.putalpha(mask_image) 
    return result_image

def convert_to_rgba(images):
    rgba_images = []
    for img in images:
        if img.mode == 'RGB':
            rgba_img = img.convert('RGBA')
            rgba_images.append(rgba_img)
        else:
            rgba_images.append(img)
    return rgba_images

def resize_image_with_aspect_ratio(image: Image, basewidth: int = 512, baseheight: int = 512) -> Image:
    """
    Resizes an image while maintaining its aspect ratio. This may not fill the entire image height.

    Args:
    - image (PIL.Image): The input image.
    - basewidth (int): The desired width of the output image. Defaults to 512.
    - baseheight (int): The desired height of the output image. Defaults to 512.

    Returns:
    - PIL.Image: The resized image.

    Raises:
    - ValueError: If `basewidth` or `baseheight` is less than or equal to 0.

    """
    if basewidth <= 0 or baseheight <= 0:
        raise ValueError("resize_image_with_aspect_ratio error: basewidth and baseheight must be greater than 0")

    # Get the original size of the image
    orig_width, orig_height = image.size
    
    # Calculate the height that corresponds to the given width while maintaining aspect ratio
    wpercent = (basewidth / float(orig_width))
    hsize = int((float(orig_height) * float(wpercent)))
    
    # Resize the image with Lanczos resampling filter
    resized_image = image.resize((basewidth, hsize), resample=Image.Resampling.LANCZOS)

    # If the height of the resized image is still larger than the given baseheight,
    # then crop the image from the top and bottom to match the baseheight
    if hsize > baseheight:
        # Calculate the number of pixels to crop from the top and bottom
        crop_height = (hsize - baseheight) // 2
        # Crop the image
        resized_image = resized_image.crop((0, crop_height, basewidth, hsize - crop_height))
    else:
        if hsize < baseheight:
            # If the height of the resized image is smaller than the given baseheight,
            # then paste the resized image in the middle of a blank image with the given baseheight
            blank_image = Image.new("RGBA", (basewidth, baseheight), (255, 255, 255, 0))
            blank_image.paste(resized_image, (0, (baseheight - hsize) // 2))
            resized_image = blank_image
    
    return resized_image

def resize_and_crop_image(image: Image, new_width: int = 512, new_height: int = 512) -> Image:
    """
    Resizes and crops an image to a specified width and height. This ensures that the entire new_width and new_height 
    dimensions are filled by the image, and the aspect ratio is maintained.

    Parameters:
    - image (PIL.Image): The image to be resized and cropped.
    - new_width (int): The desired width of the new image. Default is 512.
    - new_height (int): The desired height of the new image. Default is 512.

    Returns:
    - cropped_image (PIL.Image): The resized and cropped image.
    """
    # Get the dimensions of the original image
    orig_width, orig_height = image.size

    # Calculate the aspect ratios of the original and new images
    orig_aspect_ratio = orig_width / float(orig_height)
    new_aspect_ratio = new_width / float(new_height)

    # Calculate the new size of the image while maintaining aspect ratio
    if orig_aspect_ratio > new_aspect_ratio:
        # The original image is wider than the new image, so we need to crop the sides
        resized_width = int(new_height * orig_aspect_ratio)
        resized_height = new_height
        left_offset = (resized_width - new_width) // 2
        top_offset = 0
    else:
        # The original image is taller than the new image, so we need to crop the top and bottom
        resized_width = new_width
        resized_height = int(new_width / orig_aspect_ratio)
        left_offset = 0
        top_offset = (resized_height - new_height) // 2

    # Resize the image with Lanczos resampling filter
    resized_image = image.resize((resized_width, resized_height), resample=Image.Resampling.LANCZOS)

    # Crop the image to fill the entire height and width of the new image
    cropped_image = resized_image.crop((left_offset, top_offset, left_offset + new_width, top_offset + new_height))

    return cropped_image

def grayscale_to_gradient(image, gradient_colors):
    """
    Converts a grayscale PIL Image into a two color image using the specified gradient colors.
    
    Args:
        image (PIL.Image.Image): The input grayscale image.
        gradient_colors (list): A list of two tuples representing the gradient colors.
        
    Returns:
        PIL.Image.Image: A two color image with the same dimensions as the input grayscale image.
    """
    # Create a new image with a palette
    result = Image.new("P", image.size)
    result.putpalette([c for color in gradient_colors for c in color])
    
    # Convert the input image to a list of pixel values
    pixel_values = list(image.getdata())
    
    # Convert the pixel values to indices in the palette and assign them to the output image
    result.putdata([gradient_colors[int(p * (len(gradient_colors) - 1))] for p in pixel_values])
    
    return result

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# _, (h, k) = ellipse_bbox(0,0,768,512,math.radians(0.0)) # Ellipse center 
def ellipse_bbox(h, k, a, b, theta):
    """
    Computes the bounding box of an ellipse centered at (h,k) with semi-major axis 'a',
    semi-minor axis 'b', and rotation angle 'theta' (in radians).
    
    Args:
        h (float): x-coordinate of the ellipse center.
        k (float): y-coordinate of the ellipse center.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        theta (float): Angle of rotation (in radians) of the ellipse.
    
    Returns:
        tuple: A tuple of two tuples representing the top left and bottom right corners of
        the bounding box of the ellipse, respectively.
    """
    ux = a * math.cos(theta)
    uy = a * math.sin(theta)
    vx = b * math.cos(theta + math.pi / 2)
    vy = b * math.sin(theta + math.pi / 2)
    box_halfwidth = np.ceil(math.sqrt(ux**2 + vx**2))
    box_halfheight = np.ceil(math.sqrt(uy**2 + vy**2))
    return ((int(h - box_halfwidth), int(k - box_halfheight))
        , (int(h + box_halfwidth), int(k + box_halfheight)))


# intel = make_gradient_v2(768,512,h/2,k/2,h*2/3,k*2/3,math.radians(0.0))
def make_gradient_v1(width, height, h, k, a, b, theta):
    """
    Generates a gradient image with an elliptical shape.
    
    Args:
        width (int): Width of the output image.
        height (int): Height of the output image.
        h (float): x-coordinate of the center of the ellipse.
        k (float): y-coordinate of the center of the ellipse.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        theta (float): Angle of rotation (in radians) of the ellipse.
    
    Returns:
        PIL.Image.Image: A PIL Image object representing the gradient with an elliptical shape.
    """
    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a**2, b**2
    
    # Initialize an empty array to hold the weights
    weights = np.zeros((height, width), np.float64)
    
    # Calculate the weight for each pixel
    for y in range(height):
        for x in range(width):
            weights[y, x] = ((((x-h) * ct + (y-k) * st) ** 2) / aa
                + (((x-h) * st - (y-k) * ct) ** 2) / bb)
    
    # Convert the weights to pixel values and create a PIL Image
    pixel_values = np.uint8(np.clip(1.0 - weights, 0, 1) * 255)
    return Image.fromarray(pixel_values, mode='L')


# make_gradient_v2(768,512,h/2,k/2,h-192,k-192,math.radians(30.0))
def make_gradient_v2(width, height, h, k, a, b, theta):
    """
    Generates a gradient image with an elliptical shape.
    
    Args:
        width (int): Width of the output image.
        height (int): Height of the output image.
        h (float): x-coordinate of the center of the ellipse.
        k (float): y-coordinate of the center of the ellipse.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        theta (float): Angle of rotation (in radians) of the ellipse.
    
    Returns:
        PIL.Image.Image: A PIL Image object representing the gradient with an elliptical shape.
    """
    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a**2, b**2        
    # Generate (x,y) coordinate arrays
    y,x = np.mgrid[-k:height-k,-h:width-h]    
    # Calculate the weight for each pixel
    weights = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)
    # Convert the weights to pixel values and create a PIL Image
    pixel_values = np.uint8(np.clip(1.0 - weights, 0, 1) * 255)
    return Image.fromarray(pixel_values, mode='L')

def make_gradient_v3(width, height, h, k, a, b, theta, gradient_colors=[(255, 255, 255, 1), (0, 0, 0, 1)]):
    """
    Generates a gradient image with an elliptical shape and the specified gradient colors.
    
    Args:
        width (int): Width of the output image.
        height (int): Height of the output image.
        h (float): x-coordinate of the center of the ellipse.
        k (float): y-coordinate of the center of the ellipse.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        theta (float): Angle of rotation (in radians) of the ellipse.
        gradient_colors (list): A list of two tuples representing the gradient colors.
    
    Returns:
        PIL.Image.Image: A two color gradient image with an elliptical shape.
    """
    # Precalculate constants
    st, ct = math.sin(theta), math.cos(theta)
    aa, bb = a**2, b**2        
    # Generate (x,y) coordinate arrays
    y, x = np.mgrid[-k:height-k, -h:width-h]    
    # Calculate the weight for each pixel
    weights = (((x * ct + y * st) ** 2) / aa) + (((x * st - y * ct) ** 2) / bb)    
    # Normalize the weights to the range [0, 1]
    weights = 1.0 - np.clip(weights / np.max(weights), 0.0, 1.0)    
    # Create a grayscale image from the weights array
    grayscale_image = Image.fromarray(np.uint8(weights * 255))    
    # Convert the grayscale image into a two color gradient image using the specified gradient colors
    gradient_image = grayscale_to_gradient(grayscale_image, gradient_colors)
    
    return gradient_image

def draw_gradient_ellipse(width=512, height=512, white_amount=1.0, rotation = 0.0, contrast = 1.0):
    """
    Draw an ellipse with a radial gradient fill, and a variable amount of white in the center.

    :param height: The height of the output image. Default is 512.
    :param width: The width of the output image. Default is 512.
    :param white_amount: The amount of white in the center of the ellipse, as a float between 0.0 and 1.0. Default is 1.0.
    :return: An RGBA image with the gradient ellipse.
    """
    # Create a new image for outer ellipse
    size = (width, height)
    image = Image.new('RGBA', size, (255, 255, 255, 0))
    theta = rotation * (math.pi / 180)
    # Define the ellipse parameters
    center = (int(width // 2), int(height // 2))    
    # Draw the ellipse and fill it with the radial gradient
    image = make_gradient_v2(width, height, center[0], center[1], width * white_amount, height * white_amount, theta)
    # Apply brightness method of ImageEnhance class
    image = ImageEnhance.Contrast(image).enhance(contrast).convert('RGBA') 
    # Apply the alpha mask to the image
    image = apply_alpha_mask(image, image) 
    # Define the radial gradient parameters
    #ellipse_width, ellipse_height = (int((width * white_amount) // 1.5), int((height * white_amount) // 1.5))
    #ellipse_colors = [(255, 255, 255, 255), (0, 0, 0, 0)]
    # Create a new image for inner ellipse
    #inner_ellipse = Image.new("L", size, 0)
    #inner_ellipse = make_gradient_v2(width, height, center[0], center[1], ellipse_width, ellipse_height, theta)
    #inner_ellipse = apply_alpha_mask(inner_ellipse, inner_ellipse)    
    #image.paste(inner_ellipse, center, mask=inner_ellipse)    
    # Creating object of Brightness class
    # Return the result image
    return image

def crop_fethear_ellipse(image: Image.Image, feather_margin: int = 30, width_offset: int = 0, height_offset: int = 0) -> Image.Image:
    """
    Crop an elliptical region from the input image with a feathered edge.

    Args:
        image (PIL.Image.Image): The input image.
        feather_margin (int): The size of the feathered edge, in pixels. Default is 30.
        width_offset (int): The offset from the left and right edges of the image to the elliptical region. Default is 0.
        height_offset (int): The offset from the top and bottom edges of the image to the elliptical region. Default is 0.

    Returns:
        A new PIL Image containing the cropped elliptical region with a feathered edge.
    """

    # Create a blank mask image with the same size as the original image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Calculate the ellipse's bounding box
    ellipse_box = (
        width_offset,
        height_offset,
        image.width - width_offset,
        image.height - height_offset,
    )

    # Draw the ellipse on the mask
    draw.ellipse(ellipse_box, fill=255)

    # Apply the mask to the original image
    result = Image.new("RGBA", image.size)
    result.paste(image, mask=mask)

    # Crop the resulting image to the ellipse's bounding box
    cropped_image = result.crop(ellipse_box)

    # Create a new mask image with a black background (0)
    mask = Image.new("L", cropped_image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw an ellipse on the mask image with a feathered edge
    draw.ellipse(
        (
            0 + feather_margin,
            0 + feather_margin,
            cropped_image.width - feather_margin,
            cropped_image.height - feather_margin,
        ),
        fill=255,
        outline=0,
    )

    # Apply a Gaussian blur to the mask image
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_margin / 2))
    cropped_image.putalpha(mask)

    # Paste the cropped image onto a new image with the same size as the input image
    res = Image.new(cropped_image.mode, (image.width, image.height))
    paste_pos = (
        int((res.width - cropped_image.width) / 2),
        int((res.height - cropped_image.height) / 2),
    )
    res.paste(cropped_image, paste_pos)

    return res

def crop_inner_image(image: Image, width_offset: int, height_offset: int) -> Image:
    """
    Crops an input image to the center, with the specified width and height offsets.

    Args:
        image (PIL.Image): The input image to be cropped.
        width_offset (int): The width offset used for cropping.
        height_offset (int): The height offset used for cropping.

    Returns:
        PIL.Image: The cropped image, resized to the original image size using Lanczos resampling.
    """
    # Get the size of the input image
    width, height = image.size

    # Calculate the center coordinates of the image
    center_x, center_y = int(width / 2), int(height / 2)

    # Crop the image to the center using the specified offsets
    cropped_image = image.crop(
        (
            center_x - width_offset,
            center_y - height_offset,
            center_x + width_offset,
            center_y + height_offset,
        )
    )

    # Resize the cropped image to the original image size using Lanczos resampling
    resized_image = cropped_image.resize((width, height), resample=Image.Resampling.LANCZOS)

    return resized_image

def blend_images(start_image: Image, stop_image: Image, gray_image: Image, num_frames: int) -> list:
    """
    Blend two images together by using the gray image as the alpha amount of each frame.
    This function takes in three parameters:
    - start_image: the starting PIL image in RGBA mode
    - stop_image: the target PIL image in RGBA mode
    - gray_image: a gray scale PIL image of the same size as start_image and stop_image
    - num_frames: the number of frames to generate in the blending animation
    
    The function returns a list of PIL images representing the blending animation.
    """
    # Initialize the list of blended frames
    blended_frames = []

    #set alpha layers of images to be blended
    start_image = apply_alpha_mask(start_image, gray_image)
    stop_image = apply_alpha_mask(stop_image, gray_image, invert = True)

    # Generate each frame of the blending animation
    for i in range(num_frames):
        # Calculate the alpha amount for this frame
        alpha = i / float(num_frames - 1)

        # Blend the two images using the alpha amount
        blended_image = Image.blend(start_image, stop_image, alpha)

        # Append the blended frame to the list
        blended_frames.append(blended_image)

    # Return the list of blended frames
    return blended_frames