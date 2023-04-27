from PIL import Image, ImageDraw, ImageEnhance
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

def apply_alpha_mask(image, mask_image):
    # Resize the mask to match the current image size
    mask_image = mask_image.resize(image.size)
    # Apply the mask as the alpha layer of the current image
    result_image = image.copy()
    result_image.putalpha(mask_image.convert('L')) # convert to grayscale
    return result_image

def resize_image_with_aspect_ratio(image, basewidth=512, baseheight=512):
    # Get the original size of the image
    orig_width, orig_height = image.size
    
    # Calculate the height that corresponds to the given width while maintaining aspect ratio
    wpercent = (basewidth / float(orig_width))
    hsize = int((float(orig_height) * float(wpercent)))
    
    # Resize the image with Lanczos resampling filter
    resized_image = image.resize((basewidth, hsize), resample=Image.LANCZOS)

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

def resize_and_crop_image(image, new_width=512, new_height=512):
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
        left_offset = (resized_width - new_width) / 2
        top_offset = 0
    else:
        # The original image is taller than the new image, so we need to crop the top and bottom
        resized_width = new_width
        resized_height = int(new_width / orig_aspect_ratio)
        left_offset = 0
        top_offset = (resized_height - new_height) / 2    
    # Resize the image with Lanczos resampling filter
    resized_image = image.resize((resized_width, resized_height), resample=Image.LANCZOS)    
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