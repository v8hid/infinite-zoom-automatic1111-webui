from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageDraw, ImageFont, ImageOps, ImageMath
import requests
import base64
import numpy as np
import math
from io import BytesIO
from modules.processing import apply_overlay, slerp


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

def lerp(value1, value2, factor):
    """
    Linearly interpolate between value1 and value2 by factor.
    """
    return np.interp(factor, [0, 1], [value1, value2])

def lerp(a, b, t):
    t = np.clip(t, 0, 1)  # clip t to the range [0, 1]
    return ((1 - t) * np.array(a) + t * np.array(b))

def lerpy(img1, img2, alpha):
    vector = np.vectorize(np.int_)
    if type(img1) is PIL.Image.Image:
        img1 = np.array(img1)[:, :, 3]
    if type(img2) is PIL.Image.Image:
        img2 = np.array(img2)[:, :, 3]
    beta = 1.0 - alpha
    gamma = beta / img1.shape[1]
    delta = gamma / img1.shape[0]
    for j in range(img1.shape[0]):
        for i in range(img1.shape[1]):
             img1[j][i]=vector(((img1[j][i]*alpha)+(img2[j][i]*beta))+gamma)
    #cv2.imshow('linear interpolation',img1[:, :, :, None])
    #cv2.waitKey(0) & 0xFF
    return Image.fromarray(img1[:, :], mode='L')
    #return Image.fromarray(img1[:, :], 'RGBA')
    #return img1[:, :]

def lerp_imagemath(img1, img2, alpha:int = 50):
    # must use ImageMath.eval to avoid overflow and alpha must be an int from 0 to 100
    return ImageMath.eval("((im * a) / 100) + (im2 * (100 - a)) / 100", im=img1.convert('L'), im2= img2.convert('L'), a=alpha)

def lerp_color(color1, color2, t):
    """
    Performs a linear interpolation (lerp) between two colors at a given progress value.

    Args:
    color1 (tuple): A tuple of 4 floats representing the first color in RGBA format.
    color2 (tuple): A tuple of 4 floats representing the second color in RGBA format.
    t (float): A value between 0.0 and 1.0 representing the progress of the lerp operation.

    Returns:
    A tuple of 4 floats representing the resulting color in RGBA format.
    """
    r = (1 - t) * color1[0] + t * color2[0]
    g = (1 - t) * color1[1] + t * color2[1]
    b = (1 - t) * color1[2] + t * color2[2]
    a = (1 - t) * color1[3] + t * color2[3]
    return (r, g, b, a)

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

def multiply_alpha_ImageMath(image, factor):
    """
    Multiply the alpha layer of a PIL RGBA image by a given factor and clip it between 0 and 255.
    Returns a modified image.
    """
    # Split the image into separate bands
    r, g, b, a = image.split()
    # Multiply the alpha band by the factor using ImageMath
    a = ImageMath.eval("convert(float(a) * factor, 'L')", a=a, factor=factor)
    # Clip the alpha band between 0 and 255
    a = ImageMath.eval("convert(min(max(a, 0), 255), 'L')", a=a)
    # Merge the bands back into an RGBA image
    return Image.merge("RGBA", (r, g, b, a))

def multiply_alpha(image, factor):
    """
    Multiplies the alpha layer of an RGBA image by the given factor.

    Args:
        image (PIL.Image.Image): The input image.
        factor (float): The multiplication factor for the alpha layer.

    Returns:
        PIL.Image.Image: The modified image.
    """
    # Convert the image to a numpy array
    np_image = np.array(image)
    # Extract the alpha channel from the image
    alpha = np_image[:, :, 3].astype(float)
    # Multiply the alpha channel by the given factor
    alpha *= factor
    # Clip the alpha channel between 0 and 255
    alpha = np.clip(alpha, 0, 255)
    # Replace the original alpha channel with the modified one
    np_image[:, :, 3] = alpha.astype(np.uint8)
    # Convert the numpy array back to a PIL image
    result_image = Image.fromarray(np_image)
    return result_image

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
    #start_image = apply_alpha_mask(start_image, gray_image)
    #stop_image = apply_alpha_mask(stop_image, gray_image, invert = True)

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

def alpha_composite_images(start_image: Image, stop_image: Image, gray_image: Image, num_frames: int) -> list:
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
    ac_frames = []

    #set alpha layers of images to be blended
    start_image = apply_alpha_mask(start_image, gray_image)
    stop_image = apply_alpha_mask(stop_image, gray_image, invert = False)

    # Generate each frame of the blending animation
    for i in range(num_frames):
        # Calculate the alpha amount for this frame
        alpha = i / float(num_frames - 1)
        start_adj_image = multiply_alpha(start_image.copy(), 1 - alpha)
        stop_adj_image = multiply_alpha(stop_image.copy(), alpha)

        # Blend the two images using the alpha amount
        ac_image = Image.alpha_composite(start_adj_image, stop_adj_image)

        # Append the blended frame to the list
        ac_frames.append(ac_image)

    # Return the list of blended frames
    return ac_frames

def luma_wipe_images(start_image: Image, stop_image: Image, alpha: Image, num_frames: int) -> list:
    #progress(0, status='Generating luma wipe...')
    lw_frames = []
    for i in range(num_frames):
        # Compute the luma value for this frame
        luma_progress = i / (num_frames - 1)
        # Create a new image for the transition
        transition = Image.new("RGBA", start_image.size)
        # Loop over each pixel in the alpha layer
        for x in range(alpha.width):
            for y in range(alpha.height):
                # Compute the luma value for this pixel
                luma = alpha.getpixel((x, y))[0] / 255.0
                if luma_progress >= luma:
                    # Interpolate between the two images based on the luma value
                    pixel = (
                        int(start_image.getpixel((x, y))[0] * (1 - luma) + stop_image.getpixel((x, y))[0] * luma),
                        int(start_image.getpixel((x, y))[1] * (1 - luma) + stop_image.getpixel((x, y))[1] * luma),
                        int(start_image.getpixel((x, y))[2] * (1 - luma) + stop_image.getpixel((x, y))[2] * luma),
                        int(255 * luma_progress)  # Set the alpha value based on the luma value
                    )        
                    # Set the new pixel in the transition image
                    transition.putpixel((x, y), pixel)
                else:
                    # Set the start pixel in the transition image
                    transition.putpixel((x, y), start_image.getpixel((x, y)))
        # Append the transition image to the list   
        lw_frames.append(transition)
        #progress((x + 1) / num_frames)
    return lw_frames

def srgb_nonlinear_to_linear_channel(u):
    return (u / 12.92) if (u <= 0.04045) else pow((u + 0.055) / 1.055, 2.4)

def srgb_nonlinear_to_linear(v):
    return [srgb_nonlinear_to_linear_channel(x) for x in v]

#result_img = eval("convert('RGBA')", lambda x, y: PSLumaWipe(img_a.getpixel((x,y)), img_b.getpixel((x,y)), test_g_image.getpixel((x,y))[0]/255,(1,0,0,.5), 0.25, False, 0.1, 0.01, 0.01))
#list(np.divide((255,255,245,225),255))
def PSLumaWipe(a_color, b_color, luma, l_color=(255, 255, 255, 255), progress=0.0, invert=False, softness=0.01, start_adjust = 0.01, stop_adjust = 0.0):
    # - adjust for min and max. Do not process if luma value is outside min or max
    if ((luma >= (start_adjust)) and (luma <= (1 - stop_adjust))):
        if (invert):
            luma = 1.0 - luma
        # user color with luma
        out_color = np.array([l_color[0], l_color[1], l_color[2], luma * 255])
        time = lerp(0.0, 1.0 + softness, progress)
        #print(f"softness: {str(softness)}  out_color: {str(out_color)} a_color: {str(a_color)} b_color: {str(b_color)} time: {str(time)} luma: {str(luma)} progress: {str(progress)}")
        # if luma less than time, do not blend color
        if (luma <= time - softness):
            alpha_behind = np.clip(1.0 - (time - softness - luma) / softness, 0.0, 1.0)
            return tuple(np.round(lerp(b_color, out_color, alpha_behind)).astype(int))
        # if luma greater than time, show original color
        if (luma >= time):
            return a_color
        alpha = (time - luma) / softness
        out_color = lerp(a_color, b_color + out_color, alpha)
        #print(f"alpha: {str(alpha)}  out_color: {str(out_color)} time: {str(time)} luma: {str(luma)}")
        out_color = srgb_nonlinear_to_linear(out_color)
        return tuple(np.round(out_color).astype(int))
    else:
        # return original pixel color
        return a_color

def PSLumaWipe_images(start_image: Image, stop_image: Image, luma_wipe_image: Image, num_frames: int, transition_color: tuple[int, int, int, int] = (255,255,255,255)) -> list:
    #progress(0, status='Generating luma wipe...')
    # fix transition_color to relative 0.0 - 1.0    
    #luma_color = list(np.divide(transition_color,255))
    softness = 0.03
    lw_frames = []
    lw_frames.append(start_image)
    width, height = start_image.size
    #compensate for different image sizes for LumaWipe
    if (start_image.size != luma_wipe_image.size):
        luma_wipe_image = resize_and_crop_image(luma_wipe_image,width,height)
    # call PSLumaWipe for each pixel
    for i in range(num_frames):
        # Compute the luma value for this frame
        luma_progress = i / (num_frames - 1)
        transition = Image.new(start_image.mode, (width, height))        
        # apply to each pixel in the image
        for x in range(width):
            for y in range(height):
                # call PSLumaWipe for each pixel
                pixel = PSLumaWipe(start_image.getpixel((x, y)), stop_image.getpixel((x, y)), luma_wipe_image.getpixel((x, y))[0]/255, transition_color, luma_progress, False, softness, 0.01, 0.00)
                transition.putpixel((x, y), pixel)
        lw_frames.append(transition)
        print(f"Luma Wipe frame:{len(lw_frames)}")
        #lw_frames[-1].show()
    lw_frames.append(stop_image)
    return lw_frames