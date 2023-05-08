import numpy as np
import imageio
from .image import blend_images, draw_gradient_ellipse, alpha_composite_images, luma_wipe_images, PSLumaWipe_images
import math

def write_video(file_path, frames, fps, reversed=True, start_frame_dupe_amount=15, last_frame_dupe_amount=30, num_interpol_frames=2, blend=False, blend_image= None):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param reversed: if order of images to be reversed (default = True)
    """
    if reversed == True:
        frames = frames[::-1]

    # Drop missformed frames
    frames = [frame.convert("RGBA") for frame in frames if frame.size == frames[0].size]

    # Create an imageio video writer, avoid block size of 512.
    writer = imageio.get_writer(file_path, fps=fps, macro_block_size=None)

    # Duplicate the start and end frames
    if blend:
        num_frames_replaced = num_interpol_frames + 2
        if blend_image is None:
            blend_image = draw_gradient_ellipse(*frames[0].size, 0.63)
        next_frame = frames[num_frames_replaced]
        next_to_last_frame = frames[(-1 * num_frames_replaced)]
        
        print(f"Blending start: {math.ceil(start_frame_dupe_amount)} next frame:{(num_interpol_frames -1)}")
        #start_frames = alpha_composite_images(frames[0], next_frame, blend_image, math.ceil(start_frame_dupe_amount))
        #start_frames = luma_wipe_images(frames[0], next_frame, blend_image, math.ceil(start_frame_dupe_amount))
        start_frames = PSLumaWipe_images(frames[0], next_frame, blend_image, math.ceil(start_frame_dupe_amount),(200,200,0,128))
        del frames[:num_frames_replaced]

        print(f"Blending end: {math.ceil(last_frame_dupe_amount)} next to last frame:{-1 * (num_interpol_frames + 1)}")
        #end_frames = alpha_composite_images(next_to_last_frame, frames[-1], blend_image, math.ceil(last_frame_dupe_amount))
        #end_frames = luma_wipe_images(next_to_last_frame, frames[-1], blend_image, math.ceil(last_frame_dupe_amount))
        end_frames = PSLumaWipe_images(next_to_last_frame, frames[-1], blend_image, math.ceil(last_frame_dupe_amount),(200,200,0,128))
        frames = frames[:(-1 * num_frames_replaced)]
    else:
        start_frames = [frames[0]] * start_frame_dupe_amount
        end_frames = [frames[-1]] * last_frame_dupe_amount    

    # Write the duplicated frames to the video writer
    for frame in start_frames:
        # Convert PIL image to numpy array
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Write the frames to the video writer
    for frame in frames:
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Write the duplicated frames to the video writer
    for frame in  end_frames:
        np_frame = np.array(frame)
        writer.append_data(np_frame)

    # Close the video writer
    writer.close()