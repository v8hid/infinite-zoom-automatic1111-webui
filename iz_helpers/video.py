import numpy as np
import imageio
from .image import blend_images, draw_gradient_ellipse
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
        if blend_image is None:
            blend_image = draw_gradient_ellipse(*frames[0].size, 0.63)
        next_frame = frames[num_interpol_frames]
        next_to_last_frame = frames[-num_interpol_frames]
        print(f"Blending start: {math.ceil(start_frame_dupe_amount)}")
        start_frames = blend_images(frames[0], next_frame, blend_image, math.ceil(start_frame_dupe_amount))
        print(f"Blending end: {math.ceil(last_frame_dupe_amount)}")
        end_frames = blend_images(next_to_last_frame, frames[-1], blend_image, math.ceil(last_frame_dupe_amount))
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