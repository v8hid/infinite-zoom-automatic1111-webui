import numpy as np
import imageio
from PIL import Image

def closest_upper_divisible_by_eight(num):
    if num % 8 == 0:
        return num
    else:
        return math.ceil(num / 8) * 8

def write_video(file_path, frames, fps, reversed=True, start_frame_dupe_amount=15, last_frame_dupe_amount=30):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    :param reversed: if order of images to be reversed (default = True)
    """
    if reversed == True:
        frames = frames[::-1]

    first_frame_size = frames[0].size
    even_frame_size = (closest_upper_divisible_by_eight(first_frame_size[0]), closest_upper_divisible_by_eight(first_frame_size[1]))
    
    # Drop random garbage frames that are outsize of the requested generation size to prevent ffmpeg from throwing an error.
    filtered_frames = [frame for frame in frames if frame.size == first_frame_size]
    frames = filtered_frames
    
    # Resize all frames to the even dimensions
    #resized_frames = [frame.resize(even_frame_size, Image.ANTIALIAS) for frame in frames]
    #frames = resized_frames
    
    
    #frames.pop()

    # Create an imageio video writer, avoid block size of 512.
    writer = imageio.get_writer(file_path, fps=fps, macro_block_size=None)

    # Duplicate the start and end frames
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