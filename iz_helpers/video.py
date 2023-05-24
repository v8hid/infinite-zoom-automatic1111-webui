import numpy as np
import imageio
import subprocess
from .image import draw_gradient_ellipse, alpha_composite_images, blend_images, PSLumaWipe_images2
import math

def write_video(file_path, frames, fps, reversed=True, start_frame_dupe_amount=15, last_frame_dupe_amount=30, num_interpol_frames=2, blend_invert: bool = False, blend_image= None, blend_type:int = 0, blend_gradient_size: int = 63, blend_color = "#ffff00"):
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
    if blend_type != 0:
        num_frames_replaced = num_interpol_frames
        if blend_image is None:
            blend_image = draw_gradient_ellipse(*frames[0].size, blend_gradient_size)
        next_frame = frames[num_frames_replaced]
        next_to_last_frame = frames[(-1 * num_frames_replaced)]
        
        print(f"Blending start: {math.ceil(start_frame_dupe_amount)} next frame:{(num_frames_replaced)}")
        if blend_type == 1:
            start_frames = blend_images(frames[0], next_frame, math.ceil(start_frame_dupe_amount), blend_invert)
        elif blend_type == 2:
            start_frames = alpha_composite_images(frames[0], next_frame, blend_image, math.ceil(start_frame_dupe_amount), blend_invert)
        elif blend_type == 3:
            start_frames = PSLumaWipe_images2(frames[0], next_frame, blend_image, math.ceil(start_frame_dupe_amount), blend_invert,blend_color)
        del frames[:num_frames_replaced]

        print(f"Blending end: {math.ceil(last_frame_dupe_amount)} next to last frame:{-1 * (num_frames_replaced)}")
        if blend_type == 1:
            end_frames = blend_images(next_to_last_frame, frames[-1], math.ceil(last_frame_dupe_amount), blend_invert)
        elif blend_type == 2:
            end_frames = alpha_composite_images(next_to_last_frame, frames[-1], blend_image, math.ceil(last_frame_dupe_amount), blend_invert)
        elif blend_type == 3:
            end_frames = PSLumaWipe_images2(next_to_last_frame, frames[-1], blend_image, math.ceil(last_frame_dupe_amount), blend_invert, blend_color)
        frames = frames[:(-1 * num_frames_replaced)]
    else:
        start_frames = [frames[0]] * start_frame_dupe_amount
        end_frames = [frames[-1]] * last_frame_dupe_amount    

    # Write the blended start frames to the video writer
    for frame in start_frames:
        # Convert PIL image to numpy array
        np_frame = np.array(frame.convert("RGB"))
        writer.append_data(np_frame)

    # Write the frames to the video writer
    for frame in frames:
        np_frame = np.array(frame.convert("RGB"))
        writer.append_data(np_frame)

    # Write the blended end frames to the video writer
    for frame in end_frames:
        np_frame = np.array(frame.convert("RGB"))
        writer.append_data(np_frame)

    # Close the video writer
    writer.close()


class ContinuousVideoWriter:

    _writer = None
    
    def __init__(self, file_path, initframe, nextframe, fps, start_frame_dupe_amount=15, video_ffmpeg_opts="", blend_invert: bool = False, blend_image= None, blend_type:int = 0, blend_gradient_size: int = 63, blend_color = "#ffff00" ):
        """
        Writes initial frame to a new mp4 video file
        :param file_path: Path to output video, must end with .mp4
        :param frame: Start image PIL.Image objects
        :param fps: Desired frame rate
        :param reversed: if order of images to be reversed (default = True)
        """
        ffopts = []
        if video_ffmpeg_opts is not "":
            ffopts= video_ffmpeg_opts.split(" ")

        writer = imageio.get_writer(file_path, fps=fps, macro_block_size=None, ffmpeg_params=ffopts)
        # Duplicate the start frames
        if blend_type != 0:
            if blend_image is None:
                blend_image = draw_gradient_ellipse(*initframe.size, blend_gradient_size)
        
            if blend_type == 1:
                start_frames = blend_images(initframe, nextframe, math.ceil(start_frame_dupe_amount), blend_invert)
            elif blend_type == 2:
                start_frames = alpha_composite_images(initframe, nextframe, blend_image, math.ceil(start_frame_dupe_amount), blend_invert)
            elif blend_type == 3:
                start_frames = PSLumaWipe_images2(initframe, nextframe, blend_image, math.ceil(start_frame_dupe_amount), blend_invert,blend_color)            
        else:
            start_frames = [initframe] * start_frame_dupe_amount
        for f in start_frames:
            writer.append_data(np.array(f))
        self._writer = writer
    
    def append(self, frames):
        """
        Append a list of image PIL.Image objects to the end of the file.
        :param frames: List of image PIL.Image objects
        """
        for i,f in enumerate(frames):
            self._writer.append_data(np.array(f))
    
    def finish(self, exitframe, next_to_last_frame, last_frame_dupe_amount=30, blend_invert: bool = False, blend_image= None, blend_type:int = 0, blend_gradient_size: int = 63, blend_color = "#ffff00"  ):
        """
        Closes the file writer.
        """
        # Duplicate the exit frames
        if blend_type != 0:
            if blend_type == 1:
                end_frames = blend_images(next_to_last_frame, exitframe, math.ceil(last_frame_dupe_amount), blend_invert)
            elif blend_type == 2:
                end_frames = alpha_composite_images(next_to_last_frame, exitframe, blend_image, math.ceil(last_frame_dupe_amount), blend_invert)
            elif blend_type == 3:
                end_frames = PSLumaWipe_images2(next_to_last_frame, exitframe, blend_image, math.ceil(last_frame_dupe_amount), blend_invert, blend_color)
        else:
            end_frames = [exitframe] * last_frame_dupe_amount
        for f in end_frames:
            self._writer.append_data(np.array(f))       
        self._writer.close()

def add_audio_to_video(video_path, audio_path, output_path, ffmpeg_location = 'ffmpeg'):
    command = [ffmpeg_location, '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', '-shortest', output_path]
    subprocess.run(command)
    return output_path

def resize_video(input_path, output_path, width:int, height:int, flags:str="lanczos"):
    scaling = f'{width}:{height}'
    command = ['ffmpeg', '-i', input_path, '-vf', f'scale={scaling}:flags={flags}', output_path]
    subprocess.run(command)
    return output_path