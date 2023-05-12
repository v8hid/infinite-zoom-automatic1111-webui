from .run import (InfZoomer)
from .InfZoomConfig import InfZoomConfig
from PIL import Image

# to be called from Gradio or other client
def createZoom(
    common_prompt_pre:str,
    prompts_array:list[str],
    common_prompt_suf:str,
    negative_prompt:str,
    num_outpainting_steps: int,
    guidance_scale:float,
    num_inference_steps:int,
    custom_init_image:Image,
    custom_exit_image:Image,
    video_frame_rate:int,
    video_zoom_mode:int,
    video_start_frame_dupe_amount:int,
    video_last_frame_dupe_amount:int,
    inpainting_mask_blur:int,
    inpainting_fill_mode:int,
    zoom_speed:float,
    seed:int,
    outputsizeW:int,
    outputsizeH:int,
    batchcount:int,
    sampler:str,
    upscale_do:bool,
    upscaler_name:str,
    upscale_by:float,
    overmask:int,
    outpaintStrategy:str,
    inpainting_denoising_strength:float=1,
    inpainting_full_res:int =0,
    inpainting_padding:int=0,
    progress:any=None
):
    izc = InfZoomConfig(
        common_prompt_pre,
        prompts_array,
        common_prompt_suf,
        negative_prompt,
        num_outpainting_steps,
        guidance_scale,
        num_inference_steps,
        custom_init_image,
        custom_exit_image,
        video_frame_rate,
        video_zoom_mode,
        video_start_frame_dupe_amount,
        video_last_frame_dupe_amount,
        inpainting_mask_blur,
        inpainting_fill_mode,
        zoom_speed,
        seed,
        outputsizeW,
        outputsizeH,
        batchcount,
        sampler,
        upscale_do,
        upscaler_name,
        upscale_by,
        overmask,
        outpaintStrategy,
        inpainting_denoising_strength,
        inpainting_full_res,
        inpainting_padding,
        progress
    )
    iz= InfZoomer(izc)
    r = iz.create_zoom()
    del iz
    return r
