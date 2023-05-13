from dataclasses import dataclass
from PIL import Image
@dataclass
class InfZoomConfig():
    common_prompt_pre:str
    prompts_array:list[str]
    common_prompt_suf:str
    negative_prompt:str
    num_outpainting_steps: int
    guidance_scale:float
    num_inference_steps:int
    custom_init_image:Image
    custom_exit_image:Image
    video_frame_rate:int
    video_zoom_mode:int  #0: ZoomOut, 1: ZoomIn
    video_start_frame_dupe_amount:int
    video_last_frame_dupe_amount:int
    inpainting_mask_blur:int
    inpainting_fill_mode:int
    zoom_speed:float
    seed:int
    outputsizeW:int
    outputsizeH:int
    batchcount:int
    sampler:str
    upscale_do:bool
    upscaler_name:str
    upscale_by:float
    overmask:int
    outpaintStrategy: str
    inpainting_denoising_strength:float=1
    inpainting_full_res:int =0
    inpainting_padding:int=0
    progress:any=None
