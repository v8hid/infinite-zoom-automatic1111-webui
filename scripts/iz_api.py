from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import FileResponse
from iz_helpers.run import create_zoom, progress  # Import progress variable
from PIL import Image
import gradio as gr
import os
import logging
import io
import base64
import asyncio

def base64_to_image(base64_str):
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    return image

def infinite_zoom_api(_: gr.Blocks, app: FastAPI):
    @app.post("/infinite_zoom/create")
    async def infinite_zoom(
            common_prompt_pre: str = Body(...),
            prompts_array: list = Body(...),
            common_prompt_suf: str = Body(...),
            negative_prompt: str = Body(...),
            num_outpainting_steps: int = Body(...),
            guidance_scale: float = Body(...),
            num_inference_steps: int = Body(...),
            custom_init_image: str = Body(...),
            custom_exit_image: str = Body(...),
            video_frame_rate: int = Body(...),
            video_zoom_mode: str = Body(...),
            video_start_frame_dupe_amount: int = Body(...),
            video_last_frame_dupe_amount: int = Body(...),
            inpainting_mask_blur: int = Body(...),
            inpainting_fill_mode: str = Body(...),
            zoom_speed: float = Body(...),
            seed: int = Body(...),
            outputsizeW: int = Body(...),
            outputsizeH: int = Body(...),
            batchcount: int = Body(...),
            sampler: str = Body(...),
            upscale_do: bool = Body(...),
            upscaler_name: str = Body(...),
            upscale_by: float = Body(...)
    ):
        loop = asyncio.get_event_loop()
        video_filename, main_frames, js, info, _ = await loop.run_in_executor(
            None,
            create_zoom,
            common_prompt_pre,
            prompts_array,
            common_prompt_suf,
            negative_prompt,
            num_outpainting_steps,
            guidance_scale,
            num_inference_steps,
            base64_to_image(custom_init_image),
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
            upscale_by
        )
        return {"video_name": video_filename}
    
    @app.get("/infinite_zoom/download_video/{video_filename}")
    async def download_video(video_filename: str):
        video_path = os.path.join('outputs', 'videos', video_filename)
        if os.path.exists(video_path):
            return FileResponse(video_path)
        else:
            raise HTTPException(status_code=404, detail="Video not found")
        
    @app.get("/infinite_zoom/progress")
    async def get_progress():
        return progress
    
try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(infinite_zoom_api)
except Exception as e:
    logging.exception("Failed to call on_app_started in script_callbacks module")