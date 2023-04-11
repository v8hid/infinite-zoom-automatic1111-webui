import sys
import os
import time
basedir = os.getcwd()
sys.path.extend(basedir + '/extensions/infinite-zoom-sd-webui/')
import numpy as np
import gradio as gr
from PIL import Image

from iz_helpers.image import shrink_and_paste_on_blank
from iz_helpers.video import write_video
from webui import wrap_gradio_gpu_call
from modules import script_callbacks
import modules.shared as shared
import modules.scripts as scripts
from modules.processing import process_images, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.ui import create_output_panel, plaintext_to_html, wrap_gradio_call


output_path = basedir + '/extensions/infinite-zoom-sd-webui/out'
default_prompt = "A psychedelic jungle with trees that have glowing, fractal-like patterns, Simon stalenhag poster 1920s style, street level view, hyper futuristic, 8k resolution, hyper realistic"
default_negative_prompt = "frames, borderline, text, character, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur"


def renderTxt2Img(prompt, negative_prompt, sampler, steps, cfg_scale, width, height):
    processetd = None
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=output_path,
        outpath_grids=output_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        # seed=-1,
        sampler_name=sampler,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
    )
    # script_runner = scripts.scripts_img2img
    # p.scripts = script_runner
    # shared.state.begin()
    processed = process_images(p)
    # shared.state.end()
    return processed


def renderImg2Img(prompt, negative_prompt, sampler, steps, cfg_scale, width, height, init_image, mask_image):
    processetd = None
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=output_path,
        outpath_grids=output_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        # seed=-1,
        sampler_name=sampler,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        init_images=[init_image],
        mask=mask_image
    )
    # script_runner = scripts.scripts_txt2img
    # p.scripts = script_runner
    # shared.state.begin()
    processed = process_images(p)
    # shared.state.end()
    return processed


def create_zoom(
    prompts_array,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    custom_init_image
):
    prompts = {}
    for x in prompts_array:
        try:
            key = int(x[0])
            value = str(x[1])
            prompts[key] = value
        except ValueError:
            pass
    assert len(prompts_array) > 0, "prompts is empty"

    width = 512
    height = 512
    current_image = Image.new(mode="RGBA", size=(height, width))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255-mask_image).convert("RGB")
    current_image = current_image.convert("RGB")

    if (custom_init_image):
        current_image = custom_init_image.resize(
            (width, height), resample=Image.LANCZOS)
    else:
        processed = renderTxt2Img(prompts[min(k for k in prompts.keys() if k >= 0)],
                                  negative_prompt, "Euler a", num_inference_steps, guidance_scale, width, height)
        current_image = processed.images[0]
    mask_width = 128
    num_interpol_frames = 30

    all_frames = []
    all_frames.append(current_image)
    for i in range(num_outpainting_steps):
    #     print('Outpaint step: ' + str(i+1) +
    #           ' / ' + str(num_outpainting_steps))

        prev_image_fix = current_image

        prev_image = shrink_and_paste_on_blank(current_image, mask_width)

        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255-mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")
        # images = pipe(prompt=prompts[max(k for k in prompts.keys() if k <= i)],
        #               negative_prompt=negative_prompt,
        #               image=current_image,
        #               guidance_scale=guidance_scale,
        #               height=height,
        #               width=width,
        #               # generator = g_cuda.manual_seed(seed),
        #               mask_image=mask_image,
        #               num_inference_steps=num_inference_steps)[0]
        # current_image = images[0]
        processed = renderImg2Img(prompts[max(k for k in prompts.keys() if k <= i)], negative_prompt, "Euler a", num_inference_steps, guidance_scale, width, height, current_image, mask_image)
        current_image = processed.images[0]

        current_image.paste(prev_image, mask=prev_image)

        # interpolation steps bewteen 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_image
            interpol_width = round(
                (1 - (1-2*mask_width/height)**(1-(j+1)/num_interpol_frames))*height/2
            )
            interpol_image = interpol_image.crop((interpol_width,
                                                  interpol_width,
                                                  width - interpol_width,
                                                  height - interpol_width))

            interpol_image = interpol_image.resize((height, width))
            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round(
                (1 - (height-2*mask_width) / (height-2*interpol_width)) / 2*height
            )
            prev_image_fix_crop = shrink_and_paste_on_blank(
                prev_image_fix, interpol_width2)
            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

            all_frames.append(interpol_image)
        all_frames.append(current_image)
    video_file_name = "infinite_zoom_" + str(time.time())
    fps = 30
    save_path = output_path + video_file_name + ".mp4"
    start_frame_dupe_amount = 15
    last_frame_dupe_amount = 15

    write_video(save_path, all_frames, fps, False,
                start_frame_dupe_amount, last_frame_dupe_amount)

    ## to debug
    # img = custom_init_image.resize(
    #     (width, height), resample=Image.LANCZOS)
    # img = shrink_and_paste_on_blank(img, 128)
    # mask_image = np.array(img)[:, :, 3]
    # mask_image = Image.fromarray(255-mask_image).convert("RGB")
    
    # processed = renderImg2Img(prompts[min(k for k in prompts.keys(
    # ) if k >= 0)], negative_prompt, "Euler a", num_inference_steps, guidance_scale, width, height, img, mask_image)
    ## to debug
    return save_path , processed.images, processed.js(), plaintext_to_html(processed.info), plaintext_to_html("")


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as infinite_zoom_interface:
        gr.HTML(
            """
        <p style='text-align: center'>
        Text to Video - Infinite zoom effect
        </p>
        """
        )
        with gr.Row():
            with gr.Column(scale=1, variant='panel'):
                outpaint_prompts = gr.Dataframe(
                    type="array",
                    headers=["outpaint steps", "prompt"],
                    datatype=["number", "str"],
                    row_count=1,
                    col_count=(2, "fixed"),
                    value=[[0, default_prompt]],
                    wrap=True
                )

                outpaint_negative_prompt = gr.Textbox(
                    lines=1,
                    value=default_negative_prompt,
                    label='Negative Prompt'
                )

                outpaint_steps = gr.Slider(
                    minimum=5,
                    maximum=25,
                    step=1,
                    value=12,
                    label='Total Outpaint Steps'
                )
                with gr.Accordion("Advanced Options", open=False):
                    guidance_scale = gr.Slider(
                        minimum=0.1,
                        maximum=15,
                        step=0.1,
                        value=7,
                        label='Guidance Scale'
                    )

                    sampling_step = gr.Slider(
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=50,
                        label='Sampling Steps for each outpaint'
                    )
                    init_image = gr.Image(
                        type="pil", label="custom initial image")
                generate_btn = gr.Button(value='Generate video')

            with gr.Column(scale=1, variant='compact'):
                output_video = gr.Video(label='Output', format="mp4").style(
                    width=512, height=512, interactive=False)
                # output_video = gr.Image(label="output", interactive=False)
                out_image, generation_info, html_info, html_log = create_output_panel(
                    "infinit-zoom", output_path)
        generate_btn.click(
            fn=wrap_gradio_gpu_call(create_zoom, extra_outputs=[None, '', '']),
            inputs=[
                outpaint_prompts,
                outpaint_negative_prompt,
                outpaint_steps,
                guidance_scale,
                sampling_step,
                init_image
            ],
            outputs=[
                output_video,
                out_image,
                generation_info,
                html_info,
                html_log
            ],
        )

    return [(infinite_zoom_interface, "Infinite Zoom", "iz_interface")]


script_callbacks.on_ui_tabs(on_ui_tabs)
