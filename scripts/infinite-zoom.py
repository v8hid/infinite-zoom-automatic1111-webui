import sys
import os
import time
import json
from jsonschema import validate

import numpy as np
import gradio as gr
from PIL import Image
import math
import json

from iz_helpers import shrink_and_paste_on_blank, write_video
from webui import wrap_gradio_gpu_call
from modules import script_callbacks, scripts
import modules.shared as shared
from modules.processing import (
    process_images,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
)

from scripts import postprocessing_upscale

from modules.ui import create_output_panel, plaintext_to_html
import modules.sd_models
import modules.sd_samplers

from modules import scripts

usefulDirs = scripts.basedir().split(os.sep)[
    -2:
]  # contains install and our extension foldername
jsonprompt_schemafile = (
    usefulDirs[0] + "/" + usefulDirs[1] + "/scripts/promptschema.json"
)

available_samplers = [s.name for s in modules.sd_samplers.samplers]

default_prompt = """
{
    "prompts":{
        "headers":["outpaint steps","prompt"],
        "data":[
            [0,"Cat"],
            ["1","Dog"],
            [2,"Happy Pets"]
        ]
    },
    "negPrompt":"ugly"
}
"""

empty_prompt = (
    '{"prompts":{"data":[],"headers":["outpaint steps","prompt"]},"negPrompt":""}'
)

# must be python dict
invalid_prompt = {
    "prompts": {
        "data": [[0, "Your prompt-json is invalid, please check Settings"]],
        "headers": ["outpaint steps", "prompt"],
    },
    "negPrompt": "Invalid prompt-json",
}


def closest_upper_divisible_by_eight(num):
    if num % 8 == 0:
        return num
    else:
        return math.ceil(num / 8) * 8


def do_upscaleImg(curImg, upscale_do, upscaler_name, upscale_by):
    if not upscale_do:
        return curImg
    pp = postprocessing_upscale.scripts_postprocessing.PostprocessedImage(
        curImg
    )
    ups = postprocessing_upscale.ScriptPostprocessingUpscale()
    ups.process(
        pp,
        upscale_mode=2,
        upscale_by=upscale_by,
        upscale_to_width=None,
        upscale_to_height=None,
        upscale_crop=False,
        upscaler_1_name=upscaler_name,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    )
    return pp.image


def renderTxt2Img(prompt, negative_prompt, sampler, steps, cfg_scale, width, height):
    processed = None
    p = StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_txt2img_grids,
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
    processed = process_images(p)
    return processed


def renderImg2Img(
    prompt,
    negative_prompt,
    sampler,
    steps,
    cfg_scale,
    width,
    height,
    init_image,
    mask_image,
    inpainting_denoising_strength,
    inpainting_mask_blur,
    inpainting_fill_mode,
    inpainting_full_res,
    inpainting_padding,
):
    processed = None

    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_img2img_samples,
        outpath_grids=shared.opts.outdir_img2img_grids,
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
        denoising_strength=inpainting_denoising_strength,
        mask_blur=inpainting_mask_blur,
        inpainting_fill=inpainting_fill_mode,
        inpaint_full_res=inpainting_full_res,
        inpaint_full_res_padding=inpainting_padding,
        mask=mask_image,
    )
    # p.latent_mask = Image.new("RGB", (p.width, p.height), "white")

    processed = process_images(p)
    return processed


def fix_env_Path_ffprobe():
    envpath = os.environ["PATH"]
    ffppath = shared.opts.data.get("infzoom_ffprobepath", "")

    if ffppath and not ffppath in envpath:
        path_sep = ";" if os.name == "nt" else ":"
        os.environ["PATH"] = envpath + path_sep + ffppath


def load_model_from_setting(model_field_name, progress, progress_desc):
    # fix typo in Automatic1111 vs Vlad111
    if hasattr(modules.sd_models, "checkpoint_alisases"):
        checkPList = modules.sd_models.checkpoint_alisases
    elif hasattr(modules.sd_models, "checkpoint_aliases"):
        checkPList = modules.sd_models.checkpoint_aliases
    else:
        raise Exception("This is not a compatible StableDiffusion Platform, can not access checkpoints")
    
    model_name = shared.opts.data.get(model_field_name)
    if model_name is not None and model_name != "":
        checkinfo = checkPList[model_name]

        if not checkinfo:
            raise NameError(model_field_name + " Does not exist in your models.")

        if progress: 
            progress(0, desc=progress_desc + checkinfo.name)

        modules.sd_models.load_model(checkinfo)


def create_zoom(
    prompts_array,
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
    inpainting_denoising_strength,
    inpainting_mask_blur,
    inpainting_fill_mode,
    inpainting_full_res,
    inpainting_padding,
    zoom_speed,
    outputsizeW,
    outputsizeH,
    batchcount,
    sampler,
    upscale_do,
    upscaler_name,
    upscale_by,
    progress=None,
):
    for i in range(batchcount):
        print(f"Batch {i+1}/{batchcount}")
        result = create_zoom_single(
            prompts_array,
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
            inpainting_denoising_strength,
            inpainting_mask_blur,
            inpainting_fill_mode,
            inpainting_full_res,
            inpainting_padding,
            zoom_speed,
            outputsizeW,
            outputsizeH,
            sampler,
            upscale_do,
            upscaler_name,
            upscale_by,
            progress,
        )
    return result


def create_zoom_single(
    prompts_array,
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
    inpainting_denoising_strength,
    inpainting_mask_blur,
    inpainting_fill_mode,
    inpainting_full_res,
    inpainting_padding,
    zoom_speed,
    outputsizeW,
    outputsizeH,
    sampler,
    upscale_do,
    upscaler_name,
    upscale_by,
    progress=None,
):
    # try:
    #     if gr.Progress() is not None:
    #         progress = gr.Progress()
    #         progress(0, desc="Preparing Initial Image")
    # except Exception:
    #     pass
    fix_env_Path_ffprobe()

    prompts = {}
    for x in prompts_array:
        try:
            key = int(x[0])
            value = str(x[1])
            prompts[key] = value
        except ValueError:
            pass
    assert len(prompts_array) > 0, "prompts is empty"

    width = closest_upper_divisible_by_eight(outputsizeW)
    height = closest_upper_divisible_by_eight(outputsizeH)

    current_image = Image.new(mode="RGBA", size=(width, height))
    mask_image = np.array(current_image)[:, :, 3]
    mask_image = Image.fromarray(255 - mask_image).convert("RGB")
    current_image = current_image.convert("RGB")

    if custom_init_image:
        current_image = custom_init_image.resize(
            (width, height), resample=Image.LANCZOS
        )
    else:
        load_model_from_setting("infzoom_txt2img_model", progress, "Loading Model for txt2img: ")

        processed = renderTxt2Img(
            prompts[min(k for k in prompts.keys() if k >= 0)],
            negative_prompt,
            sampler,
            num_inference_steps,
            guidance_scale,
            width,
            height,
        )
        current_image = processed.images[0]

    mask_width = math.trunc(width / 4)  # was initially 512px => 128px
    mask_height = math.trunc(height / 4)  # was initially 512px => 128px

    num_interpol_frames = round(video_frame_rate * zoom_speed)

    all_frames = []

    if upscale_do and progress:
        progress(0, desc="upscaling inital image")

    all_frames.append(
        do_upscaleImg(current_image, upscale_do, upscaler_name, upscale_by)
        if upscale_do
        else current_image
    )

    load_model_from_setting("infzoom_inpainting_model", progress, "Loading Model for inpainting/img2img: " )

    for i in range(num_outpainting_steps):
        print_out = "Outpaint step: " + str(i + 1) + " / " + str(num_outpainting_steps)
        print(print_out)
        if progress:
            progress(((i + 1) / num_outpainting_steps), desc=print_out)

        prev_image_fix = current_image
        prev_image = shrink_and_paste_on_blank(current_image, mask_width, mask_height)
        current_image = prev_image

        # create mask (black image with white mask_width width edges)
        mask_image = np.array(current_image)[:, :, 3]
        mask_image = Image.fromarray(255 - mask_image).convert("RGB")

        # inpainting step
        current_image = current_image.convert("RGB")

        processed = renderImg2Img(
            prompts[max(k for k in prompts.keys() if k <= i)],
            negative_prompt,
            sampler,
            num_inference_steps,
            guidance_scale,
            width,
            height,
            current_image,
            mask_image,
            inpainting_denoising_strength,
            inpainting_mask_blur,
            inpainting_fill_mode,
            inpainting_full_res,
            inpainting_padding,
        )
        current_image = processed.images[0]

        current_image.paste(prev_image, mask=prev_image)

        # interpolation steps between 2 inpainted images (=sequential zoom and crop)
        for j in range(num_interpol_frames - 1):
            interpol_image = current_image

            interpol_width = round(
                (
                    1
                    - (1 - 2 * mask_width / width)
                    ** (1 - (j + 1) / num_interpol_frames)
                )
                * width
                / 2
            )

            interpol_height = round(
                (
                    1
                    - (1 - 2 * mask_height / height)
                    ** (1 - (j + 1) / num_interpol_frames)
                )
                * height
                / 2
            )

            interpol_image = interpol_image.crop(
                (
                    interpol_width,
                    interpol_height,
                    width - interpol_width,
                    height - interpol_height,
                )
            )

            interpol_image = interpol_image.resize((width, height))

            # paste the higher resolution previous image in the middle to avoid drop in quality caused by zooming
            interpol_width2 = round(
                (1 - (width - 2 * mask_width) / (width - 2 * interpol_width))
                / 2
                * width
            )

            interpol_height2 = round(
                (1 - (height - 2 * mask_height) / (height - 2 * interpol_height))
                / 2
                * height
            )

            prev_image_fix_crop = shrink_and_paste_on_blank(
                prev_image_fix, interpol_width2, interpol_height2
            )

            interpol_image.paste(prev_image_fix_crop, mask=prev_image_fix_crop)

            if upscale_do and progress:
                progress(((i + 1) / num_outpainting_steps), desc="upscaling interpol")

            all_frames.append(
                do_upscaleImg(interpol_image, upscale_do, upscaler_name, upscale_by)
                if upscale_do
                else interpol_image
            )

        if upscale_do and progress:
            progress(((i + 1) / num_outpainting_steps), desc="upscaling current")

        all_frames.append(
            do_upscaleImg(current_image, upscale_do, upscaler_name, upscale_by)
            if upscale_do
            else current_image
        )

    video_file_name = "infinite_zoom_" + str(int(time.time())) + ".mp4"
    output_path = shared.opts.data.get(
        "infzoom_outpath", shared.opts.data.get("outdir_img2img_samples")
    )
    save_path = os.path.join(
        output_path, shared.opts.data.get("infzoom_outSUBpath", "infinite-zooms")
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    out = os.path.join(save_path, video_file_name)
    write_video(
        out,
        all_frames,
        video_frame_rate,
        video_zoom_mode,
        int(video_start_frame_dupe_amount),
        int(video_last_frame_dupe_amount),
    )

    return (
        out,
        processed.images,
        processed.js(),
        plaintext_to_html(processed.info),
        plaintext_to_html(""),
    )


def validatePromptJson_throws(data):
    with open(jsonprompt_schemafile, "r") as s:
        schema = json.load(s)
    validate(instance=data, schema=schema)


def putPrompts(files):
    try:
        with open(files.name, "r") as f:
            file_contents = f.read()
            data = json.loads(file_contents)
            validatePromptJson_throws(data)
            return [
                gr.DataFrame.update(data["prompts"]),
                gr.Textbox.update(data["negPrompt"]),
            ]

    except Exception:
        gr.Error(
            "loading your prompt failed. It seems to be invalid. Your prompt table is preserved."
        )
        print(
            "[InfiniteZoom:] Loading your prompt failed. It seems to be invalid. Your prompt table is preserved."
        )
        return [gr.DataFrame.update(), gr.Textbox.update()]


def clearPrompts():
    return [
        gr.DataFrame.update(value=[[0, "Infinite Zoom. Start over"]]),
        gr.Textbox.update(""),
    ]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as infinite_zoom_interface:
        gr.HTML(
            """
            <p style="text-align: center;">
                <a target="_blank" href="https://github.com/v8hid/infinite-zoom-automatic1111-webui"><img src="https://img.shields.io/static/v1?label=github&message=repository&color=blue&style=flat&logo=github&logoColor=white" style="display: inline;" alt="GitHub Repo"/></a>
                <a href="https://discord.gg/v2nHqSrWdW"><img src="https://img.shields.io/discord/1095469311830806630?color=blue&label=discord&logo=discord&logoColor=white" style="display: inline;" alt="Discord server"></a>
            </p>

            """
        )
        with gr.Row():
            generate_btn = gr.Button(value="Generate video", variant="primary")
            interrupt = gr.Button(value="Interrupt", elem_id="interrupt_training")
        with gr.Row():
            with gr.Column(scale=1, variant="panel"):
                with gr.Tab("Main"):
                    main_outpaint_steps = gr.Slider(
                        minimum=2,
                        maximum=100,
                        step=1,
                        value=8,
                        label="Total Outpaint Steps",
                        info="The more it is, the longer your videos will be",
                    )

                    # safe reading json prompt
                    pr = shared.opts.data.get("infzoom_defPrompt", default_prompt)
                    if not pr:
                        pr = empty_prompt

                    try:
                        jpr = json.loads(pr)
                        validatePromptJson_throws(jpr)
                    except Exception:
                        jpr = invalid_prompt

                    main_prompts = gr.Dataframe(
                        type="array",
                        headers=["outpaint step", "prompt"],
                        datatype=["number", "str"],
                        row_count=1,
                        col_count=(2, "fixed"),
                        value=jpr["prompts"],
                        wrap=True,
                    )

                    main_negative_prompt = gr.Textbox(
                        value=jpr["negPrompt"], label="Negative Prompt"
                    )

                    # these button will be moved using JS unde the dataframe view as small ones
                    exportPrompts_button = gr.Button(
                        value="Export prompts",
                        variant="secondary",
                        elem_classes="sm infzoom_tab_butt",
                        elem_id="infzoom_exP_butt",
                    )
                    importPrompts_button = gr.UploadButton(
                        label="Import prompts",
                        variant="secondary",
                        elem_classes="sm infzoom_tab_butt",
                        elem_id="infzoom_imP_butt",
                    )
                    exportPrompts_button.click(
                        None,
                        _js="exportPrompts",
                        inputs=[main_prompts, main_negative_prompt],
                        outputs=None,
                    )
                    importPrompts_button.upload(
                        fn=putPrompts,
                        outputs=[main_prompts, main_negative_prompt],
                        inputs=[importPrompts_button],
                    )

                    clearPrompts_button = gr.Button(
                        value="Clear prompts",
                        variant="secondary",
                        elem_classes="sm infzoom_tab_butt",
                        elem_id="infzoom_clP_butt",
                    )
                    clearPrompts_button.click(
                        fn=clearPrompts,
                        inputs=[],
                        outputs=[main_prompts, main_negative_prompt],
                    )

                    main_sampler = gr.Dropdown(
                        label="Sampler",
                        choices=available_samplers,
                        value="Euler a",
                        type="value",
                    )
                    with gr.Row():
                        main_width = gr.Slider(
                            minimum=16,
                            maximum=2048,
                            value=shared.opts.data.get("infzoom_outsizeW", 512),
                            step=16,
                            label="Output Width",
                        )
                        main_height = gr.Slider(
                            minimum=16,
                            maximum=2048,
                            value=shared.opts.data.get("infzoom_outsizeH", 512),
                            step=16,
                            label="Output Height",
                        )
                    with gr.Row():
                        main_guidance_scale = gr.Slider(
                            minimum=0.1,
                            maximum=15,
                            step=0.1,
                            value=7,
                            label="Guidance Scale",
                        )
                        sampling_step = gr.Slider(
                            minimum=1,
                            maximum=100,
                            step=1,
                            value=50,
                            label="Sampling Steps for each outpaint",
                        )
                    with gr.Row():
                        init_image = gr.Image(type="pil", label="custom initial image")
                        exit_image = gr.Image(
                            type="pil", label="custom exit image", visible=False
                        )  # TODO: implement exit-image rendering

                    batchcount_slider = gr.Slider(
                        minimum=1,
                        maximum=25,
                        value=shared.opts.data.get("infzoom_batchcount", 1),
                        step=1,
                        label="Batch Count",
                    )
                with gr.Tab("Video"):
                    video_frame_rate = gr.Slider(
                        label="Frames per second",
                        value=30,
                        minimum=1,
                        maximum=60,
                    )
                    video_zoom_mode = gr.Radio(
                        label="Zoom mode",
                        choices=["Zoom-out", "Zoom-in"],
                        value="Zoom-out",
                        type="index",
                    )
                    video_start_frame_dupe_amount = gr.Slider(
                        label="number of start frame dupe",
                        info="Frames to freeze at the start of the video",
                        value=0,
                        minimum=1,
                        maximum=60,
                    )
                    video_last_frame_dupe_amount = gr.Slider(
                        label="number of last frame dupe",
                        info="Frames to freeze at the end of the video",
                        value=0,
                        minimum=1,
                        maximum=60,
                    )
                    video_zoom_speed = gr.Slider(
                        label="Zoom Speed",
                        value=1.0,
                        minimum=0.1,
                        maximum=20.0,
                        step=0.1,
                        info="Zoom speed in seconds (higher values create slower zoom)",
                    )

                with gr.Tab("Outpaint"):
                    inpainting_denoising_strength = gr.Slider(
                        label="Denoising Strength", minimum=0.75, maximum=1, value=1
                    )
                    inpainting_mask_blur = gr.Slider(
                        label="Mask Blur", minimum=0, maximum=64, value=0
                    )
                    inpainting_fill_mode = gr.Radio(
                        label="Masked content",
                        choices=["fill", "original", "latent noise", "latent nothing"],
                        value="latent noise",
                        type="index",
                    )
                    inpainting_full_res = gr.Checkbox(label="Inpaint Full Resolution")
                    inpainting_padding = gr.Slider(
                        label="masked padding", minimum=0, maximum=256, value=0
                    )

                with gr.Tab("Post proccess"):
                    upscale_do = gr.Checkbox(False, label="Enable Upscale")
                    upscaler_name = gr.Dropdown(
                        label="Upscaler",
                        elem_id="infZ_upscaler",
                        choices=[x.name for x in shared.sd_upscalers],
                        value=shared.sd_upscalers[0].name,
                    )

                    upscale_by = gr.Slider(
                        label="Upscale by factor", minimum=1, maximum=8, value=1
                    )
                    with gr.Accordion("Help", open=False):
                        gr.Markdown(
                            """# Performance critical
Depending on amount of frames and which upscaler you choose it might took a long time to render.  
Our best experience and trade-off is the R-ERSGAn4x upscaler.
"""
                        )

            with gr.Column(scale=1, variant="compact"):
                output_video = gr.Video(label="Output").style(width=512, height=512)
                (
                    out_image,
                    generation_info,
                    html_info,
                    html_log,
                ) = create_output_panel(
                    "infinite-zoom", shared.opts.outdir_img2img_samples
                )
        generate_btn.click(
            fn=wrap_gradio_gpu_call(create_zoom, extra_outputs=[None, "", ""]),
            inputs=[
                main_prompts,
                main_negative_prompt,
                main_outpaint_steps,
                main_guidance_scale,
                sampling_step,
                init_image,
                exit_image,
                video_frame_rate,
                video_zoom_mode,
                video_start_frame_dupe_amount,
                video_last_frame_dupe_amount,
                inpainting_denoising_strength,
                inpainting_mask_blur,
                inpainting_fill_mode,
                inpainting_full_res,
                inpainting_padding,
                video_zoom_speed,
                main_width,
                main_height,
                batchcount_slider,
                main_sampler,
                upscale_do,
                upscaler_name,
                upscale_by,
            ],
            outputs=[output_video, out_image, generation_info, html_info, html_log],
        )
        interrupt.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])
    infinite_zoom_interface.queue()
    return [(infinite_zoom_interface, "Infinite Zoom", "iz_interface")]


def on_ui_settings():
    section = ("infinite-zoom", "Infinite Zoom")

    shared.opts.add_option(
        "infzoom_outpath",
        shared.OptionInfo(
            "",
            "Path where to store your infinite video. Let empty to use img2img-output",
            gr.Textbox,
            {"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_outSUBpath",
        shared.OptionInfo(
            "infinite-zooms",
            "Which subfolder name to be created in the outpath. Default is 'infinite-zooms'",
            gr.Textbox,
            {"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_outsizeW",
        shared.OptionInfo(
            512,
            "Default width of your video",
            gr.Slider,
            {"minimum": 16, "maximum": 2048, "step": 16},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_outsizeH",
        shared.OptionInfo(
            512,
            "Default height your video",
            gr.Slider,
            {"minimum": 16, "maximum": 2048, "step": 16},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_ffprobepath",
        shared.OptionInfo(
            "",
            "Writing videos has  dependency to an existing FFPROBE executable on your machine. D/L here (https://github.com/BtbN/FFmpeg-Builds/releases) your OS variant and point to your installation path",
            gr.Textbox,
            {"interactive": True},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_txt2img_model",
        shared.OptionInfo(
            None,
            "Name of your desired model to render keyframes (txt2img)",
            gr.Dropdown,
            lambda: {"choices": shared.list_checkpoint_tiles()},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_inpainting_model",
        shared.OptionInfo(
            None,
            "Name of your desired inpaint model (img2img-inpaint). Default is vanilla sd-v1-5-inpainting.ckpt ",
            gr.Dropdown,
            lambda: {"choices": shared.list_checkpoint_tiles()},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_defPrompt",
        shared.OptionInfo(
            default_prompt,
            "Default prompt-setup to start with'",
            gr.Code,
            {"interactive": True, "language": "json"},
            section=section,
        ),
    )


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
