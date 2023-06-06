import json
from msilib.schema import File
import gradio as gr
import modules.shared as shared
from webui import wrap_gradio_gpu_call
from modules.ui import create_output_panel
from .run_interface import createZoom

from .static_variables import (
    default_prompt,
    empty_prompt,
    invalid_prompt,
    available_samplers,
    default_total_outpaints,
    default_sampling_steps,
    default_cfg_scale,
    default_mask_blur,
    default_sampler,
    default_overmask,
    default_gradient_size,
    default_outpaint_amount,
)
from .helpers import validatePromptJson_throws, putPrompts, clearPrompts, renumberDataframe, closest_upper_divisible_by_eight
from .prompt_util import readJsonPrompt
from .static_variables import promptTableHeaders


def on_ui_tabs():
    main_seed = gr.Number()
    audio_filename = gr.Textbox(None)    

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
                    with gr.Row():
                        batchcount_slider = gr.Slider(
                            minimum=1,
                            maximum=25,
                            value=shared.opts.data.get("infzoom_batchcount", 1),
                            step=1,
                            label="Batch Count",
                        )

                        main_outpaint_steps = gr.Slider(
                            minimum=2,
                            maximum=120,
                            step=1,
                            label="Total video length [s]",
                            value=default_total_outpaints,
                            precision=0,
                            interactive=True,
                        )

                    # safe reading json prompt
                    pr = shared.opts.data.get("infzoom_defPrompt", default_prompt)
                    jpr = readJsonPrompt(pr, True)

                    main_common_prompt_pre = gr.Textbox(
                        value=jpr["prePrompt"], label="Common Prompt Prefix"
                    )
                    main_prompts = gr.Dataframe(
                        type="array",                   
                        headers=promptTableHeaders[0],
                        datatype=promptTableHeaders[1],
                        row_count=1,
                        col_count=(5, "fixed"),
                        value=jpr["prompts"],
                        wrap=True,
                        elem_id = "infzoom_prompt_table",                        
                    )

                    main_common_prompt_suf = gr.Textbox(
                        value=jpr["postPrompt"], label="Common Prompt Suffix"
                    )

                    main_negative_prompt = gr.Textbox(
                        value=jpr["negPrompt"], label="Negative Prompt"
                    )

                    with gr.Accordion("Render settings"):
                        with gr.Row():
                            main_seed = gr.Number(
                                label="Seed", 
                                value=jpr["seed"],
                                elem_id="infzoom_main_seed",
                                precision=0, 
                                interactive=True
                            )
                            main_sampler = gr.Dropdown(
                                label="Sampler",
                                choices=available_samplers,
                                value=default_sampler,
                                type="value",
                            )
                        with gr.Row():
                            main_width = gr.Slider(
                                minimum=16,
                                maximum=2048,
                                value=shared.opts.data.get("infzoom_outsizeW", 512),
                                step=8,
                                label="Output Width",
                            )                            
                            main_height = gr.Slider(
                                minimum=16,
                                maximum=2048,
                                value=shared.opts.data.get("infzoom_outsizeH", 512),
                                step=8,
                                label="Output Height",
                            )
                        with gr.Row():
                            main_guidance_scale = gr.Slider(
                                minimum=0.1,
                                maximum=15,
                                step=0.1,
                                value=default_cfg_scale,
                                label="Guidance Scale",
                            )
                            sampling_step = gr.Slider(
                                minimum=1,
                                maximum=150,
                                step=1,
                                value=default_sampling_steps,
                                label="Sampling Steps for each outpaint",
                            )
                        with gr.Row():
                            init_image = gr.Image(type="pil", label="custom initial image")
                            exit_image = gr.Image(type="pil", label="custom exit image")

                with gr.Tab("Video"):
                    video_frame_rate = gr.Slider(
                        label="Frames per second",
                        value=30,
                        minimum=1,
                        maximum=60,
                        step=1
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
                        maximum=120,
                        step=1
                    )
                    video_last_frame_dupe_amount = gr.Slider(
                        label="number of last frame dupe",
                        info="Frames to freeze at the end of the video",
                        value=0,
                        minimum=1,
                        maximum=120,
                        step=1
                    )
                    with gr.Row():
                        video_zoom_speed = gr.Slider(
                            label="Zoom Speed",
                            value=1.0,
                            minimum=0.1,
                            maximum=20.0,
                            step=0.1,
                            info="Zoom speed in seconds (higher values create slower zoom)",
                        )
                        video_est_length = gr.Number(
                            label="Estimated video length [s]",
                            info="a basic estimation of the video length",
                            value=1.0,
                            precision=1,
                            readonly=True,
                            id="infzoom_est_length",
                        )
                    with gr.Accordion("FFMPEG Expert", open=False):
                        gr.Markdown(
                            """# I need FFMPEG control
You can put CLI options here as documented <a href='https://ffmpeg.org/ffmpeg.html#Options'>FFMPEG OPTIONS</a> and <a href='https://ffmpeg.org/ffmpeg-filters.html'>FILTER OPTIONS</a>

## Examples:
* ```-vf crop=200:200```  crop down to 200x200 pixel from center (useful to cutoff jumpy borders)
* ```-vf scale=320:240```  scales your video to 320x240
* ```-c:v libx264 -preset veryslow -qp 0``` uses lossless compression

You might give multiple options in one line.

"""
)
                        video_ffmpeg_opts=gr.Textbox(
                            value="", label="FFMPEG Opts"
                        )
                    with gr.Accordion("Blend settings"):
                        with gr.Row():
                            blend_image = gr.Image(type="pil", label="Custom in/out Blend Image")
                            blend_mode = gr.Radio(
                                label="Blend Mode",
                                choices=["None", "Simple Blend", "Alpha Composite", "Luma Wipe"],
                                value="Luma Wipe",
                                type="index",
                            )
                            blend_invert_do = gr.Checkbox(False, label="Reverse Blend/Wipe")
                        with gr.Row():
                            blend_gradient_size = gr.Slider(
                                label="Blend Gradient size",
                                minimum=25,
                                maximum=75,
                                value=default_gradient_size,
                                step=1
                            )
                            blend_color = gr.ColorPicker(
                                label='Blend Edge Color', 
                                default='#ffff00'
                            )
                            video_zoom_speed.change(calc_est_video_length,inputs=[blend_mode,video_zoom_speed, video_start_frame_dupe_amount,video_last_frame_dupe_amount,video_frame_rate,main_outpaint_steps],outputs=[video_est_length])
                            main_outpaint_steps.change(calc_est_video_length,inputs=[blend_mode,video_zoom_speed, video_start_frame_dupe_amount,video_last_frame_dupe_amount,video_frame_rate,main_outpaint_steps],outputs=[video_est_length])
                            video_frame_rate.change(calc_est_video_length,inputs=[blend_mode,video_zoom_speed, video_start_frame_dupe_amount,video_last_frame_dupe_amount,video_frame_rate,main_outpaint_steps],outputs=[video_est_length])
                            video_start_frame_dupe_amount.change(calc_est_video_length,inputs=[blend_mode,video_zoom_speed, video_start_frame_dupe_amount,video_last_frame_dupe_amount,video_frame_rate,main_outpaint_steps],outputs=[video_est_length])
                            video_last_frame_dupe_amount.change(calc_est_video_length,inputs=[blend_mode,video_zoom_speed, video_start_frame_dupe_amount,video_last_frame_dupe_amount,video_frame_rate,main_outpaint_steps],outputs=[video_est_length])
                            blend_mode.change(calc_est_video_length,inputs=[blend_mode,video_zoom_speed, video_start_frame_dupe_amount,video_last_frame_dupe_amount,video_frame_rate,main_outpaint_steps],outputs=[video_est_length])  
                    with gr.Accordion("Blend Info", open=False):
                        gr.Markdown(
                            """# Important Blend Info:
Number of Start and Stop Frame Duplication number of frames used for the blend/wipe effect. At 30 Frames per second, 30 frames is 1 second.
Blend Gradient size determines if blends extend to the border of the images. 61 is typical, higher values may result in frames around steps of your video

Free to use grayscale blend images can be found here: https://github.com/Oncorporation/obs-studio/tree/master/plugins/obs-transitions/data/luma_wipes
Ideas for custom blend images: https://www.pexels.com/search/gradient/
"""
                        )

                with gr.Tab("Audio"):
                    with gr.Row():
                        audio_filename = gr.Textbox(
                            value=jpr["audioFileName"], 
                            label="Audio File Name",
                            elem_id="infzoom_audioFileName")
                        audio_file = gr.File(
                            value=None,
                            file_count="single",
                            file_types=["audio"],
                            type="file",
                            label="Audio File")
                        audio_file.change(get_filename, inputs=[audio_file], outputs=[audio_filename])
                    with gr.Row():
                        audio_volume = gr.Slider(
                            label="Audio volume",
                            minimum=0.0,
                            maximum=2.0,
                            step=.05,
                            value=1.0)

                with gr.Tab("Outpaint"):
                    outpaint_amount_px = gr.Slider(
                        label="Outpaint pixels",
                        minimum=8,
                        maximum=512,
                        step=8,
                        value=default_outpaint_amount,
                        elem_id="infzoom_outpaintAmount"
                    )

                    inpainting_mask_blur = gr.Slider(
                        label="Mask Blur",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=default_mask_blur,
                    )
                    overmask = gr.Slider(
                        label="Overmask (px) paint a bit into centered image",
                        minimum=0,
                        maximum=64,
                        step=1,
                        value=default_overmask,
                        elem_id="infzoom_outpaintOvermask"
                    )
                    inpainting_fill_mode = gr.Radio(
                        label="Masked content",
                        choices=["fill", "original", "latent noise", "latent nothing"],
                        value="latent noise",
                        type="index",
                    )

                    outpaintStrategy= gr.Radio(
                        label="Outpaint Strategy",
                        choices=["Center", "Corners"],
                        value="Corners",
                        type="value",
                        elem_id="infzoom_outpaintStrategy"
                    )
                    main_width.change(get_min_outpaint_amount,inputs=[main_width, outpaint_amount_px, outpaintStrategy],outputs=[outpaint_amount_px])



                with gr.Tab("Post proccess"):
                    upscale_do = gr.Checkbox(False, label="Enable Upscale")
                    upscaler_name = gr.Dropdown(
                        label="Upscaler",
                        elem_id="infZ_upscaler",
                        choices=[x.name for x in shared.sd_upscalers],
                        value=shared.sd_upscalers[0].name,
                    )
                    upscale_by = gr.Slider(
                        label="Upscale by factor",
                        minimum=1,
                        maximum=8,
                        step=0.25,
                        value=1,
                    )
                    with gr.Accordion("Help", open=False):
                        gr.Markdown(
                            """# Performance critical
Depending on amount of frames and which upscaler you choose it might took a long time to render.  
Our best experience and trade-off is the R-ERSGAn4x upscaler.
"""
                        )


                # these buttons will be moved using JS under the dataframe view as small ones
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
                    inputs=[
                        main_common_prompt_pre,
                        main_prompts,
                        main_common_prompt_suf,
                        main_negative_prompt,
                        audio_filename,
                        main_seed
                    ],
                    outputs=None,
                )
                importPrompts_button.upload(
                    fn=putPrompts,
                    outputs=[
                        main_common_prompt_pre,
                        main_prompts,
                        main_common_prompt_suf,
                        main_negative_prompt,
                        main_outpaint_steps,
                        audio_filename,
                        main_seed
                    ],
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
                    outputs=[
                        main_prompts,
                        main_negative_prompt,
                        main_common_prompt_pre,
                        main_common_prompt_suf,
                        audio_filename,
                        main_seed
                    ],
                )
                renumberPrompts_button = gr.Button(
                    value= "Renumber Prompts",
                    variant="secondary",
                    elem_classes="sm infzoom_tab_butt",
                    elem_id="infzoom_rnP_butt",
                )
                renumberPrompts_button.click(
                    fn=renumberDataframe,
                    inputs=[main_prompts],
                    outputs=[main_prompts]
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
            fn=wrap_gradio_gpu_call(createZoom, extra_outputs=[None, "", ""]),
            inputs=[
                main_common_prompt_pre,
                main_prompts,
                main_common_prompt_suf,
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
                video_ffmpeg_opts,
                inpainting_mask_blur,
                inpainting_fill_mode,
                video_zoom_speed,
                main_seed,
                main_width,
                main_height,
                batchcount_slider,
                main_sampler,
                upscale_do,
                upscaler_name,
                upscale_by,
                overmask,
                outpaintStrategy,
                outpaint_amount_px,
                blend_image,
                blend_mode,
                blend_gradient_size,
                blend_invert_do,
                blend_color,
                audio_filename,
                audio_volume,
            ],
            outputs=[output_video, out_image, generation_info, html_info, html_log],
        )

        main_prompts.change(
            fn=checkPrompts, inputs=[main_prompts], outputs=[generate_btn]
        )

        interrupt.click(fn=lambda: shared.state.interrupt(), inputs=[], outputs=[])
    infinite_zoom_interface.queue()
    return [(infinite_zoom_interface, "Infinite Zoom", "iz_interface")]


def checkPrompts(p):
    return gr.Button.update(
        interactive=any(0 in sublist for sublist in p)
        or any("0" in sublist for sublist in p)
    )

def get_filename(file):
    return file.name

def get_min_outpaint_amount(width, outpaint_amount, strategy):
    #automatically sets the minimum outpaint amount based on the width for Center strategy
    min_outpaint_px = outpaint_amount
    if strategy == "Center":
        min_outpaint_px = closest_upper_divisible_by_eight(max(outpaint_amount, width // 4))
    return min_outpaint_px

def calc_est_video_length(blend_mode, video_zoom_speed, video_start_frame_dupe_amount,video_last_frame_dupe_amount,fps, main_outpaint_steps):
    #calculates the estimated video length based on the blend mode, zoom speed, and outpaint steps
    #this is just an estimate, the actual length will vary
    steps = main_outpaint_steps
    estimate = (steps * video_zoom_speed) + ((video_start_frame_dupe_amount + video_last_frame_dupe_amount) / fps)
    if blend_mode != 0:
        steps = (main_outpaint_steps - 3)
        estimate = (steps * video_zoom_speed) + (((video_start_frame_dupe_amount + video_last_frame_dupe_amount) / fps) - 1.0)

    return estimate