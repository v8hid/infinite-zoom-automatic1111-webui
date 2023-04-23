import json
import gradio as gr
from .run import create_zoom
import modules.shared as shared
from webui import wrap_gradio_gpu_call
from modules.ui import create_output_panel
from .static_variables import (
    default_prompt,
    empty_prompt,
    invalid_prompt,
    available_samplers,
)
from .helpers import validatePromptJson_throws, putPrompts, clearPrompts


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
                        headers=["outpaint step", "prompt", "image location"],
                        datatype=["number", "str", "str"],
                        row_count=1,
                        col_count=(3, "fixed"),
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
                    with gr.Row():
                        seed = gr.Number(
                            label="Seed", value=-1, precision=0, interactive=True
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
                        exit_image = gr.Image(type="pil", label="custom exit image")

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
                seed,
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
