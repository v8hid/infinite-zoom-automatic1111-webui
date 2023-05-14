import gradio as gr
import modules.shared as shared
from webui import wrap_gradio_gpu_call
from modules.ui import create_output_panel
from .run_interface import createZoom

from .static_variables import (
    default_prompt,
    available_samplers,
    default_total_outpaints,
    default_sampling_steps,
    default_cfg_scale,
    default_mask_blur,
    default_sampler,
    default_overmask
)
from .helpers import putPrompts, clearPrompts
from .prompt_util import readJsonPrompt
from .static_variables import promptTableHeaders


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
                    with gr.Row():
                        batchcount_slider = gr.Slider(
                            minimum=1,
                            maximum=25,
                            value=shared.opts.data.get("infzoom_batchcount", 1),
                            step=1,
                            label="Batch Count",
                        )
                        main_outpaint_steps = gr.Number(
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
                        headers=promptTableHeaders,
                        datatype=["number", "str"],
                        row_count=1,
                        col_count=(2, "fixed"),
                        value=jpr["prompts"],
                        wrap=True,
                    )

                    main_common_prompt_suf = gr.Textbox(
                        value=jpr["postPrompt"], label="Common Prompt Suffix"
                    )

                    main_negative_prompt = gr.Textbox(
                        value=jpr["negPrompt"], label="Negative Prompt"
                    )

                    # these button will be moved using JS under the dataframe view as small ones
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
                        ],
                    )

                    with gr.Accordion("Render settings"):
                        with gr.Row():
                            seed = gr.Number(
                                label="Seed", value=-1, precision=0, interactive=True
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
                            init_image = gr.Image(
                                type="pil", label="Custom initial image"
                            )
                            exit_image = gr.Image(
                                type="pil", label="Custom exit image", visible=False
                            )
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
                        maximum=60,
                        step=1
                    )
                    video_last_frame_dupe_amount = gr.Slider(
                        label="number of last frame dupe",
                        info="Frames to freeze at the end of the video",
                        value=0,
                        minimum=1,
                        maximum=60,
                        step=1
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
                    outpaint_amount_px = gr.Slider(
                        label="Outpaint pixels",
                        minimum=4,
                        maximum=508,
                        step=8,
                        value=64,
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
                        type="value" 
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
                        label="Upscale by factor",
                        minimum=1,
                        maximum=8,
                        step=0.5,
                        value=2,
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
                inpainting_mask_blur,
                inpainting_fill_mode,
                video_zoom_speed,
                seed,
                main_width,
                main_height,
                batchcount_slider,
                main_sampler,
                upscale_do,
                upscaler_name,
                upscale_by,
                overmask,
                outpaintStrategy,
                outpaint_amount_px
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
