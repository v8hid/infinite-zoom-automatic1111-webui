import gradio as gr
from types import SimpleNamespace
import sys
import os
from modules import script_callbacks
from webui import wrap_gradio_gpu_call
basedirs = [os.getcwd()]


def zoom(
    model_id,
    prompts_array,
    negative_prompt,
    num_outpainting_steps,
    guidance_scale,
    num_inference_steps,
    custom_init_image
):
    pass


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as infinite_zoom_interface:
        gr.HTML(
            """
        <p style='text-align: center'>
       Text to Video - Infinite zoom effect
        </p>
        """
        )
    # with gr.Blocks():
        with gr.Row():
            with gr.Column():
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
                    model_id = gr.Dropdown(
                        choices=inpaint_model_list,
                        value=inpaint_model_list[0],
                        label='Pre-trained Model ID'
                    )

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

            with gr.Column():
                output_video = gr.Video(label='Output', format="mp4").style(
                    width=512, height=512)

    generate_btn.click(
        fn=wrap_gradio_gpu_call(zoom, extra_outputs=[None, '', '']),
        inputs=[
            model_id,
            outpaint_prompts,
            outpaint_negative_prompt,
            outpaint_steps,
            guidance_scale,
            sampling_step,
            init_image
        ],
        outputs=[
            output_video,
        ],
    )

    return [(infinite_zoom_interface, "Infinite Zoom", "iz_interface")]


script_callbacks.on_ui_tabs(on_ui_tabs)
