import gradio as gr
import modules.shared as shared
from .static_variables import default_prompt


def on_ui_settings():
    section = ("infinite-zoom", "Infinite Zoom")

    shared.opts.add_option(
        "infzoom_outpath",
        shared.OptionInfo(
            "outputs",
            "Path where to store your infinite video. Default is Outputs",
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
            lambda: {"choices": [x for x in list(shared.list_checkpoint_tiles()) if "inpainting" not in x]},
            section=section,
        ),
    )

    shared.opts.add_option(
        "infzoom_inpainting_model",
        shared.OptionInfo(
            None,
            "Name of your desired inpaint model (img2img-inpaint). Default is vanilla sd-v1-5-inpainting.ckpt ",
            gr.Dropdown,
            lambda: {"choices": [x for x in list(shared.list_checkpoint_tiles()) if "inpainting" in x]},
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
    
    
    shared.opts.add_option(
        "infzoom_collectAllResources",
        shared.OptionInfo(
            False,
            "!!! Store all images (txt2img, init_image,exit_image, inpainting, interpolation) into one folder in your OUTPUT Path. Very slow, a lot of data. Dont do this on long runs !!!",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )
     
