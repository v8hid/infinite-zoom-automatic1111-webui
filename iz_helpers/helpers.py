import math
import os
import json
from jsonschema import validate
import modules.shared as shared
import modules.sd_models
import gradio as gr
from scripts import postprocessing_upscale
from .static_variables import jsonprompt_schemafile


def fix_env_Path_ffprobe():
    envpath = os.environ["PATH"]
    ffppath = shared.opts.data.get("infzoom_ffprobepath", "")

    if ffppath and not ffppath in envpath:
        path_sep = ";" if os.name == "nt" else ":"
        os.environ["PATH"] = envpath + path_sep + ffppath


def closest_upper_divisible_by_eight(num):
    if num % 8 == 0:
        return num
    else:
        return math.ceil(num / 8) * 8


def load_model_from_setting(model_field_name, progress, progress_desc):
    # fix typo in Automatic1111 vs Vlad111
    if hasattr(modules.sd_models, "checkpoint_alisases"):
        checkPList = modules.sd_models.checkpoint_alisases
    elif hasattr(modules.sd_models, "checkpoint_aliases"):
        checkPList = modules.sd_models.checkpoint_aliases
    else:
        raise Exception(
            "This is not a compatible StableDiffusion Platform, can not access checkpoints"
        )

    model_name = shared.opts.data.get(model_field_name)
    if model_name is not None and model_name != "":
        checkinfo = checkPList[model_name]

        if not checkinfo:
            raise NameError(model_field_name + " Does not exist in your models.")

        if progress:
            progress(0, desc=progress_desc + checkinfo.name)

        modules.sd_models.load_model(checkinfo)


def predict_upscalesize(width: int, height: int, scale: float):
    # ensure even width and even height for ffmpeg
    # if odd, switch to scale to mode
    rwidth = round(width * scale)
    rheight = round(height * scale)
    ups_mode = 2  # upscale_by
 
    if (rwidth % 2) == 1:
        ups_mode = 1
        rwidth += 1
    if (rheight % 2) == 1:
        ups_mode = 1
        rheight += 1
    return rwidth,rheight, ups_mode

def do_upscaleImg(curImg, upscale_do, upscaler_name, upscale_by):
    if not upscale_do:
        return curImg

    if isinstance(upscale_by, (int,float)):
        # ensure even width and even height for ffmpeg
        # if odd, switch to scale to mode
        rwidth, rheight, ups_mode = predict_upscalesize(curImg.width,curImg.height,upscale_by)

        if 1 == ups_mode:
            print(
                "Infinite Zoom: aligning output size to even width and height: "
                + str(rwidth)
                + " x "
                + str(rheight),
                end="\r",
            )

    else:
        if isinstance(upscale_by, tuple): # resize to
            rwidth,rheight = upscale_by
            ups_mode = 1

        else:
            raise f"InfZoom unexpected scaletype:{upscale_by}"

    pp = postprocessing_upscale.scripts_postprocessing.PostprocessedImage(curImg)
    ups = postprocessing_upscale.ScriptPostprocessingUpscale()
    ups.process(
        pp,
        upscale_mode=ups_mode,
        upscale_by=upscale_by,
        upscale_to_width=rwidth,
        upscale_to_height=rheight,
        upscale_crop=False,
        upscaler_1_name=upscaler_name,
        upscaler_2_name=None,
        upscaler_2_visibility=0.0,
    )
    return pp.image


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
