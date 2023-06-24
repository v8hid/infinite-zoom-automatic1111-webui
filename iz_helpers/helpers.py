import math
import os
import modules.shared as shared
import modules.sd_models
import gradio as gr
from scripts import postprocessing_upscale
from pkg_resources import resource_filename
from .prompt_util import readJsonPrompt, process_keys
from .static_variables import jsonprompt_schemafile
import asyncio

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


def do_upscaleImg(curImg, upscale_do, upscaler_name, upscale_by):
    if not upscale_do:
        return curImg

    # ensure even width and even height for ffmpeg
    # if odd, switch to scale to mode
    rwidth = round(curImg.width * upscale_by)
    rheight = round(curImg.height * upscale_by)

    ups_mode = 2  # upscale_by
    if (rwidth % 2) == 1:
        ups_mode = 1
        rwidth += 1
    if (rheight % 2) == 1:
        ups_mode = 1
        rheight += 1

    if 1 == ups_mode:
        print(
            "Infinite Zoom: aligning output size to even width and height: "
            + str(rwidth)
            + " x "
            + str(rheight),
            end="\r",
        )
    try:
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
    except Exception as e:
        print("Infinite Zoom: upscaling failed: " + str(e))
        return curImg
    return pp.image

async def showGradioErrorAsync(txt, delay=1):
    await asyncio.sleep(delay)  # sleep for 1 second
    raise gr.Error(txt)

def validatePromptJson_throws(data):
    with open(jsonprompt_schemafile, "r") as s:
        schema = json.load(s)
    validate(instance=data, schema=schema)

def recalcPromptKeys(data):
    prompts_keys = process_keys(data)
    return prompts_keys[0]

def putPrompts(files):                
    try:
        with open(files.name, "r") as f:
            file_contents = f.read()

            data = readJsonPrompt(file_contents,False)
            prompts_keys = process_keys(data["prompts"]["data"])
            return [
                gr.Textbox.update(data["prePrompt"]),
                gr.DataFrame.update(data["prompts"]),
                gr.Textbox.update(data["postPrompt"]),
                gr.Textbox.update(data["negPrompt"]),
                gr.Slider.update(value=prompts_keys[0]),
                gr.Textbox.update(data["audioFileName"]),
                gr.Number.update(data["seed"]),
                gr.Slider.update(data["width"]),
                gr.Slider.update(data["height"]),
                gr.Dropdown.update(data["sampler"]),
                gr.Slider.update(data["guidanceScale"]),
                gr.Slider.update(data["steps"]),
                gr.Textbox.update(data["lutFileName"]),
                gr.Slider.update(data["outpaintAmount"]),
                gr.Slider.update(data["maskBlur"]),
                gr.Slider.update(data["overmask"]),
                gr.Radio.update(data["outpaintStrategy"]),
                gr.Radio.update(data["zoomMode"]),
                gr.Slider.update(data["fps"]),
                gr.Slider.update(data["zoomSpeed"]),
                gr.Slider.update(data["startFrames"]),
                gr.Slider.update(data["lastFrames"]),
                gr.Radio.update(data["blendMode"]),
                gr.ColorPicker.update(data["blendColor"]),
                gr.Slider.update(data["blendGradient"]),
                gr.Checkbox.update(data["blendInvert"])            
            ]

    except Exception:
        print(
            "[InfiniteZoom:] Loading your prompt failed. It seems to be invalid. Your prompt table is preserved."
        )
        # error only be shown with raise, so ui gets broken.
        #asyncio.run(showGradioErrorAsync("Loading your prompts failed. It seems to be invalid. Your prompt table has been preserved.",5))

        return [gr.Textbox.update(), gr.DataFrame.update(), gr.Textbox.update(),gr.Textbox.update(), gr.Textbox.update(), gr.Textbox.update(),gr.Number.update(), 
                gr.Slider.update(), gr.Slider.update(), gr.Dropdown.update(), gr.Slider.update(), gr.Slider.update(), gr.Textbox.update(), gr.Slider.update(), gr.Slider.update(), 
                gr.Slider.update(), gr.Radio.update(), gr.Radio.update(), gr.Slider.update(), gr.Slider.update(), 
                gr.Slider.update(), gr.Slider.update(), gr.Radio.update(), gr.ColorPicker.update(), gr.Slider.update(), 
                gr.Checkbox.update()]


def clearPrompts():
    return [
        gr.DataFrame.update(value=[[0, "Infinite Zoom. Start over"]]),
        gr.Textbox.update(""),
        gr.Textbox.update(""),
        gr.Textbox.update(""),
        gr.Textbox.update(None),
        gr.Number.update(-1),
        gr.Slider.update(value=768),
        gr.Slider.update(value=512),
        gr.Dropdown.update("DDIM"),
        gr.Slider.update(value=8),
        gr.Slider.update(value=35),
        gr.Textbox.update(""),
        gr.Slider.update(value=128),
        gr.Slider.update(value=48),
        gr.Slider.update(value=8),
        gr.Radio.update("Center"),
        gr.Radio.update(value=0),
        gr.Slider.update(value=30),
        gr.Slider.update(value=1),
        gr.Slider.update(value=0),
        gr.Slider.update(value=0),
        gr.Radio.update("None"),
        gr.ColorPicker.update("#ffff00"),
        gr.Slider.update(value=61),
        gr.Checkbox.update(False),
        gr.Image.update(None)
    ]

def value_to_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
    elif isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    return False

def find_ffmpeg_binary():
    try:
        import google.colab
        return 'ffmpeg'
    except:
        pass
    for package in ['imageio_ffmpeg', 'imageio-ffmpeg']:
        try:
            package_path = resource_filename(package, 'binaries')
            files = [os.path.join(package_path, f) for f in os.listdir(package_path) if f.startswith("ffmpeg-")]
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return files[0] if files else 'ffmpeg'
        except:
            return 'ffmpeg'


def renumberDataframe(data):
    # Store the first sublist as index0
    index0 = data[0]

    # Skip the first sublist (index 0)
    data = data[1:]

    try:
        # Sort the data based on the first column
        data.sort(key=lambda x: int(x[0]))

        # Renumber the index values sequentially
        for i in range(len(data)):
            data[i][0] = i + 1

    except Exception as e:
        print(f"An error occurred: {e}")
        return [index0] + data

    # Prepend index0 to the renumbered data
    data.insert(0, index0)

    return data