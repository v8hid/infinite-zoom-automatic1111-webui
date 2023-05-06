import os
from modules import scripts
import modules.sd_samplers

default_sampling_steps = 35
default_sampler = "DDIM"
default_cfg_scale = 8
default_mask_blur = 60
default_total_outpaints = 5
promptTableHeaders = ["Outpaint Steps", "Prompt", "image location", "blend mask", "is keyframe"], ["number", "str", "str", "str", "bool"]

default_prompt = """
{
    "prePrompt":"(((Best quality))), ((masterpiece)), ",
    "prompts":{
        "headers":["Start at second [0,1,...]","prompt","image location","blend mask location", "is keyframe"],
        "data":[
            [0, "Huge spectacular Waterfall in a dense tropical forest,epic perspective,(vegetation overgrowth:1.3)(intricate, ornamentation:1.1),(baroque:1.1), fantasy, (realistic:1) digital painting , (magical,mystical:1.2) , (wide angle shot:1.4), (landscape composed:1.2)(medieval:1.1), divine,cinematic,(tropical forest:1.4),(river:1.3)mythology,india, volumetric lighting, Hindu ,epic,  Alex Horley Wenjun Lin greg rutkowski Ruan Jia (Wayne Barlowe:1.2) <lora:epiNoiseoffset_v2:0.6> ","C:\\\\path\\\\to\\\\image.png", "extensions\\\\infinite-zoom-automatic1111-webui\\\\blends\\\\sun-square.png", true],
            [1, "a Lush jungle","","",false],
            [2, "a Thick rainforest","","",false],
            [4, "a Verdant canopy","","",false]
        ]
    },
    "postPrompt": "style by Alex Horley Wenjun Lin greg rutkowski Ruan Jia (Wayne Barlowe:1.2), <lora:epiNoiseoffset_v2:0.6>",
    "negPrompt": "frames, border, edges, borderline, text, character, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur, bad-artist"
}
"""

empty_prompt = (
    '{"prompts":{"data":[0,"","","",false],"headers":["Outpaintg Steps","prompt","image location", "blend mask location", "is keyframe"]},"negPrompt":"", prePrompt:"", postPrompt:""}'
)

invalid_prompt = {
    "prompts": {
        "data": [[0, "Your prompt-json is invalid, please check Settings","", "", False]],
        "headers": ["Start at second [0,1,...]", "prompt","image location","blend mask location", "is keyframe"],
    },
    "negPrompt": "Invalid prompt-json",
    "prePrompt": "Invalid prompt",
    "postPrompt": "Invalid prompt",
}

available_samplers = [
    s.name for s in modules.sd_samplers.samplers if "UniPc" not in s.name
]

current_script_dir = scripts.basedir().split(os.sep)[
    -2:
]  # contains install and our extension foldername
jsonprompt_schemafile = (
    current_script_dir[0]
    + "/"
    + current_script_dir[1]
    + "/iz_helpers/promptschema.json"
)
