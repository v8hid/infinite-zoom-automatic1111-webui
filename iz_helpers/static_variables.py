import os
from modules import scripts
import modules.sd_samplers

default_prompt = """
{
    "commonPromptPrefix":"<lora:epiNoiseoffset_v2:0.6> ",
    "prompts":{
        "headers":["outpaint steps","prompt","image location","blend mask location", "is keyframe"],
        "data":[
            [0,"Huge spectacular Waterfall in a dense tropical forest,epic perspective,(vegetation overgrowth:1.3)(intricate, ornamentation:1.1),(baroque:1.1), fantasy, (realistic:1) digital painting , (magical,mystical:1.2) , (wide angle shot:1.4), (landscape composed:1.2)(medieval:1.1), divine,cinematic,(tropical forest:1.4),(river:1.3)mythology,india, volumetric lighting, Hindu ,epic,  Alex Horley Wenjun Lin greg rutkowski Ruan Jia (Wayne Barlowe:1.2) <lora:epiNoiseoffset_v2:0.6> ","C:\\path\\to\\image.png", "C:\\path\\to\\mask_image.png", false]
        ]
    },
    "commonPromptSuffix":"style by Alex Horley Wenjun Lin greg rutkowski Ruan Jia (Wayne Barlowe:1.2)",
    "negPrompt":"frames, border, edges, text, character, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur bad-artist"
}
"""

empty_prompt = (
    '{"prompts":{"data":[],"headers":["outpaint steps","prompt","image location", "blend mask location", "is keyframe"]},"negPrompt":"", commonPromptPrefix:"", commonPromptSuffix:""}'
)

invalid_prompt = {
    "prompts": {
        "data": [[0, "Your prompt-json is invalid, please check Settings","", "", False]],
        "headers": ["outpaint steps", "prompt","image location","blend mask location", "is keyframe"],
    },
    "negPrompt": "Invalid prompt-json",
    "commonPromptPrefix": "Invalid prompt",
    "commonPromptSuffix": "Invalid prompt"
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
