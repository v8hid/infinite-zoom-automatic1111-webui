import os
from modules import scripts
import modules.sd_samplers

default_sampling_steps = 35
default_sampler = "DDIM"
default_cfg_scale = 8
default_mask_blur = 48
default_overmask = 8
default_total_outpaints = 5
promptTableHeaders = ["Start at second [0,1,...]", "Prompt"]

default_prompt = """
{
	"prePrompt": "Huge spectacular Waterfall in ",
	"prompts": {
		"data": [
			[0, "a dense tropical forest"],
			[2, "a Lush jungle"],
			[3, "a Thick rainforest"],
			[5, "a Verdant canopy"]
		]
	},
	"postPrompt": "epic perspective,(vegetation overgrowth:1.3)(intricate, ornamentation:1.1),(baroque:1.1), fantasy, (realistic:1) digital painting , (magical,mystical:1.2) , (wide angle shot:1.4), (landscape composed:1.2)(medieval:1.1),(tropical forest:1.4),(river:1.3) volumetric lighting ,epic, style by Alex Horley Wenjun Lin greg rutkowski Ruan Jia (Wayne Barlowe:1.2)",
	"negPrompt": "frames, border, edges, borderline, text, character, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur, bad-artist"
}
"""

empty_prompt = '{"prompts":{"data":[],"negPrompt":"", prePrompt:"", postPrompt:""}'

invalid_prompt = {
    "prompts": {
        "data": [[0, "Your prompt-json is invalid, please check Settings"]],
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
