import os
from modules import scripts
import modules.sd_samplers

default_prompt = """
{
	"commonPromptPrefix": "Huge spectacular Waterfall in ",
	"prompts": {
		"headers": ["outpaint steps", "prompt"],
		"data": [
			[0, "a dense tropical forest"],
			[2, "a Lush jungle"],
			[3, "a Thick rainforest"],
			[5, "a Verdant canopy"]
		]
	},
	"commonPromptSuffix": "epic perspective,(vegetation overgrowth:1.3)(intricate, ornamentation:1.1),(baroque:1.1), fantasy, (realistic:1) digital painting , (magical,mystical:1.2) , (wide angle shot:1.4), (landscape composed:1.2)(medieval:1.1),(tropical forest:1.4),(river:1.3) volumetric lighting ,epic, style by Alex Horley Wenjun Lin greg rutkowski Ruan Jia (Wayne Barlowe:1.2)",
	"negPrompt": "frames, border, edges, borderline, text, character, duplicate, error, out of frame, watermark, low quality, ugly, deformed, blur, bad-artist"
}
"""

empty_prompt = '{"prompts":{"data":[],"headers":["outpaint steps","prompt"]},"negPrompt":"", commonPromptPrefix:"", commonPromptSuffix}'

invalid_prompt = {
    "prompts": {
        "data": [[0, "Your prompt-json is invalid, please check Settings"]],
        "headers": ["outpaint steps", "prompt"],
    },
    "negPrompt": "Invalid prompt-json",
    "commonPromptPrefix": "Invalid prompt",
    "commonPromptSuffix": "Invalid prompt",
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
