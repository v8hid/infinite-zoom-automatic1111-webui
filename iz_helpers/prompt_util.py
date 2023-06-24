import json
from jsonschema import validate

from .static_variables import (
    empty_prompt,
    invalid_prompt,
    jsonprompt_schemafile,
    promptTableHeaders,
    default_total_outpaints,
    default_outpaint_amount,
    default_gradient_size,
    default_mask_blur,
    default_sampler,
    default_cfg_scale,
    default_sampling_steps,
    default_overmask
)
prompts_keys = (default_total_outpaints, default_total_outpaints)

def process_keys(data):
    #data = json.loads(txt)['data']
    keys = [int(sublist[0]) for sublist in data]
    max_key = max(keys)
    num_keys = len(keys)
    return (max_key, num_keys)

def completeOptionals(j):
    if isinstance(j, dict):
        # Remove header information, user dont pimp our ui
        if "prompts" in j:
            if "headers" in j["prompts"]:
                del j["prompts"]["headers"]
            j["prompts"]["headers"]=promptTableHeaders[0]
            
        if "negPrompt" not in j:
            j["negPrompt"]=""
            
        if "prePrompt" not in j:
            if "commonPromptPrefix" in j:
                j["prePrompt"]=j["commonPromptPrefix"]
            else:
                j["prePrompt"]=""
            
        if "postPrompt" not in j:
            if "commonPromptSuffix" in j:
                j["postPrompt"]=j["commonPromptSuffix"]
            else:
                j["postPrompt"]=""

        if "audioFileName" not in j:
                j["audioFileName"]= None

        if "seed" not in j:
                j["seed"]= -1

        if "width" not in j:
            j["width"]= 768

        if "height" not in j:
            j["height"]= 512

        if "sampler" not in j:
            j["sampler"]= default_sampler

        if "guidanceScale" not in j:
            j["guidanceScale"]= default_cfg_scale

        if "steps" not in j:
            j["steps"]= default_sampling_steps

        if "lutFileName" not in j:
            j["lutFileName"]= None

        if "outpaintAmount" not in j:
            j["outpaintAmount"]= default_outpaint_amount

        if "maskBlur" not in j:
            j["maskBlur"]= default_mask_blur

        if "overmask" not in j:
                j["overmask"]= default_overmask

        if "outpaintStrategy" not in j:
            j["outpaintStrategy"]= "Corners"

        if "zoomMode" not in j:
            j["zoomMode"]= "Zoom-out"

        if "fps" not in j:
            j["fps"]= 30

        if "zoomSpeed" not in j:
            j["zoomSpeed"]= 1

        if "startFrames" not in j:
            j["startFrames"]= 0

        if "lastFrames" not in j:
            j["lastFrames"]= 0

        if "blendMode" not in j:
            j["blendMode"]= "Not Used"

        if "blendColor" not in j:
            j["blendColor"]= "#ffff00"

        if "blendGradient" not in j:
            j["blendGradient"]= default_gradient_size

        if "blendInvert" not in j:
            j["blendInvert"]= False
        

    return j

def validatePromptJson_throws(data):
    with open(jsonprompt_schemafile, "r") as s:
        schema = json.load(s)
    try:
        validate(instance=data, schema=schema)
       
    except Exception:
        raise Exception("Your prompts are not schema valid.")

    return completeOptionals(data)


def readJsonPrompt(txt, returnFailPrompt=False):
    if not txt:
        return empty_prompt

    try:
        jpr = json.loads(txt)
    except Exception:
        if returnFailPrompt:
            print (f"Infinite Zoom: Corrupted Json structure: {txt[:24]} ...")
            return invalid_prompt
        raise (f"Infinite Zoom: Corrupted Json structure: {txt[:24]} ...")

    try:
        return validatePromptJson_throws(jpr)
    except Exception:
        if returnFailPrompt:
            return invalid_prompt
        pass
    
