import json
from jsonschema import validate

from .static_variables import (
    empty_prompt,
    invalid_prompt,
    jsonprompt_schemafile
)

def completeOptionals(j):
    if isinstance(j, dict):
        if "prompts" in j:
            if "headers" not in j["prompts"]:
                j["prompts"]["headers"] = ["outpaint steps","prompt"]
        
        if "negPrompt" not in j:
            j["prompts"]["negPrompt"]=""
            
        if "prePrompt" not in j:
            j["prompts"]["prePrompt"]=""
            
        if "postPrompt" not in j:
            j["prompts"]["postPrompt"]=""

    return j


def validatePromptJson_throws(data):
    with open(jsonprompt_schemafile, "r") as s:
        schema = json.load(s)
    try:
        validate(instance=data, schema=schema)
       
    except Exception:
        raise "Your prompts are not schema valid."
        #fixJson(data)

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
    