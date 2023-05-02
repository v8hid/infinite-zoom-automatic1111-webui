import json
from jsonschema import validate

from .static_variables import (
    empty_prompt,
    invalid_prompt,
    jsonprompt_schemafile,
    promptTableHeaders
)

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
    
