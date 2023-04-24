import json
from jsonschema import validate

from .static_variables import (
    empty_prompt,
    invalid_prompt,
    jsonprompt_schemafile
)

"""
    json is valid, but not our current schema.
    lets try something. 
    does it look like something usable?
def fixJson(j):
    fixedJ = empty_prompt
    try:
        if isinstance(j, dict):
            if "prompts" in j:
                if "data" in j["prompts"]:
                    if isinstance (j["prompts"]["data"],list):
                        fixedJ["prompts"]["data"] = j["prompts"]["data"]
                        if not isinstance (fixedJ["prompts"]["data"][0].

                if "headers" not in j["prompts"]:
                    fixedJ["prompts"]["headers"] = ["outpaint steps","prompt"]
                else:
                    fixedJ["prompts"]["headers"] = j["prompts"]["headers"]

            if "negPrompt" in j:
                fixedJ["prompts"]["headers"]
            
            if "commonPrompt" in j:
                return j
    except Exception:
        raise "JsonFix: Failed on recovering json prompt"
    return j
"""

def fixHeaders(j):
    if isinstance(j, dict):
        if "prompts" in j:
            if "headers" not in j["prompts"]:
                j["prompts"]["headers"] = ["outpaint steps","prompt"]
    return j


def validatePromptJson_throws(data):
    with open(jsonprompt_schemafile, "r") as s:
        schema = json.load(s)
    try:
        validate(instance=data, schema=schema)
    except Exception:
        raise "Your prompts are not schema valid."
        #fixJson(data)

    return fixHeaders(data)


def readJsonPrompt(txt, returnFailPrompt=False):
    if not txt:
        return empty_prompt

    try:
        jpr = json.loads(txt)
    except Exception:
        if returnFailPrompt is not None:
            print (f"Infinite Zoom: Corrupted Json structure: {txt[:24]} ...")
            return invalid_prompt
        raise (f"Infinite Zoom: Corrupted Json structure: {txt[:24]} ...")

    return validatePromptJson_throws(jpr)
