import json
from pathlib import Path

def load_json(path, default):
    try:
        if Path(path).exists():
            with open(path, "r") as f:
                return json.load(f)
    except:
        pass
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def build_system_prompt(memory):
    return f"""
You are a fitness AI companion.

User memory:
{memory}
"""

def extract_memory_from_message(text, memory):
    t = text.lower()
    updated = dict(memory)

    if "lose weight" in t:
        updated["goal"] = "fat loss"
    if "gain muscle" in t:
        updated["goal"] = "muscle gain"
    if "beginner" in t:
        updated["level"] = "beginner"

    return updated