"""Parse released natural-language prompts without target-specific cases."""
from __future__ import annotations

import re

import numpy as np

ORDINALS = {name: index for index, name in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth".split(), 1)}


def heading(text):
    value = text.lower()
    for name, pattern in (
        ("left", r"front\s+end\s+oriented\s+to\s+the\s+left|facing\s+left|faces?\s+left|oriented\s+to\s+the\s+left|leftward\s+orientation"),
        ("right", r"facing\s+right|faces?\s+right|oriented\s+to\s+the\s+right|rightward\s+orientation"),
        ("back", r"facing\s+backward|faces\s+back(?:ward)?|back\s+facing\s+the\s+camera|oriented\s+backwards|facing\s+back"),
        ("front", r"facing\s+front|facing\s+forward|faces?\s+front|faces?\s+forward|oriented\s+front|oriented\s+forward|front-facing|forward\s+orientation"),
    ):
        if re.search(pattern, value):
            return name
    return "unknown"


def category(text):
    value = text.lower()
    patterns = (
        (r"upper\s+left", "upper_left"), (r"upper\s+right", "upper_right"),
        (r"lower\s+left", "lower_left"), (r"lower\s+right", "lower_right"),
        (r"middle\s+upper", "upper_center"), (r"middle\s+lower", "lower_center"),
        (r"left\s+central|central\s+left", "middle_left"),
        (r"right\s+central|central\s+right", "middle_right"),
        (r"left\s+(?:corner|side)", "middle_left"),
        (r"right\s+(?:corner|side)", "middle_right"),
        (r"middle(?:\s+of)?\s+(?:the\s+)?(?:video|frame|screen|area|section)", "middle_center"),
        (r"upper", "upper_center"), (r"lower", "lower_center"),
    )
    matches = []
    for index, (pattern, name) in enumerate(patterns):
        match = re.search(pattern, value)
        if match:
            matches.append((match.start(), index, name))
    return min(matches)[2] if matches else "middle_center"


def segment_texts(description):
    marks = list(re.finditer(r"as the video progresses|as the video continues|towards? the end|towards? (?:the )?video's conclusion|by the end(?: of the video)?",
                             description.lower()))
    if not marks:
        return [description] * 3
    first = marks[0].start()
    second = marks[1].start() if len(marks) > 1 else len(description)
    middle = description[first:second]
    return [description[:first], middle, description[second:] if len(marks) > 1 else middle]


def segment_info(text):
    name = category(text)
    match = re.search(r"(?:the )?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth) "
                      r"(?:object|vehicle|order|position)", text.lower())
    vertical, horizontal = name.split("_")
    return name, vertical, horizontal, ORDINALS.get(match.group(1), 0) if match else 0, heading(text)


def parse_prompt(description):
    values = [float(number) for number in re.findall(r"(\d+(?:\.\d+)?)\s+meters?", description.lower())]
    dimensions = (values[:3] + [1.5, 4.5, 1.8])[:3]
    distances = values[3:] or [40.]
    first = segment_texts(description)[0].lower()
    vehicle = re.search(r"\b(van|bus|car|truck)\b", first)
    color = re.search(r"\b(silver grey|white|red|black|blue|yellow-orange|yellow orange)\b", first)
    azimuth = re.search(r"azimuth of (-?\d+(?:\.\d+)?)", description.lower())
    state = re.search(r"\b(parked|parking state|moving|driving)\b", first)
    attributes = (vehicle.group(1) if vehicle else "",
                  color.group(1).replace("-", " ") if color else "",
                  "moving" if state and state.group(1) in ("moving", "driving") else "parking",
                  float(azimuth.group(1)) if azimuth else 0.)
    segments = [segment_info(text) for text in segment_texts(description)]
    return {"dims": dimensions, "distances": distances[:3], "attrs": attributes,
            "segments": segments[:3]}


def distance_at(distances, frame):
    if len(distances) == 1:
        return float(distances[0])
    anchors = [0, 14, 29] if len(distances) >= 3 else [0, 29]
    return float(np.interp(frame, anchors, distances[:len(anchors)]))


def frame_segment(info, frame):
    return info["segments"][0 if frame <= 9 else 1 if frame <= 19 else 2]
