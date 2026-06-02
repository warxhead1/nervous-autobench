"""LLM-response parsing helpers for the noise kernel.

Moved verbatim from noise_kernel/__init__.py.
"""

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Helper: extract noise() from a response that also includes mainImage
# ---------------------------------------------------------------------------

def _extract_noise_fn(glsl: str) -> str:
    """Extract the `float noise(...)` function from a larger GLSL snippet.

    Returns the extracted function (including its closing brace) or '' if not found.
    """
    m = re.search(r"(float\s+noise\s*\(\s*vec3[^{]*\{)", glsl)
    if not m:
        return ""
    start = m.start()
    depth = 0
    i = m.start(1) + len(m.group(1)) - 1  # position of the opening brace
    while i < len(glsl):
        if glsl[i] == "{":
            depth += 1
        elif glsl[i] == "}":
            depth -= 1
            if depth == 0:
                return glsl[start:i + 1].strip()
        i += 1
    return glsl[start:].strip()
