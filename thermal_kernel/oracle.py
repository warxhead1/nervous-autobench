"""Seed programs + diagnostic seeds (split from thermal_kernel/__init__.py)."""
from __future__ import annotations

import textwrap


# ---------------------------------------------------------------------------
# Seed programs — all use CORRECT sign convention: m = 2*(0.5 - temp)
# ---------------------------------------------------------------------------

_SEEDS: list[tuple[str, str]] = [
    ("correct_allen_cahn", textwrap.dedent("""\
        float reaction(float phi, float temp) {
            float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
            // m > 0 when cold (T<0.5) → solid preferred — CORRECT convention
            float m = 2.0f * (0.5f - temp);
            return -dW + m;
        }""")),
    ("logistic_correct", textwrap.dedent("""\
        float reaction(float phi, float temp) {
            float m = 0.5f - temp;  // positive when cold → freezes
            float interface_term = -4.0f*phi*phi*phi + 6.0f*phi*phi - 2.0f*phi;
            float bulk_term = 6.0f * phi * (1.0f - phi) * m;
            return interface_term + bulk_term;
        }""")),
    ("interface_amplified", textwrap.dedent("""\
        float reaction(float phi, float temp) {
            float m = 2.0f * (0.5f - temp);  // positive when cold
            float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
            // Amplify drive near interface (phi in [0.2,0.8]) to sharpen front
            float amp = 1.0f + 4.0f * phi * (1.0f - phi);  // peak at phi=0.5
            return -dW + m * amp;
        }""")),
]

_DIAGNOSTIC_SEEDS: list[tuple[str, str]] = [
    ("ZERO_REACTION",   "float reaction(float phi, float temp) { return 0.0f; }"),
    ("WRONG_SIGN_SEED", textwrap.dedent("""\
        float reaction(float phi, float temp) {
            float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
            float m = 2.0f * (temp - 0.5f);  // WRONG sign — melts when cold
            return -dW + m;
        }""")),
]


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    return _SEEDS


def get_diagnostic_seeds() -> list[tuple[str, str]]:
    return _DIAGNOSTIC_SEEDS
