"""Latent kernel seed programs.

Evolution seeds and diagnostic (negative-control) seeds, 3-arg signature with
the correct sign convention. Moved verbatim from the package __init__
(behaviour-preserving file split).
"""
from __future__ import annotations

import textwrap


# ---------------------------------------------------------------------------
# Seed programs — 3-arg signature, correct sign convention
# ---------------------------------------------------------------------------

_SEEDS: list[tuple[str, str]] = [
    ("classical_latent", textwrap.dedent("""\
        float reaction(float phi, float temp, float lap_T) {
            // Classical Allen-Cahn — ignores lap_T (baseline)
            float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
            float m = 2.0f * (0.5f - temp);
            return -dW + m;
        }""")),
    ("thermally_responsive", textwrap.dedent("""\
        float reaction(float phi, float temp, float lap_T) {
            float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
            float m = 2.0f * (0.5f - temp);
            // Thermal brake: slow down when heat flowing in (lap_T > 0 → equilibrating)
            // Speed up when heat flowing out (lap_T < 0 → more undercooling available)
            float brake = 1.0f - 0.15f * fmaxf(0.f, lap_T);
            return (-dW + m) * fmaxf(0.1f, brake);
        }""")),
    ("interface_thermal", textwrap.dedent("""\
        float reaction(float phi, float temp, float lap_T) {
            float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
            float m = 2.0f * (0.5f - temp);
            // Amplify at interface + respond to thermal gradient
            float amp = 1.0f + 3.0f * phi * (1.0f - phi);
            float thermal_mod = 1.0f - 0.1f * lap_T;
            return (-dW + m * amp) * fmaxf(0.1f, thermal_mod);
        }""")),
]

_DIAGNOSTIC_SEEDS: list[tuple[str, str]] = [
    ("ZERO_REACTION",
     "float reaction(float phi, float temp, float lap_T) { return 0.0f; }"),
    ("WRONG_SIGN",  textwrap.dedent("""\
        float reaction(float phi, float temp, float lap_T) {
            float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
            float m = 2.0f * (temp - 0.5f);  // WRONG sign
            return -dW + m;
        }""")),
]


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    return _SEEDS


def get_diagnostic_seeds() -> list[tuple[str, str]]:
    return _DIAGNOSTIC_SEEDS
