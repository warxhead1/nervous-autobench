"""Phase field kernel — FunSearch evolution of matter-state transition driving forces.

# Domain: Allen-Cahn Phase Field  float reaction(float phi, float temp)

Evolves the bulk driving force of a phase field model against a 1D Allen-Cahn
time-stepper oracle.  The evolved function encodes the thermodynamic force
that drives matter transitions: liquid ↔ solid (water ↔ ice), vapour ↔ liquid,
or any two-phase system.

## The math — Allen-Cahn equation

  ∂φ/∂t = D·∇²φ + reaction(φ, T)

  φ ∈ [0,1]: order parameter (0 = liquid/disordered, 1 = solid/ordered)
  T ∈ [0,1]: dimensionless temperature (0 = absolute zero, 0.5 = melting point,
                                         1 = far above melting)

  The classical Allen-Cahn driving force is:
      reaction(φ, T) = −W′(φ) + m·(T − 0.5)
  where W(φ) = φ²(1−φ)²  (double-well potential)
  and W′(φ) = 2φ(1−φ)(2φ−1) = 4φ³ − 6φ² + 2φ

  Physical requirements:
    1. Equilibrium at φ=0 and φ=1 for T=0.5 (balanced at melting):
           reaction(0, 0.5) ≈ 0  and  reaction(1, 0.5) ≈ 0
    2. Correct temperature response:
           T > 0.5 → liquid preferred → pushes φ → 0 when φ > 0
           T < 0.5 → solid preferred  → pushes φ → 1 when φ < 1
    3. Interface convergence: from a sharp step, φ(x) should converge
       to a smooth tanh-profile within O(100) time steps.

## Oracle: 1D time-stepper

  1. Initialise φ[i]: step function (φ=0 for i<N/2, φ=1 for i≥N/2)
  2. Run N_steps of explicit Euler:
       φ[i] += dt · (D · ∇²φ[i] + reaction(φ[i], T))
  3. Measure interface quality:
       tanh_score = correlation of final φ(x) with ideal tanh profile
       equil_score = |φ[0:N/4]| + |1−φ[3N/4:N]|  (ends reach equilibrium)
       width_score = interface width vs target (too narrow or too wide = penalty)
  4. Fitness = 0.5·tanh_score + 0.3·equil_score + 0.2·width_score

  For the "melting" instance (T=0.7): solid half should shrink → φ→0.
  For the "freezing" instance (T=0.3): solid half should grow → φ→1.
  For the "balanced" instance (T=0.5): interface should sharpen without moving.

## Hard preconditions

  Stability: |reaction(φ, T)| < 50 for all φ∈[0,1], T∈[0,1].
  Finite: no NaN/Inf during time-stepping.

## TEngine integration path

  The evolved reaction(φ, T) function drops directly into a Slang compute
  shader that implements per-voxel phase field evolution for TEngine's
  temperature field layer:

    for each voxel:
      float T = temperature_field[voxel];
      float phi = phase_field[voxel];
      float dphi = D * laplacian(phi, voxel) + reaction(phi, T);
      phase_field[voxel] = clamp(phi + dt * dphi, 0.0, 1.0);

  The evolved function provides the thermodynamic driving force — the part
  that currently does not exist in TEngine.  This is the mathematical substrate
  for "walk up to a river and cast a freeze spell".

## Connection to other kernels

  SPH kernel   → fluid smoothing kernel (how particles reconstruct density)
  Terrain kernel → riverbed geometry that determines where water accumulates
  This kernel   → whether the water at each voxel is ice or liquid
  Phase field  → SDF kernel: ice boundary can be raymarched as evolved SDF
"""
from __future__ import annotations

import json
import logging
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..kernel_base import FunSearchKernel, KernelConfig, CandidateProgram, Island
from ..core import Verdict
from ..sandbox import SandboxedExecutor, compile_and_run
from ..tsp_kernel import ensure_sandboxed_executor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase field benchmark instance
# ---------------------------------------------------------------------------

@dataclass
class PhaseInstance:
    """A thermodynamic scenario: temperature + diffusion + run length.

    The oracle initialises a 1D phase field with a sharp liquid/solid interface
    and runs the Allen-Cahn PDE for n_steps, measuring how well the evolved
    reaction() function drives the interface toward the correct equilibrium.
    """
    name: str
    description: str
    temperature: float     # T ∈ [0,1]; 0.5 = melting point
    D: float               # Diffusion coefficient
    n_steps: int           # Time-step count
    dt: float              # Time-step size
    grid_size: int         # 1D grid resolution
    target_width: float    # Target interface width (grid cells)


# ---------------------------------------------------------------------------
# C++ evaluator: 1D Allen-Cahn time-stepper
# Evolved reaction() inserted at EVOLVED_CODE_MARKER.
# Input  (stdin): JSON with instance parameters
# Output (stdout): JSON with fitness, tanh_score, equil_score, width, valid
# ---------------------------------------------------------------------------

_PHASE_EVALUATOR_CPP = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cfloat>

using namespace std;

// Clamps are not needed in evolved code but available for safety
inline float clamp01(float x) { return fmaxf(0.f, fminf(1.f, x)); }

// ===== EVOLVED_CODE_MARKER =====

// ── 1D Allen-Cahn solver ─────────────────────────────────────────────────────
// Grid layout: phi[0] = liquid end (phi→0), phi[N-1] = solid end (phi→1)
// Initial condition: sharp interface at grid centre.
// Grid spacing dx=1 (dimensionless units). D is the diffusion coefficient in
// grid-units²/time. Interface width ≈ sqrt(D) grid cells.
// Stability: dt < 1/(2*D) required for explicit Euler.

static int run_pde(int N, float T, float D, float dt, int steps,
                   vector<float>& phi_out) {
    vector<float> phi(N), phi_new(N);
    // dx=1 in grid units — Laplacian is just phi[i-1]-2*phi[i]+phi[i+1]
    float dx2 = 1.0f;

    // Step function initial condition: liquid left, solid right
    for (int i = 0; i < N; i++)
        phi[i] = (i < N/2) ? 0.05f : 0.95f;

    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < N; i++) {
            float p = phi[i];
            // Laplacian with Neumann BC (zero-flux at endpoints)
            float pL = (i > 0)   ? phi[i-1] : phi[i];
            float pR = (i < N-1) ? phi[i+1] : phi[i];
            float lap = (pL - 2.f*p + pR) / dx2;

            float r = reaction(p, T);
            // Hard stability gate: if reaction explodes, abort
            if (!isfinite(r) || fabsf(r) > 50.f) return -1;

            phi_new[i] = p + dt * (D * lap + r);
            // Allow slight overshoot (physical phase fields can go outside [0,1])
            if (!isfinite(phi_new[i])) return -1;
        }
        swap(phi, phi_new);
    }

    phi_out = phi;
    return 0;
}

// Pearson correlation between phi(x) and an ideal tanh profile
// tanh( (x - x0) / w ) shifted and scaled to [0,1]
static float tanh_correlation(const vector<float>& phi, float temp) {
    int N = (int)phi.size();
    // Expected: if T<0.5, solid grows (x0 moves left); if T>0.5, melts (x0 moves right)
    // For correlation test, just check against a smooth profile centred at middle
    float x0 = 0.5f;  // centre of domain [0,1]
    float w  = 0.1f;  // typical interface width in [0,1] normalised coords

    double sx=0,sy=0,sxy=0,sxx=0,syy=0;
    int n = 0;
    for (int i = 0; i < N; i++) {
        float x = (float)i / (N-1.f);
        // Target tanh profile (solid on right: phi→1)
        float ref = 0.5f * (1.f + tanhf((x - x0) / w));
        float py  = phi[i];
        sx  += ref; sy += py;
        sxy += ref * py; sxx += ref*ref; syy += py*py;
        n++;
    }
    double D_val = sqrt((n*sxx - sx*sx) * (n*syy - sy*sy));
    if (D_val < 1e-9) return 0.f;
    float r = (float)((n*sxy - sx*sy) / D_val);
    return fmaxf(0.f, fminf(1.f, r));  // clamp to [0,1]: Pearson can exceed 1 numerically when phi variance ≈ 0
}

// Equilibrium score: ends of domain should reach φ≈0 and φ≈1
// Returns 0 (bad) → 1 (good)
static float equilibrium_score(const vector<float>& phi, float temp) {
    int N = (int)phi.size();
    int quarter = N / 4;

    // Left end (liquid expected)
    double left_mean = 0.0;
    for (int i = 0; i < quarter; i++) left_mean += phi[i];
    left_mean /= quarter;

    // Right end (solid expected)
    double right_mean = 0.0;
    for (int i = N-quarter; i < N; i++) right_mean += phi[i];
    right_mean /= quarter;

    // If T < 0.5: freezing — right stays ~1, left moves toward 1 eventually
    //             but in finite steps we just want right=1 and left small
    // If T > 0.5: melting — left stays ~0, right moves toward 0
    // If T = 0.5: balanced — left~0, right~1 stable
    float liquid_end = (float)left_mean;   // want near 0
    float solid_end  = (float)right_mean;  // want near 1

    float left_score  = 1.f - liquid_end;   // higher when liquid_end → 0
    float right_score = solid_end;           // higher when solid_end → 1

    // For melting (T>0.5): right should be shrinking, so some middle value is OK
    // For freezing (T<0.5): left should be growing toward 1
    // We reward agreement with expected direction
    if (temp > 0.6f) {
        // Melting: reward right_mean being smaller than 0.9 (interface moved left)
        right_score = (right_mean < 0.9f) ? 0.7f + 0.3f * (0.9f - (float)right_mean) / 0.9f : 0.5f;
    } else if (temp < 0.4f) {
        // Freezing: reward left_mean being larger than 0.1 (interface moved right)
        left_score = (left_mean > 0.1f) ? 0.7f + 0.3f * (float)left_mean : 0.5f;
    }

    return 0.5f * clamp01(left_score) + 0.5f * clamp01(right_score);
}

// Interface width score: penalise if width is far from target
static float width_score(const vector<float>& phi, float target_width_cells) {
    int N = (int)phi.size();
    // Count cells where phi ∈ [0.1, 0.9] as "interface"
    int cnt = 0;
    for (int i = 0; i < N; i++)
        if (phi[i] > 0.1f && phi[i] < 0.9f) cnt++;
    float w = (float)cnt;
    float err = (w - target_width_cells) / (target_width_cells + 1.f);
    return expf(-2.f * err * err);
}

int main() {
    string inp((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());

    auto get_float = [&](const char* key, float def) -> float {
        string k = string("\"") + key + "\"";
        size_t p = inp.find(k);
        if (p == string::npos) return def;
        p += k.size();
        while (p < inp.size() && (inp[p]==' '||inp[p]==':')) p++;
        char* end; float v = strtof(inp.c_str()+p, &end);
        return (end != inp.c_str()+p) ? v : def;
    };
    auto get_int = [&](const char* key, int def) -> int {
        return (int)get_float(key, (float)def);
    };
    auto get_str = [&](const char* key) -> string {
        string k = string("\"") + key + "\"";
        size_t p = inp.find(k);
        if (p == string::npos) return "";
        p += k.size();
        while (p < inp.size() && (inp[p]==' '||inp[p]==':')) p++;
        if (p < inp.size() && inp[p]=='"') {
            p++; string r;
            while (p < inp.size() && inp[p]!='"') r += inp[p++];
            return r;
        }
        return "";
    };

    string name       = get_str("name");
    float  temp       = get_float("temperature", 0.5f);
    float  D          = get_float("D", 0.3f);
    float  dt         = get_float("dt", 0.0005f);
    int    steps      = get_int("n_steps", 200);
    int    N          = get_int("grid_size", 128);
    float  tgt_width  = get_float("target_width", 8.0f);

    // Precondition: reaction must be finite and bounded at all corners
    float corners[4] = {
        reaction(0.0f, 0.0f), reaction(1.0f, 0.0f),
        reaction(0.0f, 1.0f), reaction(1.0f, 1.0f)
    };
    for (int i = 0; i < 4; i++) {
        if (!isfinite(corners[i]) || fabsf(corners[i]) > 50.f) {
            printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"unstable_corners\","
                   "\"name\":\"%s\"}\n", name.c_str());
            return 0;
        }
    }

    vector<float> phi;
    int status = run_pde(N, temp, D, dt, steps, phi);
    if (status != 0 || phi.empty()) {
        printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"pde_diverged\","
               "\"name\":\"%s\"}\n", name.c_str());
        return 0;
    }

    // Reject fields that collapsed to a uniform non-physical state
    // (all phi < -0.3 or all phi > 1.3 means driving force overran the double-well)
    float phi_min = phi[0], phi_max = phi[0];
    for (float v : phi) { phi_min = fminf(phi_min, v); phi_max = fmaxf(phi_max, v); }
    if (phi_min < -0.3f || phi_max > 1.3f || (phi_max - phi_min) < 0.05f) {
        printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"collapsed_field\","
               "\"name\":\"%s\"}\n", name.c_str());
        return 0;
    }

    float tc = tanh_correlation(phi, temp);
    float es = equilibrium_score(phi, temp);
    float ws = width_score(phi, tgt_width);

    float fitness = 0.5f*tc + 0.3f*es + 0.2f*ws;

    // Report a few phi values for debugging
    float phi_left  = phi[N/8];
    float phi_mid   = phi[N/2];
    float phi_right = phi[7*N/8];

    printf("{\"fitness\":%.6f,\"tanh_score\":%.4f,\"equil_score\":%.4f,"
           "\"width_score\":%.4f,\"phi_left\":%.4f,\"phi_mid\":%.4f,"
           "\"phi_right\":%.4f,\"valid\":true,\"name\":\"%s\"}\n",
           fitness, tc, es, ws, phi_left, phi_mid, phi_right, name.c_str());
    return 0;
}
"""

PHASE_FUNCTION_SIGNATURE = '''\
float reaction(float phi, float temp);

// Evolved Allen-Cahn driving force for matter state transitions.
// phi:  order parameter ∈ [0,1]  (0 = liquid, 1 = solid)
// temp: dimensionless temperature ∈ [0,1]  (0.5 = melting point)
// Returns: dphi/dt bulk driving force (scalar, any finite float)
//
// PHYSICAL REQUIREMENTS:
//   Equilibrium at melting:     reaction(0, 0.5) ≈ 0  (liquid equilibrium)
//                               reaction(1, 0.5) ≈ 0  (solid equilibrium)
//   Temperature response:
//     temp > 0.5  →  reaction pushes phi toward 0 (melting)
//     temp < 0.5  →  reaction pushes phi toward 1 (freezing)
//   Stability: |reaction(phi, temp)| < 50 for phi,temp ∈ [0,1]
//
// CLASSICAL ALLEN-CAHN (reference):
//   W′(phi) = 4*phi*phi*phi - 6*phi*phi + 2*phi   (double-well derivative)
//   m = 2.0*(0.5 - temp)   ← positive when cold (T<0.5) → solid preferred
//   reaction(phi, temp) = -W′(phi) + m
//   = -(4*phi³ - 6*phi² + 2*phi) + 2*(0.5 - temp)
//
// The oracle runs a 1D finite-difference PDE stepper and measures:
//   - How smoothly the interface relaxes to a tanh profile
//   - Whether the ends reach correct equilibrium (phi→0 liquid, phi→1 solid)
//   - Whether the interface width matches the target
//
// Available C math: sinf, cosf, expf, logf, sqrtf, fabsf, powf, fmaxf, fminf
// Function body must be under 15 lines. No loops, no static state.
'''


# ---------------------------------------------------------------------------
# Instance configurations
# ---------------------------------------------------------------------------

_PHASE_INSTANCE_CONFIGS: dict[str, dict] = {
    # Grid spacing dx=1; interface width ≈ sqrt(D) cells; stability: dt < 1/(2*D).
    # D=50  → interface width ≈ 7 cells;  dt_max = 0.010; use dt=0.006.
    # D=100 → interface width ≈ 10 cells; dt_max = 0.005; use dt=0.003.
    "water_ice_freezing": {
        "temperature": 0.30,     # Below melting: ice grows
        "D": 50.0,
        "dt": 0.006,
        "n_steps": 300,
        "grid_size": 64,
        "target_width": 7.0,
        "description": "Water supercooled at T=0.30: ice front grows (φ→1 spreads)",
    },
    "water_ice_melting": {
        "temperature": 0.70,     # Above melting: ice melts
        "D": 50.0,
        "dt": 0.006,
        "n_steps": 300,
        "grid_size": 64,
        "target_width": 7.0,
        "description": "Ice melting at T=0.70: liquid front grows (φ→0 spreads)",
    },
    "phase_balanced": {
        "temperature": 0.50,     # At melting: interface sharpens, doesn't move
        "D": 50.0,
        "dt": 0.006,
        "n_steps": 400,
        "grid_size": 64,
        "target_width": 7.0,
        "description": "Balanced at T=0.50: interface sharpens to equilibrium tanh",
    },
    "rapid_freeze": {
        "temperature": 0.10,     # Deep supercooling: very fast ice growth
        "D": 30.0,
        "dt": 0.008,
        "n_steps": 200,
        "grid_size": 64,
        "target_width": 5.0,
        "description": "Deep supercooling T=0.10: rapid ice growth, narrow interface",
    },
    "slow_melt": {
        "temperature": 0.65,     # Slow melt with wider interface
        "D": 100.0,
        "dt": 0.003,
        "n_steps": 500,
        "grid_size": 64,
        "target_width": 10.0,
        "description": "Slow melt T=0.65, wide interface: gradual phase boundary evolution",
    },
}


# ---------------------------------------------------------------------------
# Seed programs — classical phase field driving forces
# ---------------------------------------------------------------------------

# Classical Allen-Cahn: W'(phi) + temperature coupling
_ALLEN_CAHN_SEED = '''\
float reaction(float phi, float temp) {
    // Double-well derivative W'(phi) = 4phi^3 - 6phi^2 + 2phi
    float dW = phi * phi * (4.0f * phi - 6.0f) + 2.0f * phi;
    // m > 0 when cold (T<0.5) → solid preferred. CORRECT sign convention.
    float m = 2.0f * (0.5f - temp);
    return -dW + m;
}'''

# Ginzburg-Landau with asymmetric well
_GINZBURG_LANDAU_SEED = '''\
float reaction(float phi, float temp) {
    // Tilted double-well: W′ shifted by temperature
    float p3 = phi * phi * phi;
    float p2 = phi * phi;
    float bulk = -12.0f * p2 + 12.0f * phi;     // derivative of 4phi^2(1-phi)^2 scaled
    float tilt = 6.0f * (0.5f - temp);           // positive when cold → solid preferred
    return bulk * (1.0f - phi) * phi + tilt;
}'''

# Logistic-like model: phi(1-phi) as interface localiser
_LOGISTIC_PHASE_SEED = '''\
float reaction(float phi, float temp) {
    float m = 0.5f - temp;  // positive when cold → freezes
    // Interface localised to phi*(1-phi), bulk driven by temperature
    float interface_term = -4.0f * phi * phi * phi + 6.0f * phi * phi - 2.0f * phi;
    float bulk_term = 6.0f * phi * (1.0f - phi) * m;
    return interface_term + bulk_term;
}'''

SEED_PHASE_PROGRAMS: dict[str, list[tuple[str, str]]] = {
    "generic": [
        ("allen_cahn",       _ALLEN_CAHN_SEED),
        ("ginzburg_landau",  _GINZBURG_LANDAU_SEED),
        ("logistic_phase",   _LOGISTIC_PHASE_SEED),
    ],
}
for _name in _PHASE_INSTANCE_CONFIGS:
    SEED_PHASE_PROGRAMS[_name] = SEED_PHASE_PROGRAMS["generic"]


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    return SEED_PHASE_PROGRAMS.get(instance_name, SEED_PHASE_PROGRAMS["generic"])


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_instance(name: str) -> PhaseInstance:
    cfg = _PHASE_INSTANCE_CONFIGS.get(name)
    if cfg is None:
        raise ValueError(f"Unknown phase instance: {name!r}. "
                         f"Available: {list(_PHASE_INSTANCE_CONFIGS)}")
    return PhaseInstance(
        name=name,
        description=cfg["description"],
        temperature=cfg["temperature"],
        D=cfg["D"],
        dt=cfg["dt"],
        n_steps=cfg["n_steps"],
        grid_size=cfg["grid_size"],
        target_width=cfg["target_width"],
    )


# ---------------------------------------------------------------------------
# C++ source construction
# ---------------------------------------------------------------------------

def build_candidate_source(code: str, instance: PhaseInstance) -> tuple[str, str]:
    cpp = _PHASE_EVALUATOR_CPP.replace("// ===== EVOLVED_CODE_MARKER =====", code, 1)
    data = {
        "name": instance.name,
        "temperature": instance.temperature,
        "D": instance.D,
        "dt": instance.dt,
        "n_steps": instance.n_steps,
        "grid_size": instance.grid_size,
        "target_width": instance.target_width,
    }
    return cpp, json.dumps(data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Single-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_on_instance(
    code: str,
    instance: PhaseInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 30.0,
    generation: int = 999,
) -> Optional[float]:
    """Evaluate one candidate on one instance.

    Oracle phase scheduling (Layer 0→1→2→3):
      gen <  3: equil_score only         — establish correct attractor basins first
      gen <  6: 0.5*tanh + 0.5*equil    — add interface shape
      gen >= 6: 0.5*tanh + 0.3*equil + 0.2*width  — full oracle
    """
    cpp_source, stdin_data = build_candidate_source(code, instance)

    stdout, verdict, _latency = compile_and_run(
        cpp_source,
        language="cpp",
        stdin=stdin_data,
        executor=executor,
    )
    if verdict != Verdict.OK or not stdout:
        return None

    try:
        out = json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None

    if not out.get("valid", False):
        return None

    tanh_s = float(out.get("tanh_score",  0.0))
    equil_s = float(out.get("equil_score", 0.0))
    width_s = float(out.get("width_score", 0.0))

    # Oracle phase scheduling: build up complexity over generations
    if generation < 3:
        fitness = equil_s                              # Layer 1: equilibrium basins only
    elif generation < 6:
        fitness = 0.5 * tanh_s + 0.5 * equil_s       # Layer 2: add interface shape
    else:
        fitness = 0.5 * tanh_s + 0.3 * equil_s + 0.2 * width_s  # Layer 3: full oracle

    fitness = min(fitness, 1.0)  # defensive: components should already be in [0,1]
    return fitness if math.isfinite(fitness) and fitness > 0 else None


def ensure_executor(allow_unsandboxed: bool = False) -> SandboxedExecutor:
    return ensure_sandboxed_executor(allow_unsandboxed=allow_unsandboxed)


# ---------------------------------------------------------------------------
# Phase FunSearch kernel
# ---------------------------------------------------------------------------

class PhaseKernel(FunSearchKernel):
    """FunSearch kernel that evolves Allen-Cahn phase field driving forces.

    Oracle: 1D time-stepper — measures whether the evolved reaction(phi, temp)
    drives a sharp liquid/solid interface to a smooth tanh equilibrium profile
    at the correct temperature response.

    Fitness = 0.5·tanh_score + 0.3·equil_score + 0.2·width_score.

    This is the mathematical substrate for matter state transitions in TEngine:
    the "freeze the river with a spell" mechanic.
    """

    kernel_name = "phase"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        logger.info("Phase kernel: %d instances, sandbox=%s",
                    len(self.problem_instances), self.executor.sandbox_type)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def load_instances(self) -> list[PhaseInstance]:
        instances = []
        for name in self.config.instances:
            inst = generate_instance(name)
            logger.info("Phase instance %r: T=%.2f, D=%.2f, steps=%d",
                        name, inst.temperature, inst.D, inst.n_steps)
            instances.append(inst)
        return instances

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(
            code, instance, self.executor,
            run_timeout=self.config.run_timeout,
            generation=self.generation,  # oracle phase scheduling
        )

    def build_prompt(
        self,
        island: Island,
        top_programs: list[CandidateProgram],
        generation: int,
        hint: str = "",
    ) -> str:
        instance = self.problem_instances[0] if self.problem_instances else None
        inst_desc = (f"{instance.name} (T={instance.temperature:.2f}, "
                     f"D={instance.D:.2f}, steps={instance.n_steps}  — {instance.description})"
                     if instance else "unknown")

        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )

        hint_block = ""
        if hint:
            hint_block = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

        temp_str = f"{instance.temperature:.2f}" if instance else "0.5"
        goal_desc = ("solid phase grows (φ→1 spreads)" if instance and instance.temperature < 0.5
                     else ("liquid grows (φ→0 spreads)" if instance and instance.temperature > 0.5
                           else "interface sharpens at equilibrium"))

        return (
            f"You are a computational physics expert specialising in phase field models.\n"
            f"Evolve a better Allen-Cahn driving force reaction(phi, temp).\n\n"
            f"{PHASE_FUNCTION_SIGNATURE}\n\n"
            f"Target: {inst_desc}\n"
            f"Physics goal at T={temp_str}: {goal_desc}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Top programs in this island:\n\n{exemplars}\n\n"
            f"Your goal: maximise Fitness = 0.5·tanh_score + 0.3·equil_score + 0.2·width_score\n"
            f"  tanh_score:  how closely final phi(x) matches a tanh profile\n"
            f"  equil_score: how well the domain ends reach phi=0 and phi=1\n"
            f"  width_score: whether interface width ≈ {instance.target_width if instance else 8} cells\n\n"
            f"Rules:\n"
            f"- Return ONLY the reaction() function in a single ```cpp code block\n"
            f"- Signature: float reaction(float phi, float temp)\n"
            f"- No loops, no static state, no dynamic allocation\n"
            f"- |reaction(phi, temp)| < 50 for all phi,temp ∈ [0,1] (stability checked)\n"
            f"- Under 15 lines\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "reaction" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        seeds = get_seed_programs(
            self.config.instances[0] if self.config.instances else "water_ice_freezing"
        )
        programs = []
        for name, code in seeds:
            prog = CandidateProgram(
                id=str(uuid.uuid4()),
                code=code,
                island=island_id,
                generation=generation,
            )
            programs.append(prog)
        return programs

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    def save_results(self, programs: list[CandidateProgram]) -> Optional[Path]:
        if not self.config.output_dir:
            return None
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"phase_results_gen{self.generation:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "phase",
            "run_id": self.run_id,
            "generation": self.generation,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {
                    "id": p.id,
                    "fitness": p.fitness,
                    "worst_fitness": p.worst_fitness,
                    "island": p.island,
                    "generation": p.generation,
                    "reaction_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Phase results → %s (best=%.4f)", out_path,
                    top[0].fitness if top else 0)
        return out_path
