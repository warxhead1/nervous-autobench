"""Phase field oracle — C++ evaluator, function signature, seeds, source build.

Behavior-preserving split of phase_kernel/__init__.py: the C++ Allen-Cahn
time-stepper source, the LLM function-signature prompt block, the seed
programs, and C++ source construction moved here verbatim.
"""
from __future__ import annotations

import json

from .instance import PhaseInstance, _PHASE_INSTANCE_CONFIGS


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
