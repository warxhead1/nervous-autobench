"""Latent-heat-coupled 2D Allen-Cahn kernel.

# Domain: Coupled Phase-Field + Thermal Diffusion
# Function: float reaction(float phi, float temp, float lap_T)

Evolves the Allen-Cahn driving force in a setting where temperature is NOT fixed —
it evolves via coupled PDEs:

  ∂φ/∂t = D_φ·∇²φ + reaction(φ, T, ∇²T)
  ∂T/∂t = D_T·∇²T + L·∂φ/∂t

Latent heat L > 0: freezing (∂φ/∂t > 0) releases heat → T rises → drive weakens.
This is self-limiting: ice can only form until local T rises to the melting point 0.5.

## Why this oracle is hard

With fixed T (thermal_kernel), any correct-sign function trivially scores 1.0 because
the entire cold zone freezes and the hot zone melts.

With coupled T, the equilibrium coverage is:
  φ_eq = (0.5 − T_cold_initial) / L

For T_cold=0.25, L=0.4:  φ_eq = 0.625  (only 62.5% of cold zone freezes)
For T_cold=0.30, L=0.6:  φ_eq = 0.333  (only 33% freezes — very sensitive)

Functions that freeze too fast overshoot equilibrium → T rises above 0.5 → drive reverses
→ ice melts → oscillation → low score.
Functions that freeze at exactly the right rate hit the equilibrium cleanly → high score.

## Third argument: lap_T = ∇²T

This gives the function LOCAL THERMAL CONTEXT:
  lap_T > 0: T increasing at this point (heat flowing in, latent heat releasing nearby)
             → function can slow down to avoid overshoot
  lap_T < 0: T decreasing (heat flowing out, cold diffusing in)
             → function can speed up, more undercooling available
  lap_T ≈ 0: bulk region (already equilibrated)

Classical Allen-Cahn ignores lap_T. A function that uses it can stabilise the front.

## Oracle scoring (v2 — velocity-aware)

  T_balance_score  = Gaussian(mean_T_cold − 0.5, σ=0.10)     weight 0.30
  velocity_score   = exp(-(log(v/v_stefan))^2 / 0.25)         weight 0.30
                     v_stefan = D_T*(0.5-cold_temp)/(L*r_mid)
                     step(v>0): retreating fronts score 0
  sharpness_score  = narrow interface reward (target_width=3)  weight 0.25
  retention_score  = opposite zone stays in correct phase      weight 0.15

v2 changes vs v1:
  - D_phi reduced 10→4: natural AC width sqrt(4/0.5)≈2.8 cells, target_width=3 achievable
  - T_balance σ widened 0.06→0.10: correct equilibria near T=0.48-0.52 no longer penalized
  - T_balance weight 0.50→0.30: corroborating signal, not dominant
  - velocity_score added (0.30): rewards Stefan self-regulation; retreating fronts score 0
  - sharpness weight 0.35→0.25: less dominant; total still sums to 1.0
  - frontier_gradual instance added: cold_temp=0.30, L=0.6, φ_eq=0.333

Seed calibration (measured at D_phi=4, dt=0.020, freeze_latent):
  interface_thermal seed: 0.83 vs ZERO_REACTION 0.48 → gap +0.35  ✓
  classical_latent seed:  0.54 vs ZERO_REACTION 0.48 → gap +0.06  (marginal)
  gen21 discovered:       0.90 vs ZERO_REACTION 0.48 → gap +0.42  ✓
  WRONG_SIGN: collapsed → 0.001 (well below all correct-sign seeds) ✓

  Note: classical AC is a weak seed at D_phi=4 — its early-phase velocity≈0 because
  the undercooling is rapidly suppressed by latent heat before step n/4. This is physical
  (the oracle correctly penalizes slow classical AC) but means classical barely beats zero.
  Interface-amplified seeds (interface_thermal, gen21) show large gaps vs zero.
"""
from __future__ import annotations

import json
import logging
import math
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..kernel_base import FunSearchKernel, KernelConfig, CandidateProgram, Island
from ..core import Verdict
from ..sandbox import SandboxedExecutor, compile_and_run
from ..tsp_kernel import ensure_sandboxed_executor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------

@dataclass
class LatentInstance:
    name: str
    description: str
    grid_size: int
    cold_temp: float      # initial T in cold zone
    hot_temp: float       # initial T in hot zone
    cold_radius: float    # cold zone radius (cells)
    D_phi: float          # phase diffusivity
    D_T: float            # thermal diffusivity
    L: float              # latent heat coefficient
    dt: float
    n_steps: int
    initial_phi: float    # 0.02=all liquid, 0.98=all solid
    target_width: float   # target interface width (cells)

    @property
    def phi_eq(self) -> float:
        """Theoretical equilibrium coverage from energy balance."""
        if self.initial_phi < 0.5:
            return (0.5 - self.cold_temp) / self.L   # freeze: φ_eq = ΔT/L
        else:
            return (self.hot_temp - 0.5) / self.L    # melt: φ_melt = ΔT/L


# ---------------------------------------------------------------------------
# Instance registry
# ---------------------------------------------------------------------------

_LATENT_INSTANCE_CONFIGS: dict[str, dict] = {
    "freeze_latent": {
        "description": (
            "Freeze spot with latent heat: T rises as ice forms. "
            "Equilibrium coverage φ_eq=(0.5-0.25)/0.4=0.625 — only 62.5% of cold zone freezes. "
            "D_phi=4 gives natural interface width ~2.8 cells; target_width=3 is achievable."
        ),
        "grid_size": 64, "cold_temp": 0.25, "hot_temp": 0.75, "cold_radius": 18.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.4, "dt": 0.020,
        "n_steps": 500, "initial_phi": 0.02, "target_width": 3.0,
    },
    "melt_latent": {
        "description": (
            "Melt spot with latent heat: T drops as ice melts. "
            "Equilibrium melt coverage=(0.75-0.5)/0.4=0.625 — only 62.5% of hot zone melts. "
            "D_phi=4 matches freeze_latent for symmetric difficulty."
        ),
        "grid_size": 64, "cold_temp": 0.25, "hot_temp": 0.75, "cold_radius": 18.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.4, "dt": 0.020,
        "n_steps": 500, "initial_phi": 0.98, "target_width": 3.0,
    },
    "gentle_latent": {
        "description": (
            "Near-equilibrium freeze (T_cold=0.40) with latent heat. "
            "φ_eq=(0.5-0.40)/0.3=0.333 — tiny undercooling, very sensitive oracle."
        ),
        "grid_size": 64, "cold_temp": 0.40, "hot_temp": 0.60, "cold_radius": 20.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.3, "dt": 0.020,
        "n_steps": 700, "initial_phi": 0.02, "target_width": 3.0,
    },
    "frontier_gradual": {
        "description": (
            "Gradual frontier freeze: cold_temp=0.30, L=0.6 → φ_eq=0.333. "
            "Strong latent heat suppression forces gentle self-regulation. "
            "Velocity oracle is most discriminating here: v_stefan = D_T*(0.5-0.30)/(0.6*r_mid)."
        ),
        "grid_size": 64, "cold_temp": 0.30, "hot_temp": 0.70, "cold_radius": 18.0,
        "D_phi": 4.0, "D_T": 5.0, "L": 0.6, "dt": 0.020,
        "n_steps": 500, "initial_phi": 0.02, "target_width": 3.0,
    },
}


def generate_instance(name: str) -> LatentInstance:
    if name not in _LATENT_INSTANCE_CONFIGS:
        raise ValueError(f"Unknown latent instance: {name!r}. "
                         f"Known: {list(_LATENT_INSTANCE_CONFIGS)}")
    cfg = _LATENT_INSTANCE_CONFIGS[name]
    return LatentInstance(name=name, **cfg)


# ---------------------------------------------------------------------------
# C++ evaluator — coupled φ + T PDE, 3-arg reaction signature
# ---------------------------------------------------------------------------

_LATENT_EVALUATOR_CPP = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cfloat>

using namespace std;

inline float clamp01(float x) { return fmaxf(0.f, fminf(1.f, x)); }

// ===== EVOLVED_CODE_MARKER =====

static string g_json;
static float get_float(const char* k, float def) {
    string tok = "\"" + string(k) + "\":";
    auto p = g_json.find(tok);
    if (p == string::npos) return def;
    return stof(g_json.substr(p + tok.size()));
}
static int get_int(const char* k, int def) { return (int)get_float(k, (float)def); }
static string get_str(const char* k) {
    string tok = "\"" + string(k) + "\":\"";
    auto p = g_json.find(tok); if (p == string::npos) return "";
    auto q = g_json.find('"', p + tok.size());
    return (q == string::npos) ? "" : g_json.substr(p + tok.size(), q - p - tok.size());
}

// ── Coupled 2D Allen-Cahn + thermal diffusion ────────────────────────────────
// φ PDE: ∂φ/∂t = D_phi·∇²φ + reaction(φ, T, ∇²T)
// T PDE: ∂T/∂t = D_T·∇²T  + L·(∂φ/∂t)        [latent heat]
//
// Stability: D_phi*dt ≤ 0.25, D_T*dt ≤ 0.25 (2D 5-point stencil bound).
// Latent heat term is bounded by L*|reaction|*dt which is small per step.

// Compute mean radial interface position: weighted centroid where phi in (0.1, 0.9)
static float front_radius(const vector<float>& phi, int N) {
    float cx = (N-1)*0.5f, cy = (N-1)*0.5f;
    double sum_r = 0, sum_w = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float p = phi[i*N+j];
            if (p > 0.1f && p < 0.9f) {
                float r = sqrtf((i-cx)*(i-cx)+(j-cy)*(j-cy));
                float w = p * (1.f - p);  // weight by interface sharpness
                sum_r += w * r;
                sum_w += w;
            }
        }
    return (sum_w > 0) ? (float)(sum_r / sum_w) : -1.f;
}

// run_coupled_pde: evolves phi+T for `steps` time steps.
// Captures front radii at n/8 (r_early) and n/4 (r_quarter) for velocity measurement.
// These sample the EARLY GROWTH PHASE before the front peaks and recoils.
static int run_coupled_pde(int N, const vector<float>& T_init,
                            float D_phi, float D_T, float L,
                            float dt, int steps, float init_phi,
                            vector<float>& phi_out, vector<float>& T_out,
                            float& r_early_out, float& r_quarter_out) {
    int sz = N * N;
    vector<float> phi(sz, init_phi), phi_new(sz);
    vector<float> T = T_init, T_new(sz);
    r_early_out = r_quarter_out = -1.f;

    // Nucleation seed at centre (phi_seed nudges toward transition)
    int cx = N / 2, cy = N / 2;
    float seed_val = (init_phi < 0.5f) ? 0.65f : 0.35f;
    for (int di = -5; di <= 5; di++)
        for (int dj = -5; dj <= 5; dj++) {
            if (di*di + dj*dj > 25) continue;
            int ii = cx+di, jj = cy+dj;
            if (ii >= 0 && ii < N && jj >= 0 && jj < N)
                phi[ii*N+jj] = seed_val;
        }

    // Front velocity checkpoints: early (n/8) and quarter (n/4).
    // The front typically peaks around n/4-n/2; measuring growth in [n/8, n/4]
    // captures the active Stefan growth regime before latent heat suppression.
    int early_step   = steps / 8;
    int quarter_step = steps / 4;

    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float p = phi[i*N+j];
                float t = T[i*N+j];

                // 5-point stencils (Neumann BC)
                float pL = (i>0)   ? phi[(i-1)*N+j] : p;
                float pR = (i<N-1) ? phi[(i+1)*N+j] : p;
                float pD = (j>0)   ? phi[i*N+(j-1)] : p;
                float pU = (j<N-1) ? phi[i*N+(j+1)] : p;
                float lap_phi = pL + pR + pD + pU - 4.f*p;

                float tL = (i>0)   ? T[(i-1)*N+j] : t;
                float tR = (i<N-1) ? T[(i+1)*N+j] : t;
                float tD = (j>0)   ? T[i*N+(j-1)] : t;
                float tU = (j<N-1) ? T[i*N+(j+1)] : t;
                float lap_T_val = tL + tR + tD + tU - 4.f*t;

                float r = reaction(p, t, lap_T_val);
                if (!isfinite(r) || fabsf(r) > 200.f) return -1;

                float dphi = dt * (D_phi * lap_phi + r);
                phi_new[i*N+j] = p + dphi;
                if (!isfinite(phi_new[i*N+j])) return -1;

                // T update: diffusion + latent heat from phase change
                T_new[i*N+j] = t + dt * D_T * lap_T_val + L * dphi;
                if (!isfinite(T_new[i*N+j])) return -1;
            }
        }
        swap(phi, phi_new);
        swap(T, T_new);

        if (s == early_step - 1)   r_early_out   = front_radius(phi, N);
        if (s == quarter_step - 1) r_quarter_out = front_radius(phi, N);
    }

    // Collapsed-field gate
    float phi_min = phi[0], phi_max = phi[0];
    for (float v : phi) { phi_min = fminf(phi_min,v); phi_max = fmaxf(phi_max,v); }
    if (phi_min < -0.3f || phi_max > 1.3f || (phi_max - phi_min) < 0.02f)
        return -2;

    phi_out = phi;
    T_out = T;
    return 0;
}

// ── Sharpness score (same as thermal_kernel) ─────────────────────────────────
static float sharpness_score(const vector<float>& phi, int N,
                               float cold_radius, float tgt_width) {
    float cx = (N-1)*0.5f, cy = (N-1)*0.5f;
    int max_r = (int)(cold_radius * 2.0f + 2);
    if (max_r > N/2) max_r = N/2;

    vector<double> sum_phi(max_r+1, 0.0), cnt(max_r+1, 0.0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int r = (int)sqrtf((i-cx)*(i-cx)+(j-cy)*(j-cy));
            if (r <= max_r) { sum_phi[r] += phi[i*N+j]; cnt[r]++; }
        }

    float rmin=1.f, rmax=0.f;
    for (int r=0; r<=max_r; r++) {
        if (cnt[r]<1) continue;
        float v=(float)(sum_phi[r]/cnt[r]);
        rmin=fminf(rmin,v); rmax=fmaxf(rmax,v);
    }
    if (rmin > 0.3f || rmax < 0.7f) return 0.f;

    float iface_bins = 0.f;
    for (int r=0; r<=max_r; r++) {
        if (cnt[r]<1) continue;
        float v=(float)(sum_phi[r]/cnt[r]);
        if (v > 0.1f && v < 0.9f) iface_bins += 1.f;
    }
    float excess = fmaxf(0.f, iface_bins - tgt_width);
    return expf(-(excess*excess)/(tgt_width*tgt_width));
}

int main() {
    getline(cin, g_json);
    if (g_json.empty()) getline(cin, g_json);

    string name      = get_str("name");
    float cold_temp  = get_float("cold_temp",   0.25f);
    float hot_temp   = get_float("hot_temp",    0.75f);
    float cold_rad   = get_float("cold_radius", 18.0f);
    float D_phi      = get_float("D_phi",  10.0f);
    float D_T        = get_float("D_T",     5.0f);
    float L          = get_float("L",       0.4f);
    float dt         = get_float("dt",    0.018f);
    int   steps      = get_int ("n_steps",  500);
    int   N          = get_int ("grid_size",  64);
    float init_phi   = get_float("initial_phi",  0.02f);
    float tgt_width  = get_float("target_width",  2.0f);
    // phi_eq removed — oracle no longer targets specific coverage value
    // T_balance is the physically correct signal (rises to 0.5 when latent heat equilibrates)

    // Stability checks
    if (D_phi * dt > 0.25f + 1e-5f || D_T * dt > 0.25f + 1e-5f) {
        printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"stability_violated\","
               "\"name\":\"%s\"}\n", name.c_str());
        return 0;
    }

    // Corner stability check
    bool is_melt_mode = (init_phi > 0.5f);
    float test_T = is_melt_mode ? hot_temp : cold_temp;
    float test_lapT = 0.f;
    float corners[4] = {
        reaction(0.f, test_T, test_lapT), reaction(1.f, test_T, test_lapT),
        reaction(0.f, 0.5f,  test_lapT), reaction(1.f, 0.5f,  test_lapT)
    };
    for (int i=0; i<4; i++) {
        if (!isfinite(corners[i]) || fabsf(corners[i]) > 200.f) {
            printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"unstable_corners\","
                   "\"name\":\"%s\"}\n", name.c_str());
            return 0;
        }
    }

    // Build initial temperature field
    float cx = (N-1)*0.5f, cy = (N-1)*0.5f;
    vector<float> T_init(N*N);
    for (int i=0; i<N; i++) for (int j=0; j<N; j++) {
        float r = sqrtf((i-cx)*(i-cx)+(j-cy)*(j-cy));
        bool in_spot = (r < cold_rad);
        if (!is_melt_mode)
            T_init[i*N+j] = in_spot ? cold_temp : hot_temp;
        else
            T_init[i*N+j] = in_spot ? hot_temp  : cold_temp;
    }

    // Run coupled PDE; capture front radii at n/8 (r_early) and n/4 (r_quarter)
    // for early-growth-phase velocity measurement.
    vector<float> phi, T_final;
    float r_early = -1.f, r_quarter = -1.f;
    int status = run_coupled_pde(N, T_init, D_phi, D_T, L, dt, steps, init_phi,
                                  phi, T_final, r_early, r_quarter);
    if (status == -1) {
        printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"pde_diverged\","
               "\"name\":\"%s\"}\n", name.c_str());
        return 0;
    }
    if (status == -2) {
        printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"collapsed_field\","
               "\"name\":\"%s\"}\n", name.c_str());
        return 0;
    }

    // Coverage and temperature in each zone
    float T_mid = (cold_temp + hot_temp) * 0.5f;
    double sum_phi_cold=0, sum_T_cold=0, n_cold=0;
    for (int i=0; i<N*N; i++) {
        bool in_cold_zone = (T_init[i] <= T_mid);
        if (in_cold_zone) { sum_phi_cold+=phi[i]; sum_T_cold+=T_final[i]; n_cold++; }
    }
    float mean_phi_cold = n_cold>0 ? (float)(sum_phi_cold/n_cold) : 0.f;
    float mean_T_cold   = n_cold>0 ? (float)(sum_T_cold  /n_cold) : cold_temp;

    // T balance: corroborating physics signal (σ=0.10, widened from 0.06).
    // Correct Stefan freezing raises mean_T_cold → 0.5 via latent heat release.
    // σ=0.10: correct equilibria near T=0.48-0.52 are rewarded.
    // Zero-reaction: mean_T_cold drifts from diffusion only; scores ≪ 1.
    float T_balance_score = expf(-((mean_T_cold - 0.5f)*(mean_T_cold - 0.5f)) / (0.10f*0.10f));

    // Retention: opposite zone stays in correct phase
    float retention;
    if (!is_melt_mode) {
        int liq_hot=0, tot_hot=0;
        for (int i=0; i<N*N; i++) {
            if (T_init[i] > T_mid) { tot_hot++; if (phi[i]<0.5f) liq_hot++; }
        }
        retention = tot_hot>0 ? (float)liq_hot/tot_hot : 1.f;
    } else {
        int sol_cold=0, tot_cold=0;
        for (int i=0; i<N*N; i++) {
            if (T_init[i] <= T_mid) { tot_cold++; if (phi[i]>0.5f) sol_cold++; }
        }
        retention = tot_cold>0 ? (float)sol_cold/tot_cold : 1.f;
    }

    float sharp = sharpness_score(phi, N, cold_rad, tgt_width);

    // Velocity score: rewards early-growth-phase Stefan behavior.
    // Window: steps n/8 → n/4 (front is actively growing before latent heat suppression).
    // v_measured = (r_quarter - r_early) / (n/8 * dt)  [cells per time]
    // v_stefan   = D_T * |undercooling| / (L * r_mid)  [analytical Stefan velocity]
    // score = exp(-(log(v/v_stefan))^2 / 0.25)  IF v > 0, ELSE 0
    // Hard step on v > 0: functions that don't advance at all score 0.
    float velocity_score = 0.f;
    if (r_early > 0.f && r_quarter > 0.f) {
        float window_t = (float)(steps / 8) * dt;
        float v_measured = (r_quarter - r_early) / window_t;
        if (v_measured > 0.f) {
            float r_mid = 0.5f * (r_early + r_quarter);
            if (r_mid > 0.5f) {
                float undercooling = is_melt_mode ? (hot_temp - 0.5f) : (0.5f - cold_temp);
                float v_stefan = D_T * undercooling / (L * r_mid);
                if (v_stefan > 1e-6f) {
                    float log_ratio = logf(v_measured / v_stefan);
                    velocity_score = expf(-(log_ratio * log_ratio) / 0.25f);
                }
            }
        }
        // v_measured <= 0: no early-phase growth → velocity_score = 0
    }

    // Fitness v2: T_balance (0.30) + velocity (0.30) + sharpness (0.25) + retention (0.15) = 1.00
    float fitness = 0.30f*T_balance_score + 0.30f*velocity_score + 0.25f*sharp + 0.15f*retention;

    printf("{\"fitness\":%.6f,\"T_balance\":%.4f,\"velocity\":%.4f,\"sharpness\":%.4f,"
           "\"retention\":%.4f,"
           "\"mean_phi_cold\":%.4f,\"mean_T_cold\":%.4f,"
           "\"r_early\":%.2f,\"r_quarter\":%.2f,"
           "\"valid\":true,\"name\":\"%s\"}\n",
           fitness, T_balance_score, velocity_score, sharp, retention,
           mean_phi_cold, mean_T_cold, r_early, r_quarter, name.c_str());
    return 0;
}
"""

LATENT_FUNCTION_SIGNATURE = """\
float reaction(float phi, float temp, float lap_T);

// Latent-heat Allen-Cahn driving force — coupled phase + thermal evolution.
// phi:   order parameter ∈ [0,1]  (0 = liquid, 1 = solid)
// temp:  LOCAL temperature ∈ [0,1]  (evolves — rises as ice forms via latent heat)
// lap_T: Laplacian of temperature at this cell  (∇²T, dimensionless)
//          > 0: heat flowing INTO this cell (freeze front heating up)
//          < 0: heat flowing OUT  (cold diffusing in, more undercooling available)
//          ≈ 0: bulk region, already equilibrated
//
// SIGN CONVENTION (correct):
//   temp < 0.5  →  COLD  →  m > 0  →  pushes phi → 1 (solid)
//   temp > 0.5  →  HOT   →  m < 0  →  pushes phi → 0 (liquid)
//
// LATENT HEAT SELF-LIMITING:
//   Freezing (∂φ/∂t > 0) releases heat: T rises until T → 0.5 → m → 0 → equilibrium
//   Target coverage φ_eq = (0.5 − T_cold_initial) / L
//   For T_cold=0.25, L=0.4:  φ_eq = 0.625
//
// Classic Allen-Cahn (ignores lap_T):
//   W′(phi) = 4φ³ − 6φ² + 2φ
//   reaction = −W′(phi) + 2*(0.5 − temp)
//
// Available: sinf cosf expf logf sqrtf fabsf powf tanhf fmaxf fminf fmodf
// No loops, no static state. Under 20 lines."""


def build_candidate_source(code: str, inst: LatentInstance) -> tuple[str, str]:
    cpp = _LATENT_EVALUATOR_CPP.replace("// ===== EVOLVED_CODE_MARKER =====", code)
    stdin_data = json.dumps({
        "name": inst.name,
        "cold_temp": inst.cold_temp,
        "hot_temp": inst.hot_temp,
        "cold_radius": inst.cold_radius,
        "D_phi": inst.D_phi,
        "D_T": inst.D_T,
        "L": inst.L,
        "dt": inst.dt,
        "n_steps": inst.n_steps,
        "grid_size": inst.grid_size,
        "initial_phi": inst.initial_phi,
        "target_width": inst.target_width,
    })
    return cpp, stdin_data


def evaluate_on_instance(code: str, inst: LatentInstance,
                          executor: Any,
                          run_timeout: float = 60.0) -> Optional[float]:
    cpp_source, stdin_data = build_candidate_source(code, inst)
    stdout, verdict, _lat = compile_and_run(
        cpp_source, language="cpp", stdin=stdin_data, executor=executor
    )
    if verdict != Verdict.OK or not stdout:
        return None
    try:
        out = json.loads(stdout.strip().split("\n")[-1])
    except Exception:
        return None
    if not out.get("valid", False):
        return None
    fitness = float(out.get("fitness", 0.0))
    return fitness if math.isfinite(fitness) and fitness > 0 else None


def ensure_executor(allow_unsandboxed: bool = False) -> SandboxedExecutor:
    return ensure_sandboxed_executor(allow_unsandboxed=allow_unsandboxed)


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


# ---------------------------------------------------------------------------
# LatentKernel
# ---------------------------------------------------------------------------

class LatentKernel(FunSearchKernel):
    """FunSearch kernel: 2D coupled phase-thermal PDE with latent heat.

    Evolves reaction(phi, temp, lap_T) against an oracle where temperature
    EVOLVES — it rises as ice forms (latent heat) and the freeze front is
    thermodynamically self-limiting at φ_eq = (0.5 − T_cold) / L.

    Fitness v2 = 0.30·T_balance + 0.30·velocity + 0.25·sharpness + 0.15·retention

    velocity_score rewards Stefan self-regulation: the oracle tracks the front position
    at the midpoint step and final step, comparing observed velocity to the analytical
    Stefan prediction v_stefan = D_T * |undercooling| / (L * r_mid). Retreating fronts
    score 0 (hard step function on v > 0).

    D_phi reduced to 4.0 from 10.0: natural interface width sqrt(4/0.5) ≈ 2.8 cells
    makes target_width=3 achievable, unblocking the sharpness score from dead weight.
    """

    kernel_name = "latent"

    def __init__(self, config: KernelConfig) -> None:
        super().__init__(config)
        self.executor = ensure_executor(allow_unsandboxed=config.allow_unsandboxed)
        self.problem_instances = self.load_instances()
        for inst in self.problem_instances:
            logger.info(
                "Latent instance %r: cold=%.2f hot=%.2f R=%.0f L=%.2f φ_eq=%.3f",
                inst.name, inst.cold_temp, inst.hot_temp,
                inst.cold_radius, inst.L, inst.phi_eq
            )

    def load_instances(self) -> list[LatentInstance]:
        return [generate_instance(n) for n in self.config.instances]

    def evaluate_candidate(self, code: str, instance: Any) -> Optional[float]:
        return evaluate_on_instance(code, instance, self.executor)

    def build_prompt(self, island: Island, top_programs: list[CandidateProgram],
                     generation: int, hint: str = "") -> str:
        instance = self.problem_instances[0] if self.problem_instances else None
        phi_eq_str = f"{instance.phi_eq:.3f}" if instance else "0.625"
        inst_desc  = (f"{instance.name} — {instance.description}" if instance else "unknown")
        cold = instance.cold_temp if instance else 0.25
        hot  = instance.hot_temp  if instance else 0.75
        L    = instance.L         if instance else 0.4
        D_T  = instance.D_T       if instance else 5.0

        # Compute reference Stefan velocity for prompt context
        undercooling = 0.5 - cold if instance and instance.initial_phi < 0.5 else (hot - 0.5)
        v_stefan_ref = f"{D_T * undercooling / (L * 15.0):.3f}"  # at r~15 cells

        exemplars = "\n\n".join(
            f"// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}"
            for i, p in enumerate(top_programs[:3])
        )
        hint_block = f"\n## Strategic hint\n{hint}\n" if hint else ""

        return (
            f"You are a computational physicist specialising in phase-field models "
            f"and thermodynamic coupling.\n"
            f"Evolve a better Allen-Cahn driving force with latent heat feedback.\n\n"
            f"{LATENT_FUNCTION_SIGNATURE}\n\n"
            f"Target: {inst_desc}\n"
            f"  cold_temp={cold:.2f}  hot_temp={hot:.2f}  L={L:.2f}  φ_eq={phi_eq_str}\n"
            f"Island {island.id} — Generation {generation}\n"
            f"{hint_block}\n"
            f"Fitness = 0.30·T_balance + 0.30·velocity + 0.25·sharpness + 0.15·retention\n"
            f"  T_balance:  Gaussian(mean_T_cold − 0.5, σ=0.10) — latent heat equilibrium\n"
            f"  velocity:   exp(-(log(v/v_stefan))^2/0.25) if v>0, else 0\n"
            f"              v_stefan ≈ {v_stefan_ref} cells/time at r~15 cells\n"
            f"              KEY: retreating fronts (v<0) score 0 — self-regulation required\n"
            f"  sharpness:  Gaussian reward for interface width ≈ 3 cells\n"
            f"  retention:  opposite zone stays in correct phase\n\n"
            f"Top programs:\n\n{exemplars}\n\n"
            f"Rules:\n"
            f"- Return ONLY a single ```cpp code block\n"
            f"- Signature: float reaction(float phi, float temp, float lap_T)\n"
            f"- |reaction| < 50 for all inputs\n"
            f"- No loops, no static state, under 20 lines\n"
        )

    def parse_response(self, response: str) -> str:
        m = re.search(r'```(?:cpp|c\+\+)?\s*\n(.*?)```', response, re.DOTALL)
        if m:
            code = m.group(1).strip()
            if "reaction" in code and "lap_T" in code:
                return code
        return ""

    def seed_programs(self, island_id: int, generation: int) -> list[CandidateProgram]:
        programs = []
        for name, code in get_seed_programs(
            self.config.instances[0] if self.config.instances else "freeze_latent"
        ):
            prog = CandidateProgram(
                id=str(uuid.uuid4()), code=code,
                island=island_id, generation=generation,
            )
            programs.append(prog)
        return programs

    def save_results(self, programs: list[CandidateProgram]) -> Optional[Path]:
        if not self.config.output_dir:
            return None
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"latent_results_gen{self.generation:02d}.json"

        top = sorted(programs, key=lambda p: p.fitness, reverse=True)
        data = {
            "kernel": "latent",
            "run_id": self.run_id,
            "generation": self.generation,
            "stop_reason": self.stop_reason,
            "instances": self.config.instances,
            "top_programs": [
                {
                    "id": p.id, "fitness": p.fitness, "worst_fitness": p.worst_fitness,
                    "island": p.island, "generation": p.generation,
                    "reaction_code": p.code,
                }
                for p in top[:20]
            ],
        }
        out_path.write_text(json.dumps(data, indent=2))
        logger.info("Latent results → %s (best=%.4f)", out_path,
                    top[0].fitness if top else 0)
        return out_path
