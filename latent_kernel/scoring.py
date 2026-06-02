"""Latent kernel C++ evaluator and sandbox scoring.

The coupled φ+T PDE C++ source, the reaction signature shown to the LLM,
candidate-source assembly, and the sandbox evaluation entry point.
Moved verbatim from the package __init__ (behaviour-preserving file split).

`compile_and_run` and `Verdict` are resolved through the package namespace at
call time so that any test patch on `autobench.latent_kernel.compile_and_run` /
`.Verdict` takes effect (mirrors the sdf_kernel split).
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from .instance import LatentInstance


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
    # Resolve compile_and_run / Verdict through the package namespace at call
    # time so test patches on autobench.latent_kernel.* take effect.
    from . import compile_and_run, Verdict

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
