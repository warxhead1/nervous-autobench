"""C++ evaluator source + candidate scoring (split from thermal_kernel/__init__.py)."""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from ..core import Verdict
from ..engines.sandbox import compile_and_run
from .instance import ThermalInstance


# ---------------------------------------------------------------------------
# C++ evaluator — 2D Allen-Cahn time-stepper, fixed temperature field
# ---------------------------------------------------------------------------

_THERMAL_EVALUATOR_CPP = r"""
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

// ── 2D Allen-Cahn solver ─────────────────────────────────────────────────────
// Grid N×N flat array phi[i*N+j], Neumann BC, dx=1.
// 2D stability: D*dt/dx^2 <= 0.25 (5-point stencil, half the 1D bound).
static int run_pde_2d(int N, const vector<float>& T_field, float D, float dt,
                      int steps, float init_phi, vector<float>& phi_out) {
    int sz = N * N;
    vector<float> phi(sz, init_phi);
    vector<float> phi_new(sz);

    // Nucleation seed at centre — strong enough to overcome interface barrier
    // Functions with zero driving force at phi=0 (like logistic forms) need this
    int cx = N / 2, cy = N / 2;
    float seed_val = (init_phi < 0.5f) ? 0.65f : 0.35f;
    for (int di = -5; di <= 5; di++)
        for (int dj = -5; dj <= 5; dj++) {
            if (di*di + dj*dj > 25) continue;  // circle of radius 5
            int ii = cx + di, jj = cy + dj;
            if (ii >= 0 && ii < N && jj >= 0 && jj < N)
                phi[ii*N+jj] = seed_val;
        }

    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float p = phi[i*N+j];
                float pL = (i > 0)   ? phi[(i-1)*N+j] : p;
                float pR = (i < N-1) ? phi[(i+1)*N+j] : p;
                float pD = (j > 0)   ? phi[i*N+(j-1)] : p;
                float pU = (j < N-1) ? phi[i*N+(j+1)] : p;
                float lap = pL + pR + pD + pU - 4.f*p;

                float T = T_field[i*N+j];
                float r = reaction(p, T);
                if (!isfinite(r) || fabsf(r) > 200.f) return -1;
                phi_new[i*N+j] = p + dt * (D * lap + r);
                if (!isfinite(phi_new[i*N+j])) return -1;
            }
        }
        swap(phi, phi_new);
    }

    float phi_min = phi[0], phi_max = phi[0];
    for (float v : phi) { phi_min = fminf(phi_min, v); phi_max = fmaxf(phi_max, v); }
    if (phi_min < -0.3f || phi_max > 1.3f || (phi_max - phi_min) < 0.02f)
        return -2;  // collapsed

    phi_out = phi;
    return 0;
}

// ── Temperature field ─────────────────────────────────────────────────────
static void build_T_field(int N, float cold_temp, float hot_temp,
                            float cold_radius, bool is_melt_mode,
                            vector<float>& T_field) {
    T_field.resize(N * N);
    float cx = (N - 1) * 0.5f, cy = (N - 1) * 0.5f;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float r = sqrtf((i-cx)*(i-cx) + (j-cy)*(j-cy));
            bool in_spot = (r < cold_radius);
            // freeze_mode: cold inside (freezes), hot outside
            // melt_mode:   hot inside (melts), cold outside
            if (!is_melt_mode)
                T_field[i*N+j] = in_spot ? cold_temp : hot_temp;
            else
                T_field[i*N+j] = in_spot ? hot_temp : cold_temp;
        }
}

// ── Radial sharpness score ────────────────────────────────────────────────
// Measures how sharp the ice-liquid interface is in the radial profile.
// target_width = desired number of radial bins with phi ∈ (0.1, 0.9).
// Classical Allen-Cahn with D=10 produces ~5 bins; target_width=2 rewards sharper functions.
// Guard: requires a genuine phi transition (min<0.3 AND max>0.7) — prevents
// score inflation when everything is frozen or all liquid.
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

    // Check for genuine transition in radial profile
    float rmin = 1.f, rmax = 0.f;
    for (int r = 0; r <= max_r; r++) {
        if (cnt[r] < 1) continue;
        float v = (float)(sum_phi[r]/cnt[r]);
        rmin = fminf(rmin, v); rmax = fmaxf(rmax, v);
    }
    if (rmin > 0.3f || rmax < 0.7f) return 0.f;  // no transition → no sharpness score

    // Count interface bins: radial cells where phi_r ∈ (0.1, 0.9)
    float iface_bins = 0.f;
    for (int r = 0; r <= max_r; r++) {
        if (cnt[r] < 1) continue;
        float v = (float)(sum_phi[r]/cnt[r]);
        if (v > 0.1f && v < 0.9f) iface_bins += 1.f;
    }

    // Penalise excess width beyond target; reward at target
    float excess = fmaxf(0.f, iface_bins - tgt_width);
    return expf(-(excess*excess) / (tgt_width*tgt_width));
}

// ── Coverage fraction ──────────────────────────────────────────────────────
// Returns fraction of target zone where phi is in the desired state.
static float coverage(const vector<float>& phi, const vector<float>& T_field,
                       int N, float T_threshold, bool want_solid) {
    int count=0, total=0;
    for (int i=0; i<N*N; i++) {
        bool in_target = (T_field[i] <= T_threshold);
        if (!in_target) continue;
        total++;
        if (want_solid  && phi[i] > 0.5f) count++;
        if (!want_solid && phi[i] < 0.5f) count++;
    }
    return total>0 ? (float)count/total : 0.f;
}

int main() {
    getline(cin, g_json);
    if (g_json.empty()) getline(cin, g_json);

    string name      = get_str("name");
    float cold_temp  = get_float("cold_temp",   0.25f);
    float hot_temp   = get_float("hot_temp",    0.75f);
    float cold_rad   = get_float("cold_radius", 18.0f);
    float D          = get_float("D",     10.0f);
    float dt         = get_float("dt",    0.018f);
    int   steps      = get_int ("n_steps",  400);
    int   N          = get_int ("grid_size",  64);
    float init_phi   = get_float("initial_phi",  0.02f);
    float tgt_width  = get_float("target_width",  4.0f);

    // 2D stability: D*dt <= 0.25
    if (D * dt > 0.25f + 1e-5f) {
        printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"stability_violated\","
               "\"detail\":\"D*dt=%.4f>0.25\",\"name\":\"%s\"}\n", D*dt, name.c_str());
        return 0;
    }

    bool is_melt_mode = (init_phi > 0.5f);  // melt_spot starts solid, freeze_spot starts liquid

    // Corner stability check
    float test_corners[4] = {
        reaction(0.0f, cold_temp), reaction(1.0f, cold_temp),
        reaction(0.0f, 0.5f),     reaction(1.0f, 0.5f)
    };
    for (int i=0; i<4; i++) {
        if (!isfinite(test_corners[i]) || fabsf(test_corners[i]) > 200.f) {
            printf("{\"fitness\":0.001,\"valid\":false,\"reason\":\"unstable_corners\","
                   "\"name\":\"%s\"}\n", name.c_str());
            return 0;
        }
    }

    vector<float> T_field;
    build_T_field(N, cold_temp, hot_temp, cold_rad, is_melt_mode, T_field);

    vector<float> phi;
    int status = run_pde_2d(N, T_field, D, dt, steps, init_phi, phi);
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

    float T_mid = (cold_temp + hot_temp) * 0.5f;

    // Coverage: target phase fills its zone
    // Retention: opposite phase stays in its zone
    float phase_cov, retention;
    if (!is_melt_mode) {
        // freeze: cold zone → solid, hot zone → liquid
        int sol_cold=0, tot_cold=0, liq_hot=0, tot_hot=0;
        for (int i=0; i<N*N; i++) {
            if (T_field[i] <= T_mid) { tot_cold++; if (phi[i] > 0.5f) sol_cold++; }
            else                     { tot_hot++;  if (phi[i] < 0.5f) liq_hot++;  }
        }
        phase_cov = tot_cold>0 ? (float)sol_cold/tot_cold : 0.f;
        retention = tot_hot>0  ? (float)liq_hot/tot_hot   : 0.f;
    } else {
        // melt: hot zone → liquid, cold zone → solid
        int liq_hot=0, tot_hot=0, sol_cold=0, tot_cold=0;
        for (int i=0; i<N*N; i++) {
            if (T_field[i] > T_mid) { tot_hot++;  if (phi[i] < 0.5f) liq_hot++;  }
            else                    { tot_cold++; if (phi[i] > 0.5f) sol_cold++; }
        }
        phase_cov = tot_hot>0  ? (float)liq_hot/tot_hot   : 0.f;
        retention = tot_cold>0 ? (float)sol_cold/tot_cold : 0.f;
    }

    // Sharpness: interface should be narrow (target_width cells in transition zone)
    float sharp = sharpness_score(phi, N, cold_rad, tgt_width);

    // Fitness: phase coverage + zone retention + interface sharpness
    float fitness = 0.30f*phase_cov + 0.30f*retention + 0.40f*sharp;

    printf("{\"fitness\":%.6f,\"phase_coverage\":%.4f,\"retention\":%.4f,"
           "\"sharpness\":%.4f,\"valid\":true,\"name\":\"%s\"}\n",
           fitness, phase_cov, retention, sharp, name.c_str());
    return 0;
}
"""

THERMAL_FUNCTION_SIGNATURE = '''\
float reaction(float phi, float temp);

// 2D Allen-Cahn driving force — evolving ice/water phase transitions.
// phi:  order parameter ∈ [0,1]  (0 = liquid, 1 = solid)
// temp: dimensionless temperature ∈ [0,1]  (0.5 = melting point)
//
// SIGN CONVENTION — critical:
//   temp < 0.5  →  COLD  →  solid preferred  →  should push phi toward 1
//   temp > 0.5  →  HOT   →  liquid preferred  →  should push phi toward 0
//
// Classical Allen-Cahn with CORRECT freezing convention:
//   W′(phi) = 4*phi^3 - 6*phi^2 + 2*phi
//   m = 2*(0.5 - temp)   ← positive when cold → nucleates solid
//   reaction = -W′(phi) + m
//
// Available: sinf cosf expf logf sqrtf fabsf powf tanhf fmaxf fminf fmodf
// No loops, no static state. Under 20 lines.
'''


def build_candidate_source(code: str, inst: ThermalInstance) -> tuple[str, str]:
    cpp = _THERMAL_EVALUATOR_CPP.replace("// ===== EVOLVED_CODE_MARKER =====", code)
    stdin_data = json.dumps({
        "name": inst.name,
        "cold_temp": inst.cold_temp,
        "hot_temp": inst.hot_temp,
        "cold_radius": inst.cold_radius,
        "D": inst.D,
        "dt": inst.dt,
        "n_steps": inst.n_steps,
        "grid_size": inst.grid_size,
        "initial_phi": inst.initial_phi,
        "target_width": inst.target_width,
    })
    return cpp, stdin_data


def evaluate_on_instance(code: str, inst: ThermalInstance,
                          executor: Any,
                          run_timeout: float = 30.0) -> Optional[float]:
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
