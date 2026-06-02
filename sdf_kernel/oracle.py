"""Oracle / LLM-facing helpers — C++ skeleton, seed programs, prompts, scoring.

Holds the fixed C++ MSE harness skeleton, the per-instance seed programs,
island personas + prompt sketches, the prompt builder, the LLM-response parser,
and the candidate evaluation entrypoint (evaluate_on_instance).
"""

from __future__ import annotations

import json
import logging
import math
import re

from .instance import SDFInstance
from .topology import compute_topology_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C++ skeleton — fixed harness, LLM injects sdf(x, y, z)
# ---------------------------------------------------------------------------

CPP_SKELETON = r"""// Auto-generated SDF skeleton. LLM writes sdf() only.
// Reads sample points from stdin JSON; evaluates sdf() and returns MSE.
#include <bits/stdc++.h>
using namespace std;

// === LLM-EVOLVED FUNCTION — do not modify the signature ===
extern "C" float sdf(float x, float y, float z);
// ===========================================================

struct Sample { float x, y, z, expected; };

int main() {
    string s((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());

    // Hand-rolled JSON parser to avoid external dependencies.
    // Expects: {"name":"...","samples":[[x,y,z,expected],...]}
    int pos = 0;
    auto skip_ws = [&]() {
        while (pos < (int)s.size() && isspace((unsigned char)s[pos])) pos++;
    };
    auto expect_char = [&](char c) -> bool {
        skip_ws();
        if (pos < (int)s.size() && s[pos] == c) { pos++; return true; }
        return false;
    };
    auto parse_string = [&]() -> string {
        skip_ws();
        if (pos >= (int)s.size() || s[pos] != '"') return string();
        pos++; string out;
        while (pos < (int)s.size() && s[pos] != '"') {
            if (s[pos] == '\\') pos++;
            if (pos < (int)s.size()) out += s[pos++];
        }
        if (pos < (int)s.size()) pos++;
        return out;
    };
    auto parse_number = [&]() -> double {
        skip_ws();
        bool neg = false;
        if (pos < (int)s.size() && s[pos] == '-') { neg = true; pos++; }
        double v = 0;
        while (pos < (int)s.size() && isdigit((unsigned char)s[pos]))
            { v = v*10 + (s[pos]-'0'); pos++; }
        if (pos < (int)s.size() && s[pos] == '.') {
            pos++; double frac = 0.1;
            while (pos < (int)s.size() && isdigit((unsigned char)s[pos]))
                { v += (s[pos]-'0')*frac; frac*=0.1; pos++; }
        }
        if (pos < (int)s.size() && (s[pos]=='e'||s[pos]=='E')) {
            pos++; bool eneg = false;
            if (pos<(int)s.size()&&(s[pos]=='+'||s[pos]=='-')){eneg=(s[pos]=='-');pos++;}
            int exp=0;
            while(pos<(int)s.size()&&isdigit((unsigned char)s[pos])){exp=exp*10+(s[pos]-'0');pos++;}
            double mult=1; for(int i=0;i<exp;i++) mult*=10;
            if(eneg) v/=mult; else v*=mult;
        }
        return neg ? -v : v;
    };

    string inst_name;
    vector<Sample> samples;

    expect_char('{');
    while (true) {
        skip_ws();
        if (pos >= (int)s.size() || s[pos] == '}') { pos++; break; }
        string key = parse_string();
        expect_char(':');
        if (key == "name") {
            inst_name = parse_string();
        } else if (key == "samples") {
            expect_char('[');
            while (true) {
                skip_ws();
                if (pos>=(int)s.size()||s[pos]==']') { pos++; break; }
                expect_char('[');
                float x=(float)parse_number(); expect_char(',');
                float y=(float)parse_number(); expect_char(',');
                float z=(float)parse_number(); expect_char(',');
                float e=(float)parse_number();
                expect_char(']');
                samples.push_back({x,y,z,e});
                skip_ws(); if (pos<(int)s.size()&&s[pos]==',') pos++;
            }
        } else {
            // Skip unknown fields
            skip_ws();
            int depth = 0;
            if (pos<(int)s.size()&&(s[pos]=='{'||s[pos]=='[')) {
                while(pos<(int)s.size()) {
                    if(s[pos]=='{'||s[pos]=='[') depth++;
                    else if(s[pos]=='}'||s[pos]==']') { depth--; if(depth==0){pos++;break;} }
                    pos++;
                }
            } else if (pos<(int)s.size()&&s[pos]=='"') { parse_string(); }
            else { parse_number(); }
        }
        skip_ws(); if (pos<(int)s.size()&&s[pos]==',') pos++;
    }

    if (samples.empty()) {
        cerr << "ERROR: no samples parsed" << endl;
        return 1;
    }

    double mse = 0.0;
    int n = (int)samples.size();
    for (auto& smp : samples) {
        double got = (double)sdf(smp.x, smp.y, smp.z);
        double diff = got - (double)smp.expected;
        mse += diff * diff;
    }
    mse /= n;

    // Eikonal check: |∇sdf| should = 1 everywhere (valid signed-distance field).
    // Finite-difference gradient at every 5th sample point.
    const float h = 1e-3f;
    double grad_err_sum = 0.0;
    int n_grad = 0;
    for (int i = 0; i < n; i += 5) {
        auto& smp = samples[i];
        float dx = (sdf(smp.x+h,smp.y,smp.z) - sdf(smp.x-h,smp.y,smp.z)) / (2*h);
        float dy = (sdf(smp.x,smp.y+h,smp.z) - sdf(smp.x,smp.y-h,smp.z)) / (2*h);
        float dz = (sdf(smp.x,smp.y,smp.z+h) - sdf(smp.x,smp.y,smp.z-h)) / (2*h);
        float grad_mag = sqrtf(dx*dx + dy*dy + dz*dz);
        if (isfinite(grad_mag)) { grad_err_sum += fabs(grad_mag - 1.0f); n_grad++; }
    }
    double mean_grad_err = (n_grad > 0) ? grad_err_sum / n_grad : 1.0;

    printf("{\"mse\":%.9f,\"grad_err\":%.6f,\"n\":%d,\"instance\":\"%s\"}\n",
           mse, mean_grad_err, n, inst_name.c_str());
    return 0;
}
"""


SDF_FUNCTION_SIGNATURE = '''\
extern "C" float sdf(float x, float y, float z);

// Goal: return the signed distance from point (x,y,z) to the target surface.
// Positive outside the surface, negative inside, zero exactly on the surface.
// The returned value is compared against precomputed ground-truth distances.
// Fitness = 1 / (1 + MSE) — minimise MSE to maximise fitness.
//
// Available C math functions (no includes needed — skeleton provides them):
//   sqrtf, fabsf, fmaxf, fminf, sinf, cosf, atan2f, powf, expf, logf
//
// Useful SDF building blocks:
//   sphere(R):      sqrtf(x*x+y*y+z*z) - R
//   box(bx,by,bz):  q=max(|.|−b,0); length(q)+min(max(q.x,q.y,q.z),0)
//   torus(R,r):     q=sqrtf(x*x+y*y)-R; sqrtf(q*q+z*z)-r
//   smooth_union:   h=max(k-|d1-d2|,0)/k; min(d1,d2)-h*h*k/4
//   domain twist:   (xw,yw) = rotate(x,y, angle*z)
//   gyroid:         sin(s*x)*cos(s*y)+sin(s*y)*cos(s*z)+sin(s*z)*cos(s*x)
//   cloud blob:     fminf cascade of 5-8 spheres at asymmetric offsets
//   torus knot(p,q):((R+r*cos(q*t))*cos(p*t),(R+r*cos(q*t))*sin(p*t),r*sin(q*t)); min-dist tube
//   helix tube:     (R*cos(t), pitch*t/2pi, R*sin(t)); use atan2f+k-loop for nearest t
//   Scherk surf:    f=exp(z)*cos(y)-cos(x); SDF≈|f|/sqrt(sin²x+exp(2z))
'''


# ---------------------------------------------------------------------------
# Baseline seed programs — analytical solutions per instance family
# ---------------------------------------------------------------------------

SEED_SDF_PROGRAMS: dict[str, list[tuple[str, str]]] = {
    "generic": [
        ("sphere_baseline", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
        ("box_baseline", '''\
extern "C" float sdf(float x, float y, float z) {
    float qx = fabsf(x) - 0.7f;
    float qy = fabsf(y) - 0.5f;
    float qz = fabsf(z) - 0.6f;
    float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
    return sqrtf(mx*mx + my*my + mz*mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
}'''),
        ("torus_baseline", '''\
extern "C" float sdf(float x, float y, float z) {
    float q = sqrtf(x*x + y*y) - 0.9f;
    return sqrtf(q*q + z*z) - 0.3f;
}'''),
    ],
    "sphere": [
        ("sphere_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
        ("sphere_approx1", '''\
extern "C" float sdf(float x, float y, float z) {
    // Intentionally imperfect: Euclidean approximation
    float r2 = x*x + y*y + z*z;
    return sqrtf(r2) - 1.05f;
}'''),
        ("sphere_scaled", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 0.95f;
}'''),
    ],
    "gyroid": [
        ("gyroid_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    float s = 2.5f;
    float f = sinf(s*x)*cosf(s*y) + sinf(s*y)*cosf(s*z) + sinf(s*z)*cosf(s*x);
    float grad = s * 1.7320508f;  // sqrt(3)
    return fabsf(f) / grad - 0.15f;
}'''),
        ("gyroid_no_thickness", '''\
extern "C" float sdf(float x, float y, float z) {
    float s = 2.5f;
    float f = sinf(s*x)*cosf(s*y) + sinf(s*y)*cosf(s*z) + sinf(s*z)*cosf(s*x);
    return f * 0.23f;
}'''),
        ("gyroid_thick", '''\
extern "C" float sdf(float x, float y, float z) {
    float s = 2.5f;
    float f = sinf(s*x)*cosf(s*y) + sinf(s*y)*cosf(s*z) + sinf(s*z)*cosf(s*x);
    float grad = s * 1.7320508f;
    return fabsf(f) / grad - 0.25f;
}'''),
    ],
    "round_box": [
        ("round_box_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    float qx = fabsf(x) - 0.7f;
    float qy = fabsf(y) - 0.4f;
    float qz = fabsf(z) - 0.5f;
    float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
    return sqrtf(mx*mx + my*my + mz*mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f) - 0.15f;
}'''),
        ("box_no_rounding", '''\
extern "C" float sdf(float x, float y, float z) {
    float qx = fabsf(x) - 0.7f;
    float qy = fabsf(y) - 0.4f;
    float qz = fabsf(z) - 0.5f;
    float mx = fmaxf(qx, 0.0f), my = fmaxf(qy, 0.0f), mz = fmaxf(qz, 0.0f);
    return sqrtf(mx*mx + my*my + mz*mz) + fminf(fmaxf(qx, fmaxf(qy, qz)), 0.0f);
}'''),
        ("sphere_fallback", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 0.9f;
}'''),
    ],
    "warped_sphere": [
        ("sphere_no_warp", '''\
extern "C" float sdf(float x, float y, float z) {
    // Suboptimal: plain sphere (missing the warp terms) — scores ~0.62 fitness
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
        ("warped_sphere_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // Exact: sphere + 3-axis sinusoidal domain warp (amplitude=0.25, freq=3.0)
    float xw = x + 0.25f * sinf(3.0f * y);
    float yw = y + 0.25f * sinf(3.0f * z);
    float zw = z + 0.25f * sinf(3.0f * x);
    return sqrtf(xw*xw + yw*yw + zw*zw) - 1.0f;
}'''),
        ("warped_sphere_partial", '''\
extern "C" float sdf(float x, float y, float z) {
    // Partial: warp only x-axis — suboptimal, tests partial correction
    float xw = x + 0.25f * sinf(3.0f * y);
    return sqrtf(xw*xw + y*y + z*z) - 1.0f;
}'''),
    ],
    "smooth_union": [
        ("three_spheres_min", '''\
extern "C" float sdf(float x, float y, float z) {
    float d1 = sqrtf((x-0.6f)*(x-0.6f)+y*y+z*z) - 0.5f;
    float d2 = sqrtf((x+0.6f)*(x+0.6f)+y*y+z*z) - 0.5f;
    float d3 = sqrtf(x*x+(y-0.7f)*(y-0.7f)+z*z) - 0.4f;
    return fminf(fminf(d1, d2), d3);
}'''),
        ("smooth_union_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    float d1 = sqrtf((x-0.6f)*(x-0.6f)+y*y+z*z) - 0.5f;
    float d2 = sqrtf((x+0.6f)*(x+0.6f)+y*y+z*z) - 0.5f;
    float d3 = sqrtf(x*x+(y-0.7f)*(y-0.7f)+z*z) - 0.4f;
    // smooth_union with k=0.3
    float k = 0.3f;
    float h12 = fmaxf(k - fabsf(d1-d2), 0.0f) / k;
    float su12 = fminf(d1, d2) - h12*h12*k*0.25f;
    k = 0.25f;
    float h = fmaxf(k - fabsf(su12-d3), 0.0f) / k;
    return fminf(su12, d3) - h*h*k*0.25f;
}'''),
        ("single_sphere", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 0.8f;
}'''),
    ],
    "cloud_cluster": [
        ("cloud_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // True min-union of 7 spheres (eikonal-valid SDF)
    float d = sqrtf(x*x + y*y + z*z) - 0.55f;
    d = fminf(d, sqrtf((x-0.5f)*(x-0.5f)+(y-0.15f)*(y-0.15f)+(z-0.1f)*(z-0.1f))-0.45f);
    d = fminf(d, sqrtf((x+0.45f)*(x+0.45f)+(y-0.2f)*(y-0.2f)+(z-0.05f)*(z-0.05f))-0.42f);
    d = fminf(d, sqrtf((x-0.15f)*(x-0.15f)+(y-0.42f)*(y-0.42f)+(z+0.22f)*(z+0.22f))-0.38f);
    d = fminf(d, sqrtf((x+0.22f)*(x+0.22f)+(y-0.45f)*(y-0.45f)+(z-0.28f)*(z-0.28f))-0.35f);
    d = fminf(d, sqrtf((x-0.38f)*(x-0.38f)+(y-0.52f)*(y-0.52f)+(z-0.12f)*(z-0.12f))-0.30f);
    d = fminf(d, sqrtf((x+0.1f)*(x+0.1f)+(y-0.62f)*(y-0.62f)+(z+0.08f)*(z+0.08f))-0.26f);
    return d;
}'''),
        ("cloud_3sphere", '''\
extern "C" float sdf(float x, float y, float z) {
    // Partial: only 3 of 7 spheres — suboptimal, missing upper puffs
    float d = sqrtf(x*x + y*y + z*z) - 0.55f;
    d = fminf(d, sqrtf((x-0.5f)*(x-0.5f)+(y-0.15f)*(y-0.15f)+(z-0.1f)*(z-0.1f))-0.45f);
    d = fminf(d, sqrtf((x+0.45f)*(x+0.45f)+(y-0.2f)*(y-0.2f)+(z-0.05f)*(z-0.05f))-0.42f);
    return d;
}'''),
        ("cloud_sphere_fallback", '''\
extern "C" float sdf(float x, float y, float z) {
    // Worst-case: single large sphere
    return sqrtf(x*x + (y-0.2f)*(y-0.2f) + z*z) - 0.9f;
}'''),
    ],
    "torus_knot": [
        ("torus_knot_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // (2,3) torus knot: sample N=400 points on curve, find min distance
    const float R = 0.6f, rm = 0.35f, tube = 0.15f;
    const float pi2 = 6.28318530f;
    float best = 1e9f;
    for (int i = 0; i < 400; i++) {
        float t = pi2 * i / 400.0f;
        float rr = R + rm * cosf(3.0f * t);
        float kx = rr * cosf(2.0f * t);
        float ky = rr * sinf(2.0f * t);
        float kz = rm * sinf(3.0f * t);
        float d = sqrtf((x-kx)*(x-kx)+(y-ky)*(y-ky)+(z-kz)*(z-kz));
        if (d < best) best = d;
    }
    return best - tube;
}'''),
        ("torus_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    // Plain torus (missing the knot winding) — suboptimal seed
    float rho = sqrtf(x*x + y*y) - 0.6f;
    return sqrtf(rho*rho + z*z) - 0.25f;
}'''),
        ("sphere_fallback", '''\
extern "C" float sdf(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z) - 1.0f;
}'''),
    ],
    "helix_tube": [
        ("helix_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // Helix around y-axis: C(t) = (R*cos(t), pitch*t/(2π), R*sin(t)), 2 turns
    // N=800 for spacing << tube_radius
    const float R = 0.7f, pitch = 0.5f, tube = 0.15f;
    const float pi2 = 6.28318530f;
    float best = 1e9f;
    for (int i = 0; i <= 800; i++) {
        float t = 4.0f * pi2 * i / 800.0f;
        float kx = R * cosf(t);
        float ky = pitch * t / pi2;
        float kz = R * sinf(t);
        float d = sqrtf((x-kx)*(x-kx)+(y-ky)*(y-ky)+(z-kz)*(z-kz));
        if (d < best) best = d;
    }
    return best - tube;
}'''),
        ("torus_as_helix_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    // Torus (wrong topology, but similar scale) — suboptimal seed
    float rho = sqrtf(x*x + z*z) - 0.7f;
    return sqrtf(rho*rho + y*y) - 0.2f;
}'''),
        ("cylinder_approx", '''\
extern "C" float sdf(float x, float y, float z) {
    // Vertical cylinder (worst-case approximation)
    return sqrtf(x*x + z*z) - 0.7f;
}'''),
    ],
    "scherk_first": [
        ("scherk_exact", '''\
extern "C" float sdf(float x, float y, float z) {
    // Scherk's first minimal surface: exp(z)*cos(y) = cos(x)
    // f = exp(z)*cos(y) - cos(x); |grad_f|^2 = sin^2(x) + exp(2z)
    float ez = expf(fmaxf(fminf(z, 1.5f), -1.5f));
    float f = ez * cosf(y) - cosf(x);
    float grad2 = sinf(x)*sinf(x) + ez*ez;
    return fabsf(f) / sqrtf(fmaxf(grad2, 0.005f)) - 0.08f;
}'''),
        ("scherk_saddle", '''\
extern "C" float sdf(float x, float y, float z) {
    // 2nd-order Taylor approx: z ≈ 0.5*(y^2 - x^2) near origin
    float f = z - 0.5f * (y*y - x*x);
    return fabsf(f) * 0.65f - 0.08f;
}'''),
        ("flat_plane", '''\
extern "C" float sdf(float x, float y, float z) {
    // Worst-case seed: flat plane at z=0
    return fabsf(z) - 0.12f;
}'''),
    ],
}


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    """Return seed programs for a given instance, falling back to generic seeds."""
    return SEED_SDF_PROGRAMS.get(instance_name, SEED_SDF_PROGRAMS["generic"])


# ---------------------------------------------------------------------------
# Sandbox gate
# ---------------------------------------------------------------------------

def build_candidate_source(sdf_code: str) -> str:
    """Combine the fixed skeleton with the LLM-evolved sdf() implementation."""
    return CPP_SKELETON + "\n" + sdf_code + "\n"


def _instance_stdin(instance: SDFInstance) -> str:
    """Serialize an SDF instance to JSON for the C++ evaluator via stdin."""
    return json.dumps({
        "name": instance.name,
        "samples": [[x, y, z, e] for x, y, z, e in instance.samples],
    })


def evaluate_on_instance(
    sdf_code: str,
    instance: SDFInstance,
    executor: "SandboxedExecutor",
    run_timeout: float = 10.0,
) -> float | None:
    """Compile + run sdf_code against instance. Returns fitness in (0,1] or None.

    Combined fitness:
      combined = 0.6 * (1/(1+MSE)) * exp(-0.5*grad_err) + 0.4 * topology_score

    The eikonal term penalises sdf() functions that fit the zero-set but violate
    |∇f|=1 (not raymarcher-usable). The topology oracle measures sign-change
    density on a 24^3 grid and applies a Gaussian oracle against calibrated
    targets per instance — rewarding SDFs with the correct topological complexity
    (gyroid: very high density, round_box: very low, warped_sphere: low).

    On topology-harness failure the weight falls back to the eikonal-only score.
    """
    # Resolve through the package namespace at call time so tests that
    # patch("autobench.sdf_kernel.compile_and_run") take effect here.
    from . import compile_and_run
    from ..core import Verdict

    source = build_candidate_source(sdf_code)
    stdout, verdict, _latency = compile_and_run(
        source,
        "cpp",
        constraints={"max_time_seconds": run_timeout, "max_memory_mb": executor.max_memory_mb},
        stdin=_instance_stdin(instance),
        executor=executor,
    )
    if verdict != Verdict.OK:
        logger.debug("SDF candidate non-OK verdict %s on %s", verdict, instance.name)
        return None
    try:
        out = json.loads(stdout.strip())
        mse = float(out["mse"])
        if not math.isfinite(mse) or mse < 0:
            return None
        base = 1.0 / (1.0 + mse)
        grad_err = float(out.get("grad_err", 0.0))
        eikonal_penalty = math.exp(-0.5 * grad_err) if math.isfinite(grad_err) else 0.1
        eikonal_fitness = base * eikonal_penalty

        # Topology oracle — fail-safe: non-OK result yields 0.0/0.0 but
        # keeps the candidate alive on the weighted eikonal score.
        topology_score, sign_change_density = compute_topology_score(
            sdf_code, instance, executor, run_timeout=run_timeout
        )

        if topology_score > 0.0:
            fitness = 0.6 * eikonal_fitness + 0.4 * topology_score
        else:
            # topology harness failed — fall back to eikonal-only score
            fitness = eikonal_fitness

        # Stash per-instance diagnostics for publish_candidate to read back.
        # Avoids changing evaluate_candidate's return type (kernel_base expects float|None).
        instance._last_grad_err = grad_err
        instance._last_mse = mse
        instance._last_topology_score = topology_score
        instance._last_sign_change_density = sign_change_density
        return fitness
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        logger.debug("SDF candidate output unparseable on %s: %s", instance.name, exc)
        return None


# ---------------------------------------------------------------------------
# Island personas + sketch library for SDF diversity
# ---------------------------------------------------------------------------

SDF_ISLAND_PERSONAS = [
    ("implicit surface geometer",
     "Think in terms of classical implicit surfaces: gyroid, Schwarz P/D, Neovius, torus knots. "
     "Use signed-distance combinations (fminf for union, fmaxf for intersection, smooth-union blends)."),
    ("domain-warp specialist",
     "Apply domain distortions before evaluating the base shape: sinusoidal twists, cylindrical folding, "
     "noise-based displacement. Warped domains often fit periodic/complex SDFs that analytic primitives miss."),
    ("polynomial approximator",
     "Use polynomial and rational function approximations: Taylor expansions around the surface, "
     "Chebyshev-style fits, or piecewise quadratics. Think like a numerical analyst fitting a curve."),
    ("trig-lattice explorer",
     "Exploit periodic structures: Fourier-style sums of sin/cos at multiple frequencies and axes. "
     "Gyroid and similar TPMS surfaces are naturally represented this way."),
    ("geometric combinator",
     "Combine simple primitives (spheres, boxes, cylinders, capsules) via smooth-min/max with "
     "varying blend radius k. Multi-stage smooth unions with different k values create rich surfaces."),
    ("volumetric blob sculptor",
     "Build organic shapes by smooth-union-blending 5-10 spheres at organic offsets with varying radii. "
     "Cloud, flesh, and cumulus structures emerge from asymmetric multi-sphere compositions with k=0.3-0.5. "
     "Position spheres to create flat bases and billowing tops: lower spheres wide, upper spheres small."),
    ("curve-tube sweeper",
     "Define a parametric curve C(t) = (R*cos(at), pitch*t, R*sin(at)) for helices, "
     "or C(t) = ((R+r*cos(qt))*cos(pt), (R+r*cos(qt))*sin(pt), r*sin(qt)) for torus knots. "
     "SDF = min_t dist(p, C(t)) - tube_radius. Unroll via atan2f to find nearest turn index."),
    ("minimal surface analyst",
     "Explore doubly- and triply-periodic minimal surfaces: gyroid, Schwarz-P, Scherk's saddle. "
     "Represent as f(x,y,z) = 0 where f is a sum of trig/exp terms, then approximate SDF as |f|/|∇f| "
     "using analytically computed gradient magnitude."),
]

SDF_PROMPT_SKETCHES = [
    ("smooth-union cascade",
     "Build from 2-3 simple primitives (spheres/boxes) combined with smooth-min (k=0.1–0.5). "
     "Vary k per level: tight blends near the surface, loose blends for large-scale shape."),
    ("trigonometric lattice",
     "Represent the surface as a linear combination of sin(a*x+b)*cos(c*y+d) terms across all 3 axes. "
     "Gyroid = sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x); try variations with frequency scaling."),
    ("domain-twisted primitive",
     "Start from a known primitive (sphere radius r), then warp the domain: "
     "x' = x + 0.2*sin(4*y), y' = y + 0.2*sin(4*z), then evaluate the primitive at (x',y',z')."),
    ("Fourier series SDF",
     "Decompose the target into frequency components: use 2-4 terms of sin/cos per axis at "
     "frequencies 1, 2, 4. Combine additively with learned coefficients (try ±0.3–1.0 range)."),
    ("level-set blending",
     "Define multiple implicit surfaces f1, f2, f3 and blend: 0.4*f1 + 0.3*f2 + 0.3*f3. "
     "The blended level set at 0 is a weighted combination of their zero-crossings."),
    ("blob cluster cascade",
     "Place 5-9 spheres at organic offset positions with varying radii (0.2-0.6). "
     "Apply smooth-min (k=0.35-0.5) in a cascade: accumulate smallest-to-largest. "
     "For cloud/organic shapes: bias lower spheres wider, upper spheres smaller and offset upward."),
    ("parametric curve tube",
     "Define knot or helix curve C(t): for helix C(t)=(R*cos(t), pitch*t/2π, R*sin(t)); "
     "for (p,q) torus knot C(t)=((R+r*cos(qt))*cos(pt), (R+r*cos(qt))*sin(pt), r*sin(qt)). "
     "SDF = min_t ||p-C(t)||₂ - tube_r. Use atan2f(z,x) to find starting t, then loop ±3 turns."),
    ("exponential minimal surface",
     "Use the level-set form: f = exp(z)*cos(y) - cos(x) for Scherk, "
     "or f = cos(x) + cos(y) + cos(z) for Schwarz-P, f = sin(x)cos(y) + ... for gyroid. "
     "Approximate SDF = |f| / |∇f| where ∇f is computed analytically per term."),
]


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

def build_llm_prompt(
    island: "Island",
    top_programs: list["CandidateProgram"],
    generation: int,
    instance_names: list[str],
    hint: str = "",
) -> str:
    """Build the prompt for evolving a new sdf() function."""
    exemplars = ""
    for i, p in enumerate(sorted(top_programs, key=lambda x: -x.fitness)[:3]):
        exemplars += f"\n// Exemplar {i+1} (fitness={p.fitness:.4f}):\n{p.code}\n"

    persona_name, persona_hint = SDF_ISLAND_PERSONAS[island.id % len(SDF_ISLAND_PERSONAS)]
    sketch_name, sketch_desc = SDF_PROMPT_SKETCHES[(island.id + generation) % len(SDF_PROMPT_SKETCHES)]

    if len(instance_names) == 1:
        instance_header = f"Target instance (this island evaluates ONLY this): **{instance_names[0]}**"
    else:
        instance_header = f"Benchmark instances: {', '.join(instance_names)}"

    hint_section = ""
    if hint:
        hint_section = f"\n## Strategic advice (plateau breaker)\n{hint}\n"

    return f"""You are a shader math expert acting as a "{persona_name}".
{persona_hint}

Generate a new C++ `sdf(x, y, z)` function that approximates a target signed-distance field.

{SDF_FUNCTION_SIGNATURE}

{instance_header}
Island {island.id} — Generation {generation}

Approach hint ({sketch_name}): {sketch_desc}
{hint_section}
Top programs in this island:
{exemplars}

Your goal: write a sdf() function that MINIMISES the Mean Squared Error between its output and the ground-truth SDF values at a set of 3D sample points.

Fitness = 1 / (1 + MSE). Perfect match = fitness 1.0.

Rules:
- Return ONLY the C++ sdf() implementation in a single ```cpp code block
- Use `extern "C" float sdf(float x, float y, float z)`
- Include `<cmath>` functions (sinf, cosf, sqrtf, fabsf, fmaxf, fminf, powf)
- Do NOT redefine structs or main() — only provide sdf()
- Keep it under 40 lines
- Try to identify the geometric structure from the instance name and fitness signal
- Combining domain twists, smooth unions, and trig-based implicit surfaces usually beats axis-aligned primitives
"""


def parse_llm_response(response: str) -> str:
    """Extract a C++ sdf() function from an LLM response.

    Tries markdown fences first, then explicit markers, then bare function.
    Returns "" on failure — the caller treats that as a 0.0 fitness candidate.
    """
    if not response or not response.strip():
        return ""
    text = response.strip()

    # 1. Markdown code fence (the common case for all major models)
    fence = re.search(r"```[a-zA-Z0-9_+\-]*[ \t]*\n?(.*?)```", text, re.DOTALL)
    if fence and fence.group(1).strip():
        return fence.group(1).strip()

    # 2. Bare function definition starting with extern "C" or float sdf
    sig = re.search(r'(?:extern\s+"C"\s+)?(?:inline\s+)?float\s+sdf\s*\(', text)
    if sig:
        return text[sig.start():].strip()

    # 3. Last resort: strip stray fences and hope it compiles
    return text.replace("```cpp", "").replace("```c++", "").replace("```", "").strip()
