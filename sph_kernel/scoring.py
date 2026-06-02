"""SPH scoring — C++ evaluator template, source construction, and fitness.

Moved verbatim from ``sph_kernel/__init__.py`` as part of a behavior-preserving
file split. Contains the embedded C++ evaluator, the evolved-function signature
shown to the LLM, candidate-source construction, and single-instance evaluation.
"""
from __future__ import annotations

import json
import math
from typing import Optional

from ..core import Verdict
from ..engines.sandbox import SandboxedExecutor, compile_and_run
from .instance import SPHInstance


# ---------------------------------------------------------------------------
# C++ evaluator template
# The evolved sph_kernel() is injected at EVOLVED_CODE_MARKER.
# Input (stdin): JSON with particles, probes, h, name.
# Output: JSON line with mse, compact_ok, positive_ok, n, instance.
# ---------------------------------------------------------------------------

_SPH_EVALUATOR_CPP = r"""
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <array>

using namespace std;

// ===== EVOLVED_CODE_MARKER =====

struct Particle { float x, y, z, mass; };
struct Probe    { float x, y, z, rho;  };

// Minimal JSON parser (no dependencies)
struct P {
    const string& s; int pos;
    P(const string& src): s(src), pos(0) {}
    void ws(){ while(pos<(int)s.size()&&(s[pos]==' '||s[pos]=='\n'||s[pos]=='\r'||s[pos]=='\t'))pos++; }
    void expect(char c){ ws(); if(pos>=(int)s.size()||s[pos]!=c){fprintf(stderr,"expected '%c'\n",c);exit(1);} pos++; }
    string str(){ expect('"'); string r; while(pos<(int)s.size()&&s[pos]!='"')r+=s[pos++]; expect('"'); return r; }
    double num(){
        ws(); bool neg=pos<(int)s.size()&&s[pos]=='-'; if(neg)pos++;
        double v=0;
        while(pos<(int)s.size()&&isdigit((unsigned char)s[pos]))v=v*10+(s[pos++]-'0');
        if(pos<(int)s.size()&&s[pos]=='.'){pos++;double f=0.1;while(pos<(int)s.size()&&isdigit((unsigned char)s[pos])){v+=f*(s[pos++]-'0');f*=0.1;}}
        if(pos<(int)s.size()&&(s[pos]=='e'||s[pos]=='E')){pos++;bool en=s[pos]=='-';if(en||s[pos]=='+')pos++;int e=0;while(pos<(int)s.size()&&isdigit((unsigned char)s[pos]))e=e*10+(s[pos++]-'0');double m=1;for(int i=0;i<e;i++)m*=10;if(en)v/=m;else v*=m;}
        return neg?-v:v;
    }
    vector<array<float,4>> arr4(){
        expect('['); vector<array<float,4>> r; ws();
        while(pos<(int)s.size()&&s[pos]!=']'){
            expect('[');
            float a=(float)num(); expect(',');
            float b=(float)num(); expect(',');
            float c=(float)num(); expect(',');
            float d=(float)num();
            ws(); if(pos<(int)s.size()&&s[pos]==']')pos++;
            r.push_back({a,b,c,d});
            ws(); if(pos<(int)s.size()&&s[pos]==',')pos++;
            ws();
        }
        expect(']'); return r;
    }
    void skip(){ ws(); if(pos>=(int)s.size())return;
        if(s[pos]=='"'){str();return;}
        if(s[pos]=='['||s[pos]=='{'){int d=0;while(pos<(int)s.size()){if(s[pos]=='['||s[pos]=='{')d++;else if(s[pos]==']'||s[pos]=='}'){d--;pos++;if(!d)break;}else pos++;}return;}
        while(pos<(int)s.size()&&s[pos]!=','&&s[pos]!='}'&&s[pos]!=']')pos++;
    }
};

int main() {
    string inp((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());
    P p(inp);
    p.expect('{');

    string inst_name; float h=0.4f;
    vector<Particle> parts; vector<Probe> prbs;

    while(true){
        p.ws(); if(p.pos>=(int)inp.size()||inp[p.pos]=='}'){p.pos++;break;}
        string k=p.str(); p.expect(':');
        if(k=="name")      inst_name=p.str();
        else if(k=="h")    h=(float)p.num();
        else if(k=="particles"){ for(auto& t:p.arr4()) parts.push_back({t[0],t[1],t[2],t[3]}); }
        else if(k=="probes"){    for(auto& t:p.arr4()) prbs.push_back({t[0],t[1],t[2],t[3]}); }
        else p.skip();
        p.ws(); if(p.pos<(int)inp.size()&&inp[p.pos]==',')p.pos++;
    }

    if(parts.empty()||prbs.empty()){
        fprintf(stderr,"ERROR: no particles/probes parsed (%d/%d)\n",(int)parts.size(),(int)prbs.size());
        return 1;
    }

    // Hard preconditions
    int compact_ok = (fabsf(sph_kernel(h*1.001f, h)) < 1e-3f) ? 1 : 0;
    int positive_ok= (sph_kernel(0.0f, h) >= 0.0f) ? 1 : 0;

    if(!compact_ok || !positive_ok){
        printf("{\"mse\":100.0,\"compact_ok\":%d,\"positive_ok\":%d,\"n\":0,\"instance\":\"%s\"}\n",
               compact_ok, positive_ok, inst_name.c_str());
        return 0;
    }

    // Density reconstruction MSE
    double mse=0.0;
    int n=(int)prbs.size();
    for(int i=0;i<n;i++){
        Probe& q=prbs[i];
        float rho_est=0.0f;
        for(auto& pt:parts){
            float dx=q.x-pt.x, dy=q.y-pt.y, dz=q.z-pt.z;
            float r=sqrtf(dx*dx+dy*dy+dz*dz);
            rho_est += pt.mass * sph_kernel(r, h);
        }
        double err=(double)rho_est - (double)q.rho;
        mse += err*err;
    }
    mse /= n;

    // Monotonicity score: W(r,h) must be monotone decreasing r ∈ [0,h].
    // Violation means W has a local maximum away from r=0 → SPH pressure
    // gradient direction flips sign, breaking momentum conservation.
    // Sample 32 equidistant points; each non-monotone step costs 1/32.
    float mono_score = 1.0f;
    {
        int N_m = 32;
        float prev_w = sph_kernel(0.0f, h);
        int violations = 0;
        for (int k = 1; k <= N_m; k++) {
            float r_k = h * (float)k / N_m;
            float w_k = sph_kernel(r_k, h);
            if (w_k > prev_w + 1e-6f) violations++;
            prev_w = w_k;
        }
        mono_score = 1.0f - (float)violations / N_m;
    }

    // Gradient direction check: d/dr W(r,h) at r=h/2 should be negative.
    // Uses central difference. Kernels with W'(h/2) > 0 push particles apart
    // when they should attract → wrong pressure forces.
    float eps_r = h * 0.01f;
    float half_h = h * 0.5f;
    float dW_dr = (sph_kernel(half_h + eps_r, h) - sph_kernel(half_h - eps_r, h)) / (2.0f * eps_r);
    float grad_ok = (dW_dr < 0.0f) ? 1.0f : 0.0f;

    printf("{\"mse\":%.9f,\"compact_ok\":%d,\"positive_ok\":%d,\"mono_score\":%.4f,"
           "\"grad_ok\":%.1f,\"n\":%d,\"instance\":\"%s\"}\n",
           mse, compact_ok, positive_ok, mono_score, grad_ok, n, inst_name.c_str());
    return 0;
}
"""

SPH_FUNCTION_SIGNATURE = '''\
extern "C" float sph_kernel(float r, float h);

// Evolved SPH smoothing kernel W(r, h).
// r: distance between particle and evaluation point (r >= 0)
// h: smoothing length (compact support radius; W must be 0 for r >= h)
//
// PHYSICAL REQUIREMENTS (checked as hard preconditions):
//   Compact support: sph_kernel(h, h) == 0  (exactly zero outside support)
//   Positivity:      sph_kernel(0, h) >= 0
//
// WHAT MAKES A GOOD KERNEL:
//   The oracle places particles with mass ∝ ρ_true at jittered positions,
//   then checks how accurately Σ mⱼ·W(|x-xⱼ|,h) reconstructs ρ_true at
//   off-lattice probe points. Kernel *shape* determines accuracy — a flat
//   top-hat or tent function pass the preconditions but fail the oracle.
//
// Known kernels for reference:
//   Cubic spline:   σ·{6(q³-q²)+1 for q<0.5; 2(1-q)³ for 0.5≤q<1; 0 for q≥1}
//                   σ = 8/(πh³) in 3D
//   Wendland C²:    σ·(1-q)⁴(4q+1) for q<1; 0 otherwise
//                   σ = 21/(2πh³) in 3D
//   where q = r/h
//
// Available C math:  sqrtf, fabsf, fmaxf, fminf, sinf, cosf, powf, expf, logf
'''


# ---------------------------------------------------------------------------
# C++ source construction
# ---------------------------------------------------------------------------

def build_candidate_source(code: str, instance: SPHInstance) -> tuple[str, str]:
    """Return (cpp_source, json_stdin) for one evaluation.

    The evolved sph_kernel() is inserted into the evaluator template;
    instance data is serialised as JSON for stdin.
    """
    cpp = _SPH_EVALUATOR_CPP.replace("// ===== EVOLVED_CODE_MARKER =====", code, 1)

    data = {
        "name": instance.name,
        "h": instance.h,
        "particles": [[p[0], p[1], p[2], p[3]] for p in instance.particles],
        "probes":    [[q[0], q[1], q[2], q[3]] for q in instance.probes],
    }
    return cpp, json.dumps(data, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Single-instance evaluation
# ---------------------------------------------------------------------------

def evaluate_on_instance(
    code: str,
    instance: SPHInstance,
    executor: SandboxedExecutor,
    run_timeout: float = 30.0,
    compile_timeout: float = 20.0,
) -> Optional[float]:
    """Compile + run evolved kernel; return fitness in (0,1] or None on failure."""
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

    compact_ok  = out.get("compact_ok", 0)
    positive_ok = out.get("positive_ok", 0)
    if not compact_ok or not positive_ok:
        return None     # hard precondition violated

    mse = float(out.get("mse", 10.0))
    if not math.isfinite(mse) or mse < 0:
        return None

    density_fitness = 1.0 / (1.0 + mse)
    mono_score      = float(out.get("mono_score", 0.0))
    grad_ok         = float(out.get("grad_ok",    0.0))

    # Fitness = 0.70·density + 0.20·monotone + 0.10·gradient_direction
    # Monotone decreasing W ensures SPH pressure forces have correct sign.
    # A kernel with W'(h/2) > 0 would push particles apart under compression.
    return 0.70 * density_fitness + 0.20 * mono_score + 0.10 * grad_ok
