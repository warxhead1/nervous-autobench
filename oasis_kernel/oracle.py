"""Oasis oracle — C++ shallow-water evaluator, signature, seeds, source build.

The evolved program is the constitutive flow law:

    float flux(float dhead, float depth, float visc)

returning the volumetric flux from a higher-head cell to a lower neighbour. The
C++ harness runs a mass-conserving 2D shallow-water sim over a dune basin with an
artesian spring + day/night evaporation, then measures the oasis dynamic:
basin_capture · stability · pool_fraction · breathing (multiplicatively gated).
"""
from __future__ import annotations

import json

from .instance import OasisInstance


# ---------------------------------------------------------------------------
# C++ evaluator. Evolved flux() inserted at EVOLVED_CODE_MARKER.
# stdin: JSON instance params (+ optional "render":1). stdout: JSON score; in
# render mode also dumps H and water-depth frames for the gallery renderer.
# ---------------------------------------------------------------------------

_OASIS_EVALUATOR_CPP = r"""
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline float clamp01(float x){ return fmaxf(0.f, fminf(1.f, x)); }

// ===== EVOLVED_CODE_MARKER =====

// ── dune-basin terrain ───────────────────────────────────────────────────────
static void make_terrain(int N, float dune_amp, float dune_freq, vector<float>& H){
    H.assign(N*N, 0.f);
    float hmin=1e9f, hmax=-1e9f;
    for(int i=0;i<N;i++) for(int j=0;j<N;j++){
        float px=-1.f+2.f*j/(N-1), py=-1.f+2.f*i/(N-1);
        float r2=px*px+py*py;
        float bowl=0.55f*r2;
        float phase=dune_freq*px+1.2f*sinf(2.1f*py);
        float m=fmodf(phase/(float)M_PI,2.0f); if(m<0) m+=2.0f;
        float dunes=dune_amp*powf(fabsf(m-1.f),1.7f);
        float ripple=0.025f*sinf(9.0f*py+0.5f*px);
        float v=bowl+dunes+ripple;
        H[i*N+j]=v; hmin=fminf(hmin,v); hmax=fmaxf(hmax,v);
    }
    float hr=hmax-hmin; if(hr<1e-6f) hr=1.f;
    for(auto& v:H) v=(v-hmin)/hr;
}

static string g_in;
static float jget(const char* key, float def){
    string k=string("\"")+key+"\"";
    size_t p=g_in.find(k); if(p==string::npos) return def;
    p+=k.size(); while(p<g_in.size()&&(g_in[p]==' '||g_in[p]==':')) p++;
    char* e; float v=strtof(g_in.c_str()+p,&e); return (e!=g_in.c_str()+p)?v:def;
}

int main(){
    g_in.assign((istreambuf_iterator<char>(cin)), istreambuf_iterator<char>());
    int   N      = (int)jget("grid_size", 64);
    float visc   = jget("viscosity", 0.02f);
    float spring = jget("spring", 4.0f);
    float headmax= jget("artesian", 0.5f);
    float evap0  = jget("evap", 0.004f);
    float dayamp = jget("day_amp", 0.45f);
    float period = jget("day_period", 150.f);
    float dune_amp  = jget("dune_amp", 0.07f);
    float dune_freq = jget("dune_freq", 6.5f);
    int   steps  = (int)jget("n_steps", 700);
    bool  RENDER = jget("render", 0.f) > 0.5f;
    const float kcfl=0.125f, dt=1.0f;

    // Precondition: flux must be finite/non-negative at sample points.
    float probe[3]={ flux(0.1f,0.1f,visc), flux(0.5f,0.05f,visc), flux(0.02f,0.2f,visc) };
    for(int i=0;i<3;i++) if(!isfinite(probe[i])||probe[i]<0.f){
        printf("{\"valid\":false,\"reason\":\"flux_precondition\"}\n"); return 0; }

    vector<float> H; make_terrain(N, dune_amp, dune_freq, H);
    vector<float> w(N*N,0.f), dw(N*N,0.f), head(N*N,0.f);

    vector<float> hs=H; sort(hs.begin(),hs.end());
    float low_thr=hs[(int)(0.30f*N*N)];
    int spring_k=0; for(int k=1;k<N*N;k++) if(H[k]<H[spring_k]) spring_k=k;

    int NF=24, snap_every=(steps>NF)?steps/NF:1;
    vector<vector<float>> snaps;
    vector<float> vol(steps), wet(steps);
    const int OFF[4][2]={{1,0},{-1,0},{0,1},{0,-1}};

    for(int s=0;s<steps;s++){
        for(int k=0;k<N*N;k++) head[k]=H[k]+w[k];
        fill(dw.begin(),dw.end(),0.f);
        for(int d=0;d<4;d++){
            int oi=OFF[d][0], oj=OFF[d][1];
            for(int i=0;i<N;i++) for(int j=0;j<N;j++){
                int ni=i+oi, nj=j+oj;
                if(ni<0||ni>=N||nj<0||nj>=N) continue;
                float dh=head[i*N+j]-head[ni*N+nj];
                if(dh<=0) continue;
                float q=flux(dh, w[i*N+j], visc);
                if(!isfinite(q)||q<0){ printf("{\"valid\":false,\"reason\":\"flux_bad\"}\n"); return 0; }
                q=fminf(q, w[i*N+j]*kcfl);
                dw[i*N+j]-=q; dw[ni*N+nj]+=q;
            }
        }
        for(int k=0;k<N*N;k++) w[k]=fmaxf(0.f, w[k]+dw[k]);
        w[spring_k]+=spring*fmaxf(0.f, headmax-(H[spring_k]+w[spring_k]))*dt;
        float day=sinf(2.f*(float)M_PI*s/period);
        double V=0; int wc=0;
        for(int k=0;k<N*N;k++){
            float ev=(w[k]>1e-4f)? evap0*(1.f+dayamp*day)*expf(-w[k]/0.08f) : 0.f;
            w[k]=fmaxf(0.f, w[k]-dt*ev);
            if(!isfinite(w[k])){ printf("{\"valid\":false,\"reason\":\"w_nan\"}\n"); return 0; }
            V+=w[k]; if(w[k]>5e-3f) wc++;
        }
        vol[s]=(float)V; wet[s]=(float)wc/(N*N);
        if(V>(double)N*N*5.0){ printf("{\"valid\":false,\"reason\":\"flood_overflow\"}\n"); return 0; }
        if(RENDER && (s%snap_every==0)) snaps.push_back(w);
    }

    int t0=(int)(0.6f*steps), n=steps-t0;
    double Vbar=0; for(int s=t0;s<steps;s++) Vbar+=vol[s]; Vbar/=n; Vbar+=1e-9;

    double wlow=0, wall=0;
    for(int k=0;k<N*N;k++){ wall+=w[k]; if(H[k]<low_thr) wlow+=w[k]; }
    float cap=(wall>1e-9)?(float)(wlow/wall):0.f;

    double sx=0,sy=0,sxx=0,sxy=0;
    for(int s=t0;s<steps;s++){ double x=s-t0,y=vol[s]; sx+=x;sy+=y;sxx+=x*x;sxy+=x*y; }
    double denom=(n*sxx-sx*sx); double slope=(fabs(denom)>1e-9)?(n*sxy-sx*sy)/denom:0;
    double drift=fabs(slope)*n/Vbar; float stab=(float)exp(-3.0*drift);

    double wf=0; for(int s=t0;s<steps;s++) wf+=wet[s]; wf/=n;
    float pool=(float)exp(-pow(wf-0.06,2)/(2*0.035*0.035));

    double var=0; for(int s=t0;s<steps;s++) var+=pow(vol[s]-Vbar,2); var/=n;
    double amp=sqrt(var)/Vbar; float breathe=(float)exp(-pow(amp-0.045,2)/(2*0.035*0.035));

    float alive=(wf>0.01 && Vbar>1.0)?1.f:0.f;
    float fitness=alive*cap*(0.4f+0.6f*stab)*pool*(0.4f+0.6f*breathe);

    printf("{\"fitness\":%.6f,\"basin_capture\":%.4f,\"stability\":%.4f,"
           "\"pool_fraction\":%.4f,\"breathing\":%.4f,\"wet\":%.4f,\"amp\":%.4f,"
           "\"valid\":true}\n",
           fitness, cap, stab, pool, breathe, (float)wf, (float)amp);

    if(RENDER){
        printf("H %d %d\n", N, (int)snaps.size());
        for(int k=0;k<N*N;k++) printf("%.4f ", H[k]); printf("\n");
        for(auto& sn:snaps){ for(int k=0;k<N*N;k++) printf("%.4f ", sn[k]); printf("\n"); }
    }
    return 0;
}
"""

OASIS_FUNCTION_SIGNATURE = '''\
float flux(float dhead, float depth, float visc);

// Evolved constitutive flow law for a 2D shallow-water oasis.
// dhead: head difference (donor minus lower neighbour), always > 0  [normalised]
// depth: water depth in the donor cell (>= 0)
// visc:  kinematic viscosity of the fluid (instance-fixed; higher = sluggish)
// Returns: volumetric flux donor->neighbour this step. MUST be finite and >= 0.
//
// The harness integrates this on a 4-neighbour stencil with mass conservation
// (per-edge flux is capped at 12.5% of donor depth), then applies an artesian
// spring source and depth-shielded day/night evaporation. A good law fills the
// basin into a stable pool that breathes with the day cycle; a do-nothing law
// leaves a spring spike (no pool).
//
// PHYSICAL REFERENCE — thin-film lubrication (Reynolds equation):
//     q = depth^3 * dhead / (3 * visc)
//   Deep water flows freely; shallow shores barely move (depth^3). Viscosity
//   divides: more viscous → slower flow.
//
// Available C math: sinf, cosf, expf, logf, sqrtf, fabsf, powf, fmaxf, fminf.
// No loops, no static state, no allocation. Under 15 lines.
'''


# ---------------------------------------------------------------------------
# Seed flow laws
# ---------------------------------------------------------------------------

_LUBRICATION_SEED = '''\
float flux(float dhead, float depth, float visc) {
    // Thin-film lubrication (Reynolds): q ~ depth^3 * dhead / (3 visc)
    return depth * depth * depth * dhead / (3.0f * visc);
}'''

_DARCY_SEED = '''\
float flux(float dhead, float depth, float visc) {
    // Darcy-like porous flow: linear in depth and head gradient
    return depth * dhead / visc;
}'''

_MANNING_SEED = '''\
float flux(float dhead, float depth, float visc) {
    // Manning-like open-channel: depth^(5/3) * sqrt(slope), viscosity as friction
    return powf(depth, 1.667f) * sqrtf(dhead) / (visc + 0.001f);
}'''

SEED_OASIS_PROGRAMS: dict[str, list[tuple[str, str]]] = {
    "generic": [
        ("lubrication", _LUBRICATION_SEED),
        ("darcy",       _DARCY_SEED),
        ("manning",     _MANNING_SEED),
    ],
}


def get_seed_programs(instance_name: str) -> list[tuple[str, str]]:
    return SEED_OASIS_PROGRAMS.get(instance_name, SEED_OASIS_PROGRAMS["generic"])


# ---------------------------------------------------------------------------
# C++ source construction
# ---------------------------------------------------------------------------

def build_candidate_source(code: str, instance: OasisInstance,
                           render: bool = False) -> tuple[str, str]:
    cpp = _OASIS_EVALUATOR_CPP.replace("// ===== EVOLVED_CODE_MARKER =====", code, 1)
    data = {
        "name": instance.name,
        "grid_size": instance.grid_size,
        "viscosity": instance.viscosity,
        "spring": instance.spring,
        "artesian": instance.artesian,
        "evap": instance.evap,
        "day_amp": instance.day_amp,
        "day_period": instance.day_period,
        "dune_amp": instance.dune_amp,
        "dune_freq": instance.dune_freq,
        "n_steps": instance.n_steps,
    }
    if render:
        data["render"] = 1
    return cpp, json.dumps(data, separators=(",", ":"))
