"""Build a self-contained Shadertoy GLSL heightfield raymarcher for an oasis.

The evolved flux law produces a settled water field over a dune basin; this turns
that result into a draggable, lit 3D scene:

  * terrain is re-derived **analytically** in GLSL (the same dune-basin formula the
    C++ harness uses), so it is crisp at any zoom;
  * the simulated water depth is baked as a small const lookup and sampled
    bilinearly, then gently breathed with iTime to echo the day-cycle dynamic;
  * water is shaded with Fresnel sky reflection, Beer-Lambert depth absorption, a
    sun glint and a foam shoreline; the dunes get soft sun shadows + AO.

Drag to orbit (iMouse); it auto-rotates otherwise. Output is a complete Shadertoy
``mainImage`` body — the app's /live WebGL2 harness and Shadertoy both render it as-is.
"""
from __future__ import annotations

from .instance import OasisInstance

WN = 32  # baked water-field resolution


_TEMPLATE = r"""// Oasis — evolved shallow-water flow, raymarched. Drag to orbit.
#define PI 3.14159265
#define HSCALE 0.42
#define WSCALE 0.42
#define WN __WN__
#define DUNE_AMP __DUNE_AMP__
#define DUNE_FREQ __DUNE_FREQ__
#define HMIN __HMIN__
#define HRANGE __HRANGE__
#define BREATHE __BREATHE__

const float W[WN*WN] = float[WN*WN](__WDATA__);

float terrainRaw(vec2 p){
    float bowl   = 0.55 * dot(p,p);
    float phase  = DUNE_FREQ*p.x + 1.2*sin(2.1*p.y);
    float m      = mod(phase/PI, 2.0);
    float dunes  = DUNE_AMP * pow(abs(m-1.0), 1.7);
    float ripple = 0.025 * sin(9.0*p.y + 0.5*p.x);
    return bowl + dunes + ripple;
}
float terrainN(vec2 p){ return clamp((terrainRaw(p)-HMIN)/HRANGE, 0.0, 1.2); }
// clamp the domain so terrain doesn't explode into an infinite plateau off-grid
float ground(vec2 p){ return terrainN(clamp(p, vec2(-1.05), vec2(1.05))) * HSCALE; }

float waterRaw(vec2 q){
    vec2 g = clamp((q*0.5+0.5)*float(WN-1), vec2(0.0), vec2(float(WN-1)));
    ivec2 i0 = ivec2(floor(g)); ivec2 i1 = min(i0+1, ivec2(WN-1));
    vec2 f = fract(g);
    float a = mix(W[i0.y*WN+i0.x], W[i0.y*WN+i1.x], f.x);
    float b = mix(W[i1.y*WN+i0.x], W[i1.y*WN+i1.x], f.x);
    return mix(a, b, f.y);
}
// animated water depth (breathes with the day cycle)
float wdepth(vec2 p, float tm){ return waterRaw(p) * (1.0 + BREATHE*sin(tm*0.7)); }
float waterTop(vec2 p, float tm){ return ground(p) + max(wdepth(p,tm),0.0)*WSCALE; }

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
float grainN(vec2 p){  // fine sand detail
    vec2 q = p*10.0; vec2 i = floor(q), f = fract(q);
    f = f*f*(3.0-2.0*f);
    float a = mix(hash(i), hash(i+vec2(1,0)), f.x);
    float b = mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x);
    return mix(a, b, f.y) - 0.5;
}
vec3 terrainNormal(vec2 p){
    float e = 0.006;
    float hx = ground(p+vec2(e,0)) - ground(p-vec2(e,0));
    float hy = ground(p+vec2(0,e)) - ground(p-vec2(0,e));
    return normalize(vec3(-hx, 2.0*e, -hy));
}
vec3 waterNormal(vec2 p, float tm){
    float e = 0.006;
    float hx = waterTop(p+vec2(e,0),tm) - waterTop(p-vec2(e,0),tm);
    float hy = waterTop(p+vec2(0,e),tm) - waterTop(p-vec2(0,e),tm);
    vec3 n = vec3(-hx, 2.0*e, -hy);
    float r  = sin(34.0*p.x+22.0*p.y+2.0*tm) + 0.6*sin(26.0*p.x-31.0*p.y+1.3*tm);
    float rx = 34.0*cos(34.0*p.x+22.0*p.y+2.0*tm) - 0.6*31.0*cos(26.0*p.x-31.0*p.y+1.3*tm);
    float ry = 22.0*cos(34.0*p.x+22.0*p.y+2.0*tm) - 31.0*0.6*cos(26.0*p.x-31.0*p.y+1.3*tm);
    n.x += 0.00012*rx; n.z += 0.00012*ry;
    return normalize(n);
}

vec3 sandColor(float h){ return mix(vec3(0.54,0.41,0.26), vec3(0.96,0.86,0.66), h); }

// march a heightfield with shrinking steps: returns hit t or -1
float marchGround(vec3 ro, vec3 rd){
    float t = 0.01, dt = 0.02;
    for(int i=0;i<220;i++){
        vec3 p = ro + t*rd;
        float h = p.y - ground(p.xz);
        if(h < 0.0006) return t - dt*0.5;  // step back to the crossing
        dt = max(0.004, 0.5*h);
        t += dt;
        if(t > 8.0) break;
    }
    return -1.0;
}
float sunShadow(vec3 p, vec3 L){
    float res = 1.0, t = 0.02;
    for(int i=0;i<24;i++){
        vec3 q = p + L*t;
        float h = q.y - ground(q.xz);
        if(h < 0.0) return 0.18;
        res = min(res, 9.0*h/t);
        t += 0.03;
        if(t > 1.4) break;
    }
    return 0.18 + 0.82*clamp(res, 0.0, 1.0);
}

vec3 skyColor(vec3 rd, vec3 L){
    float h = clamp(rd.y*0.5+0.5, 0.0, 1.0);
    vec3 sky = mix(vec3(0.80,0.86,0.95), vec3(0.30,0.52,0.86), h);
    float sun = pow(max(dot(rd,L),0.0), 220.0);
    return sky + vec3(1.0,0.85,0.6)*sun;
}

void mainImage(out vec4 O, in vec2 fragCoord){
    vec2 uv = (fragCoord - 0.5*iResolution.xy)/iResolution.y;
    float tm = iTime;

    float yaw = 2.2 + 0.10*iTime, pitch = 0.95, rad = 1.85;
    if(iMouse.z > 0.0){
        yaw   = 6.2832*(iMouse.x/iResolution.x);
        pitch = mix(0.35, 1.45, clamp(iMouse.y/iResolution.y, 0.0, 1.0));
    }
    vec3 ta = vec3(0.0, 0.05, 0.0);
    vec3 ro = ta + rad*vec3(cos(yaw)*cos(pitch), sin(pitch), sin(yaw)*cos(pitch));
    vec3 ww = normalize(ta-ro), uu = normalize(cross(ww,vec3(0,1,0))), vv = cross(uu,ww);
    vec3 rd = normalize(uv.x*uu + uv.y*vv + 1.6*ww);

    float day = 0.5 + 0.42*sin(0.18*iTime);
    float alt = mix(0.20, 1.30, day);
    vec3 L = normalize(vec3(cos(alt)*0.7, sin(alt), cos(alt)*0.5));
    vec3 sunCol = mix(vec3(1.0,0.55,0.26), vec3(1.0,0.97,0.92), clamp((sin(alt)-0.12)/0.5,0.0,1.0));

    float tg = marchGround(ro, rd);
    vec3 col = skyColor(rd, L);

    if(tg > 0.0){
        vec3 p = ro + tg*rd;
        float hN = terrainN(p.xz);
        vec3 tn = terrainNormal(p.xz);
        float sh = sunShadow(p + tn*0.01, L);
        float ao = clamp(0.45 + 0.6*hN, 0.0, 1.0);
        vec3 amb = vec3(0.45,0.55,0.72)*(0.5+0.5*tn.y);
        // sand (also the pool bottom)
        float dif = clamp(dot(tn,L), 0.0, 1.0);
        vec3 sand = sandColor(hN) * (0.42*amb + sunCol*dif*sh) * ao;
        sand += 0.02*grainN(p.xz);

        // is this point underwater? -> shade water over the sand bottom
        float wd = wdepth(p.xz, tm) * WSCALE;
        if(wd > 6e-4){
            vec3 wn = waterNormal(p.xz, tm);
            vec3 refl = skyColor(reflect(rd, wn), L);
            float fres = 0.02 + 0.42*pow(1.0 - max(dot(wn,-rd),0.0), 5.0);
            // Beer-Lambert through the water column along the view slant
            float path = wd / max(0.15, -rd.y + 0.15);
            vec3 trans = exp(-vec3(3.4,1.4,0.9) * path * 7.0);
            vec3 body = sand*trans + vec3(0.02,0.18,0.24)*(1.0-trans);
            float spec = pow(max(dot(reflect(-L,wn), -rd), 0.0), 140.0);
            float foam = clamp(1.0 - wd/0.010, 0.0, 1.0)*clamp(wd/0.002,0.0,1.0);
            vec3 water = mix(body, refl, fres);
            water += sunCol*spec*sh*1.8;
            water += vec3(0.92,0.96,1.0)*foam*0.6;
            col = water;
        } else {
            col = sand;
        }
        col = mix(col, skyColor(rd,L), smoothstep(5.0,8.0,tg));
    }

    // ACES tonemap
    col = clamp((col*(2.51*col+0.03))/(col*(2.43*col+0.59)+0.14), 0.0, 1.0);
    col *= 1.0 - 0.14*dot(uv,uv);
    O = vec4(col, 1.0);
}
"""


def build_oasis_glsl(H, water_field, instance: OasisInstance, np) -> str:
    """Build the Shadertoy GLSL string. H, water_field are sim grids (numpy)."""
    N = H.shape[0]
    iy, ix = np.mgrid[0:N, 0:N].astype(np.float64)
    px = -1 + 2 * ix / (N - 1)
    py = -1 + 2 * iy / (N - 1)
    bowl = 0.55 * (px * px + py * py)
    phase = instance.dune_freq * px + 1.2 * np.sin(2.1 * py)
    m = np.mod(phase / np.pi, 2.0)
    dunes = instance.dune_amp * np.power(np.abs(m - 1.0), 1.7)
    ripple = 0.025 * np.sin(9.0 * py + 0.5 * px)
    raw = bowl + dunes + ripple
    hmin = float(raw.min())
    hrange = float(raw.max() - raw.min()) or 1.0

    # downsample water to WN×WN (bilinear)
    ys = np.linspace(0, N - 1, WN)
    xs = np.linspace(0, N - 1, WN)
    gx, gy = np.meshgrid(xs, ys)
    x0 = np.clip(np.floor(gx).astype(int), 0, N - 1)
    y0 = np.clip(np.floor(gy).astype(int), 0, N - 1)
    x1 = np.clip(x0 + 1, 0, N - 1)
    y1 = np.clip(y0 + 1, 0, N - 1)
    fx, fy = gx - x0, gy - y0
    a = water_field[y0, x0] * (1 - fx) + water_field[y0, x1] * fx
    b = water_field[y1, x0] * (1 - fx) + water_field[y1, x1] * fx
    wsmall = a * (1 - fy) + b * fy
    wdata = ",".join(f"{v:.4f}" for v in wsmall.ravel())

    return (_TEMPLATE
            .replace("__WN__", str(WN))
            .replace("__DUNE_AMP__", f"{instance.dune_amp:.4f}")
            .replace("__DUNE_FREQ__", f"{instance.dune_freq:.4f}")
            .replace("__HMIN__", f"{hmin:.5f}")
            .replace("__HRANGE__", f"{hrange:.5f}")
            .replace("__BREATHE__", "0.12")
            .replace("__WDATA__", wdata))
