"""artifact_store — persistent visual artifact pipeline for evolved programs.

On every new best fitness, renders the program to a PNG and publishes
funsearch.artifact.v1 to the bus. Consumed by TEngine, hearth-loom,
and deer-flow for cross-system knowledge sharing.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ARTIFACTS_ROOT = Path(__file__).parent.parent / "benchmarks" / "artifacts"


@dataclass
class ArtifactRecord:
    kernel: str          # sdf / noise / tsp
    run_id: str
    generation: int
    fitness: float
    instance: str
    artifact_path: str   # relative to repo root
    render_type: str     # "sdf_raymarch" | "noise_render" | "none"
    metadata: dict
    sdf_code: str = ""   # raw C++ sdf() for playground push (SDF kernel only)


def save_artifact_record(
    record: ArtifactRecord,
    nervous_bin: str | None = None,
    extra_data: dict | None = None,
) -> None:
    """Write record to artifacts index, publish bus event, and push to playground.

    ``extra_data`` (optional) merges extra top-level keys into the artifact
    ``data`` block — used for gallery-grouping discriminators like ``island`` and
    ``candidate_id`` that aren't part of the base ArtifactRecord. funsearch.artifact.v1
    allows additional properties on ``data``, so these pass through to consumers.
    """
    index_path = ARTIFACTS_ROOT / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "id": uuid.uuid4().hex[:12],
        "kernel": record.kernel,
        "run_id": record.run_id,
        "generation": record.generation,
        "fitness": record.fitness,
        "instance": record.instance,
        "artifact_path": record.artifact_path,
        "render_type": record.render_type,
        "metadata": record.metadata,
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra_data:
        # Don't clobber the canonical fields; extras only add new keys.
        for k, v in extra_data.items():
            entry.setdefault(k, v)

    with open(index_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Artifact saved: %s fitness=%.4f", record.artifact_path, record.fitness)
    _publish_artifact_event(entry, nervous_bin)

    if record.kernel == "sdf" and record.sdf_code:
        try:
            from autobench.playground_push import push_to_playground
            title = (f"[{record.instance}] fitness={record.fitness:.4f} "
                     f"gen={record.generation} run={record.run_id[:8]}")
            push_to_playground(record.sdf_code, title,
                               {"instance": record.instance,
                                "fitness": record.fitness,
                                "generation": record.generation,
                                "run_id": record.run_id})
        except Exception as e:
            logger.debug("playground push skipped: %s", e)


def _publish_artifact_event(entry: dict, nervous_bin: str | None) -> None:
    """Publish funsearch.artifact.v1 to the nervous bus."""
    # Write to debug log always
    debug_path = Path.home() / ".cache" / "nervous-bus" / "debug.jsonl"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "specversion": "1.0",
        "id": uuid.uuid4().urn,
        "source": f"/autobench/{entry['kernel']}_kernel",
        "type": "funsearch.artifact.v1",
        "datacontenttype": "application/json",
        "time": entry["time"],
        "data": entry,
    }
    try:
        with open(debug_path, "a") as f:
            f.write(json.dumps(envelope) + "\n")
    except Exception:
        pass

    if nervous_bin:
        import subprocess, os
        try:
            env = dict(os.environ)
            env["NBUS_SKIP_VALIDATION"] = "1"
            env["NERVOUS_NO_ZELLIJ"] = "1"
            env["NERVOUS_NO_REDIS"] = "1"
            proc = subprocess.Popen(
                [nervous_bin, "publish", "funsearch.artifact.v1", json.dumps(envelope)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
            )
            proc.wait(timeout=2)
        except Exception:
            pass


# Per-instance camera distance — keeps shapes in frame for their bbox.
# Rule: bbox_max * 2.2, minimum 3.0.
_INSTANCE_CAMERA_DIST: dict[str, float] = {
    "sphere":        3.0,
    "gyroid":        3.0,
    "round_box":     3.0,
    "warped_sphere": 3.0,
    "smooth_union":  3.5,
    "cloud_cluster": 4.5,   # bbox 2.0
    "torus_knot":    2.0,   # compact, tube only fills ~±1.0
    "helix_tube":    2.5,
    "scherk_first":  2.2,   # bbox 1.2
}


def render_sdf_to_png(
    sdf_glsl: str,
    out_path: Path,
    viewport: tuple[int, int] = (512, 512),
    i_time: float = 1.2,
    instance_name: str = "",
) -> bool:
    """Render an SDF function to PNG.

    Tries two paths in order:
    1. ShaderExecutor (GPU, fast) — used when silo_tester / EGL is available.
    2. In-house C++ sphere tracer (CPU, always works, zero VRAM).

    The C++ tracer matches the GLSL probe math exactly: same finite-difference
    normals (e=0.001), same march step (0.75×), same termination (|h|<0.0015×d).
    """
    # GLSL probe (GPU path, may fail headless)
    probe = f"""
{sdf_glsl}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {{
    vec2 uv = (fragCoord - 0.5*iResolution.xy) / iResolution.y;
    float t = {i_time};
    vec3 ro = vec3(2.5*cos(t), 1.4, 2.5*sin(t));
    vec3 ww=normalize(-ro); vec3 uu=normalize(cross(ww,vec3(0,1,0))); vec3 vv=cross(uu,ww);
    vec3 rd=normalize(uv.x*uu+uv.y*vv+1.7*ww);
    float d=0.001;
    vec3 col=vec3(0.04,0.02,0.07);
    for(int i=0;i<96;i++){{
        vec3 p=ro+d*rd; float h=sdf(p);
        if(abs(h)<0.0015*d||d>9.0) break; d+=h*0.75;
    }}
    if(d<9.0){{
        vec3 p=ro+d*rd; float e=0.001;
        vec3 n=normalize(vec3(sdf(p+vec3(e,0,0))-sdf(p-vec3(e,0,0)),
                              sdf(p+vec3(0,e,0))-sdf(p-vec3(0,e,0)),
                              sdf(p+vec3(0,0,e))-sdf(p-vec3(0,0,e))));
        vec3 ld=normalize(vec3(1.5,2.0,-0.5));
        float diff=max(dot(n,ld),0.0),spec=pow(max(dot(reflect(-ld,n),-rd),0.0),48.0);
        col=vec3(0.15,0.55,0.85)*diff+vec3(1.0)*spec*0.7+vec3(0.08,0.04,0.12)*(1.0-diff);
        col*=exp(-d*0.12);
    }}
    fragColor=vec4(pow(clamp(col,0.0,1.0),vec3(0.4545)),1.0);
}}
"""
    try:
        from autobench.engines.shader_executor import ShaderExecutor
        ex = ShaderExecutor()
        result = ex.render_only(probe, out_path=str(out_path), viewport=viewport, i_time=i_time)
        if result.frame_path != "" and Path(result.frame_path).exists():
            return True
    except Exception as e:
        logger.debug("ShaderExecutor render unavailable (%s), falling back to CPU tracer", e)

    # CPU sphere tracer fallback — translates GLSL to C++ by extracting the sdf() body
    # The GLSL sdf_glsl contains `float sdf(vec3 p) { ... }` — convert to C++ signature.
    cpp_code = sdf_glsl.replace(
        "float sdf(vec3 p)",
        "extern \"C\" float sdf(float x, float y, float z)"
    )
    if "float x=p.x" not in cpp_code and "float x, float y, float z" in cpp_code:
        pass  # already C++ signature
    elif "vec3 p" not in cpp_code and "float x, float y, float z" not in cpp_code:
        cpp_code = cpp_code  # passthrough
    # Inject x=p.x binding when needed
    if "extern \"C\" float sdf(float x, float y, float z)" in cpp_code and "float x=p.x" not in cpp_code:
        cpp_code = cpp_code.replace(
            "extern \"C\" float sdf(float x, float y, float z) {",
            "extern \"C\" float sdf(float x, float y, float z) {\n    (void)x; (void)y; (void)z;",
        )

    from autobench.engines.sdf_tracer import render_sdf_cpp_to_png
    cam_dist = _INSTANCE_CAMERA_DIST.get(instance_name, 3.5)
    return render_sdf_cpp_to_png(
        cpp_code,
        Path(out_path),
        viewport=viewport,
        i_time=i_time,
        camera_dist=cam_dist,
    )


def artifact_path_for(kernel: str, run_id: str, generation: int, fitness: float) -> Path:
    """Standard path for a kernel artifact."""
    slug = f"gen{generation:03d}_f{fitness:.4f}".replace(".", "p")
    return ARTIFACTS_ROOT / kernel / run_id / f"{slug}.png"
