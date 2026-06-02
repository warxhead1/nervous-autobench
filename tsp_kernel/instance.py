"""TSP data containers: TSPLIB instances, candidates, islands, ULIDs.

Pure data + parsing — no sandbox, no LLM, no kernel loop. Kept dependency-free
so scoring/oracle/loop can all import from here without circular edges.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path


# Crockford base32 (no I, L, O, U) — ULID alphabet.
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    """Return a 26-char ULID: 48-bit ms timestamp + 80 random bits.

    Sortable and monotonic-ish by creation time, unlike uuid4 — the schemas
    declare run_id as a ULID, so run ordering by id is meaningful.
    """
    value = ((int(time.time() * 1000) & ((1 << 48) - 1)) << 80) | int.from_bytes(os.urandom(10), "big")
    out = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


# ---------------------------------------------------------------------------
# TSPLIB instance format
# ---------------------------------------------------------------------------

TSPLIB_URL_BASE = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"


def fetch_tsplib_instance(name: str, cache_dir: Path | None = None) -> "TSPInstance":
    """Load a TSPLIB instance from the local cache (or download if missing)."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "instances"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for .tsp.gz first (compressed archive format)
    gz_path = cache_dir / f"{name}.tsp.gz"
    tsp_path = cache_dir / f"{name}.tsp"

    if gz_path.exists():
        path = gz_path
    elif tsp_path.exists():
        path = tsp_path
    else:
        # No local copy — embedded benchmark data (no network fetch in kernel)
        raise FileNotFoundError(
            f"Instance '{name}' not found in {cache_dir}. "
            f"Run 'python -m autobench.tsp_kernel bootstrap' first to download."
        )

    inst = TSPInstance.from_file(path)

    # Load optimal tour length if .opt.tour.gz exists
    opt_gz = cache_dir / f"{name}.opt.tour.gz"
    if opt_gz.exists():
        import gzip
        with gzip.open(opt_gz, 'rt', encoding='utf-8', errors='replace') as f:
            content = f.read()
        # Optimal tour format: one integer per line, after header
        reading_tour = False
        tour = []
        for line in content.splitlines():
            if reading_tour and line.strip():
                try:
                    tour.append(int(line.strip()))
                except ValueError:
                    pass
            if "TOUR_SECTION" in line:
                reading_tour = True
        if tour and len(tour) > 1:
            # Compute optimal length from coords
            opt_len = 0.0
            for i in range(len(tour) - 1):
                opt_len += inst.distance(tour[i] - 1, tour[i + 1] - 1)
            inst.optimal_tour_length = opt_len
            inst.optimal_tour = tour

    return inst


@dataclass
class TSPInstance:
    name: str
    n: int
    coords: list[tuple[float, float]]
    dist_matrix: list[list[float]] | None = None
    optimal_tour_length: float | None = None
    optimal_tour: list[int] | None = None

    def distance(self, i: int, j: int) -> float:
        if self.dist_matrix is not None:
            return self.dist_matrix[i][j]
        cx, cy = self.coords[i]
        dx, dy = self.coords[j]
        return ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5

    def compute_dist_matrix(self) -> None:
        n = self.n
        self.dist_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                cx, cy = self.coords[i]
                dx, dy = self.coords[j]
                d = ((cx - dx) ** 2 + (cy - dy) ** 2) ** 0.5
                self.dist_matrix[i][j] = d
                self.dist_matrix[j][i] = d

    @classmethod
    def from_file(cls, path: Path) -> "TSPInstance":
        # Handle .tsp.gz and .tsp extensions
        name = path.stem  # 'berlin52.tsp' or 'berlin52'
        if name.endswith('.tsp'):
            name = name[:-4]
        # Decompress if needed
        if path.suffix == '.gz':
            import gzip
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
                content = f.read()
        else:
            content = path.read_text(encoding='utf-8', errors='replace')
        lines = content.splitlines()

        coords: list[tuple[float, float]] = []
        dimension: int | None = None
        reading_coords = False

        for line in lines:
            line = line.strip()
            if line.startswith("DIMENSION"):
                dimension = int(line.split()[-1])
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                reading_coords = False
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        idx, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                        coords.append((x, y))
                    except ValueError:
                        continue
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True

        # If dimension was declared but coords don't match, truncate or pad
        if dimension and len(coords) != dimension:
            if len(coords) < dimension:
                coords = coords + [(0.0, 0.0)] * (dimension - len(coords))

        n = len(coords)
        inst = cls(name=name, n=n, coords=coords)
        inst.compute_dist_matrix()
        return inst


@dataclass
class Tourevaluation:
    tour: list[int]
    length: float
    instance_name: str


@dataclass
class CandidateProgram:
    id: str
    priority_code: str
    island: int
    generation: int
    fitness: float = 0.0
    fitness_variance: float = 0.0
    worst_fitness: float = 0.0
    computation_time_ms: float = 0.0
    source: str = "llm"  # "llm" | "baseline" | "mutated" | "migrated"
    evaluated: bool = False  # set once fitness is computed; skip re-evaluation
    per_instance_fitness: dict = field(default_factory=dict)  # instance_name → fitness ratio

    @property
    def code(self) -> str:
        """Alias for priority_code — satisfies FunSearchKernel.evaluate_fitness interface."""
        return self.priority_code


@dataclass
class Island:
    id: int
    population: list[CandidateProgram] = field(default_factory=list)
    best_program: CandidateProgram | None = None


# Known optimal tour lengths for benchmark instances
KNOWN_OPTIMALS: dict[str, float] = {
    "berlin52": 7542.0,
    "eil101": 629.0,
    "kroa100": 21282.0,
    "ch130": 6110.0,
    "ts225": 126643.0,
    "pr1002": 259045.0,
}
