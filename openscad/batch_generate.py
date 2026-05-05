"""Drive OpenSCAD CLI over every fragment in ``clusters.json``.

For each fragment we call ``openscad -o frag.stl -D ... stele.scad`` so the
geometry is regenerated from a single source-of-truth ``stele.scad``
file. This is the entire OpenSCAD half of the pipeline:

    bridge/clusters.json      (the data contract)
        │
        ▼
    openscad/stele.scad  +  six -D parameters per fragment
        │
        ▼
    outputs/stl/<fragment_id>.stl

Run from the project root:

    python openscad/batch_generate.py             # full run
    python openscad/batch_generate.py --dry-run   # only print commands
    python openscad/batch_generate.py --limit 3   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger("openscad")

DEFAULT_CANDIDATES = [
    r"C:\Program Files\OpenSCAD\openscad.exe",
    r"C:\Program Files (x86)\OpenSCAD\openscad.exe",
    r"C:\Program Files\OpenSCAD (Nightly)\openscad.exe",
    r"E:\OpenSCAD\openscad.exe",
    r"D:\OpenSCAD\openscad.exe",
    "/usr/bin/openscad",
    "/usr/local/bin/openscad",
    "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",
]


def find_openscad(explicit: str | None = None) -> str:
    """Locate the OpenSCAD binary.

    Order: explicit CLI arg ▶ ``OPENSCAD_BIN`` env var ▶ ``PATH`` ▶
    default install paths. Returns the absolute path or raises with a
    clear install hint.
    """
    if explicit:
        if Path(explicit).exists():
            return explicit
        raise FileNotFoundError(f"--openscad-bin path not found: {explicit}")

    env = os.environ.get("OPENSCAD_BIN")
    if env and Path(env).exists():
        return env

    on_path = shutil.which("openscad") or shutil.which("openscad.exe")
    if on_path:
        return on_path

    for cand in DEFAULT_CANDIDATES:
        if Path(cand).exists():
            return cand

    raise FileNotFoundError(
        "OpenSCAD not found. Install from https://openscad.org/downloads.html "
        "(≈30 MB) or set the OPENSCAD_BIN environment variable. "
        "On Windows the default install puts it at "
        r"C:\Program Files\OpenSCAD\openscad.exe."
    )


def render_fragment(
    openscad_bin: str,
    scad_path: Path,
    out_stl: Path,
    params: dict[str, float],
    seed: int,
    timeout_s: int = 300,
) -> bool:
    """Run ``openscad -o out.stl -D k=v ... scad_path``. Returns True on
    success."""
    cmd: list[str] = [openscad_bin, "-o", str(out_stl)]
    for k, v in params.items():
        cmd.extend(["-D", f"{k}={float(v):.6f}"])
    cmd.extend(["-D", f"seed={int(seed)}"])
    cmd.append(str(scad_path))

    log.info("[openscad] %s  →  %s", scad_path.name, out_stl.name)
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        log.error("openscad timed out (%ds) for %s", timeout_s, out_stl.name)
        return False

    if result.returncode != 0:
        log.error("openscad returned %d for %s\nSTDERR:\n%s",
                  result.returncode, out_stl.name, result.stderr.strip())
        return False
    if not out_stl.exists() or out_stl.stat().st_size < 200:
        log.error("openscad produced no/empty STL for %s", out_stl.name)
        return False
    return True


def build_all(
    payload: dict,
    scad_path: Path,
    out_dir: Path,
    openscad_bin: str,
    dry_run: bool,
    limit: int | None,
) -> tuple[int, int]:
    fragments = payload["fragments"]
    if limit is not None:
        fragments = fragments[:limit]
    ensure_dir(out_dir)

    n_ok, n_total = 0, len(fragments)
    for f in fragments:
        out_stl = out_dir / f"{f['fragment_id']}.stl"
        seed = (abs(hash(f["fragment_id"])) % (2**31 - 1))
        if dry_run:
            params_str = " ".join(
                f"-D {k}={float(v):.4f}" for k, v in f["params"].items()
            )
            print(f'openscad -o "{out_stl}" {params_str} -D seed={seed} '
                  f'"{scad_path}"')
            n_ok += 1
            continue
        ok = render_fragment(
            openscad_bin=openscad_bin,
            scad_path=scad_path,
            out_stl=out_stl,
            params=f["params"],
            seed=seed,
        )
        n_ok += int(ok)
    return n_ok, n_total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--clusters-json",
        default=str(PROJECT_ROOT / "outputs" / "fragments_params" / "clusters.json"),
    )
    p.add_argument(
        "--scad",
        default=str(PROJECT_ROOT / "openscad" / "stele.scad"),
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "outputs" / "stl"),
    )
    p.add_argument("--openscad-bin", default=None,
                   help="absolute path to openscad executable")
    p.add_argument("--dry-run", action="store_true",
                   help="print the openscad commands without running them")
    p.add_argument("--limit", type=int, default=None,
                   help="render only the first N fragments (smoke test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.clusters_json).read_text(encoding="utf-8"))

    if args.dry_run:
        bin_path = "openscad"
    else:
        bin_path = find_openscad(args.openscad_bin)
        log.info("using openscad: %s", bin_path)

    n_ok, n_total = build_all(
        payload=payload,
        scad_path=Path(args.scad),
        out_dir=Path(args.out_dir),
        openscad_bin=bin_path,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    log.info("[openscad] %d/%d fragments OK", n_ok, n_total)
    print(f"[openscad] {n_ok}/{n_total} fragments OK")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
