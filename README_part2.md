# Part 2 — Sentiment Steles  (OpenSCAD CSG  +  Blender modifier/driver)

> **Theme**: Each TF-IDF cluster of movie-trailer comments becomes a
> "review stele" (影评碑). 27 steles arranged on three concentric rings
> (action / romance / horror) form a 1-minute mood arc — height tracks
> intensity, twist tracks controversy, surface erosion tracks
> negativity, emission tracks positive sentiment.

## How this differs from demo130 / demo131

The brief says software **environments** can be reused; "thinking" cannot.
Below is the explicit differentiation table — every row swaps either
the tool or the paradigm:

| Layer | demo130 (Brutalism) | demo131 (Millennium-China) | **demo124 (this Part 2)** |
|---|---|---|---|
| Theme | Western Brutalism | China 2000–2010 landmarks | **Movie trailer reviews → influence steles** |
| Design tool 1 | Blender (`bmesh` imperative) | Grasshopper (visual nodes) | **OpenSCAD (declarative CSG: `union/difference/linear_extrude`)** |
| Design tool 2 | TouchDesigner (`.toe` audio-reactive) | Blender (Geometry-Node trees) | **Blender (modifier stack + F-Curve drivers + NLA)** |
| Geometry paradigm | vertex-level edits | per-vertex node graph | **boolean ops on primitive solids → STL** |
| Animation paradigm | keyframed phase blocks | Reveal slider + helical orbit | **NLA-driven mood Action + Bezier follow-path camera** |
| Layout | 5×4 grid | linear stack + dendrogram | **3 concentric rings (one per genre)** |
| Time variable | per-fragment delay | dendrogram leaf order | **single global `Mood_Driver.location.x` empty** |
| Output container | `.toe` + `.blend` | `.gh` + `.blend` | **`.scad` + `.stl` × 27 + `.blend`** |

No file-format collision with the previous two demos.

## Pipeline

```
Part-1 outputs                          (movie_review_pipeline/data, outputs/models/*)
        │
        ▼
bridge/export_clusters_json.py          → outputs/fragments_params/clusters.json   (27 fragments)
        │
        ├──────────────► openscad/batch_generate.py
        │                  ├── reads clusters.json
        │                  ├── invokes  openscad -D height=… -D twist=… stele.scad
        │                  └── writes   outputs/stl/<fragment_id>.stl   × 27
        │
        ▼
blender/import_steles.py            ← STL fragments imported into Blender
blender/modifier_stack.py           ← per-cluster preset stack
blender/driver_bindings.py          ← drivers tie data → modifiers
blender/camera_and_animation.py     ← NLA mood arc + Bezier camera path
blender/render_eevee_next.py        ← 1080p / 24 fps / EEVEE Next
        │
        ▼
outputs/renders/movie_review.mp4    (≥ 65 s)
```

## Run

### 1. Build the data contract (10 s)

```powershell
cd movie_review_pipeline
.\.venv\Scripts\python.exe bridge\export_clusters_json.py
# -> outputs/fragments_params/clusters.json
```

### 2. Generate 27 STL fragments via OpenSCAD (1–2 min)

> Install OpenSCAD first: <https://openscad.org/downloads.html> (≈30 MB).
> The script auto-discovers it in default install paths or via the
> `OPENSCAD_BIN` environment variable.

```powershell
.\.venv\Scripts\python.exe openscad\batch_generate.py
# -> outputs/stl/stele-<genre>-<tmdb_id>-c<cluster>-r<rank>.stl
```

Smoke check before the full run:

```powershell
.\.venv\Scripts\python.exe openscad\batch_generate.py --dry-run --limit 3
```

### 3. Stage the Blender scene (2 min)

Open `Blender 4.x` → **Scripting** workspace → **Open** → run the five
scripts **in order**:

1. `blender/import_steles.py`
2. `blender/modifier_stack.py`
3. `blender/driver_bindings.py`
4. `blender/camera_and_animation.py`
5. `blender/render_eevee_next.py`

Then **Render → Render Animation** (or run
`bpy.ops.render.render(animation=True)` from a sixth script).
Output: `outputs/renders/movie_review.mp4` (1080p / 24 fps / 65 s).

## Data → form mapping (auditable in `clusters.json`)

| Part-1 statistic | Part-2 form | Operator |
|---|---|---|
| `avg_intensity` (1..5) | stele height (100–300 mm) | OpenSCAD `linear_extrude(height=…)` + Blender Z scale driver |
| `cluster_id` (0..5) | torsion ±100° | `linear_extrude(twist=…)` + driven `SimpleDeform` |
| `vocab_diversity` (TTR) | helical window count + size | OpenSCAD `for (i = [0:windows_n-1]) difference{ ... }` |
| `n_reviews` (log-scaled) | plinth width | OpenSCAD `square([2*half_base, …])` |
| `pos_ratio − neg_ratio` | top-chamfer angle | OpenSCAD `rotate([0, chamfer_ang, 0]) cube(...)` |
| `std_intensity` | erosion bumps + emission | OpenSCAD `sphere(r=rd)` + Principled BSDF emission driver |
| poster mean-RGB | Blender base material colour | Principled BSDF Base Color |
| **global mood** | timeline-driven scale + emission | **`Mood_Driver` empty** with NLA action |

## Fallback if OpenSCAD is unavailable

`blender/import_steles.py` builds a tapered-cone primitive per fragment
when its STL is missing. The pipeline therefore renders end-to-end even
without OpenSCAD installed, **but only one design-tool environment
(Blender) is exercised**, which is below the assignment's "≥ 2"
threshold. Install OpenSCAD before the final submission.

## Output checklist (for OneDrive upload)

* `outputs/fragments_params/clusters.json`            (data contract)
* `outputs/stl/*.stl`                                 (27 fragments)
* `openscad/stele.scad`                               (the parametric source)
* `<your-blender-file>.blend`                         (saved after Step 3)
* `outputs/renders/movie_review.mp4`                  (≥ 65 s)
* `README_part2.md`, `requirements_part2.txt`         (this file & deps)

## Credits

* OpenSCAD — Marius Kintel & contributors (GPL-2.0).
* Blender 4.x Python API — Blender Foundation.
* Part-1 datasets, GPT augmentation, ML models — see `README.md`.
