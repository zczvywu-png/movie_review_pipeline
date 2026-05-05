// stele.scad — one parametric "review stele" (影评碑) per fragment.
//
// CSG paradigm: the entire form is composed by declarative boolean
// operations on primitive solids — `union()`, `difference()`,
// `linear_extrude()`, `offset()`. NO mesh manipulation, NO node graphs.
// This keeps the geometric thinking deliberately distinct from
// demo130 (Blender bmesh — imperative vertex code) and demo131
// (Blender Geometry Nodes / Grasshopper — node-based dataflow).
//
// All six inputs are 0..1 normalised values produced by
// `bridge/export_clusters_json.py`. Pass them on the OpenSCAD CLI:
//
//   openscad -o frag.stl \
//            -D height=0.7 \
//            -D twist=0.3 \
//            -D porosity=0.4 \
//            -D base_width=0.8 \
//            -D top_chamfer=0.5 \
//            -D rugged=0.2 \
//            -D seed=4242 \
//            stele.scad
//
// Architectural mapping (review-data → carved form):
//   height       avg comment intensity (1..5)             vertical mass
//   twist        TF-IDF cluster id of the discourse       torsion + spiral
//   porosity     vocabulary type-token ratio              window count + size
//   base_width   log(review count)                        plinth scale
//   top_chamfer  pos_ratio − neg_ratio                    angled crown cut
//   rugged       std(intensity) — controversy             surface erosion

// ---------- inputs (defaults are mid-range so the file previews fine) -----
height      = 0.7;
twist       = 0.3;
porosity    = 0.4;
base_width  = 0.8;
top_chamfer = 0.5;
rugged      = 0.2;
seed        = 4242;

// ---------- resolved real-world dimensions (in millimetres) ---------------
h_total      = 100 + 200 * height;                 // 100 .. 300 mm tall
half_base    = (40 + 60 * base_width) / 2;          // 20 .. 50 mm half side
twist_deg    = (twist - 0.5) * 200;                 // -100 .. +100 deg over height
chamfer_h    = 30 + 50 * top_chamfer;               // 30 .. 80 mm crown bite
chamfer_ang  = 12 + 35 * top_chamfer;               // 12 .. 47 deg slant
windows_n    = round(6 + 10 * porosity);            // 6 .. 16 voids
window_r     = 3 + 5 * porosity;                    // 3 .. 8 mm half-width
rugged_amp   = 0.4 + 4.0 * rugged;                  // 0.4 .. 4.4 mm bump radius
n_bumps      = round(8 + 16 * rugged);              // 8 .. 24 erosion bumps

// Lower facet count keeps CSG cheap for OpenSCAD 2021.01 — each subtracted
// sphere/box still has a tractable triangle budget when combined.
$fn = 14;

// ---------- one stele as a CSG tree --------------------------------------
module stele() {
    difference() {
        // (1) main body + plinth — the only additive layer
        union() {
            // twisted prism, gently tapered top so the chamfer reads cleaner
            linear_extrude(height = h_total,
                           twist  = twist_deg,
                           slices = 24,
                           scale  = [0.93, 0.93])
                offset(r = 1.5)
                    square([2 * half_base, 2 * half_base], center = true);
            // a slightly oversized plinth + a bevelled riser
            translate([0, 0, -10])
                cube([2 * (half_base + 6),
                      2 * (half_base + 6),
                      10], center = true);
            translate([0, 0, -16])
                cube([2 * (half_base + 4),
                      2 * (half_base + 4),
                      6], center = true);
        }

        // (2) angled top chamfer — rotate-and-subtract a big slab
        translate([0, 0, h_total - chamfer_h / 2])
            rotate([0, chamfer_ang, 0])
                cube([3 * half_base,
                      3 * half_base,
                      chamfer_h * 2], center = true);

        // (3) "review windows" — a helical array of carved boxes, count and
        //     size both increase with porosity (vocabulary diversity)
        for (i = [0 : windows_n - 1]) {
            zfrac = (i + 0.5) / windows_n;
            ang   = 360 * (zfrac * (1 + 0.6 * twist) + i * 0.137);
            translate([cos(ang) * (half_base * 0.62),
                       sin(ang) * (half_base * 0.62),
                       h_total * (0.08 + 0.84 * zfrac)])
                rotate([0, 0, ang])
                    cube([window_r * 2.4,
                          window_r * 0.85,
                          window_r * 1.1], center = true);
        }

        // (4) erosion bumps — small subtracted spheres, distribution
        //     deterministic via `rands(seed)`
        for (i = [0 : n_bumps - 1]) {
            zf = rands(0.05, 0.95, 1, seed + i)[0];
            af = rands(0, 360,  1, seed + 1000 + i)[0];
            rd = rands(rugged_amp * 0.5,
                       rugged_amp * 1.6,
                       1,
                       seed + 2000 + i)[0];
            translate([cos(af) * (half_base + 0.4),
                       sin(af) * (half_base + 0.4),
                       zf * h_total])
                sphere(r = rd);
        }
    }
}

stele();
