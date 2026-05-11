#!/usr/bin/env python3
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = "gpu-blas"
BUILD_FILES = {"CMakeLists.txt"}
SHAPE_DEFINITION_PATTERN = re.compile(
    r"template\s*<\s*typename\s+TypeConfigTemplate\s*,\s*typename\s+DimTypes\s*>"
    r"\s*struct\s+(\w+)"
)
WORKLOAD_REGISTRATION_PATTERN = re.compile(r"BASELINER_REGISTER_WORKLOAD")
def collect_line_counts(root: str) -> dict[str, int]:
    """Return {relative_path: line_count} for every file under root."""
    result = subprocess.run(
        ["find", root, "-type", "f", "-exec", "wc", "-l", "{}", "+"],
        capture_output=True, text=True, check=True,
    )
    mapping: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] != "total":
            mapping[parts[1]] = int(parts[0])
    return mapping

def count_pattern(path: str, pattern: re.Pattern) -> int:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return len(pattern.findall(text))


def extract_shape_names(path: str) -> list:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return SHAPE_DEFINITION_PATTERN.findall(text)

#Classify the current file into 

#     {
#         "bucket": "build"|"common"|"shapes_def"|"backend_common"|"shape_common"|"spec_common"|"spec_registration"
#         "backend":dynamic,
#         "shape": dynamic,
#         "specialization": dynamuc,
#     }
#
#
#
#
def classify(rel_path: str) -> dict:
    """
    Returns a dict with classification info:
      bucket: "build" | "common" | "shapes_def" | "backend_common"
              | "shape_common" | "spec_common" | "spec_registration"
      backend, shape, specialization: when applicable
    """
    parts = Path(rel_path).parts
    name = parts[-1]
    # parts[0] is always ROOT ("gpu-blas")
    depth = len(parts) - 1  # number of dirs below ROOT (file itself excluded)

    if name in BUILD_FILES:
        return {"bucket": "build"}

    # gpu-blas/<file>
    if depth == 1:
        if name == "BlasShapes.hpp":
            return {"bucket": "shapes_def"}
        return {"bucket": "common"}

    backend = parts[1]

    # gpu-blas/<backend>/<file>
    if depth == 2:
        return {"bucket": "backend_common", "backend": backend}

    shape = parts[2]

    # gpu-blas/<backend>/<shape>/<file>
    if depth == 3:
        return {"bucket": "shape_common", "backend": backend, "shape": shape}

    specialization = parts[3]

    # gpu-blas/<backend>/<shape>/<spec>/<file>
    # .hpp -> spec_common, .cu/.hip -> registration site
    suffix = Path(name).suffix
    if suffix in (".cu", ".hip"):
        bucket = "spec_registration"
    else:
        bucket = "spec_common"
    return {
        "bucket": bucket,
        "backend": backend,
        "shape": shape,
        "specialization": specialization,
    }


# -----------------------------------------------------------------------------
# Step 4: aggregate
# -----------------------------------------------------------------------------

def main() -> None:
    line_counts = collect_line_counts(ROOT)

    stats = {
        "build_loc": 0,
        "common_loc": 0,
        "shapes_loc": 0,        # LOC of BlasShapes.hpp
        "shape_count": 0,       # number of shape struct definitions in BlasShapes.hpp
        "shape_names": [],      # captured names of those shape structs
        "backends": defaultdict(lambda: {
            "common_loc": 0,
            "shapes": defaultdict(lambda: {
                "shape_common_loc": 0,
                "specializations": defaultdict(lambda: {
                    "common_loc": 0,
                    "registration_loc": 0,
                    "workload_count": 0,
                }),
            }),
        }),
    }

    for rel_path, loc in line_counts.items():
        info = classify(rel_path)
        bucket = info["bucket"]

        if bucket == "build":
            stats["build_loc"] += loc

        elif bucket == "common":
            stats["common_loc"] += loc

        elif bucket == "shapes_def":
            stats["shapes_loc"] = loc
            names = extract_shape_names(rel_path)
            stats["shape_names"] = names
            stats["shape_count"] = len(names)

        elif bucket == "backend_common":
            stats["backends"][info["backend"]]["common_loc"] += loc

        elif bucket == "shape_common":
            shape_data = stats["backends"][info["backend"]]["shapes"][info["shape"]]
            shape_data["shape_common_loc"] += loc

        elif bucket == "spec_common":
            spec = stats["backends"][info["backend"]]["shapes"][info["shape"]] \
                       ["specializations"][info["specialization"]]
            spec["common_loc"] += loc

        elif bucket == "spec_registration":
            spec = stats["backends"][info["backend"]]["shapes"][info["shape"]] \
                       ["specializations"][info["specialization"]]
            spec["registration_loc"] += loc
            spec["workload_count"] += count_pattern(rel_path, WORKLOAD_REGISTRATION_PATTERN)

    print_report(stats)


def print_report(stats: dict) -> None:
    #n_backends
    backends = stats["backends"]
    n_backends = len(backends)

    # Per-backend totals
    backend_totals = {}
    backend_workload_counts = {}
    for name, backend in backends.items():
        total = backend["common_loc"]
        n_workloads = 0
        for shape in backend["shapes"].values():
            total += shape["shape_common_loc"]
            for spec in shape["specializations"].values():
                total += spec["common_loc"] + spec["registration_loc"]
                n_workloads += spec["workload_count"]
        backend_totals[name] = total
        backend_workload_counts[name] = n_workloads

    # Per-shape totals (across backends)
    shape_names = set()
    for backend in backends.values():
        shape_names.update(backend["shapes"].keys())

    shape_totals_per_backend = {s: {} for s in shape_names}
    spec_names_per_shape = {s: set() for s in shape_names}
    for backend_name, backend in backends.items():
        for shape_name, shape in backend["shapes"].items():
            t = shape["shape_common_loc"]
            for spec_name, spec in shape["specializations"].items():
                t += spec["common_loc"] + spec["registration_loc"]
                spec_names_per_shape[shape_name].add(spec_name)
            shape_totals_per_backend[shape_name][backend_name] = t

    grand_total_workloads = sum(backend_workload_counts.values())
    grand_total_source_loc = (
        stats["common_loc"] + stats["shapes_loc"] + sum(backend_totals.values())
    )

   ## BASELINER-BLAS -- Implementation Effort
    print()
    print("+" + "-" * 68 + "+")
    print("|" + "  baseliner-blas — Implementation Effort".ljust(68) + "|")
    print("+" + "-" * 68 + "+")

    print(f"\n  Total source LOC                        {grand_total_source_loc:>6}")
    print(f"  Total build LOC (CMakeLists.txt)        {stats['build_loc']:>6}")
    print(f"  Backends supported                      {n_backends:>6}  "
          f"({', '.join(backends.keys())})")
    shape_list = ", ".join(stats["shape_names"]) if stats["shape_names"] else "(none)"
    print(f"  Shape kinds defined (BlasShapes.hpp)    {stats['shape_count']:>6}  ({shape_list})")
    print(f"  Workloads registered (total)            {grand_total_workloads:>6}")

   
    section("Top-level structure (gpu-blas/)")
    rows = [
        ("Common headers",            stats["common_loc"]),
        ("BlasShapes.hpp",            stats["shapes_loc"]),
        ("Backends (combined)",       sum(backend_totals.values())),
        ("Build files",               stats["build_loc"]),
    ]
    print_table(rows, total_label="TOTAL", total_value=grand_total_source_loc + stats["build_loc"])

  
    section("Per backend")
    rows = []
    for name in backends:
        rows.append((name, backend_totals[name]))
    print_table(rows, total_label="All backends", total_value=sum(backend_totals.values()))

  
    section("Per shape (summed across backends)")
    rows = []
    for shape_name in sorted(shape_names):
        per_backend = shape_totals_per_backend[shape_name]
        n_specs = len(spec_names_per_shape[shape_name])
        total = sum(per_backend.values())
        breakdown = ", ".join(f"{b}: {v}" for b, v in per_backend.items())
        rows.append((f"{shape_name}  ({n_specs} specs — {breakdown})", total))
    print_table(rows)

  
    section("Detailed breakdown")
    for backend_name, backend in backends.items():
        print(f"\n  {backend_name}/  —  {backend_totals[backend_name]} LOC, "
              f"{backend_workload_counts[backend_name]} workloads")
        print(f"    backend-common headers . . . . . . . . . . {backend['common_loc']:>5} LOC")

        for shape_name, shape in backend["shapes"].items():
            shape_total = shape["shape_common_loc"]
            for spec in shape["specializations"].values():
                shape_total += spec["common_loc"] + spec["registration_loc"]

            print(f"    {shape_name}/   ({shape_total} LOC)")
            print(f"      shape-common headers . . . . . . . . . {shape['shape_common_loc']:>5} LOC")

            for spec_name, spec in shape["specializations"].items():
                total = spec["common_loc"] + spec["registration_loc"]
                print(f"      {spec_name + '/':<22}"
                      f"  hpp: {spec['common_loc']:>3}  "
                      f"impl: {spec['registration_loc']:>3}  "
                      f"workloads: {spec['workload_count']:>2}  "
                      f"total: {total:>4}")

    section("Paper numbers")

    # Average LOC per specialization
    all_spec_totals = []
    for backend in backends.values():
        for shape in backend["shapes"].values():
            for spec in shape["specializations"].values():
                all_spec_totals.append(spec["common_loc"] + spec["registration_loc"])
    avg_spec = sum(all_spec_totals) / len(all_spec_totals) if all_spec_totals else 0

    # Average LOC per registration site
    all_reg_locs = []
    for backend in backends.values():
        for shape in backend["shapes"].values():
            for spec in shape["specializations"].values():
                if spec["workload_count"] > 0:
                    all_reg_locs.append(spec["registration_loc"])
    avg_reg = sum(all_reg_locs) / len(all_reg_locs) if all_reg_locs else 0

    # Average LOC per workload (two definitions)
    total_reg_loc = sum(
        spec["registration_loc"]
        for backend in backends.values()
        for shape in backend["shapes"].values()
        for spec in shape["specializations"].values()
    )
    avg_full_per_workload = (
        grand_total_source_loc / grand_total_workloads if grand_total_workloads else 0
    )

    print(f"  Total source LOC . . . . . . . . . . . . . . . {grand_total_source_loc}")
    print(f"  Cross-backend shared LOC (gpu-blas/*) . . . .  "
          f"{stats['common_loc'] + stats['shapes_loc']}  "
          f"({pct(stats['common_loc'] + stats['shapes_loc'], grand_total_source_loc)} of source)")
    print(f"  Backend-specific LOC . . . . . . . . . . . . . "
          f"{sum(backend_totals.values())}  "
          f"({pct(sum(backend_totals.values()), grand_total_source_loc)} of source)",   f"{sum(backend_totals.values())/len(backend_totals)} per backend ",f"({pct(sum(backend_totals.values())/len(backend_totals), grand_total_source_loc)} of source)",)
    print(f"  Shape kinds (struct templates) . . . . . . . . . {stats['shape_count']}")
    print(f"  LOC per Shape  . . . . . . . . . . . . . . . . . {stats['shapes_loc'] / stats['shape_count']}")
    print(f"  Workloads registered . . . . . . . . . . . . . . {grand_total_workloads}")
    print(f"  Specializations across all backends  . . . . . . {len(all_spec_totals)}")
    print(f"  Avg LOC per specialization . . . . . . . . . . . {avg_spec:.1f}")
    print(f"  Avg LOC per registration site (.cu/.hip) . . . . {avg_reg:.1f}")
    print(f"  Avg LOC per workload (Total LOC / nb_workloads). {avg_full_per_workload:.1f}")
    print()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n  ── {title} " + "─" * max(0, 60 - len(title)))


def print_table(rows, total_label: str | None = None, total_value: int | None = None) -> None:
    if not rows:
        return
    label_w = max(len(str(label)) for label, _ in rows)
    if total_label:
        label_w = max(label_w, len(total_label))
    for label, value in rows:
        print(f"    {str(label).ljust(label_w)}   {value:>6}")
    if total_label is not None and total_value is not None:
        print(f"    {'─' * label_w}   ──────")
        print(f"    {total_label.ljust(label_w)}   {total_value:>6}")


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0%"
    return f"{100 * part / whole:.0f}%"


if __name__ == "__main__":
    main()