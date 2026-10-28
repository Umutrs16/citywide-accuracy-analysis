"""Recompute the reported audit numbers from the saved JSON and CBC logs."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def log_objective(text):
    match = re.search(r"Objective value:\s+([+-]?\d+(?:\.\d+)?)", text)
    return None if match is None else float(match.group(1))


def main():
    rev4 = load("revision4_summary.json")
    rev5 = load("revision5_summary.json")
    rows = rev5["slice_a_rows"]
    ghsl = np.array([row["unweighted_delta_census"] for row in rows if row["population"] == "ghsl"], dtype=float)
    world = np.array([row["unweighted_delta_census"] for row in rows if row["population"] == "worldpop"], dtype=float)
    stored_g = rev5["slice_a"]["by_population"]["ghsl"]["unweighted_delta_census"]["p50"]
    stored_w = rev5["slice_a"]["by_population"]["worldpop"]["unweighted_delta_census"]["p50"]
    print(f"subset A rows {len(rows)}")
    print(f"proven rows {sum(bool(row['proven_optimal']) for row in rows)}")
    print(f"GHSL median gap {np.median(ghsl):.3f} stored {stored_g:.3f}")
    print(f"WorldPop median gap {np.median(world):.3f} stored {stored_w:.3f}")
    if abs(np.median(ghsl) - stored_g) > 1e-6 or abs(np.median(world) - stored_w) > 1e-6:
        raise SystemExit("Median gap does not match the stored summary.")
    if len(ghsl) != 12 or len(world) != 12:
        raise SystemExit("Expected 12 scenarios on each population surface.")
    if not all(row["proven_optimal"] and row["result_line"] == "Result - Optimal solution found" for row in rows):
        raise SystemExit("A subset-A row is not marked proven.")
    for row in rows:
        path = RES / "cbc_logs" / Path(str(row["log"])).name
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Result - Optimal solution found" not in text:
            raise SystemExit(f"Missing optimal line: {path.name}")
        objective = log_objective(text)
        if objective is None or abs(objective - float(row["objective"])) > 1e-3:
            raise SystemExit(f"Objective mismatch: {path.name}")
    logs = list((RES / "cbc_logs").glob("*.log"))
    unproven = [path.name for path in logs if "Result - Optimal solution found" not in path.read_text(encoding="utf-8", errors="replace")]
    if unproven:
        raise SystemExit(f"Logs without an optimal line: {unproven}")
    jaccard = rev4["nearest_point"]["joint_400"]["jaccard_mean"]
    print(f"nearest-point joint mean Jaccard {jaccard:.6f} n {rev4['nearest_point']['joint_400']['n']}")
    print(f"CBC logs {len(logs)}")
    print("checks passed")


if __name__ == "__main__":
    main()
