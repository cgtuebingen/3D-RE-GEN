
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Literal
import pandas as pd
import csv

import logging

def _make_timestamped_dir(base_dir: Path) -> Path:
    """
    Create ``base_dir/YY_MM_DD_HHMMSS`` and return the Path.
    """
    ts = datetime.now().strftime("%y_%m_%d_%H%M%S")
    out_dir = base_dir / ts
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def dump_evaluation(
    metrics: Dict[str, float],
    config: Dict[str, Any],
    out_root: str = "output/evaluation",
) -> Path:
    """
    Save ``metrics`` (as JSON) and a copy of the config (as JSON) into a
    timestamped folder.  Returns the folder Path for later use.
    """
    base_dir = Path(out_root)
    out_dir = _make_timestamped_dir(base_dir)

    #  metrics → JSON
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    # copy config file
    config_path = out_dir / "config.yaml"
    # copy file instead of rewriting it
    shutil.copyfile(config["config_path"], config_path)

    logging.info(f"✅  Evaluation saved to: {out_dir}")

    # save to csv
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            # Scale PSNR value to be between 0 and 1

            writer.writerow([k.replace('_', ' ').title(), f"{v:.6f}"])
    

    return out_dir



def load_metrics(evaluation_dir: Path) -> Dict[str, float]:
    """Read the ``metrics.json`` file from a given evaluation folder."""
    with (evaluation_dir / "metrics.json").open() as f:
        return json.load(f)


def _sorted_subdirs(parent: Path) -> List[Path]:
    """Return sub‑folders sorted chronologically (oldest → newest)."""
    return sorted([p for p in parent.iterdir() if p.is_dir()], key=lambda p: p.name)


def get_previous_evaluation(current_dir: Path) -> Path | None:
    """
    Find the *most recent* evaluation folder that is older than ``current_dir``.
    Returns ``None`` if there is no earlier run.
    """
    parent = current_dir.parent
    sorted_dirs = _sorted_subdirs(parent)
    # keep only those that are strictly earlier
    earlier = [d for d in sorted_dirs if d < current_dir]
    # try to return the last one
    item = earlier[-1] if earlier else None
    if item is None:
        logging.warning("ℹ️  No previous evaluation found for comparison.")

    return item


def compare_metrics_to_csv(
    prev: Dict[str, float],
    cur: Dict[str, float],
    prev_name: str = "previous",
    cur_name: str = "current",
    csv_path: Path = Path("comparison.csv"),
) -> None:
    """
    Build a CSV table that shows ``prev`` and ``cur`` side‑by‑side.
    The larger value in each row is highlighted.
    """
    # Ensure both dicts contain the same keys (missing keys get NaN)
    all_keys = list(cur.keys())

    rows = []
    for k in all_keys:
        v_prev = prev.get(k, float("nan"))
        v_cur = cur.get(k, float("nan"))

        # decide which one is larger (ignore NaN)
        if not (pd.isna(v_prev) or pd.isna(v_cur)):
            if v_cur > v_prev:
                cur_fmt = f"{v_cur:.6f}"
                prev_fmt = f"{v_prev:.6f}"
            elif v_prev > v_cur:
                prev_fmt = f"{v_prev:.6f}"
                cur_fmt = f"{v_cur:.6f}"
            else:  # equal
                prev_fmt = cur_fmt = f"{v_cur:.6f}"
        else:
            # one of them is missing → just print what we have
            prev_fmt = f"{v_prev:.6f}" if not pd.isna(v_prev) else "--"
            cur_fmt = f"{v_cur:.6f}" if not pd.isna(v_cur) else "--"

        rows.append([k.replace('_', ' ').title(), prev_fmt, cur_fmt])

    with csv_path.open("w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", prev_name.title(), cur_name.title()])
        writer.writerows(rows)

    print(f"✅  Comparison table saved to: {csv_path}")