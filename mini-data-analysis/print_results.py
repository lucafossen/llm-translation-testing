import argparse
import json
import re
import sys
from collections import defaultdict

# -------- helpers --------
def bits_from_name(name: str) -> int | None:
    """Extract the Q-bit width from a model name, or None if not present."""
    m = re.search(r"\bQ(\d+)", name)
    return int(m.group(1)) if m else (16 if "HF" in name or "FP16" in name else None)


def load_rows(path, scale):
    """Flatten the nested JSON into a list of {direction, metric, model, score, bits} dicts."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for metric, srcs in data.items():                 # bleu / chrf
        for src, tgts in srcs.items():               # eng_Latn
            for tgt, models in tgts.items():         # mri_Latn
                direction = f"{src.split('_')[0]}→{tgt.split('_')[0]}"
                for model_name, values in models.items():
                    score = float(values[0]) * scale
                    rows.append(
                        dict(
                            direction=direction,
                            metric=metric,
                            model=model_name,
                            bits=bits_from_name(model_name),
                            score=score,
                        )
                    )
    return rows


def make_tables(rows, baseline_substring):
    """Group rows by (direction, metric) and print comparison tables."""
    by_dir_metric = defaultdict(list)
    for r in rows:
        by_dir_metric[(r["direction"], r["metric"])].append(r)

    for (direction, metric), lst in by_dir_metric.items():
        # find baseline
        baseline = next(
            (r for r in lst if baseline_substring in r["model"]), None
        )
        if baseline is None:
            print(
                f"[WARN] Baseline '{baseline_substring}' not found for "
                f"{direction}/{metric}",
                file=sys.stderr,
            )
            continue
        base_val = baseline["score"]

        print(f"\n### {direction} — {metric.upper()}  "
              f"(baseline = {baseline['model']} @ {base_val:.2f})")
        print("| Model | bits | score | Δabs | Δrel % |")
        print("|-------|------|-------|------|--------|")

        # sort by bit-width (FP-16 at the top), then by name
        for r in sorted(lst, key=lambda x: (x["bits"] or 99, x["model"])):
            delta_abs = r["score"] - base_val
            delta_rel = 100.0 * delta_abs / base_val if base_val else 0.0
            print(
                f"| {r['model']} | {r['bits'] or '?'} | {r['score']:.2f} | "
                f"{delta_abs:+.2f} | {delta_rel:+.1f} |"
            )


# -------- main entry-point --------
def main():
    p = argparse.ArgumentParser(
        description="Summarise BLEU/ChrF scores per quantisation level."
    )
    p.add_argument("json_file", help="Path to the benchmark JSON file.")
    p.add_argument(
        "--baseline",
        default="LLaMA 2 13B Chat - HF",
        help="Substring that identifies the FP-16 baseline model.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1,
        help="Multiply raw scores by this factor (use 1.0 to keep 0–1 scale).",
    )
    args = p.parse_args()

    rows = load_rows(args.json_file, args.scale)
    make_tables(rows, args.baseline)


if __name__ == "__main__":
    main()
