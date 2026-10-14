"""Submission manifest: walk a final run's lineage back to the trained runs and write it as Markdown.

Every derived run in this project records where it came from in its run directory: `prc blend` writes the member runs
and weights (metrics.json "blend"), the lab scripts write a "stack" record (source run, steps, parameters, and for the
ADS-B stage the full argv, seeds and learner), the combined base writes winter_base.json (stack + month-pair members).
This script follows those records from a submitted run down to the trained configurations and writes one Markdown
file with: the upload (version, board score, file digest), the lineage tree with the step and parameters at every node,
the trained runs (configuration file, seed, holdout scores, git commit), and the package versions. Nothing is
re-derived from memory, so the manifest stays true to what was submitted (data-science review, 5 Oct: "record the
weights in a manifest"; findings 10.23).

Usage: python lab/manifest.py <run_id> [--out docs/submission_manifest.md] [--version N]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import paths  # noqa: E402


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def config_files() -> dict[str, str]:
    """Run name -> configuration file (configs/ and configs/_generated/)."""
    out = {}
    for p in sorted(paths.ROOT.joinpath("configs").rglob("*.yaml")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("name:"):
                out.setdefault(line.split(":", 1)[1].strip().strip('"\''), str(p.relative_to(paths.ROOT)).replace("\\", "/"))
                break
    return out


class Walker:
    def __init__(self):
        self.seen: set[str] = set()
        self.leaves: dict[str, dict] = {}
        self.lines: list[str] = []
        self.cfg = config_files()
        ix = paths.RUNS / "index.csv"
        self.ix = pd.read_csv(ix).set_index("run_id") if ix.exists() else pd.DataFrame()

    def score(self, run: str) -> str:
        if run in self.ix.index:
            r = self.ix.loc[run]
            r = r.iloc[-1] if isinstance(r, pd.DataFrame) else r
            return f"holdout {float(r['rmse_ex_dayoffset']):.2f} without wrong-date rows, bulk {float(r['rmse_bulk']):.2f}"
        p = paths.run_dir(run) / "oof_predictions.parquet"   # lab runs are not in index.csv: score their OOF directly
        if p.exists():
            o = pd.read_parquet(p, columns=None)
            if {"y_true", "y_pred", "dayoffset"} <= set(o.columns) and len(o) <= 400_000:   # holdout only, not a combined base
                e = o["y_pred"].astype(float) - o["y_true"].astype(float)
                exw = float((e[~o["dayoffset"].astype(bool)] ** 2).mean() ** 0.5)
                bulk = float((e[o["y_true"] < 3600] ** 2).mean() ** 0.5)
                return f"holdout {exw:.2f} without wrong-date rows, bulk {bulk:.2f}"
        return ""

    def out(self, depth: int, text: str):
        self.lines.append("  " * depth + "- " + text)

    def walk(self, run: str, depth: int = 0, role: str = ""):
        d = paths.run_dir(run)
        head = f"{role}: " if role else ""
        if run in self.seen:
            self.out(depth, f"{head}`{run}` (expanded above)")
            return
        self.seen.add(run)
        m = read_json(d / "metrics.json") or {}
        wb = read_json(d / "winter_base.json")
        st, bl = m.get("stack"), m.get("blend")
        if wb:
            self.out(depth, f"{head}`{run}`: combined base for the ADS-B stage ({wb['n_stack']:,} holdout rows from the "
                            f"stack's out-of-fold prediction + {wb['n_winter']:,} rows from {len(wb['winter_runs'])} "
                            f"month-pair members); `python lab/build_winter_base.py <stack> <groups>`")
            self.walk(wb["stack"], depth + 1, "stack (holdout days and 2026)")
            for w in wb["winter_runs"]:
                self.walk(w, depth + 1, "month-pair member (other 2025 days)")
            return
        if st:
            steps = st.get("steps", [])
            if "adsb_own_movement" in steps:
                kind = "direct" if "--direct" in st.get("argv", []) else "residual"
                self.out(depth, f"{head}`{run}`: ADS-B stage, {kind} formulation, learner {st.get('learner', 'catboost')}, "
                                f"seeds {st.get('seeds')}, first seed {st.get('seed0', 42)}, fold seed {st.get('fold_seed', 0)}, "
                                f"{st.get('n_rank_corrected', 0):,} 2026 rows corrected; {self.score(run)}\n"
                                + "  " * (depth + 1) + f"`python lab/adsb_apply.py {' '.join(st.get('argv', []))}`")
                self.walk(st["source"], depth + 1, "base")
            elif "blend_direct" in steps:
                self.out(depth, f"{head}`{run}`: covered rows = {1 - st['w_direct']:.1f} x residual member + "
                                f"{st['w_direct']:.1f} x direct member; `python lab/blend_direct.py <residual> <direct> --write`")
                self.walk(st["source"], depth + 1, "residual member")
                self.walk(st["direct"], depth + 1, f"direct member (weight {st['w_direct']})")
            elif "base_swap_uncovered" in steps:
                self.out(depth, f"{head}`{run}`: on the rows neither ADS-B stage corrects, add (new stack - old stack), "
                                f"clipped to the NM join window; `python lab/base_swap.py "
                                + " ".join(f"--swap {k}=<run>" for k in st["swaps"]) + " ...`")
                self.walk(st["source"], depth + 1, "candidate")
                self.walk(st["new_stack"], depth + 1, "new stack")
            elif "forced_long_shift" in steps:
                sh = ", ".join(f"{x:.0f}" for x in st.get("shift", []))
                self.out(depth, f"{head}`{run}`: forced-long shift on rows whose join window forces a taxi above "
                                f"{st['min_l']:.0f} s (shrinkage k = {st['k']:.0f}, band shifts {sh} s); "
                                f"`python lab/forced_long_shift.py <source> --min-l {st['min_l']:.0f} --k {st['k']:.0f} --write`")
                self.walk(st["source"], depth + 1, "source")
            elif "ltfm_partner_candidate" in steps:
                w = st["w_specialist"]
                self.out(depth, f"{head}`{run}`: the {st['airport']} rows ({st['rank_rows_changed']:,} ranking rows) = "
                                f"{1 - w:.1f} x source + {w:.1f} x the airport specialist with the partner-runway block, "
                                f"clipped to the NM join window; "
                                f"`python lab/ltfm_partner_candidate.py <source> <specialist> --write <name> --w {w}`")
                self.walk(st["source"], depth + 1, "source")
                self.walk(st["ltfm_specialist"], depth + 1, f"specialist (weight {w})")
            elif steps[:1] == ["lobt_window"]:
                self.out(depth, f"{head}`{run}`: stack = NM join-window clip + in-window estimate + linked-row regime "
                                f"exclusion (rates {st.get('rates')}); `python lab/final_stack.py <blend> --inwindow 0.5 --write`")
                self.walk(st["source"], depth + 1, "blend")
            else:
                self.out(depth, f"{head}`{run}`: steps {steps}, record {json.dumps({k: v for k, v in st.items() if k != 'source'})}")
                if "source" in st:
                    self.walk(st["source"], depth + 1, "source")
            return
        if bl and bl.get("runs"):
            ws = bl.get("weights") or [1 / len(bl["runs"])] * len(bl["runs"])
            extra = ", NM join-window bound" if bl.get("lobt_bound") else ""
            self.out(depth, f"{head}`{run}`: blend of {len(bl['runs'])} runs{extra}; {self.score(run)}")
            for r, w in zip(bl["runs"], ws):
                self.walk(r, depth + 1, f"weight {w:.3f}")
            return
        c = read_json(d / "config.resolved.json") or {}
        env = read_json(d / "env.json") or {}
        refit = (d / "refit.json").exists()
        self.leaves[run] = {"name": c.get("name", run.split("__")[1]), "seed": c.get("seed"),
                            "config": self.cfg.get(c.get("name", ""), "?"), "git": env.get("git_sha", "")[:8],
                            "refit": refit, "score": self.score(run), "packages": env.get("packages", {}),
                            "months": "+".join(m[-2:] for m in c.get("validation", {}).get("holdout_months", []))}
        self.out(depth, f"{head}`{run}`: trained, `{self.leaves[run]['config']}`, seed {c.get('seed')}"
                        f"{', refit on all twelve months' if refit else ''}; validated on 2025 months {self.leaves[run]['months']}: {self.score(run)}")


def main():
    final = sys.argv[1]
    outp = paths.ROOT / arg("--out", "docs/submission_manifest.md")
    w = Walker()
    w.walk(final)
    up = pd.read_csv(paths.ROOT / "submissions" / "uploads.csv")
    u = up[up["run_id"] == final]
    sub = paths.ROOT / "submissions" / f"{final}__submitting.parquet"
    digest = hashlib.sha256(sub.read_bytes()).hexdigest() if sub.exists() else "(file not written yet)"
    lines = [f"# Submission manifest: `{final}`", "",
             "Generated by `python lab/manifest.py " + " ".join(sys.argv[1:]) + "` from the run records; do not edit by hand.", ""]
    if len(u):
        r = u.iloc[-1]
        lines += [f"- Upload: `{r['object']}` at {r['uploaded_at_utc']}, status {r['status']}, "
                  f"board RMSE **{float(r['score']):.4f}** on {int(r['used_pairs']):,} rows."]
    lines += [f"- Submission file: `submissions/{final}__submitting.parquet`, SHA-256 `{digest}`.",
              "- Board scores of every upload: `docs/board_history.csv`, the tracked copy of the local upload log.", "",
              "## Lineage", "",
              "Each node names the run directory under `runs/` (kept locally and not published), the step that produced "
              "it and the command; children are its inputs. Weights are the blend weights as recorded.", ""]
    lines += w.lines
    lines += ["", "To reproduce, run the commands from the leaves upwards: the `prc train` runs below (each with its "
              "configuration and seed, `python -m prc train -c <configuration> -o seed=<seed>`), then the blends "
              "(`python -m prc blend --runs ... --weights ... --lobt-bound`), the stacks, the combined bases, the ADS-B "
              "stage runs (which need `python lab/adsb_features.py` for the crawled days), and the post-processing steps, "
              "then `python -m prc submit --run <final run>`. Scores of the month-pair members are on their own "
              "validation months, not on January + July."]
    lines += ["", f"## Trained runs ({len(w.leaves)})", "",
              "The git column names the development repository's commit at each run's start; that history is not "
              "published.", "",
              "| run | configuration | seed | refit | git | validation months | score on them |", "|---|---|---|---|---|---|---|"]
    for run, L in sorted(w.leaves.items()):
        lines.append(f"| `{run}` | `{L['config']}` | {L['seed']} | {'yes' if L['refit'] else 'no'} | {L['git']} | {L['months']} | {L['score'].replace('holdout ', '')} |")
    pk = next((L["packages"] for L in w.leaves.values() if L["packages"]), {})
    if pk:
        lines += ["", "## Software", "", "Recorded in the trained runs' `env.json`: " +
                  ", ".join(f"{k} {v}" for k, v in sorted(pk.items())) + "; Python 3.13, Windows 11, CatBoost on an "
                  "RTX 4070 Laptop GPU (GPU training is not bit-reproducible; seed spreads are reported in findings.md)."]
    outp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {outp.relative_to(paths.ROOT) if outp.is_relative_to(paths.ROOT) else outp}: {len(w.seen)} runs, {len(w.leaves)} trained")


if __name__ == "__main__":
    main()
