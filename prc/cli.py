"""Command line entry points: `python -m prc <command>`."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from prc import paths


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", required=True, help="path to an experiment YAML (relative to configs/ is fine)")
    p.add_argument(
        "--override",
        "-o",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. model.params.depth=10 data.sample_frac=0.2",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prc", description="PRC Data Challenge 2026 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="build data/interim/ from the raw competition parquets")
    p.add_argument("--force", action="store_true", help="rebuild even if the artifacts exist")

    p = sub.add_parser("fetch-weather", help="download METAR/TAF archives once and normalise them to parquet")
    p.add_argument("--only", choices=["metar", "taf"], help="restrict to one source (default: both)")
    p.add_argument("--force", action="store_true", help="re-download chunks that already exist")
    p.add_argument("--offline", action="store_true", help="no network: rebuild the parquets from the raw files")
    p.add_argument("--check", action="store_true", help="coverage report only: no network, no rebuild")

    p = sub.add_parser("fetch-geometry", help="download OSM stands/taxiways and OurAirports runways once and normalise them")
    p.add_argument("--only", choices=["osm", "ourairports"], help="restrict to one source (default: both)")
    p.add_argument("--force", action="store_true", help="re-download files that already exist")
    p.add_argument("--offline", action="store_true", help="no network: rebuild the parquets from the raw files")
    p.add_argument("--check", action="store_true", help="match/coverage report only: no network, no rebuild")
    p.add_argument("--radius-km", type=float, default=None, help="half-size of the OSM box around each airport (default 7)")
    p.add_argument("--config", "-c", default="base.yaml", help="config whose geometry.stands_file to use (default: base.yaml)")
    p.add_argument("--stands-file", default=None, help="hand-placed stands CSV (apt, stand_mvt, lat, lon); overrides the config")

    p = sub.add_parser("fetch-era5", help="download the Open-Meteo ERA5 hourly archive per airport (snow depth, snowfall, frost) once and normalise it")
    p.add_argument("--force", action="store_true", help="re-download airports that already exist")
    p.add_argument("--offline", action="store_true", help="no network: rebuild the parquet from the raw files")
    p.add_argument("--check", action="store_true", help="coverage report only: no network, no rebuild")

    p = sub.add_parser("fetch-adsb", help="stream adsb.lol daily history and keep the points at the ten airports (resumable)")
    p.add_argument("--days", nargs="*", default=None, help="YYYY-MM-DD days (default: Jan+Jul 2025 and 2026, interleaved)")
    p.add_argument("--workers", type=int, default=3, help="parallel days (the link is the limit)")
    p.add_argument("--kind", choices=["prod", "staging"], default="prod", help="adsb.lol release: prod (points/) or the second aggregator instance staging (points_staging/)")

    p = sub.add_parser("adsb-surface", help="build a snapshot of the ADS-B surface events of other aircraft (per day, resumable) and consolidate it")
    p.add_argument("--days", nargs="*", default=None, help="YYYY-MM-DD days or A..B ranges (inclusive) to add to the snapshot's day set "
                   "(a new snapshot without --days takes every day with points)")
    p.add_argument("--workers", type=int, default=1, help="parallel days (CPU; keep 1 next to other work)")
    p.add_argument("--snapshot", default=None, help="snapshot tag under cache/adsb_surface/ (default: $PRC_ADSB_SNAPSHOT or 's1')")
    p.add_argument("--check", action="store_true", help="coverage report of the snapshot only: no build")
    p.add_argument("--support", action="store_true", help="compute the snapshot's airport x family support table from the "
                   "2025 training departures (needs `prepare`): no build; rerun after every rebuild")
    p.add_argument("--force", action="store_true", help="rebuild every day of the set even if its inputs are unchanged")
    p.add_argument("--freeze", action="store_true", help="mark the snapshot frozen (never modified again; needs a support "
                   "table of the current consolidated table): build, --check, --support, then --freeze")

    p = sub.add_parser("fetch-pru", help="download Eurocontrol PRU open CSVs (taxi-out additional time, pre-departure delays, airport ATFM delays) once and normalise them")
    p.add_argument(
        "--only",
        choices=["taxi_out_additional_time", "atc_pre_departure_delays", "all_pre_departure_delays", "apt_dly"],
        help="restrict the download to one dataset (default: all four)",
    )
    p.add_argument("--force", action="store_true", help="re-download files that already exist (the 2026 files are republished monthly)")
    p.add_argument("--offline", action="store_true", help="no network: rebuild the parquets from the raw files")
    p.add_argument("--check", action="store_true", help="coverage report only: no network, no rebuild")

    p = sub.add_parser("levelcheck", help="compare a run's predicted monthly mean taxi-out per airport with the PRU's published means")
    p.add_argument("--run", required=True, help="run id (directory name under runs/) with test_predictions.parquet")

    p = sub.add_parser("features", help="build or inspect the feature cache")
    _add_config_args(p)
    p.add_argument("--list", action="store_true", help="show resolved blocks and cache status, build nothing")

    p = sub.add_parser("train", help="run one experiment")
    _add_config_args(p)
    p.add_argument("--seeds", type=int, default=1, help="run N seeds and record the noise floor")
    p.add_argument("--no-test", action="store_true", help="skip ranking-set predictions (faster for sweeps)")
    p.add_argument("--keep-test", action="store_true", help="with --seeds: keep every seed's ranking predictions (and refit) for `prc blend`")

    p = sub.add_parser("queue", help="run several configs one after another, live in this terminal, with a summary at the end")
    p.add_argument("--config", "-c", nargs="+", required=True, metavar="CFG", help="configs to run, in order")
    p.add_argument("--override", "-o", nargs="*", default=[], metavar="KEY=VALUE", help="dotted overrides applied to every config")
    p.add_argument("--seeds", type=int, default=1, help="run each config under N seeds (records the noise floor)")
    p.add_argument("--no-test", action="store_true", help="skip ranking-set predictions")
    p.add_argument("--keep-test", action="store_true", help="with --seeds: keep every seed's ranking predictions (and refit)")
    p.add_argument("--stop-on-error", action="store_true", help="stop at the first failing config (default: continue)")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")

    p = sub.add_parser("blend", help="average the predictions of several finished runs into one new run directory")
    p.add_argument("--runs", nargs="+", required=True, metavar="RUN_ID", help="run ids to average (same folds, same rows)")
    p.add_argument("--name", default=None, help="name of the blend run (default: blend_<first run's config name>)")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="one weight per run (normalised); default equal weights")
    p.add_argument("--lobt-bound", action="store_true", help="clip the blended predictions to the NM join window (prc/postprocess.py); allows a single run")

    p = sub.add_parser("ablate", help="measure per-block value by retraining without / with each block")
    _add_config_args(p)
    p.add_argument("--mode", choices=["leave-one-out", "add-one-in", "add-to-base"], default="leave-one-out")
    p.add_argument("--blocks", help="comma-separated blocks to test (add-to-base: registered blocks to add, required)")
    p.add_argument("--emit-configs", action="store_true", help="write the variants to configs/_generated/ and print the `prc queue` line; train nothing")

    p = sub.add_parser("leaderboard", help="compare runs, or per-feature usefulness")
    p.add_argument("--features", action="store_true", help="show the per-block effect table instead of runs")
    p.add_argument("--airports", action="store_true", help="show per-airport RMSE across runs")
    p.add_argument("--sort", default="rmse")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--name", help="filter runs by config name substring")
    p.add_argument("--rebuild", action="store_true", help="regenerate index.csv from the run directories")

    p = sub.add_parser("progress", help="how far along an in-flight training run is")
    p.add_argument("--run", help="run id (default: the most recent run)")
    p.add_argument("--tail", type=int, default=5, help="learning-curve rows to show")

    p = sub.add_parser("audit", help="data-quality report: extreme taxi times, the untyped segment, what cleaning rules would flag")
    p.add_argument("--config", "-c", default=None, help="a config whose data.clean rules to evaluate (optional)")
    p.add_argument("--override", "-o", nargs="*", default=[], metavar="KEY=VALUE", help="dotted config overrides")
    p.add_argument("--run", help="also show where this run's squared error lives (run id)")

    p = sub.add_parser("submit", help="write a submission parquet from a finished run")
    p.add_argument("--run", required=True, help="run id (directory name under runs/)")

    p = sub.add_parser(
        "predict",
        help="apply a finished run's refit model to the ranking rows of the current data tree (no retraining)",
    )
    p.add_argument("--run", required=True, help="run id (directory name under runs/) or a run directory path")
    p.add_argument(
        "--override", "-o", nargs="*", default=[], metavar="KEY=VALUE",
        help="data.version=<v> (required in a data tree other than the run's) and notes=...; nothing else may change",
    )
    p.add_argument("--tree", default=None,
                   help="the data tree this data.version is for (required with a data.version other than the run's): "
                        "refused unless PRC_DATA_DIR points at it")
    p.add_argument("--name", default=None, help="name of the new run (default: predict_<the run's config name>)")
    p.add_argument("--threads", type=int, default=-1, help="CatBoost prediction threads (default -1: all cores)")
    p.add_argument("--allow-drift", action="store_true", help="predict although feature code changed since the run (see predict.json)")
    p.add_argument("--allow-unseen", action="store_true",
                   help="predict although unseen categorical levels cover far more ranking rows than on the run's own")
    p.add_argument("--no-count-rescale", action="store_true",
                   help="keep this tree's whole-frame counts (stand/runway/carrier/type frequencies) on the ranking rows")
    p.add_argument("--no-parity", action="store_true", help="skip the training-row parity report against the run's cache")
    p.add_argument("--features-only", action="store_true",
                   help="build (warm) the run's feature matrix in this data tree, measure the parity, and stop")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Redirecting to a file makes stdout block-buffered, so a long run shows
    # nothing until it exits -- which makes a slow fit indistinguishable from a
    # hung one. Line buffering costs nothing here and keeps `> run.log` useful.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):  # pragma: no cover - unusual streams
        pass
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)

    if args.command == "prepare":
        from prc import data

        try:
            data.prepare(force=args.force)
        except ValueError as exc:
            print(f"prc prepare: {exc}", file=sys.stderr)
            return 1
        return 0

    if args.command == "fetch-weather":
        from prc import weather

        if args.check:
            print(weather.check(only=args.only))
            return 0
        if not args.offline:
            weather.fetch(only=args.only, force=args.force)
        weather.normalise(only=args.only)
        return 0

    if args.command == "fetch-geometry":
        from prc import geometry
        from prc.config import load_config

        if args.check:
            print(geometry.check())
            return 0
        if not args.offline:
            geometry.fetch(only=args.only, force=args.force, radius_km=args.radius_km)
        stands_file = args.stands_file or load_config(args.config).geometry.stands_file
        geometry.normalise(stands_file=stands_file)
        return 0

    if args.command == "fetch-adsb":
        from prc.adsb import fetch as adsb_fetch

        adsb_fetch.fetch(days=args.days, workers=args.workers, kind=args.kind)
        return 0

    if args.command == "adsb-surface":
        from prc.adsb import surface

        if args.check:
            print(surface.check(args.snapshot))
            return 0
        if args.support or (args.freeze and not (args.days or args.force)):  # no build: the support and/or the freeze
            if args.support:
                surface.build_support(args.snapshot)
            if args.freeze:
                surface.freeze_snapshot(args.snapshot)
            return 0
        surface.build(days=surface.parse_days(args.days), snapshot=args.snapshot, workers=args.workers,
                      force=args.force, freeze=args.freeze)
        return 0

    if args.command == "fetch-era5":
        from prc import reanalysis

        if args.check:
            print(reanalysis.check())
            return 0
        if not args.offline:
            reanalysis.fetch(force=args.force)
        reanalysis.normalise()
        return 0

    if args.command == "fetch-pru":
        from prc import pru

        if args.check:
            print(pru.check())
            return 0
        if not args.offline:
            pru.fetch(only=args.only, force=args.force)
        pru.normalise()
        return 0

    if args.command == "levelcheck":
        from prc.levelcheck import report

        print(report(args.run))
        return 0

    if args.command == "features":
        from prc import data
        from prc.config import load_config
        from prc.features import resolve
        from prc.features.build import build_matrix, cache_status

        cfg = load_config(args.config, args.override)
        specs = resolve(cfg.features.include, cfg.features.exclude)
        status = cache_status(specs, data_version=cfg.data.version, params=cfg.features.params)
        print(status.to_string(index=False))
        if args.list:
            target_blocks = [s.name for s in specs if s.uses_target]
            if target_blocks:
                print(f"\nfold-dependent (recomputed per fold): {target_blocks}")
            return 0

        mvt = data.load_movements()
        y = data.load_target()
        fit_mask = pd.Series(False, index=mvt.index)
        fit_mask.loc[y.index] = True
        dep_idx = data.dep_index(mvt)
        print()
        fm = build_matrix(
            specs, mvt, dep_idx,
            data_version=cfg.data.version, params=cfg.features.params,
            fold="warm", fit_mask=fit_mask, target=y.reindex(dep_idx), cfg=cfg,
        )
        print(f"\nmatrix: {fm.X.shape[0]:,} rows x {fm.X.shape[1]} columns")
        return 0

    if args.command == "train":
        from prc.ablate import seed_noise
        from prc.config import load_config
        from prc.train import run_experiment

        cfg = load_config(args.config, args.override)
        if args.seeds > 1:
            seed_noise(cfg, [cfg.seed + i for i in range(args.seeds)], compute_test=args.keep_test and not args.no_test)
        else:
            run = run_experiment(cfg, compute_test=not args.no_test)
            print(f"\nartifacts: {run.dir}")
        return 0

    if args.command == "queue":
        from prc.queue import run as run_queue

        return run_queue(
            args.config, seeds=args.seeds, no_test=args.no_test, keep_test=args.keep_test,
            overrides=args.override, stop_on_error=args.stop_on_error, dry_run=args.dry_run,
        )

    if args.command == "blend":
        from prc.blend import blend_runs

        run = blend_runs(args.runs, name=args.name, lobt_bound=args.lobt_bound, weights=args.weights)
        print(f"\nartifacts: {run.dir}")
        return 0

    if args.command == "ablate":
        from prc.ablate import run_ablation
        from prc.config import load_config

        cfg = load_config(args.config, args.override)
        blocks = [b.strip() for b in args.blocks.split(",") if b.strip()] if args.blocks else None
        frame = run_ablation(cfg, mode=args.mode, blocks=blocks, emit_only=args.emit_configs)
        cols = ["block", "config"] if args.emit_configs else ["block", "method", "metric", "value"]
        print("\n" + frame[cols].to_string(index=False))
        return 0

    if args.command == "leaderboard":
        from prc import leaderboard, tracking

        if args.rebuild:
            tracking.rebuild_index()
        if args.features:
            table, noise = leaderboard.features()
            print(leaderboard.format_features(table, noise))
        elif args.airports:
            frame = leaderboard.by_airport(limit=args.limit)
            print(frame.to_string(index=False) if not frame.empty else "no per-airport metrics recorded yet")
        else:
            frame = leaderboard.runs(sort_by=args.sort, limit=args.limit, name=args.name)
            print(frame.to_string(index=False) if not frame.empty else f"no runs yet under {paths.RUNS}")
        return 0

    if args.command == "progress":
        from prc.progress import report

        print(report(args.run, tail=args.tail))
        return 0

    if args.command == "audit":
        from prc.audit import report
        from prc.config import load_config

        cfg = load_config(args.config, args.override) if args.config else None
        print(report(cfg, args.run))
        return 0

    if args.command == "submit":
        from prc.submit import build_submission

        build_submission(args.run)
        return 0

    if args.command == "predict":
        from prc.predict import PredictRefused, predict_run

        try:
            run = predict_run(
                args.run, args.override, name=args.name, threads=args.threads,
                allow_drift=args.allow_drift, features_only=args.features_only, tree=args.tree,
                allow_unseen=args.allow_unseen, count_rescale=not args.no_count_rescale, parity=not args.no_parity,
            )
        except PredictRefused as exc:
            print(f"prc predict refused: {exc}", file=sys.stderr)
            return 2
        if run is not None:
            print(f"\nartifacts: {run.dir}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
