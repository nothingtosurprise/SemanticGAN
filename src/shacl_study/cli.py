import argparse
import csv
import importlib.metadata
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from shacl_study import __version__
from shacl_study.discovery import config_digest, format_plan, load_config, plan_queries, run_discovery, select_queries
from shacl_study.github_client import API_VERSION, GitHubAuthError, GitHubClient
from shacl_study.logging_utils import configure_logging
from shacl_study.storage import write_manifest
from shacl_study.statistics import (
    aggregate_batch_a,
    aggregate_pilot,
    census_tree,
    compare_cap_recheck,
    describe_tree_workload,
    estimate_heavy_runtime,
    load_verified_repositories,
    partition_statistics_batches,
    reusable_complete_results,
    select_heavy_benchmark,
    select_statistics_pilot,
    summarize_workload,
    unique_verified_trees,
)
from shacl_study.verification import (
    GitHubContent,
    config_sha256,
    load_verification_config,
    run_verification,
    select_benchmark,
    select_code_signal,
    select_phase2,
    select_phase2_pilot,
    select_pilot,
    resolve_tree_census,
    tree_identity_report,
)

NOT_IN_MILESTONE = ("inspect", "audit", "run-all")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="shacl_study")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="Discover candidate GitHub repositories.")
    discover.add_argument("--config", default="config/discovery_queries.yaml")
    discover.add_argument("--data-dir", default="data")
    discover.add_argument("--dry-run", action="store_true")
    discover.add_argument("--fresh", action="store_true", help="Ignore saved discovery state.")
    discover.add_argument("--max-pages", type=int, default=None)
    discover.add_argument("--per-page", type=int, default=100)
    discover.add_argument(
        "--query-id",
        action="append",
        default=None,
        help="Run one planned query id. Repeat to run several. Date partitions of a selected repository query still run.",
    )
    verify = subparsers.add_parser("verify", help="Inspect candidate repository content for SHACL.")
    verify.add_argument("--config", default="config/verification.yaml")
    verify.add_argument("--candidates", default="data/results/candidate_repositories.csv")
    verify.add_argument("--data-dir", default="data")
    verify.add_argument("--pilot", type=int, default=None, help="Inspect a stratified pilot of this many repositories.")
    verify.add_argument("--phase", type=int, choices=[1, 2], default=None, help="Phase 1 is code-search repositories. Phase 2 is the remainder. Phase 2 with --pilot does not start the full remainder.")
    verify.add_argument(
        "--benchmark",
        type=int,
        default=None,
        help="Time the two-pass strategy on this many code-signal repositories. Does not start Phase 1.",
    )
    verify.add_argument("--full-corpus", action="store_true")
    verify.add_argument("--fresh", action="store_true", help="Ignore saved verification state.")
    verify.add_argument("--repository-id", action="append", default=None)
    verify.add_argument("--tarball-threshold", type=int, default=None, help="Use one tarball when the candidate tier reaches this size.")
    verify.add_argument("--cache-dir", default=None, help="GitHub response cache. Defaults to data/raw/github-cache.")
    verify.add_argument("--benchmark-name", default="phase1_benchmark")
    verify.add_argument(
        "--tree-census",
        action="store_true",
        help="Phase 2 only. Resolve commit and tree SHAs without downloading repository files.",
    )
    stats = subparsers.add_parser("stats", help="Census shapes on unique verified trees. Pilot only.")
    stats.add_argument("--config", default="config/verification.yaml")
    stats.add_argument("--data-dir", default="data")
    stats.add_argument("--pilot", type=int, help="Number of unique verified trees. The full census is not authorized.")
    stats.add_argument("--recheck-capped", action="store_true", help="Re-enumerate only the pilot trees that hit the 400-file cap.")
    stats.add_argument(
        "--workload-census",
        action="store_true",
        help="Count candidate RDF files for every verified tree. Does not parse RDF.",
    )
    stats.add_argument(
        "--batch-a",
        action="store_true",
        help="Extract statistics for verified trees with at most 5,000 candidate RDF files.",
    )
    stats.add_argument(
        "--heavy-benchmark",
        action="store_true",
        help="Compare access strategies on 12 heavy trees. Does not run the rest of the heavy bucket.",
    )
    stats.add_argument(
        "--heavy",
        action="store_true",
        help="Extract the 5,001+ candidate-file trees with the same parser and enumeration rules as Batch A.",
    )
    stats.add_argument("--seed", type=int, default=20261009)
    for name in NOT_IN_MILESTONE:
        subparsers.add_parser(name, help="Not implemented.")

    args = parser.parse_args(argv)
    if args.command in NOT_IN_MILESTONE:
        raise SystemExit(f"{args.command} is not implemented.")
    if args.command == "verify":
        _verify(args)
        return
    if args.command == "stats":
        exclusive = [
            bool(args.pilot is not None),
            args.recheck_capped,
            args.workload_census,
            args.batch_a,
            args.heavy_benchmark,
            args.heavy,
        ]
        if sum(exclusive) > 1:
            raise SystemExit("Choose one statistics mode.")
        if args.workload_census:
            _workload_census(args)
            return
        if args.batch_a:
            _batch_a(args)
            return
        if args.heavy:
            _heavy(args)
            return
        if args.heavy_benchmark:
            _heavy_benchmark(args)
            return
        if args.recheck_capped:
            _recheck_capped(args)
            return
        if args.pilot is None:
            raise SystemExit(
                "stats requires --pilot N, --recheck-capped, --workload-census, --batch-a, --heavy, or --heavy-benchmark. "
                "The monolithic 6,343-tree run is not authorized."
            )
        _stats(args)
        return
    if args.max_pages is not None and args.max_pages < 1:
        raise SystemExit("--max-pages must be at least 1.")
    if args.per_page < 1 or args.per_page > 100:
        raise SystemExit("--per-page must be between 1 and 100.")

    config_path = Path(args.config)
    config = load_config(config_path)
    planned = select_queries(plan_queries(config), args.query_id)
    if args.query_id and not planned:
        raise SystemExit("No planned query matches --query-id.")

    if args.dry_run:
        sys.stdout.write(format_plan(planned))
        return

    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    data_dir = Path(args.data_dir)
    logger = configure_logging(data_dir / "logs" / "discovery.log")
    try:
        client = GitHubClient(token, data_dir / "raw" / "github-cache")
        state = run_discovery(
            config=config,
            config_sha256=config_digest(config_path),
            client=client,
            data_dir=data_dir,
            fresh=args.fresh,
            per_page=args.per_page,
            max_pages=args.max_pages,
            query_ids=args.query_id,
            window_end=date.today(),
            logger=logger,
        )
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc

    manifest = _manifest(config_path, state.partition_window_start, state.partition_window_end)
    write_manifest(data_dir / "results" / "run_manifest.json", manifest)
    truncated = sum(1 for item in state.executions.values() if item.potentially_truncated)
    print(
        f"candidates={len(state.candidates)} "
        f"nonpublic={len(state.nonpublic)} "
        f"queries={len(state.executions)} "
        f"truncated_or_split={truncated}"
    )


def _verify(args) -> None:
    phase2_pilot = args.phase == 2 and args.pilot is not None
    phase2_census = args.phase == 2 and args.tree_census
    selected = [
        phase2_census,
        phase2_pilot,
        args.pilot is not None and not phase2_pilot,
        args.phase == 1,
        args.phase == 2 and args.pilot is None and not args.tree_census,
        args.full_corpus,
        bool(args.repository_id),
        args.benchmark is not None,
    ]
    if sum(bool(item) for item in selected) != 1:
        raise SystemExit(
            "Refusing an unscoped verification run. Pass --pilot N, --phase 1, --phase 2, --phase 2 --pilot N, "
            "--phase 2 --tree-census, --benchmark N, or --repository-id. --full-corpus is separate."
        )
    if args.pilot is not None and args.full_corpus:
        raise SystemExit("Pass either --pilot or --full-corpus.")
    if args.pilot is not None and args.pilot < 1:
        raise SystemExit("--pilot must be at least 1.")
    if args.benchmark is not None and args.benchmark < 1:
        raise SystemExit("--benchmark must be at least 1.")
    if args.tarball_threshold is not None and args.tarball_threshold < 1:
        raise SystemExit("--tarball-threshold must be at least 1.")
    config_path = Path(args.config)
    config = load_verification_config(config_path)
    if args.tarball_threshold is not None:
        config["tarball_candidate_threshold"] = args.tarball_threshold
    data_dir = Path(args.data_dir)
    groups: dict[str, str] = {}
    with Path(args.candidates).open(encoding="utf-8", newline="") as handle:
        candidates = list(csv.DictReader(handle))
    if args.repository_id:
        wanted = set(args.repository_id)
        rows = [row for row in candidates if row["repository_id"] in wanted]
        missing = wanted - {row["repository_id"] for row in rows}
        if missing:
            raise SystemExit(f"Unknown repository id: {', '.join(sorted(missing))}")
    elif phase2_census:
        rows = select_phase2(candidates)
        print(f"phase=2 tree_census repositories={len(rows)}")
        print("Resolving commit and tree SHAs only. Repository files are not downloaded.")
    elif phase2_pilot:
        rows = select_phase2_pilot(candidates, args.pilot)
        print(f"phase=2 pilot repositories={len(rows)}")
    elif args.pilot is not None:
        rows = select_pilot(candidates, args.pilot)
    elif args.phase == 1:
        rows = select_code_signal(candidates)
        print(f"phase=1 code_signal_repositories={len(rows)}")
    elif args.phase == 2:
        rows = select_phase2(candidates)
        print(f"phase=2 repositories_without_code_signal={len(rows)}")
        print("Full Phase 2 is selected. This command inspects every remaining repository.")
    elif args.benchmark is not None:
        pilot_ids = _repository_ids(data_dir / "pilot" / "milestone2" / "repository_verification.csv")
        full_ids = _state_repository_ids(data_dir / "candidates" / "verification_state.json")
        rows = select_benchmark(
            candidates,
            args.benchmark,
            pilot_ids=pilot_ids,
            full_inspection_ids=full_ids,
        )
        groups = {str(row["repository_id"]): row.get("benchmark_group", "") for row in rows}
        print(
            "benchmark "
            f"repositories={len(rows)} "
            f"full_inspection={sum(group == 'full_inspection' for group in groups.values())} "
            f"pilot={sum(group == 'pilot' for group in groups.values())} "
            f"cold={sum(group == 'cold' for group in groups.values())}"
        )
    else:
        rows = candidates
    if not rows:
        raise SystemExit("No repositories selected.")
    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    if phase2_census:
        isolated = data_dir / "results" / "phase2_tree_census"
        log_path = data_dir / "logs" / "phase2-tree-census.log"
    elif phase2_pilot:
        isolated = data_dir / "results" / "phase2_pilot"
        log_path = data_dir / "logs" / "phase2-pilot.log"
    elif args.phase == 2:
        isolated = data_dir / "results" / "phase2"
        log_path = data_dir / "logs" / "phase2.log"
    elif args.benchmark is not None:
        isolated = data_dir / "results" / (args.benchmark_name or "phase1_benchmark")
        log_path = data_dir / "logs" / "verification.log"
    else:
        isolated = None
        log_path = data_dir / "logs" / "verification.log"
    logger = configure_logging(log_path)
    commit, dirty = _git_info(Path.cwd())
    try:
        cache_dir = Path(args.cache_dir) if args.cache_dir else data_dir / "raw" / "github-cache"
        client = GitHubClient(token, cache_dir)
        if phase2_census:
            _run_tree_census(args, rows, client, data_dir, isolated, logger)
            return
        summary = run_verification(
            rows=rows,
            source=GitHubContent(client, config),
            vocab=None,
            config=config,
            config_digest=config_sha256(config_path),
            data_dir=data_dir,
            fresh=args.fresh,
            clock=lambda: datetime.now(timezone.utc),
            logger=logger,
            phase=str(args.phase or ""),
            git_commit=commit,
            git_dirty=dirty,
            state_path=(isolated / "state.json") if isolated is not None else None,
            results_dir=isolated,
        )
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc
    counts = " ".join(f"{key}={value}" for key, value in sorted(summary["status_counts"].items()))
    print(f"verified_repositories={summary['repository_count']} {counts}")
    if args.benchmark is not None:
        report = _benchmark_report(
            data_dir,
            groups,
            results_dir=isolated,
            pilot_ids=_repository_ids(data_dir / "pilot" / "milestone2" / "repository_verification.csv"),
            full_ids=_state_repository_ids(data_dir / "candidates" / "verification_state.json"),
        )
        report_path = isolated / "benchmark_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        cold = report["cold_seconds"]
        print(
            "benchmark_report "
            f"discovery_hit={report['verified_from_discovery_hit']} "
            f"tree_inspection={report['tree_inspection']} "
            f"cold_median_s={cold['median']} "
            f"cold_mean_s={cold['mean']} "
            f"disagreements={report['disagreement_count']}"
        )


def _stats(args) -> None:
    if args.pilot > 200:
        raise SystemExit(
            "Refusing a statistics run above the pilot bound. "
            "The full 6,343-tree census is not authorized until this pilot is reviewed."
        )
    data_dir = Path(args.data_dir)
    results = data_dir / "results"
    out_dir = results / "statistics_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "logs" / "statistics-pilot.log"
    logger = configure_logging(log_path)
    verified = load_verified_repositories(results)
    trees = unique_verified_trees(verified)
    selected = select_statistics_pilot(trees, args.pilot, seed=args.seed)
    done_path = out_dir / "trees.jsonl"
    done = _load_jsonl_keys(done_path)
    logger.info(
        "statistics pilot selected=%s population_trees=%s population_repositories=%s already_saved=%s",
        len(selected),
        len(trees),
        len(verified),
        len(done),
    )
    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    config_path = Path(args.config)
    config = load_verification_config(config_path)
    try:
        client = GitHubClient(token, data_dir / "raw" / "github-cache")
        source = GitHubContent(client, config)
        clock = lambda: datetime.now(timezone.utc)
        for index, tree in enumerate(selected, start=1):
            if tree["tree_sha"] in done:
                continue
            logger.info(
                "statistics %s/%s %s donor=%s phase=%s fork=%s",
                index,
                len(selected),
                tree["tree_sha"][:12],
                tree["donor"]["repository_full_name"],
                tree["phase_origin"],
                tree["fork_origin"],
            )
            result = census_tree(tree, source=source, config=config, vocab=None, clock=clock)
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            logger.info(
                "statistics status %s nodes=%s properties=%s distinct=%s seconds=%s capped=%s",
                result["donor"],
                result["node_shapes"],
                result["property_shapes"],
                result["distinct_shapes"],
                result["seconds"],
                result["capped"],
            )
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc
    saved = _load_jsonl(done_path)
    summary = aggregate_pilot(saved, population_trees=len(trees), population_repositories=len(verified))
    summary["seed"] = args.seed
    summary["pilot_requested"] = args.pilot
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        "statistics_pilot "
        f"trees={summary['pilot_trees_extracted']}/{summary['pilot_trees_attempted']} "
        f"node_shapes={summary['shape_totals']['node_shapes']} "
        f"property_shapes={summary['shape_totals']['property_shapes']} "
        f"median_s={summary['runtime_seconds'].get('median')} "
        f"projected_hours_at_mean={summary['projection'].get('hours_at_sample_mean')}"
    )


def _recheck_capped(args) -> None:
    """Uncapped enumeration of the pilot trees whose counts stopped at 400 files."""
    data_dir = Path(args.data_dir)
    results = data_dir / "results"
    pilot_path = results / "statistics_pilot" / "trees.jsonl"
    pilot_rows = _load_jsonl(pilot_path)
    old_rows = [row for row in pilot_rows if row.get("capped")]
    if not old_rows:
        raise SystemExit(f"No capped trees in {pilot_path}.")
    out_dir = results / "statistics_pilot" / "cap_recheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "logs" / "statistics-cap-recheck.log"
    logger = configure_logging(log_path)
    verified = load_verified_repositories(results)
    trees = {item["tree_sha"]: item for item in unique_verified_trees(verified)}
    done_path = out_dir / "trees.jsonl"
    done = _load_jsonl_keys(done_path)
    logger.info("cap recheck trees=%s already_saved=%s", len(old_rows), len(done))
    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    config = load_verification_config(Path(args.config))
    try:
        client = GitHubClient(token, data_dir / "raw" / "github-cache")
        source = GitHubContent(client, config)
        clock = lambda: datetime.now(timezone.utc)
        for index, old in enumerate(old_rows, start=1):
            tree = trees.get(old["tree_sha"])
            if tree is None:
                raise SystemExit(f"Capped pilot tree {old['tree_sha']} is not in the verified corpus.")
            if old["tree_sha"] in done:
                continue
            logger.info(
                "cap recheck %s/%s %s donor=%s",
                index,
                len(old_rows),
                old["tree_sha"][:12],
                old["donor"],
            )
            result = census_tree(
                tree,
                source=source,
                config=config,
                vocab=None,
                clock=clock,
                checkpoint_path=out_dir / "checkpoints" / f"{old['tree_sha']}.json",
            )
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            logger.info(
                "cap recheck status %s nodes=%s->%s properties=%s->%s distinct=%s->%s complete=%s seconds=%s",
                result["donor"],
                old["node_shapes"],
                result["node_shapes"],
                old["property_shapes"],
                result["property_shapes"],
                old["distinct_shapes"],
                result["distinct_shapes"],
                result["enumeration_complete"],
                result["seconds"],
            )
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc
    new_rows = _load_jsonl(done_path)
    summary = compare_cap_recheck(old_rows, new_rows, pilot_rows, population_trees=len(trees))
    summary["verified_trees_already_at_cap"] = sum(1 for item in trees.values() if item["recorded_candidates"] >= 400)
    (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    complete = sum(1 for row in new_rows if row.get("enumeration_complete"))
    print(
        "cap_recheck "
        f"trees={len(new_rows)}/{len(old_rows)} "
        f"complete={complete} "
        f"extra_seconds={summary['extra_seconds']} "
        f"blended_hours_at_mean={summary['projection_blended_pilot'].get('hours_at_sample_mean')}"
    )


def _workload_census(args) -> None:
    """Metadata-only candidate counts. This does not parse RDF or write shape statistics."""
    data_dir = Path(args.data_dir)
    results = data_dir / "results"
    out_dir = results / "statistics_pilot" / "workload_census"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(data_dir / "logs" / "statistics-workload-census.log")
    verified = load_verified_repositories(results)
    trees = unique_verified_trees(verified)
    done_path = out_dir / "trees.jsonl"
    done = _load_jsonl_keys(done_path)
    logger.info("workload census trees=%s already_saved=%s", len(trees), len(done))
    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    config = load_verification_config(Path(args.config))
    try:
        client = GitHubClient(token, data_dir / "raw" / "github-cache")
        for index, tree in enumerate(trees, start=1):
            if tree["tree_sha"] in done:
                continue
            if index == 1 or index % 200 == 0:
                logger.info("workload census %s/%s %s", index, len(trees), tree["donor"]["repository_full_name"])
            row = describe_tree_workload(client, tree, config)
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc
    saved = _load_jsonl(done_path)
    pilot = _load_jsonl(results / "statistics_pilot" / "trees.jsonl")
    recheck = _load_jsonl(results / "statistics_pilot" / "cap_recheck" / "trees.jsonl")
    measured = {row["tree_sha"]: row["seconds"] for row in pilot}
    for row in recheck:
        if row.get("enumeration_complete"):
            measured[row["tree_sha"]] = row["seconds"]
    summary = summarize_workload(saved, measured)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    dist = summary["candidate_files"]
    runtime = summary["runtime"]
    print(
        "workload_census "
        f"trees={summary['trees']} complete={summary['listing_complete']} "
        f"median_files={dist.get('median')} p99_files={dist.get('p99')} max_files={dist.get('max')} "
        f"over_5000={summary['thresholds']['over_5000']} "
        f"hours_p25={runtime['estimated_hours_p25_rate']} "
        f"hours_median={runtime['estimated_hours_median_rate']} "
        f"hours_p75={runtime['estimated_hours_p75_rate']}"
    )


def _batch_a(args) -> None:
    """Complete enumeration for trees at or under 5,000 candidate RDF files."""
    data_dir = Path(args.data_dir)
    results = data_dir / "results"
    out_dir = results / "statistics_batch_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(data_dir / "logs" / "statistics-batch-a.log")
    verified = load_verified_repositories(results)
    trees = unique_verified_trees(verified)
    workload = _load_jsonl(results / "statistics_pilot" / "workload_census" / "trees.jsonl")
    if len(workload) < len(trees):
        raise SystemExit("The workload census is incomplete. Batch A is not started.")
    parts = partition_statistics_batches(trees, workload)
    selected = parts["batch_a"]
    done_path = out_dir / "trees.jsonl"
    done = _load_jsonl_keys(done_path)
    if not done:
        prior = _load_jsonl(results / "statistics_pilot" / "trees.jsonl")
        prior.extend(_load_jsonl(results / "statistics_pilot" / "cap_recheck" / "trees.jsonl"))
        reusable = reusable_complete_results({tree["tree_sha"] for tree in selected}, prior)
        if reusable:
            with done_path.open("a", encoding="utf-8") as handle:
                for row in reusable:
                    handle.write(json.dumps(row) + "\n")
            done = {row["tree_sha"] for row in reusable}
            logger.info("batch A reused complete enumerations=%s", len(reusable))
    logger.info(
        "batch A trees=%s heavy_excluded=%s unresolved_excluded=%s already_saved=%s",
        len(selected),
        len(parts["heavy"]),
        len(parts["unresolved"]),
        len(done),
    )
    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    config = load_verification_config(Path(args.config))

    def write_summary() -> None:
        saved = _load_jsonl(done_path)
        summary = aggregate_batch_a(
            saved,
            batch_a_trees=len(selected),
            heavy_trees=len(parts["heavy"]),
            unresolved_trees=len(parts["unresolved"]),
            unresolved_also_heavy=len(parts["unresolved_also_heavy"]),
            population_trees=len(trees),
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_summary()
    try:
        client = GitHubClient(token, data_dir / "raw" / "github-cache")
        source = GitHubContent(client, config)
        clock = lambda: datetime.now(timezone.utc)
        for index, tree in enumerate(selected, start=1):
            sha = tree["tree_sha"]
            if sha in done:
                continue
            logger.info(
                "batch A %s/%s %s donor=%s",
                index,
                len(selected),
                sha[:12],
                tree["donor"]["repository_full_name"],
            )
            checkpoint = out_dir / "checkpoints" / f"{sha}.json"
            result = census_tree(
                tree,
                source=source,
                config=config,
                vocab=None,
                clock=clock,
                checkpoint_path=checkpoint,
            )
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            checkpoint.unlink(missing_ok=True)
            done.add(sha)
            logger.info(
                "batch A status %s nodes=%s properties=%s distinct=%s complete=%s partial=%s seconds=%s",
                result["donor"],
                result["node_shapes"],
                result["property_shapes"],
                result["distinct_shapes"],
                result["enumeration_complete"],
                result["counts_partial"],
                result["seconds"],
            )
            if len(done) % 100 == 0:
                write_summary()
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc
    write_summary()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    totals = summary["shape_totals"]
    print(
        "batch_a "
        f"trees={summary['trees_completed']}/{summary['batch_a_trees']} "
        f"complete={summary['trees_with_complete_enumeration']} "
        f"partial={summary['trees_with_partial_counts']} "
        f"node_shapes={totals['node_shapes']} "
        f"property_shapes={totals['property_shapes']} "
        f"distinct_shapes={totals['distinct_shapes']} "
        f"hours={summary['runtime_hours']} "
        f"partial_corpus=true"
    )


def _heavy(args) -> None:
    """Same enumeration and parser mapping as Batch A, for the 5,001+ file trees only."""
    data_dir = Path(args.data_dir)
    results = data_dir / "results"
    out_dir = results / "statistics_heavy"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(data_dir / "logs" / "statistics-heavy.log")
    verified = load_verified_repositories(results)
    trees = unique_verified_trees(verified)
    workload = _load_jsonl(results / "statistics_pilot" / "workload_census" / "trees.jsonl")
    if len(workload) < len(trees):
        raise SystemExit("The workload census is incomplete. The heavy extraction is not started.")
    parts = partition_statistics_batches(trees, workload)
    selected = parts["heavy"]
    done_path = out_dir / "trees.jsonl"
    done = _load_jsonl_keys(done_path)
    if not done:
        prior = _load_jsonl(results / "statistics_pilot" / "cap_recheck" / "trees.jsonl")
        reusable = reusable_complete_results({tree["tree_sha"] for tree in selected}, prior)
        if reusable:
            with done_path.open("a", encoding="utf-8") as handle:
                for row in reusable:
                    handle.write(json.dumps(row) + "\n")
            done = {row["tree_sha"] for row in reusable}
            logger.info("heavy reused complete enumerations=%s", len(reusable))
    logger.info(
        "heavy trees=%s unresolved_also_heavy=%s already_saved=%s",
        len(selected),
        len(parts["unresolved_also_heavy"]),
        len(done),
    )
    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    config = load_verification_config(Path(args.config))

    def write_summary() -> None:
        saved = _load_jsonl(done_path)
        summary = aggregate_batch_a(
            saved,
            batch_a_trees=len(selected),
            heavy_trees=len(selected),
            unresolved_trees=len(parts["unresolved"]),
            unresolved_also_heavy=len(parts["unresolved_also_heavy"]),
            population_trees=len(trees),
        )
        summary["corpus_scope"] = "heavy_trees"
        summary["partial_corpus"] = True
        summary["label"] = (
            "Heavy trees only: verified trees with at least 5,001 candidate RDF files. "
            "Parser mapping and enumeration rules are the same as Batch A. "
            "Percentages use this heavy set as the denominator."
        )
        summary["heavy_trees"] = len(selected)
        summary.pop("batch_a_trees", None)
        summary.pop("excluded_heavy_trees", None)
        for row in summary.get("constraints", []) + summary.get("targets", []):
            if "percent_of_batch_a_trees" in row:
                row["percent_of_heavy_trees"] = row.pop("percent_of_batch_a_trees")
            if "percent_of_batch_a_repositories" in row:
                row["percent_of_heavy_repositories"] = row.pop("percent_of_batch_a_repositories")
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    write_summary()
    try:
        client = GitHubClient(token, data_dir / "raw" / "github-cache")
        source = GitHubContent(client, config)
        clock = lambda: datetime.now(timezone.utc)
        for index, tree in enumerate(selected, start=1):
            sha = tree["tree_sha"]
            if sha in done:
                continue
            logger.info(
                "heavy %s/%s %s donor=%s",
                index,
                len(selected),
                sha[:12],
                tree["donor"]["repository_full_name"],
            )
            checkpoint = out_dir / "checkpoints" / f"{sha}.json"
            result = census_tree(
                tree,
                source=source,
                config=config,
                vocab=None,
                clock=clock,
                checkpoint_path=checkpoint,
            )
            with done_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
            checkpoint.unlink(missing_ok=True)
            done.add(sha)
            logger.info(
                "heavy status %s nodes=%s properties=%s distinct=%s complete=%s partial=%s seconds=%s",
                result["donor"],
                result["node_shapes"],
                result["property_shapes"],
                result["distinct_shapes"],
                result["enumeration_complete"],
                result["counts_partial"],
                result["seconds"],
            )
            if len(done) % 10 == 0:
                write_summary()
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc
    write_summary()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    totals = summary["shape_totals"]
    print(
        "heavy "
        f"trees={summary['trees_completed']}/{summary['heavy_trees']} "
        f"complete={summary['trees_with_complete_enumeration']} "
        f"partial={summary['trees_with_partial_counts']} "
        f"node_shapes={totals['node_shapes']} "
        f"property_shapes={totals['property_shapes']} "
        f"distinct_shapes={totals['distinct_shapes']} "
        f"hours={summary['runtime_hours']} "
        f"partial_corpus=true"
    )


def _heavy_benchmark(args) -> None:
    """Time three complete access strategies on 12 heavy trees. Does not run the other 95."""
    data_dir = Path(args.data_dir)
    results = data_dir / "results"
    out_dir = results / "statistics_heavy_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(data_dir / "logs" / "statistics-heavy-benchmark.log")
    verified = load_verified_repositories(results)
    trees = {item["tree_sha"]: item for item in unique_verified_trees(verified)}
    workload = _load_jsonl(results / "statistics_pilot" / "workload_census" / "trees.jsonl")
    selected = select_heavy_benchmark(workload)
    if len(selected) != 12:
        raise SystemExit(f"Heavy benchmark selected {len(selected)} trees, expected 12.")
    missing = [item["donor"] for item in selected if item["tree_sha"] not in trees]
    if missing:
        raise SystemExit(f"Benchmark trees are not in the verified corpus: {', '.join(missing)}")
    (out_dir / "selection.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    logger.info("heavy benchmark trees=%s", ", ".join(f"{item['role']}={item['donor']}" for item in selected))
    _load_dotenv(Path(".env"))
    token = os.environ.get("GITHUB_TOKEN", "")
    config = load_verification_config(Path(args.config))
    strategies = ("tarball", "strong_first", "current")
    try:
        client = GitHubClient(token, data_dir / "raw" / "github-cache")
        source = GitHubContent(client, config)
        clock = lambda: datetime.now(timezone.utc)
        # Tarball, then the current path, then per-file strong-first.
        # Strong-first is last because it issues one request per candidate file.
        for strategy in strategies:
            for spec in selected:
                strategy_dir = out_dir / strategy
                strategy_dir.mkdir(parents=True, exist_ok=True)
                done_path = strategy_dir / "trees.jsonl"
                done = _load_jsonl_keys(done_path)
                if spec["tree_sha"] in done:
                    continue
                logger.info("heavy benchmark %s strategy=%s donor=%s", spec["role"], strategy, spec["donor"])
                tree = trees[spec["tree_sha"]]
                checkpoint = out_dir / "checkpoints" / strategy / f"{spec['tree_sha']}.json"
                result = census_tree(
                    tree,
                    source=source,
                    config=config,
                    vocab=None,
                    clock=clock,
                    checkpoint_path=checkpoint,
                    strategy=strategy,
                )
                result["benchmark_role"] = spec["role"]
                result["workload_type"] = spec["workload_type"]
                with done_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result) + "\n")
                checkpoint.unlink(missing_ok=True)
                logger.info(
                    "heavy benchmark status %s strategy=%s complete=%s fallback=%s seconds=%s download=%s parse=%s source=%s",
                    spec["donor"],
                    strategy,
                    result["enumeration_complete"],
                    result["strategy_fallback"],
                    result["seconds"],
                    result["seconds_download"],
                    result["seconds_parse"],
                    result["content_source"],
                )
    except GitHubAuthError as exc:
        raise SystemExit(str(exc)) from exc
    measured = []
    for strategy in strategies:
        for row in _load_jsonl(out_dir / strategy / "trees.jsonl"):
            measured.append(row)
    heavy_rows = [row for row in workload if int(row.get("candidate_files") or 0) >= 5001]
    estimate = estimate_heavy_runtime(heavy_rows, measured)
    brief = [
        {
            "role": row.get("benchmark_role"),
            "donor": row.get("donor"),
            "workload_type": row.get("workload_type"),
            "requested_strategy": row.get("requested_strategy"),
            "seconds": row.get("seconds"),
            "seconds_download": row.get("seconds_download"),
            "seconds_parse": row.get("seconds_parse"),
            "enumeration_complete": row.get("enumeration_complete"),
            "counts_partial": row.get("counts_partial"),
            "partial_reasons": row.get("partial_reasons"),
            "candidates_discovered": row.get("candidates_discovered"),
            "files_parsed": row.get("files_parsed"),
            "parse_failures": row.get("parse_failures"),
            "strategy_fallback": row.get("strategy_fallback"),
            "content_source": row.get("content_source"),
            "node_shapes": row.get("node_shapes"),
            "property_shapes": row.get("property_shapes"),
            "distinct_shapes": row.get("distinct_shapes"),
        }
        for row in measured
    ]
    summary = {
        "partial_corpus": True,
        "label": "Access-strategy benchmark for 12 heavy trees. The other heavy trees were not extracted.",
        "trees": selected,
        "runs": brief,
        "estimate": estimate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    hours = estimate["estimated_hours_if_one_strategy"]
    print(
        "heavy_benchmark "
        f"runs={len(measured)}/36 "
        f"hours_tarball={hours.get('tarball')} "
        f"hours_strong_first={hours.get('strong_first')} "
        f"hours_current={hours.get('current')} "
        f"hours_by_type={estimate['estimated_hours_fastest_by_workload_type']}"
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_jsonl_keys(path: Path) -> set[str]:
    return {row["tree_sha"] for row in _load_jsonl(path)}


def _repository_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["repository_id"] for row in csv.DictReader(handle) if row.get("repository_id")}


def _state_repository_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set((payload.get("repositories") or {}).keys())


def _benchmark_report(
    data_dir: Path,
    groups: dict[str, str],
    *,
    results_dir: Path,
    pilot_ids: set[str],
    full_ids: set[str],
) -> dict:
    results_path = results_dir / "repository_verification.csv"
    with results_path.open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))
    pilot = _status_map(data_dir / "pilot" / "milestone2" / "repository_verification.csv")
    full = _full_inspection_statuses(data_dir / "candidates" / "verification_state.json")
    previous_benchmark = _status_map(data_dir / "results" / "phase1_benchmark" / "repository_verification.csv")
    if results_dir.resolve() == (data_dir / "results" / "phase1_benchmark").resolve():
        previous_benchmark = {}
    disagreements = []
    discovery_hit = 0
    tree_inspection = 0
    cold_seconds: list[float] = []
    for row in results:
        repository_id = str(row["repository_id"])
        group = groups.get(repository_id, "")
        if row.get("verification_pass") == "discovery_hit":
            discovery_hit += 1
        elif row.get("verification_pass") == "tree_inspection":
            tree_inspection += 1
        if group == "cold":
            cold_seconds.append(_inspection_seconds(row))
        previous = None
        source = ""
        if repository_id in full:
            previous = full[repository_id]
            source = "full_inspection"
        elif repository_id in pilot:
            previous = pilot.get(repository_id)
            source = "pilot"
        benchmark_previous = previous_benchmark.get(repository_id)
        if benchmark_previous and benchmark_previous != row["verification_status"]:
            disagreements.append(
                {
                    "repository_id": repository_id,
                    "repository_full_name": row["repository_full_name"],
                    "benchmark_group": group,
                    "previous_source": "previous_benchmark",
                    "previous_status": benchmark_previous,
                    "benchmark_status": row["verification_status"],
                    "benchmark_reason": row["verification_reason"],
                    "verification_pass": row["verification_pass"],
                    "evidence_path": row["evidence_path"],
                }
            )
        if previous and previous != row["verification_status"]:
            disagreements.append(
                {
                    "repository_id": repository_id,
                    "repository_full_name": row["repository_full_name"],
                    "benchmark_group": group,
                    "previous_source": source,
                    "previous_status": previous,
                    "benchmark_status": row["verification_status"],
                    "benchmark_reason": row["verification_reason"],
                    "verification_pass": row["verification_pass"],
                    "evidence_path": row["evidence_path"],
                }
            )
    total = len(results) or 1
    return {
        "repository_count": len(results),
        "verified_from_discovery_hit": discovery_hit,
        "tree_inspection": tree_inspection,
        "discovery_hit_percent": round(100 * discovery_hit / total, 2),
        "tree_inspection_percent": round(100 * tree_inspection / total, 2),
        "cold_seconds": _seconds_summary(cold_seconds),
        "all_seconds": _seconds_summary([_inspection_seconds(row) for row in results]),
        "tarball_repositories": sum(1 for row in results if row.get("content_source") == "codeload_tarball"),
        "tarball_percent": round(100 * sum(1 for row in results if row.get("content_source") == "codeload_tarball") / total, 2),
        "network_raw_requests": sum(int(row.get("network_raw_requests") or 0) for row in results),
        "network_tarballs": sum(int(row.get("network_tarballs") or 0) for row in results),
        "network_bytes": sum(int(row.get("network_bytes") or 0) for row in results),
        "projected_hours_for_2908_from_cold_mean": (
            None if not cold_seconds else round((statistics.fmean(cold_seconds) * 2908) / 3600, 2)
        ),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "note": (
            "discovery_hit verifies a repository from a cached code-search path and does not enumerate "
            "every SHACL file. inspection_complete stays false for that pass. "
            "A status change from incomplete_inspection to verified_shacl means the hit established "
            "canonical evidence where the 400-file walk did not finish."
        ),
    }


def _status_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["repository_id"]: row["verification_status"] for row in csv.DictReader(handle)}


def _full_inspection_statuses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    statuses = {}
    for repository_id, blob in (payload.get("repositories") or {}).items():
        repository = blob.get("repository") or {}
        if repository.get("verification_status"):
            statuses[str(repository_id)] = repository["verification_status"]
    return statuses


def _inspection_seconds(row: dict) -> float:
    return sum(float(row.get(name) or 0) for name in ("seconds_commit", "seconds_tree", "seconds_download", "seconds_parse"))


def _seconds_summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percent / 100 * len(ordered)) - 1))
    return round(ordered[index], 3)


def _run_tree_census(args, rows, client, data_dir: Path, census_dir: Path, logger) -> None:
    census_dir.mkdir(parents=True, exist_ok=True)
    state = resolve_tree_census(
        rows,
        source=GitHubContent(client, {}),
        state_path=census_dir / "pins.json",
        fresh=args.fresh,
        clock=lambda: datetime.now(timezone.utc),
        logger=logger,
    )
    report = tree_identity_report(rows, state["pins"])
    report["runtime_estimate"] = _tree_runtime_estimate(data_dir, report)
    (census_dir / "tree_identity_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "tree_census "
        f"repositories={report['phase2_repositories']} "
        f"resolved={report['tree_sha_resolved']} "
        f"unique_trees={report['unique_tree_shas']} "
        f"avoided={report['content_inspections_avoided']} "
        f"avoided_percent={report['content_inspections_avoided_percent_of_phase2']}"
    )
    estimate = report["runtime_estimate"]
    if estimate.get("available"):
        print(
            "runtime_estimate "
            f"unique_tree_mean_hours={estimate['unique_tree_mean_hours']} "
            f"unique_tree_median_hours={estimate['unique_tree_median_hours']}"
        )


def _tree_runtime_estimate(data_dir: Path, report: dict) -> dict:
    csv_path = data_dir / "results" / "phase2_pilot" / "repository_verification.csv"
    log_path = data_dir / "logs" / "phase2-pilot.log"
    if not csv_path.exists() or not log_path.exists():
        return {"available": False}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        pilot_rows = list(csv.DictReader(handle))
    durations = _pilot_wall_seconds(log_path)
    if len(durations) != len(pilot_rows):
        return {"available": False, "reason": f"pilot timings {len(durations)} rows {len(pilot_rows)}"}
    buckets: dict[str, list[float]] = {}
    for row, duration in zip(pilot_rows, durations):
        buckets.setdefault(row["pilot_stratum"], []).append(duration)
    means = {name: statistics.mean(values) for name, values in buckets.items()}
    medians = {name: statistics.median(values) for name, values in buckets.items()}
    unique = report["unique_trees_by_representative_stratum"]
    mean_seconds = sum(count * means[name] for name, count in unique.items() if name in means)
    median_seconds = sum(count * medians[name] for name, count in unique.items() if name in medians)
    return {
        "available": True,
        "pilot_mean_seconds": {name: round(value, 3) for name, value in means.items()},
        "pilot_median_seconds": {name: round(value, 3) for name, value in medians.items()},
        "unique_tree_mean_hours": round(mean_seconds / 3600, 2),
        "unique_tree_median_hours": round(median_seconds / 3600, 2),
        "basis": (
            "Each unique tree is charged once, using the Phase 2 pilot mean or median of the "
            "stratum of its first repository. Reused repositories add no content-inspection time."
        ),
    }


def _pilot_wall_seconds(log_path: Path) -> list[float]:
    durations: list[float] = []
    started: float | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " INFO verify " not in line and " INFO status " not in line:
            continue
        stamp = line.split(" INFO ", 1)[0]
        hms, millis = stamp.split(",")
        hour, minute, second = hms.split(" ")[1].split(":")
        instant = int(hour) * 3600 + int(minute) * 60 + int(second) + int(millis) / 1000
        if " INFO verify " in line:
            started = instant
        elif started is not None:
            durations.append(instant - started)
            started = None
    return durations


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _manifest(config_path: Path, window_start: str, window_end: str) -> dict:
    commit, dirty = _git_info(Path.cwd())
    versions = {}
    for package in ("requests", "pydantic", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "milestone": 1,
        "collected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software_version": __version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "github_api_version": API_VERSION,
        "discovery_config_path": str(config_path),
        "discovery_config_sha256": config_digest(config_path),
        "partition_window_start": window_start,
        "partition_window_end": window_end,
        "python_version": sys.version,
        "dependency_versions": versions,
        "scope": "Discovery only. No SHACL verification or statistics.",
    }


def _git_info(cwd: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return commit, bool(dirty)
