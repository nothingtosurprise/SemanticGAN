"""Shape and constraint census over verified Git trees.

The primary observation is one unique verified tree SHA. Repository rows are
an expansion of that content and are never used as the content denominator.
Verification decisions are read and not rewritten.
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from shacl_study.github_client import raw_content_url, tarball_url
from shacl_study.shacl_vocab import SH, ShaclVocabulary, default_vocabulary
from shacl_study.verification import LFS_MARKER, _archive_path, inspect_repository, raw_extension_rule

CORE_TARGETS = (
    SH + "targetClass",
    SH + "targetNode",
    SH + "targetSubjectsOf",
    SH + "targetObjectsOf",
)
PATH_MARKERS = ("report", "oops", "export", "generated")


def load_verified_repositories(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    phase1 = results_dir / "repository_verification.csv"
    phase2 = results_dir / "phase2" / "repository_verification.csv"
    rows.extend(_read_verified(phase1, 1))
    rows.extend(_read_verified(phase2, 2))
    return rows


def unique_verified_trees(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        tree = (row.get("tree_sha") or "").strip()
        if tree:
            grouped[tree].append(row)
    trees = []
    for tree_sha, members in grouped.items():
        forks = [_truthy(item.get("is_fork")) for item in members]
        phases = {int(item["phase"]) for item in members}
        reasons = {item.get("verification_reason") or "" for item in members}
        candidates = max(int(item.get("candidate_files_found") or 0) for item in members)
        recorded_nodes = max(int(item.get("node_shape_count") or 0) for item in members)
        trees.append(
            {
                "tree_sha": tree_sha,
                "repositories": members,
                "repository_count": len(members),
                "phases": phases,
                "phase_origin": 1 if 1 in phases else 2,
                "fork_origin": "fork" if forks and all(forks) else "non-fork",
                "sparql_only_verification": "sparql_shape_evidence" in reasons,
                "core_and_sparql_verification": "core_and_sparql_shape_evidence" in reasons,
                "recorded_candidates": candidates,
                "recorded_node_shapes": recorded_nodes,
                "donor": _donor(members),
            }
        )
    return trees


def select_statistics_pilot(trees: list[dict], count: int, *, seed: int) -> list[dict]:
    """Stratified sample. SPARQL-only trees and the largest recorded shape files are kept."""
    if count < 1:
        raise ValueError("Pilot count must be at least 1.")
    by_sha = {item["tree_sha"]: item for item in trees}
    chosen: list[dict] = []
    seen: set[str] = set()

    def add(item: dict) -> None:
        if item["tree_sha"] in seen:
            return
        seen.add(item["tree_sha"])
        chosen.append(item)

    for item in trees:
        if item["sparql_only_verification"]:
            add(item)
    many = sorted(trees, key=lambda item: (-item["recorded_node_shapes"], item["tree_sha"]))
    for item in many:
        if len([row for row in chosen if row["recorded_node_shapes"] >= 50]) >= 8 and len(chosen) >= 11:
            break
        if item["recorded_node_shapes"] <= 0:
            break
        add(item)
        if sum(1 for row in chosen if not row["sparql_only_verification"]) >= 8:
            break

    remaining_target = max(0, count - len(chosen))
    pool = [item for item in trees if item["tree_sha"] not in seen]
    sizes = sorted(item["recorded_candidates"] for item in trees)
    bounds = _tertile_bounds(sizes)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for item in pool:
        evidence = "core_sparql" if item["core_and_sparql_verification"] else "core"
        cells[_cell(item, bounds, evidence)].append(item)
    rng = random.Random(seed)
    keys = sorted(cells)
    per_cell = max(1, remaining_target // max(1, len(keys)))
    for key in keys:
        bucket = cells[key][:]
        rng.shuffle(bucket)
        for item in bucket[:per_cell]:
            if len(chosen) >= count:
                break
            add(item)
    leftover = [item for item in pool if item["tree_sha"] not in seen]
    rng.shuffle(leftover)
    for item in leftover:
        if len(chosen) >= count:
            break
        add(item)
    if len(chosen) > count:
        forced = [item for item in chosen if item["sparql_only_verification"]]
        rest = [item for item in chosen if not item["sparql_only_verification"]]
        chosen = forced + rest[: max(0, count - len(forced))]
    return [by_sha[item["tree_sha"]] for item in chosen]


class CensusTally:
    """Running shape union for one tree. File status is kept; payloads are not."""

    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.node_uris: set[str] = set()
        self.property_uris: set[str] = set()
        self.distinct_uris: set[str] = set()
        self.node_blanks = 0
        self.property_blanks = 0
        self.distinct_blanks = 0
        self.constraints: Counter[str] = Counter()
        self.targets: Counter[str] = Counter()
        self.sparql_links = 0
        self.sparql_nodes = 0
        self.sparql_targets = 0
        self.structural = 0
        self.path_occurrences = 0
        self.validation_reports = 0
        self.provenance_paths: list[str] = []
        self.discovered = 0
        self.accounted: set[str] = set()
        self.cap_hit = False
        self.archive_too_large = False
        self.access_kind = ""
        self.content_source = ""
        self.prior_seconds = 0.0

    def add(self, item, vocab: ShaclVocabulary) -> None:
        path = item.path
        if path in self.files:
            return
        evidence = getattr(item, "evidence", None)
        status = item.parse_status
        record = {
            "path": path,
            "parse_status": status,
            "node_shapes": 0,
            "property_shapes": 0,
            "parse_error": item.parse_error or "",
        }
        if evidence is not None and status == "parsed":
            self.node_blanks += _absorb(evidence.node_shape_subjects, self.node_uris)
            self.property_blanks += _absorb(evidence.property_shape_subjects, self.property_uris)
            self.distinct_blanks += _absorb(
                evidence.node_shape_subjects | evidence.property_shape_subjects,
                self.distinct_uris,
            )
            record["node_shapes"] = len(evidence.node_shape_subjects)
            record["property_shapes"] = len(evidence.property_shape_subjects)
            for iri, count in evidence.predicate_occurrences.items():
                if iri in vocab.core_constraint_parameters:
                    self.constraints[iri] += count
                elif iri in CORE_TARGETS:
                    self.targets[iri] += count
            self.sparql_links += evidence.sparql_links
            self.sparql_nodes += evidence.sparql_constraint_nodes
            self.sparql_targets += evidence.sparql_target_occurrences
            self.structural += evidence.sparql_triples
            self.path_occurrences += evidence.path_occurrences
            self.validation_reports += evidence.validation_reports
            if evidence.validation_reports or any(marker in path.lower() for marker in PATH_MARKERS):
                if len(self.provenance_paths) < 8:
                    self.provenance_paths.append(path)
        self.files[path] = record

    def result(self, tree: dict, repository: dict, failures: list[dict], elapsed: float) -> dict:
        parsed = sum(1 for item in self.files.values() if item["parse_status"] == "parsed")
        parse_failures = sum(1 for item in self.files.values() if item["parse_status"] == "parse_error")
        skipped: Counter[str] = Counter()
        for item in self.files.values():
            status = item["parse_status"]
            if status != "parsed":
                skipped[status or "unclassified"] += 1
        listing_ok = (
            not self.cap_hit
            and not self.archive_too_large
            and self.access_kind not in {"transient", "inaccessible"}
            and bool(repository.get("commit_sha_examined"))
        )
        enumeration_complete = listing_ok and self.discovered > 0 and len(self.accounted) == self.discovered and len(self.files) == self.discovered
        partial_reasons = []
        if not enumeration_complete:
            if self.cap_hit:
                partial_reasons.append("candidate_file_cap")
            elif self.archive_too_large:
                partial_reasons.append("archive_over_size_cap")
            elif self.access_kind in {"transient", "inaccessible"}:
                partial_reasons.append(self.access_kind)
            else:
                partial_reasons.append("incomplete_listing")
        for status in ("skipped_too_large", "lfs_pointer", "parse_error", "fetch_error", "parser_unavailable"):
            if skipped.get(status):
                partial_reasons.append(status)
        hard_partial = [reason for reason in partial_reasons if reason != "parse_error"]
        counts_partial = bool(hard_partial) or "parse_error" in partial_reasons
        extracted = bool(repository.get("commit_sha_examined")) and repository.get("verification_status") not in {
            "transient_failure",
            "inaccessible",
        }
        return {
            "tree_sha": tree["tree_sha"],
            "donor": tree["donor"]["repository_full_name"],
            "commit_sha": tree["donor"]["commit_sha_examined"],
            "phase_origin": tree["phase_origin"],
            "fork_origin": tree["fork_origin"],
            "repository_count": tree["repository_count"],
            "repository_names": [item["repository_full_name"] for item in tree["repositories"]],
            "sparql_only_verification": tree["sparql_only_verification"],
            "extracted": extracted,
            "extraction_status": repository.get("verification_status", ""),
            "capped": self.cap_hit,
            "enumeration_complete": enumeration_complete,
            "counts_partial": counts_partial,
            "partial_reasons": partial_reasons,
            "content_source": self.content_source,
            "node_shapes": len(self.node_uris) + self.node_blanks,
            "property_shapes": len(self.property_uris) + self.property_blanks,
            "distinct_shapes": len(self.distinct_uris) + self.distinct_blanks,
            "constraints": dict(self.constraints),
            "targets": dict(self.targets),
            "path_occurrences": self.path_occurrences,
            "sparql_links": self.sparql_links,
            "sparql_constraint_nodes": self.sparql_nodes,
            "sparql_targets": self.sparql_targets,
            "structural_sparql": self.structural > 0,
            "sparql_only_content": self.structural > 0 and (len(self.distinct_uris) + self.distinct_blanks) == 0,
            "report_or_export": bool(self.provenance_paths),
            "provenance_paths": self.provenance_paths,
            "candidates_discovered": self.discovered,
            "files_inspected": len(self.files),
            "files_parsed": parsed,
            "parse_failures": parse_failures,
            "skipped": dict(skipped),
            "candidate_files": self.discovered,
            "seconds": round(elapsed, 3),
            "failures": [item.get("failure_kind", "") for item in failures],
            "files": [self.files[path] for path in sorted(self.files)],
        }

    def dump(self) -> str:
        payload = {
            "files": self.files,
            "node_uris": sorted(self.node_uris),
            "property_uris": sorted(self.property_uris),
            "distinct_uris": sorted(self.distinct_uris),
            "node_blanks": self.node_blanks,
            "property_blanks": self.property_blanks,
            "distinct_blanks": self.distinct_blanks,
            "constraints": dict(self.constraints),
            "targets": dict(self.targets),
            "sparql_links": self.sparql_links,
            "sparql_nodes": self.sparql_nodes,
            "sparql_targets": self.sparql_targets,
            "structural": self.structural,
            "path_occurrences": self.path_occurrences,
            "validation_reports": self.validation_reports,
            "provenance_paths": self.provenance_paths,
            "discovered": self.discovered,
            "accounted": sorted(self.accounted),
            "cap_hit": self.cap_hit,
            "archive_too_large": self.archive_too_large,
            "access_kind": self.access_kind,
            "content_source": self.content_source,
            "prior_seconds": self.prior_seconds,
        }
        return json.dumps(payload)

    @classmethod
    def load(cls, path: Path) -> "CensusTally":
        data = json.loads(path.read_text(encoding="utf-8"))
        tally = cls()
        tally.files = data["files"]
        tally.node_uris = set(data["node_uris"])
        tally.property_uris = set(data["property_uris"])
        tally.distinct_uris = set(data["distinct_uris"])
        tally.node_blanks = data["node_blanks"]
        tally.property_blanks = data["property_blanks"]
        tally.distinct_blanks = data["distinct_blanks"]
        tally.constraints = Counter(data["constraints"])
        tally.targets = Counter(data["targets"])
        tally.sparql_links = data["sparql_links"]
        tally.sparql_nodes = data["sparql_nodes"]
        tally.sparql_targets = data["sparql_targets"]
        tally.structural = data["structural"]
        tally.path_occurrences = data["path_occurrences"]
        tally.validation_reports = data["validation_reports"]
        tally.provenance_paths = data["provenance_paths"]
        tally.discovered = data["discovered"]
        tally.accounted = set(data["accounted"])
        tally.cap_hit = data["cap_hit"]
        tally.archive_too_large = data["archive_too_large"]
        tally.access_kind = data["access_kind"]
        tally.content_source = data["content_source"]
        tally.prior_seconds = float(data.get("prior_seconds") or 0)
        return tally


def census_tree(
    tree: dict,
    *,
    source,
    config: dict,
    vocab: ShaclVocabulary | None,
    clock,
    checkpoint_path: Path | None = None,
    batch_size: int = 200,
    strategy: str = "current",
) -> dict:
    """Enumerate every candidate RDF file in the pinned tree.

    The verification candidate cap is not applied. A checkpoint stores the
    running union of shape IRIs so a resumed run does not download finished files.
    """
    vocab = vocabulary_or_default(vocab)
    tally = CensusTally.load(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else CensusTally()
    donor = tree["donor"]
    row = dict(donor)
    if _truthy(donor.get("prefilter_gate")):
        row["discovered_by"] = ""
    else:
        row["discovered_by"] = donor.get("discovered_by") or "code:recorded"
    pin = {
        "commit_sha": donor["commit_sha_examined"],
        "tree_sha": donor["tree_sha"],
    }

    def on_batch(inspection) -> None:
        for item in inspection.files:
            tally.add(item, vocab)
        tally.discovered = inspection.heuristic_paths
        tally.accounted.update(inspection.census_accounted)
        tally.cap_hit = inspection.cap_hit
        tally.archive_too_large = inspection.archive_too_large
        tally.access_kind = inspection.access_kind
        tally.content_source = inspection.content_source
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(tally.dump(), encoding="utf-8")

    started = time.perf_counter()
    repository, _files, failures = inspect_repository(
        row,
        source=source,
        vocab=vocab,
        config=config,
        clock=clock,
        resolved_pin=pin,
        census=True,
        census_strategy=strategy,
        census_skip_paths=set(tally.files),
        on_census_batch=on_batch,
        census_batch_size=batch_size,
    )
    elapsed = time.perf_counter() - started + tally.prior_seconds
    repository.pop("_census_files", None)
    result = tally.result(tree, repository, failures, elapsed)
    result["requested_strategy"] = strategy
    result["strategy_fallback"] = repository.get("_strategy_fallback", "")
    result["seconds_download"] = round(float(repository.get("_seconds_download") or 0), 3)
    result["seconds_parse"] = round(float(repository.get("_seconds_parse") or 0), 3)
    return result


def aggregate_pilot(results: list[dict], *, population_trees: int, population_repositories: int) -> dict:
    usable = [item for item in results if item["extracted"]]
    node_counts = [item["node_shapes"] for item in usable]
    property_counts = [item["property_shapes"] for item in usable]
    distinct_counts = [item["distinct_shapes"] for item in usable]
    repo_nodes = []
    repo_properties = []
    repo_distinct = []
    for item in usable:
        repo_nodes.extend([item["node_shapes"]] * item["repository_count"])
        repo_properties.extend([item["property_shapes"]] * item["repository_count"])
        repo_distinct.extend([item["distinct_shapes"]] * item["repository_count"])
    vocab = default_vocabulary()
    constraints = _prevalence(
        usable,
        "constraints",
        len(usable),
        expected=sorted(vocab.core_constraint_parameters),
    )
    targets = _prevalence(
        usable,
        "targets",
        len(usable),
        expected=list(CORE_TARGETS),
    )
    times = [item["seconds"] for item in results]
    return {
        "pilot_trees_attempted": len(results),
        "pilot_trees_extracted": len(usable),
        "pilot_trees_failed": len(results) - len(usable),
        "pilot_repository_expansion": sum(item["repository_count"] for item in usable),
        "population_unique_verified_trees": population_trees,
        "population_verified_repositories": population_repositories,
        "denominators": {
            "content": "unique verified tree SHA in this pilot",
            "repository": "verified repositories whose tree is in this pilot",
            "not_the_full_corpus": True,
        },
        "shape_totals": {
            "node_shapes": sum(node_counts),
            "property_shapes": sum(property_counts),
            "distinct_shapes": sum(distinct_counts),
        },
        "shapes_per_tree": {
            "node_shapes": _distribution(node_counts),
            "property_shapes": _distribution(property_counts),
            "distinct_shapes": _distribution(distinct_counts),
        },
        "shapes_per_repository": {
            "node_shapes": _distribution(repo_nodes),
            "property_shapes": _distribution(repo_properties),
            "distinct_shapes": _distribution(repo_distinct),
        },
        "dominance": _dominance(distinct_counts),
        "targets": targets,
        "constraints": constraints,
        "sparql": {
            "trees_with_structural_constraints": sum(1 for item in usable if item["structural_sparql"]),
            "repositories_with_structural_constraints": sum(
                item["repository_count"] for item in usable if item["structural_sparql"]
            ),
            "sparql_links": sum(item["sparql_links"] for item in usable),
            "sparql_constraint_nodes": sum(item["sparql_constraint_nodes"] for item in usable),
            "sparql_only_verification_trees": sum(1 for item in usable if item["sparql_only_verification"]),
            "sparql_only_content_trees": sum(1 for item in usable if item["sparql_only_content"]),
            "sparql_target_trees": sum(1 for item in usable if item["sparql_targets"] > 0),
        },
        "flags": {
            "capped_trees": sum(1 for item in usable if item["capped"]),
            "parse_failure_files": sum(item["parse_failures"] for item in usable),
            "report_or_export_trees": sum(1 for item in usable if item["report_or_export"]),
        },
        "runtime_seconds": _distribution(times),
        "projection": _projection(times, population_trees),
        "excluded": "parse_failure_only repositories stay outside this census.",
    }


def _tree_result(tree: dict, repository: dict, files: list[dict], failures: list[dict], elapsed: float, vocab: ShaclVocabulary) -> dict:
    extracted = bool(repository.get("commit_sha_examined")) and repository.get("verification_status") not in {
        "transient_failure",
        "inaccessible",
    }
    # Census reuses the verifier's status machine. A verified tree can be
    # reported incomplete when a cap or a parse failure remains. That does
    # not change the saved verification decision. Content counts below include
    # only files that parsed.
    constraints: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    node_uris: set[str] = set()
    property_uris: set[str] = set()
    distinct_uris: set[str] = set()
    node_blanks = property_blanks = distinct_blanks = 0
    sparql_links = 0
    sparql_nodes = 0
    sparql_targets = 0
    structural = 0
    path_occurrences = 0
    validation_reports = 0
    provenance_paths = []
    parsed = 0
    parse_failures = 0
    file_rows = []
    for item in files:
        evidence = item.get("_evidence")
        path = item["path"]
        parse_status = item["parse_status"]
        if parse_status == "parse_error":
            parse_failures += 1
        if evidence is None or parse_status != "parsed":
            file_rows.append(
                {
                    "path": path,
                    "parse_status": parse_status,
                    "node_shapes": 0,
                    "property_shapes": 0,
                    "parse_error": item.get("parse_error", ""),
                }
            )
            continue
        parsed += 1
        node_blanks += _absorb(evidence.node_shape_subjects, node_uris)
        property_blanks += _absorb(evidence.property_shape_subjects, property_uris)
        distinct_blanks += _absorb(evidence.node_shape_subjects | evidence.property_shape_subjects, distinct_uris)
        for iri, count in evidence.predicate_occurrences.items():
            if iri in vocab.core_constraint_parameters:
                constraints[iri] += count
            elif iri in CORE_TARGETS:
                targets[iri] += count
        sparql_links += evidence.sparql_links
        sparql_nodes += evidence.sparql_constraint_nodes
        sparql_targets += evidence.sparql_target_occurrences
        structural += evidence.sparql_triples
        path_occurrences += evidence.path_occurrences
        validation_reports += evidence.validation_reports
        if evidence.validation_reports or any(marker in path.lower() for marker in PATH_MARKERS):
            provenance_paths.append(path)
        file_rows.append(
            {
                "path": path,
                "parse_status": parse_status,
                "node_shapes": len(evidence.node_shape_subjects),
                "property_shapes": len(evidence.property_shape_subjects),
                "parse_error": "",
            }
        )
    node_shapes = len(node_uris) + node_blanks
    property_shapes = len(property_uris) + property_blanks
    distinct_shapes = len(distinct_uris) + distinct_blanks
    return {
        "tree_sha": tree["tree_sha"],
        "donor": tree["donor"]["repository_full_name"],
        "commit_sha": tree["donor"]["commit_sha_examined"],
        "phase_origin": tree["phase_origin"],
        "fork_origin": tree["fork_origin"],
        "repository_count": tree["repository_count"],
        "repository_names": [item["repository_full_name"] for item in tree["repositories"]],
        "sparql_only_verification": tree["sparql_only_verification"],
        "extracted": extracted,
        "extraction_status": repository.get("verification_status", ""),
        "capped": "candidate_file_cap" in {item.get("failure_kind") for item in failures} or repository.get("verification_reason") == "candidate_file_cap",
        "node_shapes": node_shapes,
        "property_shapes": property_shapes,
        "distinct_shapes": distinct_shapes,
        "constraints": dict(constraints),
        "targets": dict(targets),
        "path_occurrences": path_occurrences,
        "sparql_links": sparql_links,
        "sparql_constraint_nodes": sparql_nodes,
        "sparql_targets": sparql_targets,
        "structural_sparql": structural > 0,
        "sparql_only_content": structural > 0 and distinct_shapes == 0,
        "report_or_export": bool(provenance_paths),
        "provenance_paths": provenance_paths[:8],
        "files_parsed": parsed,
        "parse_failures": parse_failures,
        "candidate_files": int(repository.get("candidate_files_found") or 0),
        "seconds": round(elapsed, 3),
        "failures": [item.get("failure_kind", "") for item in failures],
        "files": file_rows,
    }


def _absorb(subjects: set[str], uris: set[str]) -> int:
    blanks = 0
    for subject in subjects:
        if subject.startswith("bnode:"):
            blanks += 1
        else:
            uris.add(subject)
    return blanks


def _prevalence(
    results: list[dict],
    field: str,
    tree_denominator: int,
    *,
    expected: list[str] | None = None,
) -> list[dict]:
    occurrence: Counter[str] = Counter()
    trees: Counter[str] = Counter()
    repos: Counter[str] = Counter()
    for item in results:
        for iri, count in item[field].items():
            if count <= 0:
                continue
            occurrence[iri] += count
            trees[iri] += 1
            repos[iri] += item["repository_count"]
    repo_denominator = sum(item["repository_count"] for item in results)
    iris = sorted(set(occurrence) | set(expected or []))
    rows = []
    for iri in iris:
        rows.append(
            {
                "predicate": iri.rsplit("#", 1)[-1],
                "occurrences_in_unique_trees": occurrence[iri],
                "unique_trees": trees[iri],
                "percent_of_pilot_trees": _percent(trees[iri], tree_denominator),
                "verified_repositories": repos[iri],
                "percent_of_pilot_repositories": _percent(repos[iri], repo_denominator),
            }
        )
    return rows


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    mean = statistics.fmean(ordered)
    stdev = statistics.pstdev(ordered) if n > 1 else 0.0
    return {
        "n": n,
        "min": ordered[0],
        "q1": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "mean": round(mean, 3),
        "q3": _quantile(ordered, 0.75),
        "max": ordered[-1],
        "stdev": round(stdev, 3),
        "p95": _quantile(ordered, 0.95),
    }


def _dominance(values: list[int]) -> dict:
    if not values or sum(values) == 0:
        return {"top_5_percent_of_trees_share_of_shapes": 0, "share_above_mean": 0}
    ordered = sorted(values, reverse=True)
    total = sum(ordered)
    head = max(1, math.ceil(len(ordered) * 0.05))
    mean = statistics.fmean(ordered)
    return {
        "top_5_percent_of_trees_share_of_shapes": round(sum(ordered[:head]) / total, 4),
        "trees_in_that_head": head,
        "share_of_trees_above_mean": round(sum(1 for value in ordered if value > mean) / len(ordered), 4),
    }


def _projection(times: list[float], population: int) -> dict:
    if not times:
        return {}
    dist = _distribution(times)
    return {
        "sample_mean_seconds": dist["mean"],
        "sample_median_seconds": dist["median"],
        "sample_p95_seconds": dist["p95"],
        "population_trees": population,
        "hours_at_sample_mean": round(dist["mean"] * population / 3600, 2),
        "hours_at_sample_median": round(dist["median"] * population / 3600, 2),
        "note": (
            "The pilot over-represents large and SPARQL-boundary trees, so the "
            "mean projection is not a simple random-sample estimate."
        ),
    }


def _quantile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def _percent(count: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100 * count / denominator, 2)


def _tertile_bounds(values: list[int]) -> tuple[float, float]:
    if not values:
        return (0, 0)
    ordered = sorted(values)
    return (_quantile(ordered, 1 / 3), _quantile(ordered, 2 / 3))


def _cell(item: dict, bounds: tuple[float, float], evidence: str) -> tuple:
    low, high = bounds
    size = item["recorded_candidates"]
    if size <= low:
        band = "small"
    elif size <= high:
        band = "medium"
    else:
        band = "large"
    return (item["phase_origin"], item["fork_origin"], band, evidence)


def _donor(members: list[dict]) -> dict:
    inspected = [item for item in members if not _truthy(item.get("verification_reused"))]
    pool = inspected or members
    return sorted(pool, key=lambda item: (-int(item.get("rdf_files_parsed") or 0), item["repository_full_name"]))[0]


def _read_verified(path: Path, phase: int) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("verification_status") != "verified_shacl":
                continue
            row["phase"] = phase
            rows.append(row)
    return rows


def _truthy(value) -> bool:
    return str(value).strip().lower() == "true"


def vocabulary_or_default(vocab: ShaclVocabulary | None) -> ShaclVocabulary:
    return vocab or default_vocabulary()


REPRESENTATIVE_CONSTRAINTS = (
    "minCount",
    "maxCount",
    "datatype",
    "class",
    "nodeKind",
    "property",
    "targetClass",
)


def compare_cap_recheck(
    old_rows: list[dict],
    new_rows: list[dict],
    pilot_rows: list[dict],
    *,
    population_trees: int,
) -> dict:
    """Old truncated pilot rows against the uncapped recheck of the same trees."""
    fresh = {row["tree_sha"]: row for row in new_rows}
    comparisons = []
    old_seconds = 0.0
    new_seconds = 0.0
    for old in old_rows:
        new = fresh.get(old["tree_sha"])
        if new is None:
            continue
        old_seconds += old["seconds"]
        new_seconds += new["seconds"]
        constraints = []
        for name in REPRESENTATIVE_CONSTRAINTS:
            iri = SH + name
            before = old.get("targets", {}).get(iri, old.get("constraints", {}).get(iri, 0))
            after = new.get("targets", {}).get(iri, new.get("constraints", {}).get(iri, 0))
            constraints.append({"predicate": name, "old": before, "new": after, "delta": after - before})
        comparisons.append(
            {
                "donor": new["donor"],
                "tree_sha": new["tree_sha"],
                "old_node_shapes": old["node_shapes"],
                "new_node_shapes": new["node_shapes"],
                "old_property_shapes": old["property_shapes"],
                "new_property_shapes": new["property_shapes"],
                "old_distinct_shapes": old["distinct_shapes"],
                "new_distinct_shapes": new["distinct_shapes"],
                "constraints": constraints,
                "old_candidate_files": old.get("candidate_files"),
                "candidates_discovered": new.get("candidates_discovered"),
                "files_parsed": new.get("files_parsed"),
                "parse_failures": new.get("parse_failures"),
                "skipped": new.get("skipped", {}),
                "enumeration_complete": new.get("enumeration_complete"),
                "counts_partial": new.get("counts_partial"),
                "partial_reasons": new.get("partial_reasons", []),
                "old_seconds": old["seconds"],
                "new_seconds": new["seconds"],
            }
        )
    replaced = {row["tree_sha"]: row["seconds"] for row in new_rows}
    blended = [
        replaced.get(row["tree_sha"], row["seconds"])
        for row in pilot_rows
    ]
    return {
        "trees": comparisons,
        "old_seconds": round(old_seconds, 3),
        "new_seconds": round(new_seconds, 3),
        "extra_seconds": round(new_seconds - old_seconds, 3),
        "projection_blended_pilot": _projection(blended, population_trees),
        "note": (
            "The blended projection replaces the 12 truncated runtimes inside the "
            "stratified 120-tree pilot and then scales that mean to 6,343 trees. "
            "The pilot over-represents large trees, so this is not a random-sample estimate. "
            "Trees whose saved verification candidate count is already 400 are a lower bound "
            "on how many full-corpus trees need this uncapped path."
        ),
    }


# Candidate-file buckets for a workload estimate. The cuts match the
# thresholds we need to see the long tail, with finer bins where most trees sit.
_FILE_BUCKET_EDGES = (10, 20, 50, 100, 400, 1000, 2000, 5000)
_LFS_POINTER_MAX_BYTES = 2048


def file_count_bucket(count: int) -> str:
    previous = 0
    for edge in _FILE_BUCKET_EDGES:
        if count <= edge:
            return f"{previous + 1}–{edge}"
        previous = edge
    return f"{_FILE_BUCKET_EDGES[-1] + 1}+"


def describe_tree_workload(client, tree: dict, config: dict) -> dict:
    """Count candidate RDF files from a git tree listing. Does not parse RDF."""
    donor = tree["donor"]
    full_name = donor["repository_full_name"]
    commit_sha = donor["commit_sha_examined"]
    base = {
        "tree_sha": tree["tree_sha"],
        "donor": full_name,
        "commit_sha": commit_sha,
        "repository_count": tree["repository_count"],
        "phase_origin": tree["phase_origin"],
        "fork_origin": tree["fork_origin"],
    }
    try:
        payload = client.get_git_tree(full_name, tree["tree_sha"])
    except Exception as exc:
        base.update(
            {
                "candidate_files": 0,
                "candidate_bytes": 0,
                "size_unknown_files": 0,
                "listing_complete": False,
                "listing_source": "tree_request_failed",
                "listing_error": f"{type(exc).__name__}: {exc}"[:300],
                "tarball_preferred": False,
                "tarball_reason": "",
                "strong_candidates": 0,
                "oversize_files": 0,
                "lfs_observed": 0,
                "lfs_unchecked_small_files": 0,
            }
        )
        return base
    truncated = bool(payload.get("truncated"))
    blobs = _candidate_blobs(payload.get("tree") or [], config)
    source = "git_tree"
    complete = not truncated
    if truncated:
        cached = _cached_tarball_candidates(client, full_name, commit_sha, config)
        if cached is not None:
            blobs = cached
            source = "cached_tarball"
            complete = True
        else:
            source = "truncated_partial"
            complete = False
    summary = _summarize_candidates(
        blobs,
        config,
        client=client,
        full_name=full_name,
        commit_sha=commit_sha,
        truncated=truncated,
    )
    base.update(summary)
    base["listing_complete"] = complete
    base["listing_source"] = source
    base["listing_error"] = ""
    return base


def summarize_workload(rows: list[dict], measured_seconds: dict[str, float]) -> dict:
    complete = [row for row in rows if row.get("listing_complete")]
    incomplete = [row for row in rows if not row.get("listing_complete")]
    counts = [int(row["candidate_files"]) for row in complete]
    ordered = sorted(rows, key=lambda row: (-int(row["candidate_files"]), row["donor"]))
    return {
        "trees": len(rows),
        "listing_complete": len(complete),
        "listing_incomplete": len(incomplete),
        "candidate_files": _count_distribution(counts),
        "candidate_bytes": _count_distribution([int(row["candidate_bytes"]) for row in complete]),
        "thresholds": {
            "over_100": sum(1 for value in counts if value > 100),
            "over_400": sum(1 for value in counts if value > 400),
            "over_1000": sum(1 for value in counts if value > 1000),
            "over_2000": sum(1 for value in counts if value > 2000),
            "over_5000": sum(1 for value in counts if value > 5000),
        },
        "incomplete_partial_counts": [
            {
                "donor": row["donor"],
                "tree_sha": row["tree_sha"],
                "candidate_files_lower_bound": row["candidate_files"],
                "listing_source": row["listing_source"],
            }
            for row in incomplete
        ],
        "tarball_preferred": sum(1 for row in rows if row.get("tarball_preferred")),
        "trees_with_oversize_files": sum(1 for row in rows if row.get("oversize_files")),
        "oversize_files": sum(int(row.get("oversize_files") or 0) for row in rows),
        "trees_with_observed_lfs": sum(1 for row in rows if row.get("lfs_observed")),
        "lfs_pointers_observed_in_cache": sum(int(row.get("lfs_observed") or 0) for row in rows),
        "small_files_not_in_cache": sum(int(row.get("lfs_unchecked_small_files") or 0) for row in rows),
        "largest_30": [
            {
                "donor": row["donor"],
                "tree_sha": row["tree_sha"],
                "repository_count": row["repository_count"],
                "candidate_files": row["candidate_files"],
                "candidate_bytes": row["candidate_bytes"],
                "listing_complete": row["listing_complete"],
                "listing_source": row["listing_source"],
                "tarball_preferred": row["tarball_preferred"],
                "oversize_files": row["oversize_files"],
                "lfs_observed": row["lfs_observed"],
            }
            for row in ordered[:30]
        ],
        "runtime": estimate_runtime_by_file_buckets(rows, measured_seconds),
        "lfs_note": (
            "LFS pointers are detected only when the git blob is already in the raw-byte cache "
            "and is at most 2048 bytes. Small candidate files that were never fetched are counted "
            "separately and are not treated as LFS."
        ),
    }


def estimate_runtime_by_file_buckets(rows: list[dict], measured_seconds: dict[str, float]) -> dict:
    """Scale measured seconds-per-file within file-count buckets.

    A single huge tree does not set the rate for a bucket of small trees.
    """
    rates: dict[str, list[float]] = defaultdict(list)
    measured_detail: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        seconds = measured_seconds.get(row["tree_sha"])
        files = int(row["candidate_files"])
        if seconds is None or files <= 0:
            continue
        label = file_count_bucket(files)
        rates[label].append(seconds / files)
        measured_detail[label].append(
            {"donor": row["donor"], "candidate_files": files, "seconds": round(seconds, 3)}
        )
    labels = _bucket_labels()
    known = [label for label in labels if rates[label]]

    def rate_source(label: str) -> tuple[str, str]:
        if rates[label]:
            return label, "measured"
        if not known:
            return label, "unmeasured"
        index = labels.index(label)
        nearest = min(known, key=lambda item: abs(labels.index(item) - index))
        return nearest, "nearest_measured_bucket"

    buckets = []
    low_total = mid_total = high_total = 0.0
    for label in labels:
        members = [row for row in rows if file_count_bucket(int(row["candidate_files"])) == label]
        source_label, how = rate_source(label)
        sample = sorted(rates[source_label]) if rates.get(source_label) else []
        low_rate = _quantile(sample, 0.25) if sample else 0.0
        mid_rate = _quantile(sample, 0.5) if sample else 0.0
        high_rate = _quantile(sample, 0.75) if sample else 0.0
        files = sum(int(row["candidate_files"]) for row in members)
        low_seconds = low_rate * files
        mid_seconds = mid_rate * files
        high_seconds = high_rate * files
        low_total += low_seconds
        mid_total += mid_seconds
        high_total += high_seconds
        buckets.append(
            {
                "bucket": label,
                "trees": len(members),
                "candidate_files": files,
                "measured_trees": len(measured_detail[label]),
                "rate_source": how,
                "rate_bucket": source_label,
                "seconds_per_file_p25": round(low_rate, 4),
                "seconds_per_file_median": round(mid_rate, 4),
                "seconds_per_file_p75": round(high_rate, 4),
                "estimated_hours_at_median_rate": round(mid_seconds / 3600, 2),
                "measured": sorted(measured_detail[label], key=lambda item: -item["candidate_files"])[:8],
            }
        )
    return {
        "method": "median seconds per candidate file inside file-count buckets",
        "estimated_hours_p25_rate": round(low_total / 3600, 2),
        "estimated_hours_median_rate": round(mid_total / 3600, 2),
        "estimated_hours_p75_rate": round(high_total / 3600, 2),
        "buckets": buckets,
        "note": (
            "Each tree is charged its own candidate-file count times the seconds-per-file "
            "rate of its bucket. The overall pilot mean is not applied to every tree."
        ),
    }


def _candidate_blobs(entries: list[dict], config: dict) -> list[dict]:
    blobs = []
    for item in entries:
        if item.get("type") not in (None, "blob"):
            continue
        path = str(item.get("path") or "")
        if not path or path.startswith("/") or ".." in path.split("/"):
            continue
        if not raw_extension_rule(path, config):
            continue
        size = item.get("size")
        blobs.append({"path": path, "size": size if isinstance(size, int) else None})
    return blobs


def _cached_tarball_candidates(client, full_name: str, commit_sha: str, config: dict) -> list[dict] | None:
    cache = client._byte_cache_path(tarball_url(full_name, commit_sha))
    if not cache.exists():
        return None
    blobs = []
    with tarfile.open(cache, "r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = _archive_path(member.name)
            if not path or not raw_extension_rule(path, config):
                continue
            blobs.append({"path": path, "size": member.size})
    return blobs


def _cached_is_lfs(client, full_name: str, commit_sha: str, path: str) -> bool | None:
    cache = client._byte_cache_path(raw_content_url(full_name, commit_sha, path))
    if not cache.exists():
        return None
    with cache.open("rb") as handle:
        return handle.read(len(LFS_MARKER)).startswith(LFS_MARKER)


def _summarize_candidates(blobs, config, *, client, full_name: str, commit_sha: str, truncated: bool) -> dict:
    strong = sum(1 for item in blobs if raw_extension_rule(item["path"], config).startswith("extension:"))
    total = len(blobs)
    threshold = int(config.get("tarball_candidate_threshold") or 0)
    cap = int(config["max_candidate_files"])
    limit = int(config["max_file_bytes"])
    reasons = []
    if truncated:
        reasons.append("git_tree_truncated")
    if threshold and strong >= threshold:
        reasons.append("strong_extensions_at_or_above_tarball_threshold")
    if total > cap:
        reasons.append("above_verification_candidate_cap")
    lfs_observed = 0
    lfs_unchecked = 0
    for item in blobs:
        size = item["size"]
        if not isinstance(size, int) or size > _LFS_POINTER_MAX_BYTES:
            continue
        flag = _cached_is_lfs(client, full_name, commit_sha, item["path"])
        if flag is True:
            lfs_observed += 1
        elif flag is None:
            lfs_unchecked += 1
    return {
        "candidate_files": total,
        "candidate_bytes": sum(item["size"] for item in blobs if isinstance(item["size"], int)),
        "size_unknown_files": sum(1 for item in blobs if not isinstance(item["size"], int)),
        "strong_candidates": strong,
        "tarball_preferred": bool(reasons),
        "tarball_reason": "|".join(reasons),
        "oversize_files": sum(1 for item in blobs if isinstance(item["size"], int) and item["size"] > limit),
        "lfs_observed": lfs_observed,
        "lfs_unchecked_small_files": lfs_unchecked,
    }


def _count_distribution(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "q1": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "mean": round(statistics.fmean(ordered), 3),
        "q3": _quantile(ordered, 0.75),
        "p90": _quantile(ordered, 0.90),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": ordered[-1],
    }


def _bucket_labels() -> list[str]:
    labels = []
    previous = 0
    for edge in _FILE_BUCKET_EDGES:
        labels.append(f"{previous + 1}–{edge}")
        previous = edge
    labels.append(f"{previous + 1}+")
    return labels


HEAVY_CANDIDATE_MIN = 5001
GGEN_DONOR = "seanchatmangpt/ggen"
CORESE_DONOR = "handedemir/corese"


def workload_type_for(row: dict) -> str:
    files = int(row.get("candidate_files") or 0)
    strong = int(row.get("strong_candidates") or 0)
    if files <= 0:
        return "unknown"
    ratio = strong / files
    if ratio < 0.25:
        return "sniff_heavy"
    if ratio >= 0.5:
        return "strong_rdf"
    return "mixed"


def partition_statistics_batches(trees: list[dict], workload_rows: list[dict]) -> dict:
    """Split unique verified trees into Batch A, the heavy tail, and unresolved listings.

    A tree can sit in both the heavy bucket and the unresolved list when a
    truncated listing is already above 5,000 candidates. Batch A is the rest:
    a complete listing of at most 5,000 candidate RDF files.
    """
    workload = {row["tree_sha"]: row for row in workload_rows}
    batch_a = []
    heavy = []
    unresolved = []
    for tree in trees:
        row = workload.get(tree["tree_sha"])
        files = int(row["candidate_files"]) if row is not None else None
        listing_complete = bool(row and row.get("listing_complete"))
        is_heavy = files is not None and files >= HEAVY_CANDIDATE_MIN
        if is_heavy:
            heavy.append(tree)
        if not listing_complete:
            unresolved.append(tree)
        if listing_complete and not is_heavy:
            batch_a.append(tree)
    heavy_ids = {tree["tree_sha"] for tree in heavy}
    return {
        "batch_a": batch_a,
        "heavy": heavy,
        "unresolved": unresolved,
        "unresolved_also_heavy": [tree for tree in unresolved if tree["tree_sha"] in heavy_ids],
    }


def select_heavy_benchmark(workload_rows: list[dict]) -> list[dict]:
    """Twelve distinct trees that cover the heavy-tail workload types."""
    heavy = [row for row in workload_rows if int(row.get("candidate_files") or 0) >= HEAVY_CANDIDATE_MIN]
    used: set[str] = set()
    selected: list[dict] = []

    def take(row: dict | None, role: str) -> None:
        if row is None or row["tree_sha"] in used:
            return
        used.add(row["tree_sha"])
        selected.append(
            {
                "role": role,
                "tree_sha": row["tree_sha"],
                "donor": row["donor"],
                "candidate_files": int(row["candidate_files"]),
                "strong_candidates": int(row.get("strong_candidates") or 0),
                "listing_complete": bool(row.get("listing_complete")),
                "workload_type": workload_type_for(row),
            }
        )

    def unused(rows: list[dict]) -> list[dict]:
        return [row for row in rows if row["tree_sha"] not in used]

    by_donor = {row["donor"]: row for row in heavy}
    take(by_donor.get(GGEN_DONOR), "ggen_slow")
    take(by_donor.get(CORESE_DONOR), "corese_fast")
    partial = unused([row for row in heavy if not row.get("listing_complete")])
    if partial:
        take(max(partial, key=lambda row: (int(row["candidate_files"]), row["donor"])), "truncated_large")
    complete = [row for row in heavy if row.get("listing_complete")]
    small_pool = unused(complete)
    if small_pool:
        take(min(small_pool, key=lambda row: (int(row["candidate_files"]), row["donor"])), "small_end")
    if complete:
        median_files = statistics.median(int(row["candidate_files"]) for row in complete)
        median_pool = unused(complete)
        if median_pool:
            take(
                min(median_pool, key=lambda row: (abs(int(row["candidate_files"]) - median_files), row["donor"])),
                "median_heavy",
            )
    ranked = sorted(heavy, key=lambda row: (-int(row.get("strong_candidates") or 0), row["donor"]))
    slot = 1
    for row in ranked:
        if slot > 3:
            break
        if row["tree_sha"] in used:
            continue
        take(row, f"largest_strong_{slot}")
        slot += 1
    sniff_ranked = sorted(
        unused(complete),
        key=lambda row: (
            int(row.get("strong_candidates") or 0) / int(row["candidate_files"]),
            row["donor"],
        ),
    )
    for index, row in enumerate(sniff_ranked[:2], start=1):
        take(row, f"sniff_heavy_{index}")
    fill = sorted(unused(heavy), key=lambda row: (-int(row["candidate_files"]), row["donor"]))
    extra = 1
    for row in fill:
        if len(selected) >= 12:
            break
        take(row, f"additional_{extra}")
        extra += 1
    return selected


def reusable_complete_results(batch_shas: set[str], prior_rows: list[dict]) -> list[dict]:
    """Earlier complete enumerations of the same pinned tree can be kept."""
    kept: dict[str, dict] = {}
    for row in prior_rows:
        sha = row.get("tree_sha")
        if sha not in batch_shas:
            continue
        if not row.get("enumeration_complete") or row.get("capped"):
            continue
        kept[sha] = row
    return list(kept.values())


def aggregate_batch_a(
    results: list[dict],
    *,
    batch_a_trees: int,
    heavy_trees: int,
    unresolved_trees: int,
    unresolved_also_heavy: int,
    population_trees: int,
) -> dict:
    """Shape statistics for the standard trees only. The heavy tail is not in the denominator."""
    usable = [item for item in results if item.get("extracted")]
    complete = [item for item in usable if item.get("enumeration_complete")]
    partial = [item for item in results if not item.get("enumeration_complete") or item.get("counts_partial")]
    reasons: Counter[str] = Counter()
    for item in partial:
        for reason in item.get("partial_reasons") or ["unspecified"]:
            reasons[reason] += 1
    vocab = default_vocabulary()
    tree_denominator = len(usable)
    constraints = _relabel_prevalence(
        _prevalence(usable, "constraints", tree_denominator, expected=sorted(vocab.core_constraint_parameters)),
        "percent_of_batch_a_trees",
        "percent_of_batch_a_repositories",
    )
    targets = _relabel_prevalence(
        _prevalence(usable, "targets", tree_denominator, expected=list(CORE_TARGETS)),
        "percent_of_batch_a_trees",
        "percent_of_batch_a_repositories",
    )
    times = [float(item["seconds"]) for item in results if "seconds" in item]
    skipped_oversized = sum(int((item.get("skipped") or {}).get("skipped_too_large") or 0) for item in usable)
    return {
        "corpus_scope": "batch_a_standard_trees",
        "partial_corpus": True,
        "label": (
            "Partial-corpus results. Trees with more than 5,000 candidate RDF files "
            "and trees whose workload listing is unresolved are excluded. "
            "Percentages use Batch A as the denominator."
        ),
        "population_unique_verified_trees": population_trees,
        "batch_a_trees": batch_a_trees,
        "excluded_heavy_trees": heavy_trees,
        "excluded_unresolved_listings": unresolved_trees,
        "excluded_unresolved_also_in_heavy_bucket": unresolved_also_heavy,
        "trees_attempted": len(results),
        "trees_completed": len(usable),
        "trees_with_complete_enumeration": len(complete),
        "trees_with_partial_counts": len(partial),
        "partial_reason_counts": dict(reasons),
        "parsed_files": sum(int(item.get("files_parsed") or 0) for item in usable),
        "parse_failures": sum(int(item.get("parse_failures") or 0) for item in usable),
        "skipped_oversized_files": skipped_oversized,
        "shape_totals": {
            "node_shapes": sum(int(item["node_shapes"]) for item in usable),
            "property_shapes": sum(int(item["property_shapes"]) for item in usable),
            "distinct_shapes": sum(int(item["distinct_shapes"]) for item in usable),
        },
        "constraints": constraints,
        "targets": targets,
        "sparql": {
            "trees_with_structural_constraints": sum(1 for item in usable if item.get("structural_sparql")),
            "sparql_links": sum(int(item.get("sparql_links") or 0) for item in usable),
            "sparql_constraint_nodes": sum(int(item.get("sparql_constraint_nodes") or 0) for item in usable),
            "sparql_only_verification_trees": sum(1 for item in usable if item.get("sparql_only_verification")),
            "sparql_only_content_trees": sum(1 for item in usable if item.get("sparql_only_content")),
            "sparql_target_trees": sum(1 for item in usable if int(item.get("sparql_targets") or 0) > 0),
        },
        "runtime_seconds": _distribution(times),
        "runtime_hours": round(sum(times) / 3600, 2) if times else 0,
    }


def _relabel_prevalence(rows: list[dict], tree_key: str, repo_key: str) -> list[dict]:
    relabeled = []
    for row in rows:
        item = dict(row)
        item[tree_key] = item.pop("percent_of_pilot_trees")
        item[repo_key] = item.pop("percent_of_pilot_repositories")
        relabeled.append(item)
    return relabeled


def estimate_heavy_runtime(heavy_rows: list[dict], benchmark_rows: list[dict]) -> dict:
    """Apply each workload type's measured seconds-per-file rate to the heavy bucket.

    A strategy that did not finish enumeration is not used as a rate. The overall
    mean across workload types is not applied.
    """
    by_type: dict[str, dict[str, list[float]]] = {}
    for row in benchmark_rows:
        if not row.get("enumeration_complete"):
            continue
        files = int(row.get("candidate_files") or 0)
        if files <= 0:
            continue
        kind = row.get("workload_type") or "unknown"
        strategy = row.get("requested_strategy") or "current"
        by_type.setdefault(kind, {}).setdefault(strategy, []).append(float(row["seconds"]) / files)
    type_rates = {}
    for kind, strategies in by_type.items():
        rates = {name: round(statistics.median(values), 6) for name, values in strategies.items() if values}
        if not rates:
            continue
        winner = min(rates, key=rates.get)
        type_rates[kind] = {"seconds_per_file": rates, "fastest": winner, "measured_trees": {name: len(strategies[name]) for name in rates}}
    totals = {name: 0.0 for name in ("tarball", "strong_first", "current")}
    typed_fastest = 0.0
    unmeasured = 0
    for row in heavy_rows:
        files = int(row.get("candidate_files") or 0)
        kind = workload_type_for(row)
        rates = type_rates.get(kind)
        if not rates:
            unmeasured += 1
            continue
        per_strategy = rates["seconds_per_file"]
        for name in totals:
            rate = per_strategy.get(name)
            if rate is None:
                continue
            totals[name] += rate * files
        winner = rates.get("fastest")
        if winner in per_strategy:
            typed_fastest += per_strategy[winner] * files
    return {
        "workload_types": type_rates,
        "estimated_hours_if_one_strategy": {name: round(seconds / 3600, 2) for name, seconds in totals.items()},
        "estimated_hours_fastest_by_workload_type": round(typed_fastest / 3600, 2),
        "heavy_trees_without_a_measured_type": unmeasured,
        "note": (
            "Each heavy tree is charged its own candidate-file count times the median "
            "seconds-per-file of benchmark trees with the same workload type. "
            "Incomplete benchmark runs do not supply a rate."
        ),
    }
