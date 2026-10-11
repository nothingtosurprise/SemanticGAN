from datetime import datetime, timezone
from pathlib import Path

from shacl_study.rdf_evidence import classify_graph, parse_rdf
from shacl_study.shacl_vocab import default_vocabulary
from shacl_study.statistics import (
    CensusTally,
    aggregate_batch_a,
    aggregate_pilot,
    describe_tree_workload,
    estimate_heavy_runtime,
    estimate_runtime_by_file_buckets,
    partition_statistics_batches,
    select_heavy_benchmark,
    select_statistics_pilot,
    unique_verified_trees,
)
from shacl_study.verification import inspect_repository, load_verification_config

ROOT_CONFIG = load_verification_config(Path("config/verification.yaml"))
CLOCK = lambda: datetime(2026, 10, 8, tzinfo=timezone.utc)
VOCAB = default_vocabulary()


class _Source:
    def __init__(self, files):
        self.files = files

    def resolve_commit(self, full_name, branch):
        return {"commit_sha": "a" * 40, "tree_sha": "b" * 40}

    def get_tree(self, full_name, tree_sha):
        return {
            "truncated": False,
            "sha": tree_sha,
            "tree": [
                {"path": path, "type": "blob", "size": len(payload), "sha": "c" * 40}
                for path, payload in self.files.items()
            ],
        }

    def get_file(self, full_name, commit_sha, path):
        return self.files[path]

    def peek_file(self, full_name, commit_sha, path, max_bytes=65536):
        payload = self.files[path]
        return payload[:max_bytes], len(payload) <= max_bytes

    def list_tarball(self, full_name, commit_sha):
        return []


def _shape(iri: str, minimum: int = 1) -> bytes:
    return f"""\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
ex:{iri} a sh:NodeShape ;
  sh:targetClass ex:Person ;
  sh:property [
    a sh:PropertyShape ;
    sh:path ex:name ;
    sh:minCount {minimum} ;
    sh:maxCount {minimum + 1}
  ] .
""".encode()


def test_constraint_occurrences_and_sparql_boundary():
    graph = parse_rdf(_shape("PersonShape"), "turtle")
    evidence = classify_graph(graph, VOCAB)
    assert len(evidence.node_shape_subjects) == 1
    assert len(evidence.property_shape_subjects) == 1
    assert evidence.predicate_occurrences["http://www.w3.org/ns/shacl#minCount"] == 1
    assert evidence.predicate_occurrences["http://www.w3.org/ns/shacl#maxCount"] == 1
    assert evidence.predicate_occurrences["http://www.w3.org/ns/shacl#targetClass"] == 1

    catalog = parse_rdf(
        b"""\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
ex:Query a sh:SPARQLSelectExecutable ;
  sh:select "SELECT ?s WHERE { ?s ?p ?o }" .
""",
        "turtle",
    )
    catalog_evidence = classify_graph(catalog, VOCAB)
    assert catalog_evidence.sparql_triples == 0
    assert catalog_evidence.sparql_constraint_nodes == 0
    assert catalog_evidence.sparql_links == 0

    linked = parse_rdf(
        b"""\
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .
ex:ReportShape a sh:NodeShape ;
  sh:sparql [
    a sh:SPARQLConstraint ;
    sh:select "SELECT $this WHERE { }"
  ] .
""",
        "turtle",
    )
    linked_evidence = classify_graph(linked, VOCAB)
    assert linked_evidence.sparql_triples > 0
    assert linked_evidence.sparql_links == 1
    assert linked_evidence.sparql_constraint_nodes == 1
    assert len(linked_evidence.node_shape_subjects) == 1


def test_census_enumerates_past_the_verification_cap():
    plain = b"@prefix ex: <http://example.org/> .\nex:a ex:b ex:c .\n"
    files = {"a.ttl": plain, "b.ttl": plain, "c.ttl": _shape("OnlyInTheThird")}
    config = dict(ROOT_CONFIG)
    config["max_candidate_files"] = 2
    row = {
        "repository_id": "1",
        "repository_full_name": "acme/shapes",
        "default_branch": "main",
        "discovered_by": "code:shacl_class_name:primary",
        "is_fork": "false",
        "archived": "false",
    }
    capped, _, failures = inspect_repository(
        row,
        source=_Source(files),
        vocab=VOCAB,
        config=config,
        clock=CLOCK,
    )
    full, _, full_failures = inspect_repository(
        row,
        source=_Source(files),
        vocab=VOCAB,
        config=config,
        clock=CLOCK,
        census=True,
    )
    assert capped["rdf_files_parsed"] == 2
    assert capped["node_shape_count"] == 0
    assert any(item["failure_kind"] == "candidate_file_cap" for item in failures)
    assert full["rdf_files_parsed"] == 3
    assert full["node_shape_count"] == 1
    assert not any(item["failure_kind"] == "candidate_file_cap" for item in full_failures)


def test_census_parses_files_the_verification_prefilter_skips():
    files = {
        "notes.ttl": b"@prefix ex: <http://example.org/> .\nex:a ex:b ex:c .\n",
        "shape.ttl": _shape("Person"),
    }
    row = {
        "repository_id": "1",
        "repository_full_name": "acme/shapes",
        "default_branch": "main",
        "discovered_by": "",
        "is_fork": "false",
        "archived": "false",
    }
    gated, _, _ = inspect_repository(
        row,
        source=_Source(files),
        vocab=VOCAB,
        config=ROOT_CONFIG,
        clock=CLOCK,
    )
    full, _, _ = inspect_repository(
        row,
        source=_Source(files),
        vocab=VOCAB,
        config=ROOT_CONFIG,
        clock=CLOCK,
        census=True,
    )
    assert gated["rdf_files_parsed"] == 1
    assert full["rdf_files_parsed"] == 2
    assert gated["node_shape_count"] == full["node_shape_count"] == 1


def test_census_does_not_stop_at_the_first_shape():
    files = {"a.ttl": _shape("One"), "b.ttl": _shape("Two")}
    row = {
        "repository_id": "1",
        "repository_full_name": "acme/shapes",
        "default_branch": "main",
        "discovered_by": "code:shacl_class_name:primary",
        "is_fork": "false",
        "archived": "false",
    }
    early, _, _ = inspect_repository(
        row,
        source=_Source(files),
        vocab=VOCAB,
        config=ROOT_CONFIG,
        clock=CLOCK,
    )
    full, _, _ = inspect_repository(
        row,
        source=_Source(files),
        vocab=VOCAB,
        config=ROOT_CONFIG,
        clock=CLOCK,
        census=True,
    )
    assert early["node_shape_count"] == 1
    assert early["verification_stopped_early"] is True
    assert full["node_shape_count"] == 2
    assert full["verification_stopped_early"] is False
    assert "_census_files" in full
    assert full["verification_status"] == "verified_shacl"


def test_pilot_keeps_one_row_per_tree_and_the_sparql_only_cases():
    rows = []
    for index in range(12):
        rows.append(
            {
                "tree_sha": f"{index:040x}",
                "verification_status": "verified_shacl",
                "verification_reason": "core_shape_evidence",
                "phase": 1 if index < 6 else 2,
                "is_fork": "true" if index % 2 else "false",
                "candidate_files_found": str(index + 1),
                "node_shape_count": str(index * 10),
                "rdf_files_parsed": "1",
                "repository_full_name": f"acme/repo-{index}",
                "verification_reused": "false",
                "commit_sha_examined": "a" * 40,
                "prefilter_gate": "false",
            }
        )
    rows.append(
        {
            "tree_sha": "f" * 40,
            "verification_status": "verified_shacl",
            "verification_reason": "sparql_shape_evidence",
            "phase": 2,
            "is_fork": "false",
            "candidate_files_found": "3",
            "node_shape_count": "0",
            "rdf_files_parsed": "1",
            "repository_full_name": "acme/sparql-only",
            "verification_reused": "false",
            "commit_sha_examined": "b" * 40,
            "prefilter_gate": "true",
        }
    )
    trees = unique_verified_trees(rows)
    selected = select_statistics_pilot(trees, 8, seed=20261009)
    assert len({item["tree_sha"] for item in selected}) == len(selected)
    assert any(item["sparql_only_verification"] for item in selected)
    summary = aggregate_pilot(
        [
            {
                "extracted": True,
                "node_shapes": 0,
                "property_shapes": 0,
                "distinct_shapes": 0,
                "repository_count": 2,
                "constraints": {},
                "targets": {},
                "structural_sparql": True,
                "sparql_links": 1,
                "sparql_constraint_nodes": 1,
                "sparql_only_verification": True,
                "sparql_only_content": True,
                "sparql_targets": 0,
                "capped": False,
                "parse_failures": 0,
                "report_or_export": True,
                "seconds": 1.5,
            }
        ],
        population_trees=6343,
        population_repositories=7569,
    )
    assert summary["shapes_per_tree"]["distinct_shapes"]["n"] == 1
    assert summary["shapes_per_repository"]["distinct_shapes"]["n"] == 2
    assert summary["population_unique_verified_trees"] == 6343
    assert summary["sparql"]["sparql_only_content_trees"] == 1


def test_tally_counts_a_parsed_file_once():
    class Item:
        def __init__(self, path, payload):
            self.path = path
            self.parse_status = "parsed"
            self.parse_error = ""
            self.evidence = classify_graph(parse_rdf(payload, "turtle"), VOCAB)

    tally = CensusTally()
    item = Item("a.ttl", _shape("Person"))
    tally.add(item, VOCAB)
    tally.add(item, VOCAB)
    assert len(tally.files) == 1
    assert len(tally.node_uris) + tally.node_blanks == 1
    assert tally.constraints["http://www.w3.org/ns/shacl#minCount"] == 1


def _workload_tree() -> dict:
    return {
        "tree_sha": "b" * 40,
        "repository_count": 2,
        "phase_origin": 2,
        "fork_origin": "fork",
        "donor": {"repository_full_name": "acme/shapes", "commit_sha_examined": "a" * 40},
    }


def test_workload_census_counts_files_and_does_not_need_their_bytes(tmp_path):
    class Client:
        def get_git_tree(self, full_name, tree_sha):
            return {
                "truncated": False,
                "tree": [
                    {"path": "shapes/a.ttl", "type": "blob", "size": 120},
                    {"path": "shapes/b.json", "type": "blob", "size": 80},
                    {"path": "shapes/big.ttl", "type": "blob", "size": 30_000_000},
                    {"path": "README.md", "type": "blob", "size": 20},
                ],
            }

        def _byte_cache_path(self, url):
            if url.endswith("shapes/a.ttl"):
                path = tmp_path / "a.ttl"
                path.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n")
                return path
            return tmp_path / "missing"

    row = describe_tree_workload(Client(), _workload_tree(), ROOT_CONFIG)
    assert row["candidate_files"] == 3
    assert row["strong_candidates"] == 2
    assert row["oversize_files"] == 1
    assert row["listing_complete"] is True
    assert row["lfs_observed"] == 1
    assert row["lfs_unchecked_small_files"] == 1
    assert row["tarball_preferred"] is False


def test_truncated_tree_without_a_cached_tarball_stays_incomplete(tmp_path):
    class Client:
        def get_git_tree(self, full_name, tree_sha):
            return {"truncated": True, "tree": [{"path": "a.ttl", "type": "blob", "size": 10}]}

        def _byte_cache_path(self, url):
            return tmp_path / "missing"

    row = describe_tree_workload(Client(), _workload_tree(), ROOT_CONFIG)
    assert row["listing_complete"] is False
    assert row["listing_source"] == "truncated_partial"
    assert row["candidate_files"] == 1
    assert row["tarball_preferred"] is True


def test_runtime_estimate_does_not_spread_one_huge_tree_across_small_trees():
    rows = []
    measured = {}
    for index in range(120):
        sha = f"{index:040x}"
        rows.append({"tree_sha": sha, "donor": f"acme/{index}", "candidate_files": 5})
        if index < 5:
            measured[sha] = 1.0
    huge = "f" * 40
    rows.append({"tree_sha": huge, "donor": "acme/huge", "candidate_files": 6000})
    measured[huge] = 2400.0
    estimate = estimate_runtime_by_file_buckets(rows, measured)
    assert estimate["estimated_hours_median_rate"] < 2


def _batch_tree(sha: str) -> dict:
    return {"tree_sha": sha}


def _workload(sha: str, donor: str, files: int, strong: int, complete: bool = True) -> dict:
    return {
        "tree_sha": sha,
        "donor": donor,
        "candidate_files": files,
        "strong_candidates": strong,
        "listing_complete": complete,
    }


def test_batch_a_excludes_heavy_trees_and_unresolved_listings():
    trees = [_batch_tree(f"{index:040x}") for index in range(6)]
    workload = [
        _workload(trees[0]["tree_sha"], "acme/small", 31, 10, True),
        _workload(trees[1]["tree_sha"], "acme/edge", 5000, 100, True),
        _workload(trees[2]["tree_sha"], "acme/heavy", 5001, 4000, True),
        _workload(trees[3]["tree_sha"], "acme/truncated-heavy", 8000, 8000, False),
        _workload(trees[4]["tree_sha"], "acme/truncated-small", 40, 10, False),
        _workload(trees[5]["tree_sha"], "acme/failed", 0, 0, False),
    ]
    parts = partition_statistics_batches(trees, workload)
    assert [item["tree_sha"] for item in parts["batch_a"]] == [trees[0]["tree_sha"], trees[1]["tree_sha"]]
    assert {item["tree_sha"] for item in parts["heavy"]} == {trees[2]["tree_sha"], trees[3]["tree_sha"]}
    assert len(parts["unresolved"]) == 3
    assert len(parts["unresolved_also_heavy"]) == 1


def test_heavy_benchmark_covers_the_requested_workload_roles():
    rows = [
        _workload("1".rjust(40, "a"), "seanchatmangpt/ggen", 6384, 5683),
        _workload("2".rjust(40, "b"), "handedemir/corese", 6250, 6201),
        _workload("3".rjust(40, "c"), "acme/small-heavy", 5100, 5000),
        _workload("4".rjust(40, "d"), "acme/median", 8000, 7000),
        _workload("5".rjust(40, "e"), "acme/open", 45000, 45000),
        _workload("6".rjust(40, "f"), "acme/tree", 41000, 41000),
        _workload("7".rjust(40, "0"), "acme/codes", 19000, 18000),
        _workload("8".rjust(40, "1"), "acme/sniff", 23000, 20),
        _workload("9".rjust(40, "2"), "acme/json", 19000, 40),
        _workload("a".rjust(40, "3"), "acme/truncated", 44000, 44000, False),
        _workload("b".rjust(40, "4"), "acme/mid", 9000, 8000),
        _workload("c".rjust(40, "5"), "acme/other", 7000, 1000),
        _workload("d".rjust(40, "6"), "acme/extra", 12000, 6000),
    ]
    selected = select_heavy_benchmark(rows)
    assert len(selected) == 12
    assert len({item["tree_sha"] for item in selected}) == 12
    roles = {item["role"]: item["donor"] for item in selected}
    assert roles["ggen_slow"] == "seanchatmangpt/ggen"
    assert roles["corese_fast"] == "handedemir/corese"
    assert roles["small_end"] == "acme/small-heavy"
    assert roles["largest_strong_1"] == "acme/open"
    assert roles["sniff_heavy_1"] == "acme/sniff"
    assert roles["truncated_large"] == "acme/truncated"


def test_batch_a_aggregate_keeps_the_heavy_tail_out_of_the_denominator():
    summary = aggregate_batch_a(
        [
            {
                "extracted": True,
                "enumeration_complete": True,
                "counts_partial": False,
                "partial_reasons": [],
                "node_shapes": 2,
                "property_shapes": 1,
                "distinct_shapes": 3,
                "constraints": {"http://www.w3.org/ns/shacl#minCount": 4},
                "targets": {"http://www.w3.org/ns/shacl#targetClass": 2},
                "skipped": {"skipped_too_large": 1},
                "files_parsed": 5,
                "parse_failures": 1,
                "seconds": 10,
                "structural_sparql": False,
                "sparql_links": 0,
                "sparql_constraint_nodes": 0,
                "sparql_only_verification": False,
                "sparql_only_content": False,
                "sparql_targets": 0,
                "repository_count": 1,
            }
        ],
        batch_a_trees=2,
        heavy_trees=107,
        unresolved_trees=10,
        unresolved_also_heavy=3,
        population_trees=6343,
    )
    assert summary["partial_corpus"] is True
    assert summary["shape_totals"] == {"node_shapes": 2, "property_shapes": 1, "distinct_shapes": 3}
    assert summary["excluded_heavy_trees"] == 107
    assert summary["parsed_files"] == 5
    assert summary["skipped_oversized_files"] == 1
    min_count = next(row for row in summary["constraints"] if row["predicate"] == "minCount")
    assert min_count["percent_of_batch_a_trees"] == 100.0
    assert "percent_of_pilot_trees" not in min_count


def test_heavy_estimate_uses_the_rate_of_the_same_workload_type():
    heavy = [
        {"candidate_files": 10000, "strong_candidates": 9000, "listing_complete": True},
        {"candidate_files": 10000, "strong_candidates": 10, "listing_complete": True},
    ]
    benchmark = [
        {"requested_strategy": "tarball", "workload_type": "strong_rdf", "candidate_files": 100, "seconds": 10, "enumeration_complete": True},
        {"requested_strategy": "strong_first", "workload_type": "strong_rdf", "candidate_files": 100, "seconds": 100, "enumeration_complete": True},
        {"requested_strategy": "current", "workload_type": "strong_rdf", "candidate_files": 100, "seconds": 12, "enumeration_complete": True},
        {"requested_strategy": "tarball", "workload_type": "sniff_heavy", "candidate_files": 100, "seconds": 50, "enumeration_complete": True},
        {"requested_strategy": "strong_first", "workload_type": "sniff_heavy", "candidate_files": 100, "seconds": 5, "enumeration_complete": True},
        {"requested_strategy": "current", "workload_type": "sniff_heavy", "candidate_files": 100, "seconds": 40, "enumeration_complete": True},
    ]
    estimate = estimate_heavy_runtime(heavy, benchmark)
    assert estimate["workload_types"]["strong_rdf"]["fastest"] == "tarball"
    assert estimate["workload_types"]["sniff_heavy"]["fastest"] == "strong_first"
    assert estimate["estimated_hours_fastest_by_workload_type"] < estimate["estimated_hours_if_one_strategy"]["strong_first"]


def test_access_strategies_stay_complete_and_choose_different_fetches():
    plain = b"@prefix ex: <http://example.org/> .\nex:a ex:b ex:c .\n"
    files = {"a.ttl": plain, "b.ttl": _shape("Only"), "c.json": b'{"@context": {}}'}

    class Archive(_Source):
        def __init__(self):
            super().__init__(files)
            self.tarball_calls = 0
            self.raw_calls = 0

        def get_file(self, full_name, commit_sha, path):
            self.raw_calls += 1
            return super().get_file(full_name, commit_sha, path)

        def list_tarball(self, full_name, commit_sha):
            self.tarball_calls += 1
            return [
                {"path": path, "size": len(payload), "content": payload, "too_large": False}
                for path, payload in self.files.items()
            ]

    row = {
        "repository_id": "1",
        "repository_full_name": "acme/shapes",
        "default_branch": "main",
        "discovered_by": "code:shacl_class_name:primary",
        "is_fork": "false",
        "archived": "false",
    }
    config = dict(ROOT_CONFIG)
    config["tarball_candidate_threshold"] = 1
    forced = Archive()
    forced_row, _, _ = inspect_repository(
        row, source=forced, vocab=VOCAB, config=config, clock=CLOCK, census=True, census_strategy="tarball"
    )
    raw = Archive()
    raw_row, _, _ = inspect_repository(
        row, source=raw, vocab=VOCAB, config=config, clock=CLOCK, census=True, census_strategy="strong_first"
    )
    assert forced.tarball_calls == 1
    assert forced.raw_calls == 0
    assert forced_row["node_shape_count"] == 1
    assert raw.tarball_calls == 0
    assert raw.raw_calls >= 2
    assert raw_row["node_shape_count"] == 1

    class Empty(Archive):
        def list_tarball(self, full_name, commit_sha):
            self.tarball_calls += 1
            return []

    empty = Empty()
    empty_row, _, _ = inspect_repository(
        row, source=empty, vocab=VOCAB, config=config, clock=CLOCK, census=True, census_strategy="tarball"
    )
    assert empty.tarball_calls == 1
    assert empty_row["_strategy_fallback"] == "tarball_empty"
    assert empty_row["rdf_files_parsed"] == 3
