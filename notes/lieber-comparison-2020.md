# Comparison with Lieber et al. 2020

This note compares the final extraction with Lieber et al., “Statistics about Data Shape Use in RDF Data” (`paper584.pdf`). It does not rewrite those results and it does not treat the supplied Montolo archive as the poster’s corpus.

Our numbers are the extracted unique-tree aggregate in `data/results/statistics_final/summary.json`. The denominator for tree percentages is 6,336 extracted trees. The intended verified population is 6,343. Seven trees were not extracted. Counts are conservative lower bounds where `counts_partial` applies (2,724 of 6,336 trees). Shape totals are not multiplied by the number of repositories that share a tree.

## Directly comparable metrics

Comparable here means the poster states the quantity in prose and our pipeline computes a related quantity. The sampling frames are still different. No growth rate is computed.

| Quantity | Lieber et al. 2020 | This corpus | What is actually the same |
|---|---|---|---|
| Selection | GitHub search for the term “SHACL”, then manual exclusion of simple examples, plus a few well-known shapes | Defined multi-query discovery, then structural verification on the canonical `http` namespace | Both start from GitHub. The filters are not the same. |
| Scale of the analyzed set | 13 projects | 6,336 unique Git trees extracted; 7,569 verified repositories, of which 7,562 sit on an extracted tree | “Project” is not a Git tree and not a verified repository. |
| NodeShapes | 1,978 | 920,580 explicit `rdf:type sh:NodeShape` subjects, summed across trees | Both are NodeShape counts. Their counting rule is not defined. Ours unions URI subjects within a tree, counts blank nodes per file, and does not repair unparsed files. |
| Most used constraint families by occurrence | Property, cardinality, class, and datatype constraints, in the manually curated shapes, excluding OSLO and schema.org SHACL | By triple occurrences on 6,336 trees: `sh:property` 4,584,550; `sh:class` 1,755,137; `sh:maxCount` 1,605,102; `sh:datatype` 1,475,255; `sh:minCount` 1,267,604 | Same families sit at the top. The poster’s occurrence figure excludes two generated projects. Ours does not exclude generated shapes. Exact bar heights are not printed in the poster prose. |
| Prevalence versus occurrence | Both are reported: total occurrences, and percent of projects using a type at least once | Both are reported: triple occurrences, and percent of the 6,336 trees | The two measures exist on both sides. A project percentage is not a tree percentage. |
| `sh:or` prevalence | More than 75% of the analyzed repositories | 2,992 / 6,336 trees, 47.22%. Repository view: 3,591 / 7,569, 47.44% | Both are “at least once” prevalences. One schema.org project can move a 13-project percentage a lot. The poster says schema.org accounts for much of this use. |
| Literal patterns | `sh:pattern` finds some use, unlike literal restrictions in their earlier ontology study | `sh:pattern` occurs in 46.28% of extracted trees (185,857 triples) | Qualitative only. No poster percentage is printed for `sh:pattern`. |
| Coverage of the constraint vocabulary | “All constraint types are used,” with many below 20 occurrences or below 20% of projects and therefore omitted from the figures | All 33 vocabulary-derived Core parameters occur at least once. The rarest, `sh:qualifiedValueShapesDisjoint`, is in 521 trees (8.22%) | “All used” is directionally similar. The parameter lists are not shown to be identical. |

## Not comparable

| Metric | Why it is not a comparison |
|---|---|
| Percentage growth from 1,978 to 920,580 NodeShapes | The poster’s NodeShape definition is unpublished. The samples are a 13-project manual set and a 6,336-tree verified extract. Dividing them is not growth. |
| 13 projects versus 6,336 trees, as a coverage rate of GitHub | Their 13 is not a sample from our search, and our 6,336 is not “all SHACL on GitHub”. |
| Repository percentages versus their project percentages | Our 7,569 repositories are verified GitHub repositories. Several can share one tree. Their project is a manually merged shape collection. |
| Occurrence totals in their Figure 1 versus our triple sums | Figure 1 excludes OSLO and schema.org and hides types with fewer than 20 occurrences. Our sums include every extracted tree and every parsed triple. |
| Value-range constraints are “barely found” | The poster gives no percentage. Our `sh:maxInclusive` appears in 34.25% of extracted trees and `sh:minExclusive` in 13.92%. That may or may not contradict the qualitative sentence. It is not the same statistic. |
| The supplied archive’s 2,250 explicit `sh:NodeShape` triples, or its 21 dumps and 19 curated config entries | `notes/lieber-reproduction-notes.md` records that these counts do not reproduce the poster’s 13 and 1,978. They are not used as Lieber’s baseline. |
| SHACL-SPARQL | The poster studies SHACL Core constraint types. It does not report structural SPARQL. Our 35.62% of trees (2,257 / 6,336) has no poster counterpart. |
| Shape concentration | The poster does not report how NodeShapes are distributed across projects. Our top 5% of trees hold 81.1% of observed distinct shapes. There is nothing to align it with. |
| PropertyShape and distinct-shape totals | Not reported in the poster. |

## Observations

1. The poster’s analyzed set is 13 manually kept projects and 1,978 NodeShapes. The extracted corpus is 6,336 unique trees and 920,580 explicit NodeShapes. These numbers can be shown side by side. They are not a growth rate.
2. Both studies find the same frequent families: property, cardinality, class, and datatype. That agreement survives the change of frame.
3. The poster’s `sh:or` prevalence, above 75% of 13 projects, does not reappear as a tree prevalence. Here `sh:or` is in 47.22% of extracted trees. The poster itself attributes much of that use to schema.org. A 13-project percentage is sensitive to one project.
4. Every one of the 33 Core parameters occurs in at least 8% of extracted trees. The poster’s figures drop types below 20 occurrences or below 20% of projects. Rarity in that sample is not absence in this one.
5. `sh:pattern` is common here (46.28% of trees). The poster only says it finds some use. The direction agrees. The magnitude is new.
6. `sh:targetClass` is in 92.31% of extracted trees. The poster lists target-class among the frequent types in Figure 1, without a printed percentage.
7. Structural SHACL-SPARQL is in 2,257 extracted trees (35.62%). The 2020 poster cannot confirm or contradict this.
8. Observed distinct shapes are heavily concentrated: 49.6% in the top 1% of trees, 81.1% in the top 5%, 90.3% in the top 10%. Median distinct shapes per tree is 16. The mean is 371. The poster has no distribution to compare.
9. Seven verified trees and one archive that exceeded 200 MiB are outside a complete enumeration. 2,724 extracted trees have partial counts. The totals are lower bounds. The poster also dropped Turtle files that failed to parse, but only inside a 13-project set.
10. The Montolo zip in this repository does not reproduce 13 projects or 1,978 NodeShapes. A comparison that swaps in 2,250 archive triples would be a third, unofficial baseline.

## Worth discussing

The result that is new relative to the poster, and not just a larger version of the same ranking, is the concentration: most trees are small, and a few trees hold most explicit shapes. The result that agrees with the poster, on a much larger and differently selected corpus, is the constraint profile of property, cardinality, class, and datatype. The result the poster does not contain is that structural SPARQL is present in more than a third of extracted trees. The `sh:or` percentage should not be described as a decline. The NodeShape totals should not be described as growth.
