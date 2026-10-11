# Batch A coverage limitations

Batch A counts are unchanged. The parser extension map is unchanged. These files were excluded by that map or by parser failures. They are not counted as shapes.

The 3,449 parse-failed files that contain `http://www.w3.org/ns/shacl#` split as follows.

| Category | Files | Trees | Treatment |
|---|---:|---:|---|
| Malformed source under the selected syntax | 3,233 | included in the 817 | Limitation. The parser selected the syntax implied by the extension or sniff, and rejected the text. RDF was not repaired. |
| Context resolution | 81 | included in the 817 | Operational limitation. 61 local relative-context misses and 20 HTTP errors while resolving a context. Not treated as valid RDF. |
| RDFLib crash | 43 | included in the 817 | Parser limitation. AttributeError, TypeError, IndexError, and six further internal failures. Not treated as valid RDF. |
| Valid Turtle in an RDF/XML extension | 91 | 76 | Syntax-detection limitation. Documented below. |
| HTML containing the namespace string | 1 | 1 | Not an RDF graph. |

817 is the number of trees with at least one of these files. A tree can fall in more than one category.

## Valid Turtle with an RDF/XML extension

91 files in 76 trees are valid Turtle. The current map sends `.rdf`, `.owl`, and `.xml` to the RDF/XML parser only. RDF/XML parsing failed. The same bytes were accepted by the Turtle parser. A path reconstruction found 63 `.owl`, 19 `.rdf`, and 4 `.xml` files; five of the 91 paths were not recovered by that second check, so the classification count stands. No fallback was added. These files stay out of the Batch A counts. The 76 trees were not rerun.

76 / 6,229 Batch A trees is 1.2%.

## What this is not

These exclusions are not an extraction bug in the execution of the current rules. The malformed-source files are broken input. The 91 files are a narrow syntax-detection policy. The 81 and 43 are unresolved parser or operational events.
