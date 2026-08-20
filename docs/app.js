(() => {
  "use strict";

  const palette = {
    corpus: "#2f5d8a",
    boundary: "#516275",
    retrieval: "#1c7c75",
    evidence: "#c36b1d",
    answer: "#76558f",
    seal: "#324a6d",
    lessons: "#9a5a20"
  };

  const beginnerStages = [
    {
      title: "Gather the paper collection",
      analogy: "First build the library before asking it questions.",
      enters: "Released paper metadata and official source URLs.",
      happens: "TRACE collects the canonical identity for every paper in the pool and records where its PDF came from.",
      exits: "A 27,487-paper catalogue with stable paper IDs.",
      technical: "The metadata pool is the identity authority. Retrieval may discover a paper in many ways, but the submitted paper identity must come from this released catalogue."
    },
    {
      title: "Download and verify the PDFs",
      analogy: "Keep the original books, not screenshots or somebody else's notes.",
      enters: "Canonical paper IDs and source URLs.",
      happens: "The corpus builder stores each source PDF and records its byte hash, source URL, and page order.",
      exits: "An immutable local or cloud PDF snapshot.",
      technical: "A source manifest binds paper_id, URL, SHA-256, parser settings, and page count. Identity or hash failures stop that source from silently entering the corpus."
    },
    {
      title: "Turn PDFs into labeled source objects",
      analogy: "Split each book into numbered pages, paragraphs, tables, figures, equations, and references.",
      enters: "Verified PDF bytes.",
      happens: "PyMuPDF extracts page text and preserves printed object labels such as Table 3, Figure 2, Equation 6, and citation numbers.",
      exits: "One-based, page-scoped parsed paper artifacts.",
      technical: "Parsing retains sections, captions, acronyms, bibliography entries, page numbers, and visible object identifiers instead of flattening a paper into one text blob."
    },
    {
      title: "Build four complementary indexes",
      analogy: "Give the library separate catalogues for paragraphs, diagrams, nicknames, and relationships.",
      enters: "Parsed pages plus canonical metadata.",
      happens: "TRACE builds passage, object, alias, and relation indexes. Each catalogue answers a different kind of lookup.",
      exits: "Source-linked BM25 stores for text, objects, names, and citation/comparison relations.",
      technical: "The release snapshot contains page-scoped passages, table/figure captions, method-title-acronym aliases, and resolved relation edges. Every record retains its paper identity."
    },
    {
      title: "Create semantic paper embeddings",
      analogy: "Add a meaning-based shelf so similar ideas can meet even when their words differ.",
      enters: "Every paper's title and abstract.",
      happens: "BAAI/bge-small-en-v1.5 converts each paper into a 384-dimensional vector aligned with the metadata pool.",
      exits: "A reusable 27,487-row dense cache and matching ID list.",
      technical: "Dense search complements exact title, alias, passage, object, and relation routes. Strict preflight verifies the vector shape, model identity, and paper-ID alignment."
    },
    {
      title: "Read the question as a contract",
      analogy: "Before searching, understand exactly what the assignment asks you to hand in.",
      enters: "Question text, options, answer type, and table schema.",
      happens: "A temperature-zero planner identifies targets, roles, constraints, expected paper count, and useful retrieval routes.",
      exits: "A validated plan with target groups and an answer scaffold.",
      technical: "Gemini 2.5 Flash handles planning and textual reasoning. Invalid planner output degrades to a safe deterministic plan; the original organizer schema remains authoritative."
    },
    {
      title: "Retrieve separately for every target",
      analogy: "If an assignment asks about two people, keep two folders so one famous person cannot crowd out the other.",
      enters: "The plan, four sparse indexes, and dense cache.",
      happens: "Name, property, target-property, citation/baseline, object, and dense routes gather candidates while preserving which target produced each signal.",
      exits: "A candidate ledger containing support, route, rank, score, target group, and role.",
      technical: "Candidates merge by canonical paper ID only after their route provenance is retained. Hard venue/year filters, target coverage, answer-bearing rescue, cardinality, and the five-paper cap follow."
    },
    {
      title: "Choose the canonical paper set",
      analogy: "Pick the smallest set of books that covers every required part of the assignment.",
      enters: "The candidate ledger and requested paper count.",
      happens: "The selector covers the required targets first, then uses remaining capacity only for corroborated answer-bearing papers.",
      exits: "The submitted paper set P.",
      technical: "Selection enforces metadata identity, target coverage, explicit or planned cardinality, and the submission paper cap. This is the paper precision/recall boundary."
    },
    {
      title: "Locate evidence independently",
      analogy: "Put a sticky note on the exact page, table, figure, equation, or reference that supports the answer.",
      enters: "Question, selected papers P, and the exact parsed PDF pages.",
      happens: "Whole-paper localization proposes evidence; deterministic checks verify the page, object ID, quote, source type, and confidence before ranking and deduplication.",
      exits: "At most five scorer-shaped locators E(P).",
      technical: "Evidence is produced independently of answer success. Every locator must belong to P. Gemini 2.5 Flash localizes text; source checks prevent fabricated pages, object labels, and quotes."
    },
    {
      title: "Construct the typed answer",
      analogy: "Fill in the exact form the teacher supplied instead of inventing your own format.",
      enters: "P, rich evidence, parsed pages, options, and the organizer schema.",
      happens: "Multiple choice selects one legal label; tables plan the row meaning before extracting cells; freeform returns the requested text field.",
      exits: "The typed answer A.",
      technical: "Gemini 2.5 Flash handles planning, text grounding, and answers. Gemini 2.5 Pro is used only for visual table and figure reading. Table assembly preserves printed keys, types, nulls, and scorer normalization."
    },
    {
      title: "Close the grounding contract",
      analogy: "Check that every sticky note belongs to a chosen book and every answer fits the form.",
      enters: "P, E(P), A, and the original query ID.",
      happens: "TRACE normalizes the nested record, removes orphan evidence, records semantic fallbacks, and keeps every query present.",
      exits: "One scorer-shaped prediction plus a forensic trace.",
      technical: "Per-record failures are isolated. A validator-safe fallback preserves the query ID, while the trace records plan, retrieval signals, selection decisions, evidence, confidence, and diagnostics."
    },
    {
      title: "Validate, write, and attest",
      analogy: "Run the final checklist, seal the envelope, and record exactly how it was made.",
      enters: "All 71 prediction records, pool IDs, source revision, configuration, and hashes.",
      happens: "The official validator runs before the upload path appears. TRACE writes the trace, atomically exposes predictions.jsonl, then writes the provenance manifest.",
      exits: "A validator-approved prediction file, trace, and hash-bound manifest.",
      technical: "The public CLI is input-only full generation: it has no parent prediction, replay, patch, query allowlist, or answer override. Hosted-model drift can change a rerun, so the manifest makes the exact run observable."
    }
  ];

  const componentGroups = [
    { id: "corpus", title: "Corpus foundation", subtitle: "Offline and deterministic" },
    { id: "boundary", title: "Run boundary", subtitle: "Before paid inference" },
    { id: "retrieval", title: "Plan + paper identity", subtitle: "Per question" },
    { id: "evidence", title: "Evidence identity", subtitle: "Independent branch" },
    { id: "answer", title: "Answer contract", subtitle: "MC, table, freeform" },
    { id: "seal", title: "Seal + provenance", subtitle: "Fail closed" }
  ];

  const componentNodes = [
    ["metadata", "Released metadata pool", "corpus", "Canonical identity for 27,487 papers.", "Released paper_metadata.jsonl", "paper_id, title, venue, year, source URL", "Loads the organizer's canonical paper namespace.", "Stable paper identities", "Reject missing or duplicate paper IDs.", "Paper identity precision/recall", "data/paper_metadata.jsonl"],
    ["pdfs", "Immutable PDF snapshot", "corpus", "Exact paper bytes used for parsing and vision.", "Canonical IDs and source URLs", "Local directory or configured GCS snapshot", "Fetches and caches original PDF bytes.", "Source bytes plus page order", "Refuse unavailable or mismatched sources in strict runs.", "Evidence identity", "src/littraceqa/localize/fetch_corpus.py"],
    ["parser", "Level-1 PDF parser", "corpus", "Converts source bytes into page-scoped structure.", "Verified PDF bytes", "One-based pages, text, sections, captions, references", "Uses PyMuPDF and preserves visible object IDs.", "ParsedPdf artifacts", "Page/object identities must remain source-visible.", "Evidence page/object recall", "src/littraceqa/corpus/parse.py"],
    ["passages", "Passage index", "corpus", "Page-scoped BM25 text chunks.", "Parsed page text", "1,409,382 indexed records in the release snapshot", "Indexes source text with paper and page identity.", "Sparse passage retrieval", "Never detach text from paper/page provenance.", "Paper and evidence recall", "src/littraceqa/corpus/indexes/passages.py"],
    ["objects", "Object index", "corpus", "Tables, figures, captions, and visible IDs.", "Parsed source objects", "500,928 records in the release snapshot", "Indexes printed table/figure objects and captions.", "Object-aware retrieval", "Object ID must be detected on its source page.", "Evidence object recall", "src/littraceqa/corpus/indexes/objects.py"],
    ["aliases", "Alias index", "corpus", "Method, title, and acronym mappings.", "Metadata and parsed prose", "3,711,590 alias records in the release snapshot", "Resolves question names back to canonical titles.", "Name-route candidates", "Ambiguous aliases retain alternatives; they do not invent IDs.", "Paper precision", "src/littraceqa/corpus/indexes/aliases.py"],
    ["relations", "Relation index", "corpus", "Citation and comparison relationships.", "Reference lists and comparison language", "492,892 resolved edges in the release snapshot", "Connects anchors to papers that cite or compare against them.", "Citation/baseline candidates", "Edges remain bound to source paper and relation evidence.", "Paper recall on relational questions", "src/littraceqa/corpus/indexes/relations.py"],
    ["dense", "Dense embedding cache", "corpus", "Meaning-based title-plus-abstract retrieval.", "27,487 metadata rows", "BAAI/bge-small-en-v1.5 · 384 dimensions", "Embeds every paper once and aligns vectors to paper IDs.", "Cosine-search cache plus ID list", "Preflight verifies model, shape, count, and ID alignment.", "Paper recall", "src/littraceqa/retrieval/embedder_local.py"],

    ["cli", "Input-only submit CLI", "boundary", "The only public production entry point.", "Config, released inputs, pool, indexes, dense cache, source revision", "No replay or parent-output arguments", "Parses explicit paths and activates strict release checks.", "Attested run request", "Refuses forbidden parent/replay surfaces.", "Reproducibility boundary", "src/littraceqa/experiments/submit.py"],
    ["preflight", "Strict preflight", "boundary", "Stops bad runs before any model call.", "Run request plus all local assets", "Expected 71 IDs, 50 MC, 21 tables; hashes, sizes, counts, dense alignment", "Verifies the complete release profile.", "Attested asset summary", "Any mismatch aborts before Gemini spend or prediction exposure.", "Prevents silent corpus/config drift", "src/littraceqa/experiments/submit.py"],
    ["factory", "Runner factory", "boundary", "Composes the configured production services.", "Attested assets and configuration", "Flash text client, Pro vision client, retrieval, selection, localization, answer strategies", "Builds one shared runner without reading prior outputs.", "Production runner", "Unknown strategies and incompatible parameters raise.", "Whole-pipeline consistency", "src/littraceqa/experiments/new_runner.py"],
    ["driver", "Bounded parallel driver", "boundary", "Runs questions concurrently but returns input order.", "71 parsed inputs and shared runner", "Up to 8 workers; per-record timeout and isolation", "Executes run_one independently for each query.", "71 ordered results", "A failed record becomes only that query's traced fallback.", "Completeness and operational robustness", "src/littraceqa/experiments/driver.py"],
    ["input", "Defensive input parser", "boundary", "Normalizes organizer input without changing its contract.", "One released input record", "Question, options, answer types, table schema", "Normalizes option shapes and reads the authoritative schema.", "InputRecord", "Missing identity or malformed required structure fails safely.", "Answer-schema correctness", "src/littraceqa/pipeline/input.py"],

    ["planner", "Structured question planner", "retrieval", "Turns prose into explicit search and answer targets.", "InputRecord", "Criterion, venue/year, methods, routes, target roles, multiplicity", "Uses Gemini 2.5 Flash at temperature zero and validates the structured response.", "Plan or safe degraded plan", "Organizer answer type/schema always remains authoritative.", "Target coverage and answer shape", "src/littraceqa/pipeline/planner.py"],
    ["name-route", "Name + alias route", "retrieval", "Finds papers named directly or by acronym.", "Plan targets and alias store", "Exact/fuzzy title and alias matches with target-local rank", "Resolves names while retaining ambiguity and provenance.", "Name candidates", "No alias can manufacture a paper outside the pool.", "Paper precision", "src/littraceqa/retrieval/strategy.py"],
    ["property-route", "Property routes", "retrieval", "Finds papers by the requested behavior or measurement.", "Target, criterion, passage/object stores", "Property and target-property BM25; object route is gated/instrumented", "Searches source-linked passages and captions at configured depth.", "Property candidates plus support", "Support remains attached to route and target group.", "Paper recall", "src/littraceqa/retrieval/strategy.py"],
    ["relation-route", "Citation + baseline routes", "retrieval", "Follows papers that cite or compare against an anchor.", "Plan anchors and relation store", "Resolved citation/comparison edges", "Traverses source-linked relations for relational questions.", "Relation candidates", "Unresolved relations cannot silently become canonical papers.", "Paper recall on citation questions", "src/littraceqa/retrieval/strategy.py"],
    ["dense-route", "Dense route", "retrieval", "Recovers semantically related papers when words differ.", "Question/target text and aligned dense cache", "27k cosine search", "Ranks title-plus-abstract vectors as a complementary route.", "Dense candidates", "Dense presence alone never bypasses downstream gates.", "Paper recall", "src/littraceqa/retrieval/dense.py"],
    ["ledger", "Candidate ledger", "retrieval", "Merges identities without losing why a paper appeared.", "Candidates from every route", "Paper ID, support excerpts, route, rank, score, target, role", "Deduplicates by canonical paper ID and accumulates provenance.", "Per-paper retrieval dossiers", "No global rank collapse before target coverage.", "Paper precision/recall", "src/littraceqa/experiments/retrieval_dossier.py"],
    ["scope", "Hard scope filter", "retrieval", "Enforces explicit venue and year constraints.", "Candidate dossiers and planner scope", "Venue/year metadata", "Filters only when the constraint is explicit and recall-safe.", "Scope-valid candidates", "The filter never empties the entire set silently.", "Paper precision", "src/littraceqa/retrieval/selection_filter.py"],
    ["coverage", "Target-coverage floor", "retrieval", "Protects each requested target from being crowded out.", "Scoped dossiers grouped by target", "Corroborated support; name-only anchors excluded", "Selects at least one supported candidate per target before global ranking.", "Coverage floor", "Weak name-only evidence cannot claim target coverage.", "Paper recall", "src/littraceqa/retrieval/target_selection.py"],
    ["rescue", "Answer-bearing rescue", "retrieval", "Adds a demonstrably answer-bearing paper when a thin set misses it.", "Coverage floor and remaining dossiers", "Real passage/object coverage and strong margin over baseline", "Applies only to configured single-paper or constraint-only table shapes.", "Rescued candidate or no-op", "Ambiguous or weak support abstains.", "Paper recall with guarded precision", "src/littraceqa/retrieval/answer_bearing_selection.py"],
    ["cardinality", "Cardinality policy", "retrieval", "Decides how many papers may be submitted.", "Question text, plan, candidates", "Explicit question count, then planner count, then single/multi; max 5", "Applies the most authoritative count available.", "Ranked and capped paper set", "The configured maximum is a safety ceiling, not a target.", "Paper precision/recall", "src/littraceqa/retrieval/cardinality.py"],
    ["paper-set", "Canonical paper set P", "retrieval", "The paper identities visible to the scorer.", "Ranked dossiers and cardinality decision", "Canonical pool IDs only", "Serializes the selected paper set.", "P", "Every later evidence locator must close over P.", "Paper macro precision/recall/F1", "src/littraceqa/paperset/selector.py"],

    ["pdf-fetch", "Selected-PDF fetcher", "evidence", "Reads exact bytes only for selected papers.", "P and configured corpus snapshot", "Local or GCS source; no live OpenReview fetch in production", "Loads the canonical PDF bytes for each selected paper.", "Raw PDF bytes", "Missing selected sources become traced failures.", "Evidence availability", "src/littraceqa/localize/fetch_corpus.py"],
    ["page-cache", "Raw + parsed page cache", "evidence", "Ensures localization and vision see the same source.", "Selected PDF bytes", "One-based ParsedPdf plus original bytes", "Reuses deterministic parsing within the run.", "Shared pages and bytes", "No cross-paper or alternate-byte substitution.", "Evidence consistency", "src/littraceqa/localize/parse_pymupdf.py"],
    ["localizer", "Whole-paper evidence localizer", "evidence", "Proposes answer-bearing source locations once per selected paper.", "Question, paper identity, parsed pages", "Gemini 2.5 Flash · page text capped per config", "Returns proposed page, type, object, quote, and confidence.", "Rich LocatedEvidence candidates", "Proposals are not submitted until deterministically grounded.", "Evidence recall", "src/littraceqa/localize/localizer.py"],
    ["grounding", "No-fabrication grounding", "evidence", "Verifies every proposed locator against source bytes.", "Localized proposals and ParsedPdf", "Page exists; visible/detected object ID; quote grounded; finite confidence", "Normalizes and rejects unattested proposals.", "Valid rich evidence", "Invalid page, object, quote, or confidence is dropped.", "Evidence precision", "src/littraceqa/localize/evidence_contract.py"],
    ["evidence-rank", "Evidence ranker", "evidence", "Prioritizes the strongest distinct locators.", "Valid rich evidence", "Confidence, question-to-quote BM25, stable source order", "Ranks without consulting the final answer result.", "Ordered evidence candidates", "Stable ordering breaks ties reproducibly.", "Evidence precision/recall", "src/littraceqa/experiments/evidence_rank.py"],
    ["evidence-shape", "Official evidence shaper", "evidence", "Converts rich evidence to scorer-visible identities.", "Ordered evidence candidates and P", "Paper, source type, page, normalized object/citation ID", "Deduplicates by the packaged evaluator key and keeps at most five.", "Submitted E(P)", "Orphans and duplicate coarse keys are removed.", "Evidence macro precision/recall/F1", "src/littraceqa/experiments/evidence_rank.py"],

    ["context", "AnswerContext", "answer", "Carries the grounded material into answer strategies.", "Question, P, rich evidence, parsed pages, titles, PDF bytes, options/schema", "One immutable context for the selected record", "Packages source material without changing scorer evidence.", "Strategy input context", "Answer strategies cannot silently change P or submitted E(P).", "Answer correctness", "src/littraceqa/answer/interfaces.py"],
    ["mc", "Multiple-choice strategy", "answer", "Returns exactly one legal organizer label.", "AnswerContext and options", "Grounded value-to-option match; Flash fallback; first-label safety fallback", "Prefers unique grounded matching, then a temperature-zero model choice.", "MC label", "Illegal/blank output is replaced and traced as semantic fallback.", "MC accuracy", "src/littraceqa/answer/multiple_choice.py"],
    ["table-plan", "Table observation planner", "answer", "Defines what one row means before extracting values.", "Question and organizer table_schema", "Paper/entity row axis, key columns, value types, requested row hints", "Plans the schema contract before looking for cell values.", "TablePlan", "Unknown columns and incompatible row shapes are rejected.", "Table row and cell identity", "src/littraceqa/answer/table_plan.py"],
    ["table-visual", "Visual table/figure reader", "answer", "Reads values whose layout carries meaning.", "TablePlan, source locator, retained PDF bytes", "Gemini 2.5 Pro on rendered PNGs", "Renders cited pages and reads schema-shaped candidate rows.", "Visual candidate rows", "Only source-bound pages/objects may be rendered.", "Table cell recall", "src/littraceqa/answer/vision.py"],
    ["table-text", "Text/equation/reference reader", "answer", "Extracts non-visual table facts from grounded source text.", "TablePlan, parsed pages, rich evidence", "Verbatim spans, equations, citations, deterministic bibliography parsing", "Produces typed candidate rows from source-linked facts.", "Non-visual candidate rows", "Unsupported or ill-typed values become null/abstain, not invented cells.", "Table cell precision", "src/littraceqa/answer/table_extract.py"],
    ["table-assemble", "Contract-aware table assembler", "answer", "Builds the final row set accepted by the scorer.", "TablePlan plus visual/text candidate rows", "Canonical paper titles, normalized row keys, typed values, null filtering", "Merges cross-paper cells, filters requested rows, removes all-null and duplicate rows.", "Schema-valid table rows", "Duplicate scorer-normalized keys and non-schema fields are rejected.", "Table row F1 and cell accuracy", "src/littraceqa/answer/table_planned.py"],
    ["freeform", "Freeform strategy", "answer", "Returns grounded text when requested.", "AnswerContext", "Verbatim/grounded extraction with Flash", "Produces the organizer's freeform field.", "Freeform answer", "Blank or ungrounded output degrades safely.", "Freeform exact match", "src/littraceqa/answer/freeform.py"],

    ["record", "Scorer-shaped record builder", "seal", "Combines independent paper, evidence, and answer identities.", "query_id, P, E(P), A", "Official nested submission shape", "Normalizes the record and records semantic fallbacks.", "One prediction line", "Never drops a query ID; orphan evidence is forbidden.", "All primary metrics", "src/littraceqa/submission.py"],
    ["fallback", "Per-record failure isolation", "seal", "Prevents one bad question from aborting the batch.", "Exception, timeout, or invalid component output", "Validator-safe fallback for the same query ID", "Captures failure_reason and continues other records.", "Fallback line plus trace", "Fallback never borrows another query's identity.", "Completeness; fallback may score poorly", "src/littraceqa/experiments/driver.py"],
    ["order", "Input-order reassembly", "seal", "Restores the exact released question order after parallel work.", "71 completed records", "Original input positions", "Reorders results deterministically.", "Complete ordered batch", "Duplicate or missing query IDs are rejected.", "Submission completeness", "src/littraceqa/experiments/driver.py"],
    ["validator", "Packaged official validator", "seal", "Checks the complete artifact before exposure.", "Released inputs, predictions, paper pool", "Nested schema, all IDs, pool membership, evidence closure", "Runs the organizer-derived validator packaged in v1.0.1.", "Approved batch or exception", "Any validation failure leaves no prediction at the requested upload path.", "Prevents zero-score structural errors", "src/littraceqa/_vendor/validate_submission.py"],
    ["trace", "Forensic trace writer", "seal", "Records how every record was produced.", "Per-record plan, retrieval, selection, evidence, answer diagnostics", "JSONL trace written after validation", "Serializes decisions and failure reasons.", "traces.jsonl", "Secrets are not part of trace fields.", "Auditability", "src/littraceqa/experiments/submit.py"],
    ["atomic", "Atomic prediction writer", "seal", "Exposes a complete file or no file.", "Validator-approved 71-record batch", "Temporary sibling file plus os.replace", "Writes UTF-8 JSONL and atomically renames it.", "predictions.jsonl", "Schema, duplicates, and missing IDs are checked again before writing.", "Submission integrity", "src/littraceqa/submission.py"],
    ["manifest", "Provenance manifest", "seal", "Attests the exact run after the prediction exists.", "Source revision, config, inputs, pool, output, trace, preflight, summary", "SHA-256 hashes and active parameters", "Writes the generation_mode=full manifest last.", "manifest.json", "Parent prediction/trace fields are null in the public full-generation path.", "Reproducibility and drift detection", "src/littraceqa/experiments/submit.py"]
  ].map(([id, title, group, summary, consumes, state, logic, emits, gate, score, module]) => ({ id, title, group, summary, consumes, state, logic, emits, gate, score, module }));

  const componentLinks = [
    ["metadata", "pdfs"], ["pdfs", "parser"], ["parser", "passages"], ["parser", "objects"], ["parser", "aliases"], ["parser", "relations"], ["metadata", "dense"],
    ["cli", "preflight"], ["passages", "preflight"], ["objects", "preflight"], ["aliases", "preflight"], ["relations", "preflight"], ["dense", "preflight"], ["preflight", "factory"], ["factory", "driver"], ["driver", "input"],
    ["input", "planner"], ["planner", "name-route"], ["planner", "property-route"], ["planner", "relation-route"], ["planner", "dense-route"], ["aliases", "name-route"], ["passages", "property-route"], ["objects", "property-route"], ["relations", "relation-route"], ["dense", "dense-route"],
    ["name-route", "ledger"], ["property-route", "ledger"], ["relation-route", "ledger"], ["dense-route", "ledger"], ["ledger", "scope"], ["scope", "coverage"], ["coverage", "rescue"], ["rescue", "cardinality"], ["cardinality", "paper-set"],
    ["paper-set", "pdf-fetch"], ["pdf-fetch", "page-cache"], ["page-cache", "localizer"], ["localizer", "grounding"], ["grounding", "evidence-rank"], ["evidence-rank", "evidence-shape"],
    ["paper-set", "context"], ["page-cache", "context"], ["grounding", "context"], ["context", "mc"], ["context", "table-plan"], ["context", "freeform"], ["table-plan", "table-visual"], ["table-plan", "table-text"], ["table-visual", "table-assemble"], ["table-text", "table-assemble"],
    ["paper-set", "record"], ["evidence-shape", "record"], ["mc", "record"], ["table-assemble", "record"], ["freeform", "record"], ["record", "order"], ["fallback", "order"], ["driver", "fallback"], ["order", "validator"], ["validator", "trace"], ["trace", "atomic"], ["atomic", "manifest"]
  ].map(([source, target]) => ({ source, target }));

  const modelAssistedNodes = new Set(["planner", "name-route", "localizer", "mc", "table-plan", "table-visual", "table-text", "freeform"]);

  const sequenceActors = ["User", "Submit CLI", "Preflight", "Planner", "Retrieval", "Selector", "PDF + evidence", "Answer", "Validator", "Writer"];
  const sequenceEvents = [
    ["User", "Submit CLI", "Start a 71-question production run", "Supplies config, released input, metadata pool, four indexes, dense cache, source revision, output, trace, and manifest paths."],
    ["Submit CLI", "Preflight", "Verify the strict release profile", "Checks exact input/pool identity, 71 IDs, 50 MC and 21 table records, index counts and byte sizes, dense model/shape/ID alignment, and source revision."],
    ["Preflight", "Submit CLI", "Return an attested run request", "If any invariant fails, the run stops here: no Gemini call and no prediction file."],
    ["Submit CLI", "Retrieval", "Build the shared production runner", "Loads the persisted stores and dense cache; creates Flash and Pro clients; constructs selection, localization, and answer services."],
    ["Submit CLI", "Planner", "Dispatch one parsed InputRecord", "The bounded driver runs up to eight records concurrently while retaining original input positions."],
    ["Planner", "Planner", "Plan at temperature zero", "Gemini 2.5 Flash returns criterion, scope, named methods, routes, target roles, multiplicity, desired paper count, and answer scaffold. Invalid output degrades safely."],
    ["Planner", "Retrieval", "Send the validated plan", "The organizer's answer type and table schema remain authoritative even if the planner degrades."],
    ["Retrieval", "Retrieval", "Run target-aware retrieval routes", "Name/alias, property, target-property, citation/baseline, gated object, and dense routes preserve route and target provenance."],
    ["Retrieval", "Selector", "Send canonical candidate dossiers", "Duplicate paper IDs merge while retaining support excerpts, route, local rank, score, target group, and role."],
    ["Selector", "Selector", "Apply scope, coverage, rescue, and cardinality", "Hard venue/year constraints are recall-safe; required targets receive a corroborated floor; answer-bearing rescue is gated; the paper cap is five."],
    ["Selector", "PDF + evidence", "Return the selected paper set P", "Only canonical released paper IDs are allowed."],
    ["PDF + evidence", "PDF + evidence", "Load exact selected PDF bytes", "Reads the configured immutable local or GCS snapshot—never a prior prediction and no live OpenReview fetch in production."],
    ["PDF + evidence", "PDF + evidence", "Reuse one-based parsed pages", "Localization and visual answering receive the same source bytes and page ordering."],
    ["PDF + evidence", "PDF + evidence", "Localize evidence once per selected paper", "Flash proposes page, source type, visible object ID, quote, and confidence from the whole parsed paper."],
    ["PDF + evidence", "PDF + evidence", "Ground every proposal", "The page must exist; object IDs must be visible or detected; quotes must occur on the cited page; confidence must be finite and is clamped."],
    ["PDF + evidence", "Answer", "Build an AnswerContext", "Carries question, P, rich evidence, parsed pages, exact PDF bytes, titles, options, and table schema. Submitted evidence remains an independent branch."],
    ["Answer", "Answer", "Choose the registered answer strategy", "MC: grounded unique match then Flash fallback. Table: plan row axis, combine Pro visual reads with text/equation/citation facts, then assemble typed rows. Freeform: grounded text."],
    ["Answer", "Answer", "Serialize the typed answer A", "Illegal MC labels, malformed table rows, or blank outputs trigger traced semantic fallbacks rather than schema violations."],
    ["PDF + evidence", "Validator", "Shape submitted evidence E(P)", "Ranks by confidence, question-to-quote BM25, and stable order; converts to official identity; deduplicates with the packaged evaluator key; caps at five."],
    ["Answer", "Validator", "Build scorer-shaped records", "Each line contains query_id, canonical papers P, independent evidence E(P), and answer A. Orphan evidence is forbidden."],
    ["Validator", "Validator", "Validate all 71 records before exposure", "The packaged organizer validator checks complete IDs, pool membership, nested schema, source types, and evidence-paper closure."],
    ["Validator", "Writer", "Approve the complete batch", "On failure, nothing appears at the requested prediction path."],
    ["Writer", "Writer", "Write traces.jsonl", "The forensic trace is serialized only after the batch passes validation."],
    ["Writer", "Writer", "Atomically expose predictions.jsonl", "A temporary sibling is fully written and then moved into place with os.replace."],
    ["Writer", "Writer", "Write manifest.json last", "Binds source revision, config, inputs, pool, output, trace, preflight, active parameters, and run summary. generation_mode is full and parent fields are null."],
    ["Writer", "User", "Return exact paths, hashes, and summary", "A production run may contain isolated traced fallbacks, but the artifact remains complete and validator-safe."]
  ].map(([from, to, title, detail], index) => ({ from, to, title, detail, index }));

  const lessonNodes = [
    ["freeze", "Freeze a reproducible base", "Record the code revision, configuration, released inputs, source hashes, and exact starting artifact before testing a hypothesis."],
    ["hypothesis", "Choose one failure class", "Form a generic hypothesis such as a wrong locator type or table-coordinate parser defect—not a query-ID-specific answer edit."],
    ["implement", "Implement a general rule", "The rule must derive its behavior from released questions, schemas, clean traces, and immutable source PDFs. No answer constants or query allowlists."],
    ["source", "Verify against primary sources", "Check URL/SHA-bound PDFs, exact pages, visible object IDs, quotes, and negative alternatives. Abstain when the premise cannot be fully supported."],
    ["invariants", "Enforce mutation boundaries", "Prove untouched records and fields remain identical, prevent orphan evidence and duplicate normalized row keys, and bind every component to its parent hash."],
    ["validate", "Run tests and official validation", "Adversarial tests, fresh replay, byte comparison, and the packaged organizer validator must all pass before a candidate exists."],
    ["submit", "Submit only a material candidate", "Use a slot only when the expected gain justifies the uncertainty; upload one exact path and SHA-256."],
    ["learn", "Record organizer feedback", "Treat score and component metrics as experimental evidence. Keep or reject the hypothesis without putting leaderboard feedback into the public inference CLI."]
  ].map(([id, title, detail], index) => ({ id, title, detail, index }));

  let activeView = "beginner";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function selectView(view) {
    if (!["beginner", "components", "sequence", "lessons"].includes(view)) return;
    activeView = view;
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.setAttribute("aria-selected", String(button.dataset.view === view));
    });
    document.querySelectorAll("[data-panel]").forEach((panel) => {
      const selected = panel.dataset.panel === view;
      panel.classList.toggle("is-active", selected);
      panel.hidden = !selected;
    });
    document.querySelectorAll("[data-view-target]").forEach((button) => {
      button.classList.toggle("is-active", button.dataset.viewTarget === view);
    });
    if (view === "components") renderComponentMap();
    if (view === "sequence") renderSequence();
    if (view === "lessons") renderLessons();
  }

  document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));
  document.querySelectorAll("[data-view-target]").forEach((button) => button.addEventListener("click", () => {
    selectView(button.dataset.viewTarget);
    document.getElementById("low-level")?.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  }));

  function renderBeginner() {
    const root = document.getElementById("vertical-flow");
    const stepText = document.getElementById("architecture-step");
    const progress = document.getElementById("flow-progress-bar");
    const toggle = document.getElementById("architecture-toggle");
    const previous = document.getElementById("architecture-previous");
    const next = document.getElementById("architecture-next");
    if (!root || !stepText || !progress || !toggle || !previous || !next) return;

    const selection = window.d3
      ? d3.select(root).selectAll("article.flow-card").data(beginnerStages).join("article")
      : null;
    if (!selection) {
      root.innerHTML = '<p class="graph-fallback">D3 could not load. The static architecture summary remains available above.</p>';
      return;
    }
    d3.select(root).selectAll(".graph-fallback").remove();
    selection
      .attr("class", "flow-card")
      .attr("data-stage", (_, index) => index)
      .attr("aria-label", (stage, index) => `Step ${index + 1}: ${stage.title}`)
      .html((stage, index) => `
        <div class="flow-marker" aria-hidden="true"><span>${String(index + 1).padStart(2, "0")}</span></div>
        <button class="flow-summary" type="button" aria-expanded="false">
          <span class="flow-kicker">Step ${index + 1}</span>
          <span class="flow-title">${stage.title}</span>
          <p class="flow-analogy">${stage.analogy}</p>
        </button>
        <div class="flow-basics">
          <div><b>What enters</b><p>${stage.enters}</p></div>
          <div><b>What happens</b><p>${stage.happens}</p></div>
          <div><b>What exits</b><p>${stage.exits}</p></div>
        </div>
        <details><summary>Technical detail</summary><p>${stage.technical}</p></details>`);

    const cards = [...root.querySelectorAll(".flow-card")];
    let active = 0;
    let paused = reduceMotion;
    let timer;
    function status() {
      stepText.textContent = `Step ${active + 1} of ${beginnerStages.length} · ${beginnerStages[active].title}`;
      progress.style.width = `${((active + 1) / beginnerStages.length) * 100}%`;
      toggle.textContent = paused ? "Resume tour" : "Pause tour";
      toggle.setAttribute("aria-pressed", String(paused));
    }
    function show(index, scroll) {
      active = (index + beginnerStages.length) % beginnerStages.length;
      cards.forEach((card, cardIndex) => {
        const isActive = cardIndex === active;
        card.classList.toggle("is-active", isActive);
        card.querySelector(".flow-summary").setAttribute("aria-expanded", String(isActive));
      });
      status();
      if (scroll && activeView === "beginner") cards[active].scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
    }
    function schedule() {
      window.clearTimeout(timer);
      if (paused) return;
      timer = window.setTimeout(() => { show(active + 1, true); schedule(); }, 3600);
    }
    cards.forEach((card, index) => card.querySelector(".flow-summary").addEventListener("click", () => {
      paused = true;
      window.clearTimeout(timer);
      show(index, true);
    }));
    toggle.addEventListener("click", () => { paused = !paused; status(); schedule(); });
    previous.addEventListener("click", () => { paused = true; window.clearTimeout(timer); show(active - 1, true); });
    next.addEventListener("click", () => { paused = true; window.clearTimeout(timer); show(active + 1, true); });
    show(0, false);
    schedule();
  }

  let componentRendered = false;
  function renderComponentMap() {
    if (componentRendered || !window.d3) return;
    componentRendered = true;
    const host = d3.select("#component-map");
    const detail = d3.select("#component-detail");
    const width = 1780;
    const columnWidth = 278;
    const nodeWidth = 226;
    const nodeHeight = 68;
    const rowGap = 90;
    const headerHeight = 94;
    const counts = new Map(componentGroups.map((group) => [group.id, 0]));
    const layoutNodes = componentNodes.map((node) => {
      const groupIndex = componentGroups.findIndex((group) => group.id === node.group);
      const row = counts.get(node.group);
      counts.set(node.group, row + 1);
      return { ...node, x: 34 + groupIndex * columnWidth, y: headerHeight + row * rowGap };
    });
    const height = Math.max(...layoutNodes.map((node) => node.y)) + nodeHeight + 50;
    const byId = new Map(layoutNodes.map((node) => [node.id, node]));
    const svg = host.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "component-arrow").attr("viewBox", "0 -5 10 10").attr("refX", 9).attr("refY", 0).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#aab4c1");
    const viewport = svg.append("g");
    const zoomBehavior = d3.zoom().scaleExtent([0.55, 2.4]).on("zoom", (event) => viewport.attr("transform", event.transform));
    svg.call(zoomBehavior);

    viewport.selectAll("rect.component-column").data(componentGroups).join("rect")
      .attr("class", "component-column")
      .attr("x", (_, index) => 12 + index * columnWidth)
      .attr("y", 10)
      .attr("width", columnWidth - 16)
      .attr("height", height - 20)
      .attr("rx", 18)
      .attr("fill", (group) => `${palette[group.id]}0d`)
      .attr("stroke", (group) => `${palette[group.id]}44`);
    const headers = viewport.selectAll("g.component-header").data(componentGroups).join("g").attr("class", "component-header").attr("transform", (_, index) => `translate(${34 + index * columnWidth},36)`);
    headers.append("text").attr("class", "component-header-title").attr("fill", (group) => palette[group.id]).text((group) => group.title);
    headers.append("text").attr("class", "component-header-subtitle").attr("y", 22).text((group) => group.subtitle);

    const links = viewport.append("g").attr("class", "component-links").selectAll("path").data(componentLinks).join("path")
      .attr("class", "component-link")
      .attr("marker-end", "url(#component-arrow)")
      .attr("d", (link) => {
        const source = byId.get(link.source);
        const target = byId.get(link.target);
        const sx = source.x + nodeWidth;
        const sy = source.y + nodeHeight / 2;
        const tx = target.x;
        const ty = target.y + nodeHeight / 2;
        if (source.group === target.group) return `M${source.x + nodeWidth / 2},${source.y + nodeHeight} V${target.y}`;
        const mid = (sx + tx) / 2;
        return `M${sx},${sy} C${mid},${sy} ${mid},${ty} ${tx},${ty}`;
      });

    const nodes = viewport.append("g").selectAll("g.component-node").data(layoutNodes).join("g")
      .attr("class", "component-node")
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr("transform", (node) => `translate(${node.x},${node.y})`)
      .on("click", (_, node) => selectComponent(node))
      .on("keydown", (event, node) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectComponent(node); } });
    nodes.append("rect").attr("width", nodeWidth).attr("height", nodeHeight).attr("rx", 11).attr("fill", "#fff").attr("stroke", (node) => palette[node.group]);
    nodes.append("circle").attr("cx", 17).attr("cy", 17).attr("r", 5).attr("fill", (node) => palette[node.group]);
    nodes.append("text").attr("class", "component-node-title").attr("x", 30).attr("y", 21).text((node) => node.title);
    nodes.append("text").attr("class", "component-node-summary").attr("x", 14).attr("y", 43).each(function (node) { wrapText(d3.select(this), node.summary, 198, 2); });
    nodes.filter((node) => modelAssistedNodes.has(node.id)).append("rect").attr("class", "model-badge").attr("x", 169).attr("y", 8).attr("width", 48).attr("height", 17).attr("rx", 8);
    nodes.filter((node) => modelAssistedNodes.has(node.id)).append("text").attr("class", "model-badge-text").attr("x", 193).attr("y", 20).attr("text-anchor", "middle").text("MODEL");

    function selectComponent(node) {
      nodes.classed("is-selected", (candidate) => candidate.id === node.id).classed("is-muted", (candidate) => candidate.id !== node.id && !componentLinks.some((link) => (link.source === node.id && link.target === candidate.id) || (link.target === node.id && link.source === candidate.id)));
      links.classed("is-active", (link) => link.source === node.id || link.target === node.id).classed("is-muted", (link) => link.source !== node.id && link.target !== node.id);
      detail.html(detailMarkup(node, componentGroups.find((group) => group.id === node.group).title));
    }
    selectComponent(layoutNodes[0]);

    const search = document.getElementById("component-search");
    search.addEventListener("input", () => {
      const query = search.value.trim().toLowerCase();
      nodes.classed("is-search-miss", (node) => query && !Object.values(node).join(" ").toLowerCase().includes(query));
    });
    document.getElementById("component-reset").addEventListener("click", () => {
      search.value = "";
      nodes.classed("is-search-miss", false).classed("is-muted", false).classed("is-selected", false);
      links.classed("is-active", false).classed("is-muted", false);
      svg.transition().duration(reduceMotion ? 0 : 300).call(zoomBehavior.transform, d3.zoomIdentity);
      selectComponent(layoutNodes[0]);
    });
  }

  let sequenceRendered = false;
  let sequenceControl;
  function renderSequence() {
    if (sequenceRendered || !window.d3) return;
    sequenceRendered = true;
    const host = d3.select("#request-sequence");
    const detail = d3.select("#sequence-detail");
    const width = 1440;
    const actorGap = 142;
    const top = 82;
    const rowGap = 58;
    const height = top + sequenceEvents.length * rowGap + 50;
    const actorX = new Map(sequenceActors.map((actor, index) => [actor, 52 + index * actorGap]));
    const svg = host.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "sequence-arrow-d3").attr("viewBox", "0 -5 10 10").attr("refX", 9).attr("refY", 0).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#94a0b1");
    const viewport = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.65, 2.2]).on("zoom", (event) => viewport.attr("transform", event.transform)));
    const actors = viewport.selectAll("g.sequence-actor").data(sequenceActors).join("g").attr("class", "sequence-actor").attr("transform", (actor) => `translate(${actorX.get(actor)},0)`);
    actors.append("rect").attr("x", -57).attr("y", 12).attr("width", 114).attr("height", 42).attr("rx", 8);
    actors.append("text").attr("text-anchor", "middle").attr("y", 38).text((actor) => actor);
    actors.append("line").attr("y1", 54).attr("y2", height - 18).attr("class", "sequence-lifeline");
    const events = viewport.selectAll("g.sequence-row").data(sequenceEvents).join("g").attr("class", "sequence-row").attr("transform", (event) => `translate(0,${top + event.index * rowGap})`).attr("tabindex", 0).attr("role", "button");
    events.each(function (event) {
      const group = d3.select(this);
      const x1 = actorX.get(event.from);
      const x2 = actorX.get(event.to);
      const self = event.from === event.to;
      if (self) {
        group.append("path").attr("class", "sequence-arrow").attr("marker-end", "url(#sequence-arrow-d3)").attr("d", `M${x1},0 h38 v24 h-38`);
      } else {
        group.append("line").attr("class", "sequence-arrow").attr("x1", x1).attr("x2", x2).attr("marker-end", "url(#sequence-arrow-d3)");
      }
      const left = Math.min(x1, x2);
      const right = Math.max(x1, x2);
      group.append("rect").attr("class", "sequence-label-bg").attr("x", self ? x1 + 44 : left + 8).attr("y", -20).attr("width", self ? 260 : Math.max(130, right - left - 16)).attr("height", 23).attr("rx", 5);
      group.append("text").attr("class", "sequence-row-label").attr("x", self ? x1 + 52 : (x1 + x2) / 2).attr("y", -5).attr("text-anchor", self ? "start" : "middle").text(`${event.index + 1}. ${event.title}`);
    });

    let active = 0;
    let paused = reduceMotion;
    let timer;
    const stepText = document.getElementById("sequence-step");
    const toggle = document.getElementById("sequence-toggle");
    function show(index) {
      active = (index + sequenceEvents.length) % sequenceEvents.length;
      events.classed("is-current", (event) => event.index === active);
      const event = sequenceEvents[active];
      stepText.textContent = `Event ${active + 1} of ${sequenceEvents.length} · ${event.title}`;
      toggle.textContent = paused ? "Resume sequence" : "Pause sequence";
      toggle.setAttribute("aria-pressed", String(paused));
      detail.html(`<span class="detail-kind">${event.from} → ${event.to}</span><h3>${event.title}</h3><p>${event.detail}</p><dl><div><dt>Runtime position</dt><dd>${active + 1} of ${sequenceEvents.length}</dd></div><div><dt>Model call?</dt><dd>${active < 5 || active > 19 ? "No" : "Only where the event explicitly names Flash or Pro"}</dd></div></dl>`);
      if (activeView === "sequence") {
        const canvas = host.node();
        const eventX = (actorX.get(event.from) + actorX.get(event.to)) / 2;
        const eventY = top + active * rowGap;
        canvas.scrollTo({
          left: Math.max(0, eventX - canvas.clientWidth / 2),
          top: Math.max(0, eventY - canvas.clientHeight / 2),
          behavior: reduceMotion ? "auto" : "smooth"
        });
      }
    }
    function schedule() {
      window.clearTimeout(timer);
      if (paused) return;
      timer = window.setTimeout(() => { show(active + 1); schedule(); }, 1900);
    }
    events.on("click", (_, event) => { paused = true; window.clearTimeout(timer); show(event.index); }).on("keydown", (keyboard, event) => { if (keyboard.key === "Enter" || keyboard.key === " ") { keyboard.preventDefault(); paused = true; window.clearTimeout(timer); show(event.index); } });
    toggle.addEventListener("click", () => { paused = !paused; show(active); schedule(); });
    document.getElementById("sequence-previous").addEventListener("click", () => { paused = true; window.clearTimeout(timer); show(active - 1); });
    document.getElementById("sequence-next").addEventListener("click", () => { paused = true; window.clearTimeout(timer); show(active + 1); });
    sequenceControl = { show, schedule };
    show(0);
    schedule();
  }

  let lessonsRendered = false;
  function renderLessons() {
    if (lessonsRendered || !window.d3) return;
    lessonsRendered = true;
    const host = d3.select("#lessons-loop");
    const detail = d3.select("#lessons-detail");
    const width = 920;
    const height = 720;
    const cx = width / 2;
    const cy = height / 2;
    const rx = 310;
    const ry = 245;
    const positions = lessonNodes.map((node, index) => ({ ...node, x: cx + Math.cos(-Math.PI / 2 + index * Math.PI / 4) * rx, y: cy + Math.sin(-Math.PI / 2 + index * Math.PI / 4) * ry }));
    const svg = host.append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img");
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "lesson-arrow").attr("viewBox", "0 -5 10 10").attr("refX", 9).attr("refY", 0).attr("markerWidth", 6).attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", palette.lessons);
    svg.append("text").attr("class", "lesson-center-title").attr("x", cx).attr("y", cy - 10).attr("text-anchor", "middle").text("Development evidence");
    svg.append("text").attr("class", "lesson-center-subtitle").attr("x", cx).attr("y", cy + 17).attr("text-anchor", "middle").text("never a production input");
    const links = svg.append("g").selectAll("path").data(positions).join("path").attr("class", "lesson-link").attr("marker-end", "url(#lesson-arrow)").attr("d", (node, index) => {
      const next = positions[(index + 1) % positions.length];
      const mx = (node.x + next.x) / 2;
      const my = (node.y + next.y) / 2;
      return `M${node.x},${node.y} Q${mx + (mx - cx) * 0.12},${my + (my - cy) * 0.12} ${next.x},${next.y}`;
    });
    const nodes = svg.append("g").selectAll("g.lesson-node").data(positions).join("g").attr("class", "lesson-node").attr("transform", (node) => `translate(${node.x},${node.y})`).attr("tabindex", 0).attr("role", "button");
    nodes.append("rect").attr("x", -105).attr("y", -38).attr("width", 210).attr("height", 76).attr("rx", 12);
    nodes.append("circle").attr("cx", -88).attr("cy", -21).attr("r", 12);
    nodes.append("text").attr("class", "lesson-number").attr("x", -88).attr("y", -17).attr("text-anchor", "middle").text((_, index) => index + 1);
    nodes.append("text").attr("class", "lesson-title").attr("text-anchor", "middle").attr("y", -2).each(function (node) { wrapText(d3.select(this), node.title, 178, 2); });
    function choose(node) {
      nodes.classed("is-selected", (candidate) => candidate.id === node.id);
      links.classed("is-active", (candidate) => candidate.id === node.id);
      detail.html(`<span class="detail-kind">Development step ${node.index + 1}</span><h3>${node.title}</h3><p>${node.detail}</p><div class="detail-note"><b>Boundary</b><p>This step may influence the next source-code revision, but organizer feedback and prior predictions are never accepted by the public production CLI.</p></div>`);
    }
    nodes.on("click", (_, node) => choose(node)).on("keydown", (event, node) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(node); } });
    choose(positions[0]);
  }

  function detailMarkup(node, groupTitle) {
    return `<span class="detail-kind">${groupTitle}</span><h3>${node.title}</h3><p>${node.summary}</p><dl>
      <div><dt>Execution character</dt><dd>${modelAssistedNodes.has(node.id) ? "Model-assisted proposal; deterministic validation follows. Hosted reruns are not byte-guaranteed." : "Deterministic for fixed bytes, configuration, code, and runtime dependencies."}</dd></div>
      <div><dt>Consumes</dt><dd>${node.consumes}</dd></div>
      <div><dt>State</dt><dd>${node.state}</dd></div>
      <div><dt>Logic</dt><dd>${node.logic}</dd></div>
      <div><dt>Emits</dt><dd>${node.emits}</dd></div>
      <div><dt>Fail-closed gate</dt><dd>${node.gate}</dd></div>
      <div><dt>Scorer effect</dt><dd>${node.score}</dd></div>
    </dl><div class="module-path"><span>Implementation</span><code>${node.module}</code></div>`;
  }

  function wrapText(text, value, width, maxLines) {
    const words = String(value).split(/\s+/).reverse();
    let word;
    let line = [];
    let lineNumber = 0;
    const lineHeight = 14;
    const x = Number(text.attr("x") || 0);
    const y = Number(text.attr("y") || 0);
    let tspan = text.text(null).append("tspan").attr("x", x).attr("y", y);
    while ((word = words.pop())) {
      line.push(word);
      tspan.text(line.join(" "));
      if (tspan.node().getComputedTextLength() > width && line.length > 1) {
        line.pop();
        tspan.text(line.join(" "));
        line = [word];
        lineNumber += 1;
        if (lineNumber >= maxLines) {
          const current = tspan.text();
          tspan.text(`${current.replace(/[.,;:]?$/, "")}…`);
          break;
        }
        tspan = text.append("tspan").attr("x", x).attr("y", y + lineNumber * lineHeight).text(word);
      }
    }
  }

  renderBeginner();
  selectView("beginner");
})();
