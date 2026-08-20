(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const palette = { ingress: "#76558f", retrieval: "#2f5d8a", source: "#1c7c75", evidence: "#c36b1d", release: "#4c657c" };
  const nodes = [
    {id:"released", label:"Released record", group:"ingress", lane:"Contract ingress", order:0, kind:"Immutable input", consumes:"Organizer JSON: question, answer types, options, table schema", state:"RawRecord{query_id, question, schema}", logic:"Preserve organizer-owned fields byte-for-byte; no inferred aliases are written back.", emits:"Immutable typed input", gate:"Reject malformed or duplicate organizer records", impact:"Protects every scorer-facing field from schema drift"},
    {id:"contract", label:"Contract parser", group:"ingress", lane:"Contract ingress", order:1, kind:"Question contract", consumes:"Raw record and declared output schema", state:"QuestionContract{targets, modalities, cardinality, roles}", logic:"Extract required entities, quantities, comparisons, source modalities, and answer cardinality.", emits:"Typed target and answer constraints", gate:"Unknown shape becomes a conservative no-expansion plan", impact:"Prevents wrong answer type, option shape, and table contract"},
    {id:"plan", label:"Execution planner", group:"ingress", lane:"Contract ingress", order:2, kind:"Plan state", consumes:"QuestionContract and route registry", state:"QueryPlan{groups, routes, budgets, required facts}", logic:"Assign retrieval routes by target group and reserve evidence/table work only when demanded.", emits:"Bounded per-group work plan", gate:"Every route retains target-group and role provenance", impact:"Avoids cross-target paper collapse"},

    {id:"alias", label:"Alias resolver", group:"retrieval", lane:"Paper identity", order:3, kind:"Candidate producer", consumes:"Named targets, metadata titles, aliases", state:"AliasSupport{group, paper_id, match}", logic:"Normalize title variants and author/year cues into canonical candidate IDs.", emits:"Exact and alias candidate supports", gate:"No paper ID without a corpus metadata match", impact:"Raises paper recall without inventing IDs"},
    {id:"routes", label:"Route executor", group:"retrieval", lane:"Paper identity", order:4, kind:"Candidate producer", consumes:"QueryPlan plus sparse, relation, object, dense indexes", state:"RouteHit{group, route, rank, score, paper_id}", logic:"Run each bounded retrieval route independently; retain scores and target ownership.", emits:"Ranked source-scoped supports", gate:"Never merge ranks before grouping by target", impact:"Preserves target-aware paper precision"},
    {id:"ledger", label:"Candidate ledger", group:"retrieval", lane:"Paper identity", order:5, kind:"Identity state", consumes:"AliasSupport and RouteHit sets", state:"CandidateLedger[paper_id] → supports-by-group", logic:"Accumulate heterogeneous support while keeping route, group, rank, and role inspectable.", emits:"Auditable candidate coverage ledger", gate:"No global score may erase a missing target group", impact:"Makes paper selection explainable and reproducible"},
    {id:"select", label:"Set cover selector", group:"retrieval", lane:"Paper identity", order:6, kind:"Paper set", consumes:"CandidateLedger, cardinality, answer-bearing support", state:"P = ordered canonical paper IDs", logic:"Cover mandatory target groups first, then add answer-bearing sources within the capacity budget.", emits:"Selected paper set P", gate:"Constraint, coverage, metadata existence, and cap ≤ 5", impact:"Direct paper precision/recall and evidence closure"},

    {id:"manifest", label:"Source manifest", group:"source", lane:"Source foundation", order:7, kind:"Immutable source", consumes:"Selected P and canonical corpus locations", state:"SourceManifest{paper_id, URL, SHA, page_count}", logic:"Freeze source URL, byte hash, page order, and parser version for every selected paper.", emits:"Verified source acquisition plan", gate:"Hash/identity mismatch quarantines that source", impact:"Eliminates paper-ID and page-origin ambiguity"},
    {id:"parse", label:"PDF structure parser", group:"source", lane:"Source foundation", order:8, kind:"Source objects", consumes:"Verified PDFs and parser configuration", state:"ParsedPaper{pages, blocks, tables, figures, equations, refs}", logic:"Segment full text and preserve printed object identities rather than flattening the PDF.", emits:"Page-scoped structured source objects", gate:"Page index and object IDs must remain source-attested", impact:"Supports exact evidence modality and locator scoring"},
    {id:"index", label:"Object index", group:"source", lane:"Source foundation", order:9, kind:"Read-only index", consumes:"ParsedPaper and selected target facts", state:"ObjectIndex{paper_id, page, type, object_id, text}", logic:"Index text, captions, table cells, equations, figures, and citations by exact source identity.", emits:"Retrieval-ready source object inventory", gate:"No locator can reference an object outside P", impact:"Raises evidence precision through canonical locators"},

    {id:"localize", label:"Evidence localizer", group:"evidence", lane:"Evidence + extraction", order:10, kind:"Attributed evidence", consumes:"QuestionContract, ObjectIndex, selected P", state:"E(P) = locator tuples with quote/object/page", logic:"Retrieve answer-bearing regions separately from answer assembly and normalize their scorer-visible identity.", emits:"Attributed evidence candidates", gate:"Require paper closure, page, type, object ID, and source quote", impact:"Direct evidence precision/recall"},
    {id:"tableplan", label:"Observation planner", group:"evidence", lane:"Evidence + extraction", order:11, kind:"Table contract", consumes:"Table schema, question wording, source object inventory", state:"ObservationPlan{unit, row_key, columns, coordinates}", logic:"State what one row represents before values are read; select the evaluator-compatible row key.", emits:"Typed table observation contract", gate:"Reject incompatible mixed units or untyped keys", impact:"Direct table-row F1 and cell-coordinate credit"},
    {id:"extract", label:"Value extractor", group:"evidence", lane:"Evidence + extraction", order:12, kind:"Structured facts", consumes:"ObservationPlan, localized tables/text/equations", state:"FactSet{row_key, column, value, evidence}", logic:"Extract normalized values only from attested source objects and carry their exact coordinates.", emits:"Typed facts and source links", gate:"Type, row-key, coordinate, and value/source agreement", impact:"Improves cell accuracy without answer injection"},

    {id:"assemble", label:"Answer assembler", group:"release", lane:"Artifact sealing", order:13, kind:"Typed answer", consumes:"QuestionContract, P, E(P), FactSet", state:"A = organizer-shaped option, freeform, or table answer", logic:"Render only schema-valid fields; retain organizer labels and normalized value types.", emits:"Answer A", gate:"Validate field names, cardinality, option membership, and table schema", impact:"Direct MC and table answer correctness"},
    {id:"closure", label:"Closure enforcer", group:"release", lane:"Artifact sealing", order:14, kind:"Trace closure", consumes:"P, E(P), A", state:"ClosedTrace{P,E,A}", logic:"Drop evidence outside selected P; deduplicate using evaluator-visible identity; enforce evidence limits.", emits:"Closed scorer-facing record", gate:"Paper–evidence closure and per-record caps", impact:"Prevents evidence precision loss from orphan locators"},
    {id:"validate", label:"Official validator", group:"release", lane:"Artifact sealing", order:15, kind:"Pinned contract", consumes:"All closed records and organizer validator", state:"ValidationReport{errors,warnings,record_count}", logic:"Validate nested shapes and constraints before any output is accepted.", emits:"Validated 71-record collection", gate:"Any error aborts artifact creation atomically", impact:"Prevents zero-credit malformed submissions"},
    {id:"artifact", label:"Manifested JSONL", group:"release", lane:"Artifact sealing", order:16, kind:"Release artifact", consumes:"Validated collection, source revision, configuration", state:"ArtifactManifest{SHA, revision, config, validator}", logic:"Write deterministic JSONL and a provenance manifest after validation only.", emits:"Upload-ready reproducible artifact", gate:"Hash records and provenance before release", impact:"Makes the submitted output auditable"},
    {id:"audit", label:"Release audit", group:"release", lane:"Artifact sealing", order:17, kind:"Reproducibility gate", consumes:"Source tree, manifest, tests, output boundary", state:"AuditReport{scope, privacy, behavior}", logic:"Check that published code has no prediction files, credentials, private paths, IDs, or answer constants.", emits:"Clean release decision", gate:"Fail the release on prohibited material or broken tests", impact:"Protects reproducibility claim and public review"}
  ];

  const links = [
    ["released","contract"],["contract","plan"],["plan","alias"],["plan","routes"],["alias","ledger"],["routes","ledger"],["ledger","select"],
    ["select","manifest",true],["manifest","parse",true],["parse","index",true],["plan","localize"],["index","localize",true],["contract","tableplan"],["index","tableplan",true],["tableplan","extract"],["localize","extract"],["localize","assemble"],["extract","assemble"],["select","closure"],["assemble","closure"],["localize","closure"],["closure","validate"],["validate","artifact"],["artifact","audit"]
  ].map(([source,target,sourceEdge]) => ({source,target,sourceEdge}));

  const root = document.getElementById("low-level-graph");
  const detail = document.getElementById("architecture-detail");
  const tools = document.getElementById("architecture-stage-buttons");
  const stepText = document.getElementById("architecture-step");
  const toggle = document.getElementById("architecture-toggle");
  const previous = document.getElementById("architecture-previous");
  const next = document.getElementById("architecture-next");
  if (!root || !detail || !tools || !stepText || !toggle || !previous || !next || !window.d3) return;
  root.querySelector(".graph-fallback")?.remove();

  let selectedIndex = 1;
  let tourPaused = reducedMotion;
  let tourTimer;
  const incoming = new Map(nodes.map(node => [node.id, []]));
  const outgoing = new Map(nodes.map(node => [node.id, []]));
  links.forEach(link => { incoming.get(link.target).push(link.source); outgoing.get(link.source).push(link.target); });
  const neighborhood = id => new Set([id, ...incoming.get(id), ...outgoing.get(id)]);

  function updateControlLabel() {
    const node = nodes[selectedIndex];
    stepText.textContent = `${String(selectedIndex + 1).padStart(2, "0")} / ${nodes.length} · ${node.lane}`;
    toggle.textContent = tourPaused ? "Resume tour" : "Pause tour";
    toggle.setAttribute("aria-pressed", String(tourPaused));
  }

  function showDetail(node) {
    selectedIndex = nodes.findIndex(item => item.id === node.id);
    detail.innerHTML = `<span class="detail-kind">${node.kind} · ${node.lane}</span><h3>${node.label}</h3><p>${node.logic}</p><dl><div><dt>Consumes</dt><dd>${node.consumes}</dd></div><div><dt>State</dt><dd>${node.state}</dd></div><div><dt>Logic</dt><dd>${node.logic}</dd></div><div><dt>Emits</dt><dd>${node.emits}</dd></div><div><dt>Fail-closed gate</dt><dd>${node.gate}</dd></div><div><dt>Scorer impact</dt><dd>${node.impact}</dd></div></dl>`;
    const active = neighborhood(node.id);
    d3.select(root).selectAll(".node").classed("is-selected", item => item.id === node.id).classed("is-muted", item => !active.has(item.id));
    d3.select(root).selectAll(".edge").classed("is-active", edge => edge.source === node.id || edge.target === node.id).classed("is-muted", edge => !(active.has(edge.source) && active.has(edge.target)));
    d3.select(tools).selectAll("button").attr("aria-pressed", item => String(item.id === node.id));
    updateControlLabel();
  }

  d3.select(tools).selectAll("button").data(nodes).join("button").attr("type", "button").attr("aria-pressed", node => String(node.id === nodes[selectedIndex].id)).text(node => node.label).on("click", (_, node) => { tourPaused = true; window.clearTimeout(tourTimer); showDetail(node); });

  function drawGraph() {
    const width = Math.max(300, root.clientWidth - 24);
    const narrow = width < 700;
    const groupOrder = ["ingress", "retrieval", "source", "evidence", "release"];
    const height = narrow ? 1240 : 770;
    const svg = d3.select(root).selectAll("svg").data([null]).join("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img").attr("aria-label", "TRACE component-level architecture with data stores, contracts, and fail-closed gates");
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    defs.append("marker").attr("id", "arrow").attr("viewBox", "0 -5 10 10").attr("refX", 8).attr("refY", 0).attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#aab4c1");
    const nodeW = narrow ? Math.min(242, width - 66) : Math.min(172, (width - 120) / 5 - 12);
    const nodeH = 58;
    const byId = new Map();
    groupOrder.forEach((group, col) => {
      const items = nodes.filter(node => node.group === group);
      items.forEach((node, row) => {
        const x = narrow ? width / 2 : 64 + col * ((width - 128) / 4);
        const y = narrow ? 62 + node.order * 66 : 94 + row * 164;
        byId.set(node.id, {...node, x, y});
      });
    });
    const positioned = nodes.map(node => byId.get(node.id));
    const groups = groupOrder.map((group, col) => ({group, label: nodes.find(node => node.group === group).lane, color: palette[group], x: narrow ? 18 : 12 + col * ((width - 24) / 5), y: narrow ? 20 : 26, w: narrow ? width - 36 : (width - 34) / 5, h: narrow ? height - 38 : height - 52}));
    svg.append("g").selectAll("rect").data(groups).join("rect").attr("class", "component-lane").attr("x", group => group.x).attr("y", group => group.y).attr("width", group => group.w).attr("height", group => group.h).attr("fill", group => `${group.color}08`).attr("stroke", group => `${group.color}55`);
    svg.append("g").selectAll("text").data(groups).join("text").attr("class", "lane-label").attr("x", group => group.x + 12).attr("y", group => group.y + 18).attr("fill", group => group.color).text(group => group.label.toUpperCase());
    svg.append("g").selectAll("path").data(links).join("path").attr("class", link => `edge${link.sourceEdge ? " source-edge" : ""}`).attr("marker-end", "url(#arrow)").attr("d", link => {
      const source = byId.get(link.source), target = byId.get(link.target);
      if (narrow) return `M${source.x},${source.y + nodeH / 2} C${source.x},${source.y + 34} ${target.x},${target.y - 34} ${target.x},${target.y - nodeH / 2}`;
      const direction = target.x >= source.x ? 1 : -1;
      const sx = source.x + direction * nodeW / 2, tx = target.x - direction * nodeW / 2, mx = (sx + tx) / 2;
      return `M${sx},${source.y} C${mx},${source.y} ${mx},${target.y} ${tx},${target.y}`;
    });
    const node = svg.append("g").selectAll("g").data(positioned).join("g").attr("class", "node").attr("transform", item => `translate(${item.x},${item.y})`).attr("role", "button").attr("aria-label", item => `${item.label}: ${item.kind}`).style("cursor", "pointer").on("click", (_, item) => { tourPaused = true; window.clearTimeout(tourTimer); showDetail(item); }).on("mouseenter", (_, item) => showDetail(item));
    node.append("rect").attr("x", -nodeW / 2).attr("y", -nodeH / 2).attr("width", nodeW).attr("height", nodeH).attr("rx", 7).attr("fill", item => `${palette[item.group]}14`).attr("stroke", item => palette[item.group]);
    node.append("circle").attr("cx", -nodeW / 2 + 15).attr("cy", -10).attr("r", 4).attr("fill", item => palette[item.group]);
    node.append("text").attr("x", -nodeW / 2 + 26).attr("y", -6).text(item => item.label);
    node.append("text").attr("class", "node-kind").attr("x", -nodeW / 2 + 26).attr("y", 13).text(item => item.kind);
    showDetail(nodes[selectedIndex]);
  }

  function scheduleTour() {
    window.clearTimeout(tourTimer);
    if (tourPaused) return;
    tourTimer = window.setTimeout(() => { showDetail(nodes[(selectedIndex + 1) % nodes.length]); scheduleTour(); }, 2400);
  }
  toggle.addEventListener("click", () => { tourPaused = !tourPaused; updateControlLabel(); scheduleTour(); });
  previous.addEventListener("click", () => { tourPaused = true; showDetail(nodes[(selectedIndex - 1 + nodes.length) % nodes.length]); });
  next.addEventListener("click", () => { tourPaused = true; showDetail(nodes[(selectedIndex + 1) % nodes.length]); });
  drawGraph();
  scheduleTour();
  new ResizeObserver(drawGraph).observe(root);

  const sequenceRoot = document.getElementById("sequence-diagram");
  const sequenceToggle = document.getElementById("sequence-toggle");
  const sequencePrevious = document.getElementById("sequence-previous");
  const sequenceNext = document.getElementById("sequence-next");
  const sequenceStep = document.getElementById("sequence-step");
  const sequenceMessage = document.getElementById("sequence-message");
  if (!sequenceRoot || !sequenceToggle || !sequencePrevious || !sequenceNext || !sequenceStep || !sequenceMessage) return;
  sequenceRoot.querySelector(".graph-fallback")?.remove();
  const actors = ["Question", "Contract", "Routes", "Sources", "Evidence", "Answer", "Validator"];
  const events = [
    {from:0,to:1,label:"record",message:"Freeze the released record as the typed question contract."}, {from:1,to:2,label:"plan",message:"Assign target-aware retrieval routes and bounded budgets."}, {from:2,to:3,label:"P candidates",message:"Resolve and verify canonical paper identities."}, {from:3,to:4,label:"objects",message:"Parse page-scoped source objects and locate evidence."}, {from:4,to:5,label:"P + E(P) + facts",message:"Assemble only schema-compatible answer values."}, {from:5,to:6,label:"closed trace",message:"Enforce paper–evidence closure before official validation."}, {from:6,to:0,label:"validated JSONL",message:"Atomically write the manifest-backed submission artifact."}
  ];
  let sequenceIndex = 0, sequencePaused = reducedMotion, sequenceTimer;
  function updateSequenceControls() { sequenceToggle.textContent = sequencePaused ? "Resume" : "Pause"; sequenceToggle.setAttribute("aria-pressed", String(sequencePaused)); }
  function drawSequence() {
    const width = Math.max(300, sequenceRoot.clientWidth - 24), compact = width < 680, height = compact ? 420 : 305;
    const svg = d3.select(sequenceRoot).selectAll("svg").data([null]).join("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img").attr("aria-label", "Autoplaying TRACE execution sequence");
    svg.selectAll("*").remove();
    const defs = svg.append("defs"); defs.append("marker").attr("id", "sequence-arrow").attr("viewBox", "0 -5 10 10").attr("refX", 8).attr("refY", 0).attr("markerWidth", 5).attr("markerHeight", 5).attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#9ca9ba");
    const x = compact ? d3.scalePoint().domain(d3.range(4)).range([44, width - 44]) : d3.scalePoint().domain(d3.range(actors.length)).range([46, width - 46]);
    const positions = actors.map((name, index) => compact ? {name, x:x(index % 4), y:index < 4 ? 34 : 235} : {name, x:x(index), y:34});
    svg.selectAll("text.sequence-label").data(positions).join("text").attr("class", "sequence-label").attr("text-anchor", "middle").attr("x", item => item.x).attr("y", item => item.y).text(item => item.name);
    svg.selectAll("line.sequence-lane").data(positions).join("line").attr("class", "sequence-lane").attr("x1", item => item.x).attr("x2", item => item.x).attr("y1", item => item.y + 12).attr("y2", item => compact ? (item.y < 100 ? 205 : height - 20) : height - 24);
    const rowY = index => compact ? (index < 4 ? 78 + index * 32 : 276 + (index - 4) * 32) : 79 + index * 28;
    const rendered = events.map((event, index) => ({...event, index, y:rowY(index), source:positions[event.from], target:positions[event.to]}));
    svg.selectAll("line.sequence-event").data(rendered).join("line").attr("class", item => `sequence-event${item.index === sequenceIndex ? " is-current" : ""}`).attr("x1", item => item.source.x).attr("x2", item => item.target.x).attr("y1", item => item.y).attr("y2", item => item.y).attr("marker-end", "url(#sequence-arrow)");
    svg.selectAll("text.sequence-event-label").data(rendered).join("text").attr("class", item => `sequence-event-label${item.index === sequenceIndex ? " is-current" : ""}`).attr("text-anchor", "middle").attr("x", item => (item.source.x + item.target.x) / 2).attr("y", item => item.y - 6).text(item => item.label);
    const current = rendered[sequenceIndex];
    const token = svg.append("circle").attr("class", "sequence-token").attr("r", 7).attr("cx", current.source.x).attr("cy", current.y);
    if (!sequencePaused) token.transition().duration(1350).ease(d3.easeCubicInOut).attr("cx", current.target.x);
    sequenceStep.textContent = `${sequenceIndex + 1} / ${events.length}`; sequenceMessage.textContent = current.message; updateSequenceControls();
  }
  function scheduleSequence() { window.clearTimeout(sequenceTimer); if (sequencePaused) return; sequenceTimer = window.setTimeout(() => { sequenceIndex = (sequenceIndex + 1) % events.length; drawSequence(); scheduleSequence(); }, 1800); }
  sequenceToggle.addEventListener("click", () => { sequencePaused = !sequencePaused; drawSequence(); scheduleSequence(); });
  sequencePrevious.addEventListener("click", () => { sequencePaused = true; sequenceIndex = (sequenceIndex - 1 + events.length) % events.length; drawSequence(); });
  sequenceNext.addEventListener("click", () => { sequencePaused = true; sequenceIndex = (sequenceIndex + 1) % events.length; drawSequence(); });
  drawSequence(); scheduleSequence(); new ResizeObserver(drawSequence).observe(sequenceRoot);
})();
