(() => {
  "use strict";

  const nodes = [
    {id:"question", label:"Released question", group:"contract", col:0, row:0, kind:"Input", description:"The immutable question, answer types, multiple-choice options, and exact table schema.", consumes:"Released JSON record", emits:"Typed input object", gate:"Unique query ID and valid organizer shape"},
    {id:"planner", label:"Contract planner", group:"contract", col:0, row:1, kind:"Question contract", description:"Extracts requested papers, properties, cardinality, modalities, roles, and answer structure without changing organizer fields.", consumes:"Question, options, schema", emits:"Targets, roles, constraints", gate:"Typed plan or conservative fallback"},
    {id:"routes", label:"Grouped retrieval routes", group:"paper", col:1, row:0, kind:"Paper discovery", description:"Runs title and alias resolution, sparse passage search, relations, source objects, and dense metadata search per target group.", consumes:"Target groups and constraints", emits:"Route-scored candidates", gate:"Route, group, rank, score, and role retained"},
    {id:"ledger", label:"Candidate ledger", group:"paper", col:1, row:1, kind:"Paper identity", description:"Keeps heterogeneous retrieval signals separate until target coverage is evaluated.", consumes:"Grouped candidate supports", emits:"Canonical paper candidates", gate:"No cross-target rank collapse"},
    {id:"selector", label:"Paper selector", group:"paper", col:1, row:2, kind:"Paper identity", description:"Covers answer targets first, then spends remaining capacity on answer-bearing source support.", consumes:"Candidate ledger and cardinality", emits:"Selected paper set P", gate:"Constraints, coverage, and cap ≤ 5"},
    {id:"parser", label:"PDF + source parser", group:"evidence", col:2, row:0, kind:"Source foundation", description:"Parses complete selected PDFs into pages, structured blocks, and printed source objects.", consumes:"Selected paper IDs and URLs", emits:"Text, tables, figures, equations, citations", gate:"Paper ID, URL, byte hash, and page order"},
    {id:"localizer", label:"Evidence localizer", group:"evidence", col:2, row:1, kind:"Attributed evidence", description:"Retrieves answer-bearing regions independently of the answerer and normalizes their exact scorer identities.", consumes:"Question, plan, parsed sources", emits:"E(P) locator tuples", gate:"Source type, page, object ID, quote, and paper closure"},
    {id:"table", label:"Observation-unit table path", group:"evidence", col:2, row:2, kind:"Structured extraction", description:"Plans what one row represents, extracts compatible source cells, and merges only under the evaluator key contract.", consumes:"Table schema and multimodal sources", emits:"Typed row and cell facts", gate:"Row-key identity, type, coordinate, and source attestation"},
    {id:"answer", label:"Answer assembler", group:"answer", col:3, row:0, kind:"Typed answer", description:"Produces the organizer label or table rows while preserving exact field names and value types.", consumes:"Plan, papers, source facts", emits:"Answer A", gate:"Schema and normalized-key compatibility"},
    {id:"closure", label:"Trace closure", group:"answer", col:3, row:1, kind:"Artifact contract", description:"Ensures every evidence item belongs to a selected paper and deduplicates by the evaluator’s coarse identity.", consumes:"P, E(P), and A", emits:"Closed trace {P,E,A}", gate:"Paper–evidence closure and evidence cap"},
    {id:"validator", label:"Official validator", group:"answer", col:3, row:2, kind:"Release seal", description:"Checks every record against the pinned organizer contract before an upload file can be written.", consumes:"71 closed prediction records", emits:"Validated JSONL + manifest", gate:"Atomic write only after validation"}
  ];

  const links = [
    {source:"question", target:"planner"}, {source:"planner", target:"routes"},
    {source:"routes", target:"ledger"}, {source:"ledger", target:"selector"},
    {source:"selector", target:"parser", sourceEdge:true}, {source:"parser", target:"localizer", sourceEdge:true},
    {source:"parser", target:"table", sourceEdge:true}, {source:"planner", target:"table"},
    {source:"localizer", target:"answer"}, {source:"table", target:"answer"},
    {source:"selector", target:"closure"}, {source:"localizer", target:"closure"},
    {source:"answer", target:"closure"}, {source:"closure", target:"validator"}
  ];

  const palette = {contract:"#76558f", paper:"#2f5d8a", evidence:"#1c7c75", answer:"#c36b1d"};
  const detail = document.getElementById("architecture-detail");
  const root = document.getElementById("low-level-graph");
  const tools = document.getElementById("architecture-stage-buttons");
  if (!root || !detail || !tools || !window.d3) return;
  root.querySelector(".graph-fallback")?.remove();

  let selected = "planner";
  const incoming = new Map(nodes.map(n => [n.id, []]));
  const outgoing = new Map(nodes.map(n => [n.id, []]));
  links.forEach(l => { outgoing.get(l.source).push(l.target); incoming.get(l.target).push(l.source); });

  function neighborhood(id) {
    return new Set([id, ...incoming.get(id), ...outgoing.get(id)]);
  }

  function showDetail(node) {
    selected = node.id;
    detail.innerHTML = `<span class="detail-kind">${node.kind}</span><h3>${node.label}</h3><p>${node.description}</p><dl><div><dt>Consumes</dt><dd>${node.consumes}</dd></div><div><dt>Emits</dt><dd>${node.emits}</dd></div><div><dt>Gate</dt><dd>${node.gate}</dd></div></dl>`;
    const active = neighborhood(node.id);
    d3.select(root).selectAll(".node").classed("is-selected", d => d.id === node.id).classed("is-muted", d => !active.has(d.id));
    d3.select(root).selectAll(".edge").classed("is-active", d => d.source === node.id || d.target === node.id).classed("is-muted", d => !(active.has(d.source) && active.has(d.target)));
    d3.select(tools).selectAll("button").attr("aria-pressed", d => String(d.id === node.id));
  }

  d3.select(tools).selectAll("button").data(nodes).join("button")
    .attr("type", "button").attr("aria-pressed", d => String(d.id === selected))
    .text(d => d.label).on("click", (_, d) => showDetail(d));

  function draw() {
    const width = Math.max(300, root.clientWidth - 20);
    const narrow = width < 650;
    const height = narrow ? 760 : 500;
    const svg = d3.select(root).selectAll("svg").data([null]).join("svg")
      .attr("viewBox", `0 0 ${width} ${height}`).attr("role", "img")
      .attr("aria-label", "TRACE low-level component and contract graph");
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    defs.append("marker").attr("id","arrow").attr("viewBox","0 -5 10 10").attr("refX",8).attr("refY",0).attr("markerWidth",5).attr("markerHeight",5).attr("orient","auto").append("path").attr("d","M0,-5L10,0L0,5").attr("fill","#aab4c1");

    const positioned = nodes.map(n => {
      if (narrow) return {...n, x: n.col % 2 === 0 ? width * .24 : width * .72, y: 62 + (n.col * 3 + n.row) * 62};
      return {...n, x: 92 + n.col * ((width - 184) / 3), y: 92 + n.row * 150};
    });
    const byId = new Map(positioned.map(n => [n.id,n]));
    const nodeW = narrow ? Math.min(170, width * .42) : Math.min(180, width * .2);
    const nodeH = 56;

    svg.append("g").selectAll("path").data(links).join("path")
      .attr("class", d => `edge${d.sourceEdge ? " source-edge" : ""}`)
      .attr("marker-end","url(#arrow)")
      .attr("d", d => {
        const s=byId.get(d.source), t=byId.get(d.target);
        const sx=s.x + (t.x>=s.x ? nodeW/2 : -nodeW/2), sy=s.y;
        const tx=t.x + (t.x>=s.x ? -nodeW/2 : nodeW/2), ty=t.y;
        const mx=(sx+tx)/2;
        return `M${sx},${sy} C${mx},${sy} ${mx},${ty} ${tx},${ty}`;
      });

    const node = svg.append("g").selectAll("g").data(positioned).join("g")
      .attr("class","node").attr("transform",d=>`translate(${d.x},${d.y})`)
      .style("cursor","pointer").on("click",(_,d)=>showDetail(d))
      .on("mouseenter",(_,d)=>showDetail(d));
    node.append("rect").attr("x",-nodeW/2).attr("y",-nodeH/2).attr("width",nodeW).attr("height",nodeH).attr("rx",8).attr("fill",d=>`${palette[d.group]}12`).attr("stroke",d=>palette[d.group]);
    node.append("circle").attr("cx",-nodeW/2+17).attr("cy",0).attr("r",5).attr("fill",d=>palette[d.group]);
    node.append("text").attr("x",-nodeW/2+29).attr("y",4).text(d=>d.label.length>27 ? `${d.label.slice(0,26)}…` : d.label);
    showDetail(byId.get(selected));
  }

  draw();
  new ResizeObserver(draw).observe(root);

  const sequenceRoot = document.getElementById("sequence-diagram");
  const sequenceToggle = document.getElementById("sequence-toggle");
  const sequenceStep = document.getElementById("sequence-step");
  const sequenceMessage = document.getElementById("sequence-message");
  if (!sequenceRoot || !sequenceToggle || !sequenceStep || !sequenceMessage) return;
  sequenceRoot.querySelector(".graph-fallback")?.remove();

  const actors = ["Question", "Planner", "Retrieval", "Corpus", "Evidence", "Answer", "Validator"];
  const events = [
    {from:0,to:1,label:"typed input",message:"Normalize the released question contract."},
    {from:1,to:2,label:"targets + roles",message:"Plan target groups, modalities, cardinality, and schema."},
    {from:2,to:3,label:"bounded source query",message:"Run grouped title, sparse, relation, object, and dense retrieval."},
    {from:3,to:2,label:"ranked supports",message:"Return source-scoped supports without collapsing target identity."},
    {from:2,to:4,label:"selected P",message:"Select canonical papers and parse their immutable source objects."},
    {from:4,to:5,label:"E(P) + facts",message:"Localize exact evidence and assemble schema-compatible facts."},
    {from:5,to:6,label:"{P, E, A}",message:"Close paper, evidence, and answer identities into one artifact."},
    {from:6,to:0,label:"validated JSONL",message:"Write only after the pinned official validator passes."}
  ];
  let sequenceIndex = 0;
  let sequencePaused = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let sequenceTimer = null;

  function drawSequence() {
    const width = Math.max(300, sequenceRoot.clientWidth - 24);
    const compact = width < 680;
    const height = compact ? 430 : 310;
    const svg = d3.select(sequenceRoot).selectAll("svg").data([null]).join("svg")
      .attr("viewBox",`0 0 ${width} ${height}`).attr("role","img")
      .attr("aria-label","Animated message sequence across TRACE components");
    svg.selectAll("*").remove();
    const defs = svg.append("defs");
    defs.append("marker").attr("id","sequence-arrow").attr("viewBox","0 -5 10 10").attr("refX",8).attr("refY",0).attr("markerWidth",5).attr("markerHeight",5).attr("orient","auto").append("path").attr("d","M0,-5L10,0L0,5").attr("fill","#9ca9ba");
    const x = compact ? d3.scalePoint().domain(d3.range(4)).range([45,width-45]) : d3.scalePoint().domain(d3.range(actors.length)).range([48,width-48]);
    const actorPos = actors.map((name,i) => compact ? {name,x:x(i%4),y:i<4?34:238} : {name,x:x(i),y:34});
    svg.selectAll("text.sequence-label").data(actorPos).join("text").attr("class","sequence-label").attr("text-anchor","middle").attr("x",d=>d.x).attr("y",d=>d.y).text(d=>d.name);
    svg.selectAll("line.sequence-lane").data(actorPos).join("line").attr("class","sequence-lane").attr("x1",d=>d.x).attr("x2",d=>d.x).attr("y1",d=>d.y+12).attr("y2",d=>compact?(d.y<100?206:height-22):height-28);
    const rowY = i => compact ? (i<4 ? 75+i*34 : 280+(i-4)*34) : 78+i*28;
    const rendered = events.map((e,i)=>({...e,i,y:rowY(i),s:actorPos[e.from],t:actorPos[e.to]}));
    svg.selectAll("line.sequence-event").data(rendered).join("line").attr("class",d=>`sequence-event${d.i===sequenceIndex?" is-current":""}`).attr("x1",d=>d.s.x).attr("x2",d=>d.t.x).attr("y1",d=>d.y).attr("y2",d=>d.y);
    svg.selectAll("text.sequence-event-label").data(rendered).join("text").attr("class",d=>`sequence-event-label${d.i===sequenceIndex?" is-current":""}`).attr("text-anchor","middle").attr("x",d=>(d.s.x+d.t.x)/2).attr("y",d=>d.y-6).text(d=>d.label);
    const current = rendered[sequenceIndex];
    svg.append("circle").attr("class","sequence-token").attr("r",7).attr("cx",current.s.x).attr("cy",current.y)
      .transition().duration(sequencePaused?0:1150).ease(d3.easeCubicInOut).attr("cx",current.t.x);
    sequenceStep.textContent = `${sequenceIndex+1} / ${events.length}`;
    sequenceMessage.textContent = current.message;
  }

  function scheduleSequence() {
    window.clearTimeout(sequenceTimer);
    if (sequencePaused) return;
    sequenceTimer = window.setTimeout(() => {
      sequenceIndex = (sequenceIndex + 1) % events.length;
      drawSequence();
      scheduleSequence();
    }, 1750);
  }

  sequenceToggle.addEventListener("click", () => {
    sequencePaused = !sequencePaused;
    sequenceToggle.textContent = sequencePaused ? "Replay" : "Pause";
    sequenceToggle.setAttribute("aria-pressed", String(sequencePaused));
    if (!sequencePaused) {
      sequenceIndex = (sequenceIndex + 1) % events.length;
      drawSequence();
      scheduleSequence();
    } else {
      window.clearTimeout(sequenceTimer);
    }
  });
  drawSequence();
  scheduleSequence();
  new ResizeObserver(drawSequence).observe(sequenceRoot);
})();
