"use strict";
// Spiking-Evo watch: runs list, live world, individual inspector, metric charts.

const $ = (s) => document.querySelector(s);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const state = {
  runs: [], run: null, sid: null, info: null, frame: null, playing: false, busy: false,
  selected: null, layersOn: {}, series: {}, metricsSel: new Set(), expsSel: new Set(),
};

// -- helpers ---------------------------------------------------------------------------
function status(msg, error = false) {
  const el = $("#status");
  el.textContent = msg;
  el.className = error ? "error" : "";
}
async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : {});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "on") for (const [ev, fn] of Object.entries(v)) e.addEventListener(ev, fn);
    else if (k === "class") e.className = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c instanceof Node ? c : document.createTextNode(c));
  return e;
}
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "–"
  : Math.abs(x) >= 1000 ? x.toFixed(0) : x.toFixed(d);

// -- tabs ------------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab));
  if (b.dataset.tab === "metrics") renderMetrics();
  else drawWorld();
}));

// -- runs sidebar ----------------------------------------------------------------------
async function loadRuns() {
  state.runs = await api("/api/runs");
  const configs = await api("/api/configs");
  $("#configs").replaceChildren(...configs.map((c) => el("option", { value: c.path }, c.name)));
  renderRunList();
  setupMetricsControls();
  status(`${state.runs.length} run trovate`);
}
function groupRuns() {
  const groups = new Map();
  for (const r of state.runs) {
    const key = `${r.experiment} · ${r.source}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(r);
  }
  return groups;
}
function renderRunList() {
  const f = $("#filter").value.toLowerCase();
  const box = $("#runs");
  box.replaceChildren();
  for (const [exp, runs] of groupRuns()) {
    if (f && !exp.toLowerCase().includes(f)) continue;
    const seeds = el("div", { class: "seeds" }, ...runs.map((r) => el("button", {
      title: r.path, on: { click: () => selectRun(r) },
    }, r.seed)));
    const kind = runs[0].kind === "ecosystem" ? "ecosistema" : "generazioni";
    box.append(el("details", { class: "exp" },
      el("summary", {}, runs[0].experiment, " ", el("span", { class: "meta" }, `${runs[0].source} · ${runs.length} seed · ${kind}`)), seeds));
  }
}
$("#filter").addEventListener("input", renderRunList);
$("#start-config").addEventListener("click", () => startSession({ config: $("#configs").value }));

function selectRun(r) {
  state.run = r;
  const src = $("#source");
  src.replaceChildren(el("b", {}, `${r.experiment} / seed ${r.seed}`));
  if (r.kind === "generations") {
    const snaps = r.snapshots;
    const sel = el("input", { type: "range", min: 0, max: Math.max(0, snaps.length - 1), value: snaps.length - 1 });
    const lab = el("span", {}, snaps.length ? snaps[snaps.length - 1].replace(/\D/g, "").replace(/^0+(?=\d)/, "") : "–");
    sel.addEventListener("input", () => { lab.textContent = snaps[sel.value].replace(/\D/g, "").replace(/^0+(?=\d)/, ""); });
    sel.addEventListener("change", () => startSession({ run: r.path, genomes: snaps[sel.value] }));
    if (r.checkpoint) src.append(el("button", { on: { click: () => startSession({ run: r.path, genomes: "checkpoint" }) } }, "Popolazione finale"));
    src.append(el("button", { on: { click: () => startSession({ run: r.path, genomes: "random" }) } }, "Casuale (gen 0)"));
    if (snaps.length) src.append(el("label", {}, "migliore della gen.", sel, lab));
    startSession({ run: r.path, genomes: r.checkpoint ? "checkpoint" : (snaps.at(-1) || "random") });
  } else {
    src.append(el("span", { class: "hint" }, "ecosistema avviato dai fondatori della config"));
    startSession({ run: r.path });
  }
  // metrics: preselect this experiment
  state.expsSel = new Set([`${r.experiment} · ${r.source}`]);
  setupMetricsControls();
}

// -- live world ------------------------------------------------------------------------
async function startSession(body) {
  try {
    status("avvio della simulazione…");
    state.playing = false; $("#play").textContent = "▶";
    const info = await api("/api/sim", body);
    state.sid = info.sid; state.info = info; state.selected = null;
    $("#arena").max = info.arenas - 1;
    if (+$("#arena").value > info.arenas - 1) $("#arena").value = 0;
    status(info.kind === "ecosystem" ? `${info.label} · ecosistema, fino a ${info.agents} individui`
      : `${info.label} · ${info.agents} agenti in ${info.arenas} arene`);
    await tick(0);
    play(true);
  } catch (e) { status(e.message, true); }
}
async function tick(n) {
  if (!state.sid || state.busy) return;
  state.busy = true;
  try {
    const f = await api(`/api/frame?sid=${state.sid}&n=${n}&arena=${+$("#arena").value || 0}`);
    state.frame = f;
    syncLayerToggles(f.layers);
    drawWorld();
    if (state.selected !== null) await showAgent(state.selected);
  } catch (e) { status(e.message, true); play(false); } finally { state.busy = false; }
}
function play(on) {
  state.playing = on;
  $("#play").textContent = on ? "❚❚" : "▶";
  if (on) loop();
}
async function loop() {
  if (!state.playing) return;
  await tick(+$("#speed").value);
  if (state.playing) requestAnimationFrame(loop);
}
$("#play").addEventListener("click", () => play(!state.playing));
$("#step").addEventListener("click", () => { play(false); tick(1); });
$("#arena").addEventListener("change", () => { state.selected = null; tick(0); });
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target.tagName !== "INPUT") { e.preventDefault(); play(!state.playing); }
});

const LAYER_NAMES = { food: "cibo", hazard: "pericolo", soil: "suolo fertile", crop: "raccolto",
  poison: "veleno", signal: "segnali" };
function syncLayerToggles(layers) {
  const box = $("#layer-toggles");
  const names = Object.keys(layers);
  const have = [...box.querySelectorAll("input")].map((i) => i.value);
  if (names.join() === have.join()) return;
  box.replaceChildren(el("span", { class: "hint" }, "strati:"));
  for (const n of names) {
    if (!(n in state.layersOn)) state.layersOn[n] = true;
    const cb = el("input", { type: "checkbox", value: n });
    cb.checked = state.layersOn[n];
    cb.addEventListener("change", () => { state.layersOn[n] = cb.checked; drawWorld(); });
    box.append(el("label", {}, cb, LAYER_NAMES[layers[n].kind] || n));
  }
  const lines = el("input", { type: "checkbox", value: "_links" });
  lines.checked = state.layersOn._links ?? true;
  lines.addEventListener("change", () => { state.layersOn._links = lines.checked; drawWorld(); });
  if (state.info?.kind === "ecosystem") box.append(el("label", {}, lines, "genitore–figlio"));
}

function energyColor(e, max = 100) {
  // low energy -> red, high (relative to the richest visible agent) -> blue
  const t = Math.max(0, Math.min(1, e / Math.max(1, max)));
  const a = [214, 64, 54], b = [47, 111, 222];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * t)).join(",")})`;
}
function drawWorld() {
  const f = state.frame, info = state.info, cv = $("#grid");
  if (!f || !info) return;
  const wrap = cv.parentElement;
  const size = Math.max(200, Math.min(wrap.clientWidth - 24, wrap.clientHeight - 24));
  const cell = Math.max(2, Math.floor(size / Math.max(info.width, info.height)));
  const W = cell * info.width, H = cell * info.height, dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = css("--grid"); g.fillRect(0, 0, W, H);
  if (cell >= 8) {
    g.strokeStyle = css("--gridline"); g.lineWidth = 1; g.beginPath();
    for (let x = 0; x <= info.width; x++) { g.moveTo(x * cell + .5, 0); g.lineTo(x * cell + .5, H); }
    for (let y = 0; y <= info.height; y++) { g.moveTo(0, y * cell + .5); g.lineTo(W, y * cell + .5); }
    g.stroke();
  }
  const each = (data, fn) => { for (let k = 0; k < data.length; k++) if (data[k]) fn(k % info.width, Math.floor(k / info.width), data[k]); };
  const order = ["soil", "hazard", "crop", "poison", "food", "signal"];
  const layers = Object.entries(f.layers).sort((a, b) => order.indexOf(a[1].kind) - order.indexOf(b[1].kind));
  const legend = [];
  for (const [name, L] of layers) {
    if (!state.layersOn[name]) continue;
    if (L.kind === "soil") { g.fillStyle = `rgba(${css("--soil")},.22)`; each(L.data, (x, y) => g.fillRect(x * cell, y * cell, cell, cell)); legend.push([`rgba(${css("--soil")},.5)`, "suolo fertile"]); }
    if (L.kind === "hazard") { const mx = Math.max(...L.data) || 1; each(L.data, (x, y, v) => { g.fillStyle = `rgba(${css("--hazard")},${.15 + .45 * v / mx})`; g.fillRect(x * cell, y * cell, cell, cell); }); legend.push([`rgb(${css("--hazard")})`, "pericolo"]); }
    if (L.kind === "crop") { g.fillStyle = css("--crop"); each(L.data, (x, y) => g.fillRect(x * cell + cell * .25, y * cell + cell * .25, cell * .5, cell * .5)); legend.push([css("--crop"), "raccolto"]); }
    if (L.kind === "poison") { g.fillStyle = `rgba(${css("--poison")},.55)`; each(L.data, (x, y) => g.fillRect(x * cell, y * cell, cell, cell)); legend.push([`rgb(${css("--poison")})`, "veleno"]); }
    if (L.kind === "food") { g.fillStyle = css("--food"); each(L.data, (x, y, v) => { g.beginPath(); g.arc((x + .5) * cell, (y + .5) * cell, Math.min(cell * .45, cell * (.18 + .07 * v)), 0, 7); g.fill(); }); legend.push([css("--food"), "cibo"]); }
    if (L.kind === "signal") { g.strokeStyle = css("--signal"); g.lineWidth = 2; each(L.data, (x, y) => { g.beginPath(); g.arc((x + .5) * cell, (y + .5) * cell, cell * 1.4, 0, 7); g.stroke(); }); legend.push([css("--signal"), "segnale emesso"]); }
  }
  // parent-child links (ecosystem)
  const byId = new Map(f.agents.map((a) => [a.id, a]));
  if (info.kind === "ecosystem" && (state.layersOn._links ?? true)) {
    g.strokeStyle = css("--muted"); g.globalAlpha = .45; g.lineWidth = 1; g.beginPath();
    for (const a of f.agents) {
      const p = byId.get(a.parent);
      if (p && a.stage === 1 && Math.abs(p.x - a.x) < info.width / 2 && Math.abs(p.y - a.y) < info.height / 2) {
        g.moveTo((a.x + .5) * cell, (a.y + .5) * cell); g.lineTo((p.x + .5) * cell, (p.y + .5) * cell);
      }
    }
    g.stroke(); g.globalAlpha = 1;
  }
  // agents: triangle pointing to heading (0 N, 1 E, 2 S, 3 W)
  const emax = Math.max(1, ...f.agents.map((a) => a.e));
  for (const a of f.agents) {
    const cx = (a.x + .5) * cell, cy = (a.y + .5) * cell;
    const r = cell * (a.stage === 1 ? .32 : .46);
    const ang = [-Math.PI / 2, 0, Math.PI / 2, Math.PI][a.h];
    g.save(); g.translate(cx, cy); g.rotate(ang);
    g.beginPath(); g.moveTo(r, 0); g.lineTo(-r * .8, r * .75); g.lineTo(-r * .8, -r * .75); g.closePath();
    g.fillStyle = !a.alive ? css("--muted") : a.stage === 1 ? css("--juvenile") : energyColor(a.e, emax);
    g.globalAlpha = a.alive ? 1 : .35; g.fill(); g.globalAlpha = 1;
    if (a.i === state.selected) { g.lineWidth = 2; g.strokeStyle = css("--text"); g.stroke(); }
    g.restore();
    if (a.emit) { g.strokeStyle = css("--signal"); g.lineWidth = 2; g.beginPath(); g.arc(cx, cy, r * 1.6, 0, 7); g.stroke(); }
  }
  legend.push([energyColor(1, 1), "adulto: blu = più energia, rosso = meno"]);
  if (info.kind === "ecosystem") legend.push([css("--juvenile"), "piccolo"]);
  $("#legend").replaceChildren(...legend.map(([c, t]) => el("span", {}, el("i", { style: `background:${c}` }), t)));
  const bits = [`passo ${f.tick}`];
  if (f.episode !== undefined) bits.push(`episodio ${f.episode}`);
  if (f.season) bits.push(`stagione: ${f.season}`);
  if (f.population !== undefined) bits.push(`popolazione ${f.population} (piccoli ${f.juveniles}) · generazione max ${f.max_generation}`);
  else bits.push(`vivi ${f.alive}/${f.agents.length} · energia media raccolta ${fmt(f.energy_gained, 1)}`);
  $("#hud").textContent = bits.join(" · ");
  cv.onclick = (ev) => {
    const r = cv.getBoundingClientRect();
    const x = (ev.clientX - r.left) / cell, y = (ev.clientY - r.top) / cell;
    let best = null, bd = 2.5;
    for (const a of f.agents) { const d = Math.hypot(a.x + .5 - x, a.y + .5 - y); if (d < bd) { bd = d; best = a; } }
    state.selected = best ? best.i : null;
    drawWorld();
    if (best) showAgent(best.i); else $("#agent-panel").replaceChildren(el("p", { class: "hint" }, "Nessuna creatura qui."));
  };
}
window.addEventListener("resize", () => { drawWorld(); renderMetrics(); });

// -- individual inspector -------------------------------------------------------------
const HEADINGS = ["nord", "est", "sud", "ovest"];
async function showAgent(i) {
  let a;
  try { a = await api(`/api/agent?sid=${state.sid}&i=${i}`); } catch (e) { status(e.message, true); return; }
  const p = $("#agent-panel");
  const kv = el("div", { class: "kv" });
  const add = (k, v) => kv.append(el("b", {}, k), el("span", {}, v));
  add("individuo", a.id !== undefined ? `#${a.id} (genitore #${a.parent}, gen. ${a.generation})` : `agente ${a.i}`);
  add("stato", a.alive ? (a.stage === 1 ? "piccolo, vivo" : "vivo") : "morto");
  add("energia", `${fmt(a.energy, 1)} (raccolta ${fmt(a.energy_gained, 1)})`);
  add("età", `${a.age} passi`);
  add("posizione", `(${a.x}, ${a.y}) verso ${HEADINGS[a.heading]}`);
  if (a.traits) add("geni di vita", Object.entries(a.traits).map(([k, v]) => `${k} ${fmt(v)}`).join(", "));
  const b = a.brain;
  add("cervello", `${b.n_neurons} neuroni (${b.n_hidden} nascosti), ${b.src.length} sinapsi`);

  const bars = el("div", { class: "bars" });
  for (const [k, v] of Object.entries(a.inputs)) {
    bars.append(el("span", {}, k), el("div", { class: "bar" }, el("div", { style: `width:${Math.round(100 * Math.min(1, Math.abs(v)))}%` })), el("span", {}, fmt(v)));
  }
  const motor = el("div", { class: "motor" });
  const m = a.motor;
  const mv = [["avanti", m.forward === 1], ["indietro", m.forward === -1], ["gira a sinistra", m.turn === -1], ["gira a destra", m.turn === 1]];
  a.actions.forEach((on, k) => mv.push([`azione ${k}`, on]));
  for (const [t, on] of mv) motor.append(el("span", { class: on ? "on" : "" }, t));

  const brain = el("canvas", { id: "brain" });
  p.replaceChildren(el("h2", {}, "Individuo"), kv, el("h2", {}, "Sensi (ingressi adesso)"), bars,
    el("h2", {}, "Uscite di questo passo"), motor,
    el("h2", {}, "Cervello (giallo = ha sparato in questo passo)"), brain);
  drawBrain(brain, b);
}
function drawBrain(cv, b) {
  const W = cv.clientWidth, H = 280, dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const nIn = b.n_inputs, nOut = b.n_outputs, n = b.n_neurons;
  const pos = new Array(n);
  const col = (i, k, x) => [x, 16 + (H - 32) * (k === 1 ? .5 : i / (k - 1))];
  for (let i = 0; i < nIn; i++) pos[i] = col(i, nIn, 14);
  for (let i = 0; i < nOut; i++) pos[nIn + i] = col(i, nOut, W - 14);
  const hidden = n - nIn - nOut;
  for (let h = 0; h < hidden; h++) {
    const ang = (2 * Math.PI * h) / Math.max(1, hidden);
    pos[nIn + nOut + h] = [W / 2 + (W * .2) * Math.cos(ang), H / 2 + (H * .36) * Math.sin(ang)];
  }
  const mw = Math.max(1e-9, ...b.weight.map(Math.abs));
  for (let s = 0; s < b.src.length; s++) {
    const p = pos[b.src[s]], q = pos[b.dst[s]];
    if (!p || !q) continue;
    const w = b.weight[s];
    g.strokeStyle = `rgba(${css(w >= 0 ? "--pos" : "--neg")},${.15 + .7 * Math.abs(w) / mw})`;
    g.lineWidth = .5 + 2 * Math.abs(w) / mw;
    g.beginPath(); g.moveTo(...p);
    if (b.src[s] === b.dst[s]) g.arc(p[0] + 6, p[1], 6, Math.PI, 3 * Math.PI);
    else g.lineTo(...q);
    g.stroke();
  }
  g.font = "10px system-ui"; g.textBaseline = "middle";
  for (let i = 0; i < n; i++) {
    const [x, y] = pos[i];
    g.beginPath(); g.arc(x, y, 5, 0, 7);
    g.fillStyle = b.spikes[i] > 0 ? css("--spike") : b.dormant[i] ? css("--gridline") : css("--panel");
    g.fill(); g.strokeStyle = css("--muted"); g.lineWidth = 1; g.stroke();
    g.fillStyle = css("--muted");
    if (i < nIn) { g.textAlign = "left"; g.fillText(b.input_names[i] || `in${i}`, x + 8, y); }
    else if (i < nIn + nOut) { g.textAlign = "right"; g.fillText(b.output_names[i - nIn], x - 8, y); }
  }
}

// -- metrics ---------------------------------------------------------------------------
const PALETTE = ["#2f6fde", "#e0601e", "#2a9d5c", "#a33ac2", "#c9a227", "#d6406f", "#1aa3b8", "#6b6b6b"];
function experiments() { return [...groupRuns().keys()]; }
function setupMetricsControls() {
  const exps = experiments();
  if (!state.expsSel.size && exps.length) state.expsSel.add(exps[0]);
  const box = $("#m-exps");
  box.replaceChildren(el("span", { class: "hint" }, "esperimenti:"));
  exps.forEach((e, k) => {
    const cb = el("input", { type: "checkbox" }); cb.checked = state.expsSel.has(e);
    cb.addEventListener("change", () => { cb.checked ? state.expsSel.add(e) : state.expsSel.delete(e); renderMetrics(); });
    box.append(el("label", {}, cb, el("i", { style: `background:${PALETTE[k % PALETTE.length]}` }), e));
  });
  if ($("#metrics").classList.contains("active")) renderMetrics();
}
async function seriesOf(run) {
  if (!state.series[run.path]) state.series[run.path] = await api(`/api/series?run=${encodeURIComponent(run.path)}`);
  return state.series[run.path];
}
let metricsToken = 0;
async function renderMetrics() {
  if (!$("#metrics").classList.contains("active")) return;
  const token = ++metricsToken;
  const exps = experiments().filter((e) => state.expsSel.has(e));
  const groups = groupRuns();
  status("lettura dei CSV…");
  let data;
  try {
    data = await Promise.all(exps.map(async (e) => [e, await Promise.all(groups.get(e).map(seriesOf))]));
  } catch (err) { status(err.message, true); return; }
  if (token !== metricsToken) return;
  status(`${exps.length} esperimenti`);
  // files and columns available
  const files = new Set();
  for (const [, ss] of data) for (const s of ss) Object.keys(s).forEach((f) => files.add(f));
  const fileSel = $("#m-file");
  const prevFile = fileSel.value;
  fileSel.replaceChildren(...[...files].map((f) => el("option", { value: f }, f)));
  fileSel.value = files.has(prevFile) ? prevFile : [...files][0] || "";
  fileSel.onchange = () => { state.metricsSel.clear(); renderMetrics(); };
  const file = fileSel.value;
  const cols = new Set();
  for (const [, ss] of data) for (const s of ss) if (s[file]) Object.keys(s[file]).forEach((c) => cols.add(c));
  const numeric = [...cols].filter((c) => data.some(([, ss]) => ss.some((s) => s[file] && typeof s[file][c]?.find((v) => v !== null) === "number")));
  const xs = numeric.filter((c) => ["generation", "tick"].includes(c));
  const xSel = $("#m-x");
  xSel.replaceChildren(...(xs.length ? xs : ["(riga)"]).map((c) => el("option", { value: c }, c)));
  xSel.onchange = renderMetrics;
  const xcol = xs.length ? xSel.value || xs[0] : null;
  const defaults = ["fitness_mean", "energy_gained_mean", "alive_fraction", "population", "visited_cells_mean", "synapses_mean", "feed_events", "births"];
  if (!state.metricsSel.size) numeric.filter((c) => defaults.includes(c)).slice(0, 4).forEach((c) => state.metricsSel.add(c));
  if (!state.metricsSel.size) numeric.filter((c) => c !== xcol).slice(0, 2).forEach((c) => state.metricsSel.add(c));
  const colBox = $("#m-cols");
  colBox.replaceChildren(el("h2", {}, "Metriche"));
  for (const c of numeric.filter((c) => c !== xcol)) {
    const cb = el("input", { type: "checkbox" }); cb.checked = state.metricsSel.has(c);
    cb.addEventListener("change", () => { cb.checked ? state.metricsSel.add(c) : state.metricsSel.delete(c); renderMetrics(); });
    colBox.append(el("label", {}, cb, " ", c));
  }
  const charts = $("#charts");
  charts.replaceChildren();
  const colorOf = (e) => PALETTE[experiments().indexOf(e) % PALETTE.length];
  const summary = [];
  for (const c of [...state.metricsSel].filter((c) => numeric.includes(c))) {
    const lines = data.map(([e, ss]) => ({
      exp: e, color: colorOf(e),
      seeds: ss.filter((s) => s[file]?.[c]).map((s) => ({
        x: xcol ? s[file][xcol] : s[file][c].map((_, k) => k), y: s[file][c],
      })),
    }));
    const cv = el("canvas");
    charts.append(el("div", { class: "chart" }, el("h3", {}, c), cv));
    drawChart(cv, lines, xcol || "riga");
    for (const L of lines) {
      const last = L.seeds.map((s) => s.y.filter((v) => typeof v === "number").at(-1)).filter((v) => v !== undefined);
      if (last.length) summary.push([c, L.exp, last]);
    }
  }
  const tbl = el("table", {}, el("tr", {}, el("th", {}, "metrica (ultimo valore)"), el("th", {}, "esperimento"), el("th", {}, "seed"), el("th", {}, "media"), el("th", {}, "dev. std"), el("th", {}, "min"), el("th", {}, "max")));
  for (const [c, e, v] of summary) {
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) : 0;
    tbl.append(el("tr", {}, el("td", {}, c), el("td", {}, e), el("td", {}, v.length), el("td", {}, fmt(m)), el("td", {}, fmt(sd)), el("td", {}, fmt(Math.min(...v))), el("td", {}, fmt(Math.max(...v)))));
  }
  $("#m-summary").replaceChildren(tbl);
}
function drawChart(cv, lines, xlabel) {
  const W = cv.clientWidth || 400, H = 220, dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const pad = { l: 52, r: 10, t: 8, b: 30 };
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const L of lines) for (const s of L.seeds) s.x.forEach((x, k) => {
    const y = s.y[k]; if (typeof y !== "number" || typeof x !== "number") return;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  });
  if (!Number.isFinite(x0)) return;
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const X = (x) => pad.l + (W - pad.l - pad.r) * (x - x0) / Math.max(1e-9, x1 - x0);
  const Y = (y) => H - pad.b - (H - pad.t - pad.b) * (y - y0) / (y1 - y0);
  g.strokeStyle = css("--gridline"); g.fillStyle = css("--muted"); g.font = "11px system-ui"; g.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const y = y0 + (y1 - y0) * k / 4; g.beginPath(); g.moveTo(pad.l, Y(y)); g.lineTo(W - pad.r, Y(y)); g.stroke();
    g.textAlign = "right"; g.textBaseline = "middle"; g.fillText(fmt(y, Math.abs(y1 - y0) < 5 ? 2 : 0), pad.l - 6, Y(y));
  }
  g.textAlign = "center"; g.textBaseline = "top";
  for (let k = 0; k <= 4; k++) { const x = x0 + (x1 - x0) * k / 4; g.fillText(fmt(x, 0), X(x), H - pad.b + 6); }
  g.fillText(xlabel, W / 2, H - 14);
  for (const L of lines) {
    // mean and sd across seeds at each common index
    const n = Math.min(...L.seeds.map((s) => s.y.length));
    if (!Number.isFinite(n) || !n) continue;
    g.globalAlpha = L.seeds.length > 8 ? .07 : .18; g.strokeStyle = L.color; g.lineWidth = 1;
    for (const s of L.seeds) { g.beginPath(); s.y.forEach((y, k) => { if (typeof y === "number") (k ? g.lineTo(X(s.x[k]), Y(y)) : g.moveTo(X(s.x[k]), Y(y))); }); g.stroke(); }
    const mean = [], sd = [], xs = L.seeds[0].x.slice(0, n);
    for (let k = 0; k < n; k++) {
      const v = L.seeds.map((s) => s.y[k]).filter((y) => typeof y === "number");
      const m = v.reduce((a, b) => a + b, 0) / Math.max(1, v.length);
      mean.push(m); sd.push(v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) : 0);
    }
    if (L.seeds.length > 1) {
      g.globalAlpha = .15; g.fillStyle = L.color; g.beginPath();
      xs.forEach((x, k) => (k ? g.lineTo(X(x), Y(mean[k] + sd[k])) : g.moveTo(X(x), Y(mean[k] + sd[k]))));
      for (let k = n - 1; k >= 0; k--) g.lineTo(X(xs[k]), Y(mean[k] - sd[k]));
      g.closePath(); g.fill();
    }
    g.globalAlpha = 1; g.strokeStyle = L.color; g.lineWidth = 2; g.beginPath();
    xs.forEach((x, k) => (k ? g.lineTo(X(x), Y(mean[k])) : g.moveTo(X(x), Y(mean[k]))));
    g.stroke();
  }
}

loadRuns().catch((e) => status(e.message, true));
