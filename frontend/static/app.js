/* Agent pipeline page.
 *
 * 1. GET /api/options fills the record and ICU-stay menus and the LLM chip.
 * 2. Run opens an EventSource on /api/run; each "stage" event fills one card.
 *    Step mode holds arrived stages until Step is clicked.
 * 3. After "done", the chat (POST /api/chat) and the outcome reveal unlock.
 *
 * The stream is closed on done, on error and on any connection error: an
 * EventSource left open reconnects and would run the case again.
 * Text from the server (including LLM text) is only ever inserted as text
 * nodes, never as HTML.
 */
"use strict";

const STAGES = [
  [1, "Raw ECG", "The recording as stored, 360 samples a second, on ECG paper."],
  [2, "Cleaned", "A band-pass filter removes baseline wander and high-frequency noise."],
  [3, "Beats found", "Where each heartbeat's R peak is."],
  [4, "Beats cut out", "A 0.83 s window around each beat: what the model reads."],
  [5, "Model says", "The ECG model's label for every beat. The cardiologist's label is shown below it for you only; the AI never sees it."],
  [6, "Heart agent", "Rhythm facts from the beat labels."],
  [7, "Chart agent", "Facts from the ICU stay's first 24 hours (nothing after the 24-hour mark)."],
  [8, "Rules", "Draft rules that match the facts. Each names its source."],
  [9, "Report", "Written by the local LLM and fact-checked, or the plain template."],
];
const CIRCLED = "①②③④⑤⑥⑦⑧⑨";
const CLASS_VAR = { N: "--ink-3", S: "--s1", V: "--crit", F: "--s2", Q: "--q" };
const CLASS_NAME = { N: "normal", S: "supraventricular", V: "ventricular", F: "fusion", Q: "unclassifiable" };

const $ = (id) => document.getElementById(id);
const state = {
  options: null, source: null, running: false, stepMode: false, credits: 0, queue: [],
  caseId: null, stay: null, data: {}, knownIds: new Set(), beatHighlight: new Set(),
  redraws: new Map(), timer: null,
};

/* ---------- small DOM helpers ---------- */

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("data-") || key === "role") node.setAttribute(key, value);
    else node[key] = value;
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function showError(message) { $("error").textContent = message; $("error").hidden = false; }
function hideError() { $("error").hidden = true; }

function idTag(id) {
  return el("span", { class: "fid", "data-id": id, role: "button", tabIndex: 0, title: "Highlight " + id }, id);
}

/* Turn "[H1, C3]" style IDs inside server text into clickable tags. Text only. */
function linkify(text) {
  const frag = document.createDocumentFragment();
  const re = /\b([HCRG]\d{1,3})\b/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (!state.knownIds.has(m[1]) && m[1][0] !== "G") continue;
    frag.append(text.slice(last, m.index), idTag(m[1]));
    last = m.index + m[0].length;
  }
  frag.append(text.slice(last));
  return frag;
}

/* ---------- options ---------- */

async function loadOptions() {
  try {
    const resp = await fetch("/api/options");
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const o = await resp.json();
    state.options = o;
    fillRecords(o.records);
    fillStays(o.stays);
    showLlm(o.llm);
    const missing = Object.values(o.unavailable || {});
    if (missing.length) showError(missing.join(" · "));
    if (!o.outcome_available) $("reveal").title = "The outcome table is not available";
  } catch (err) {
    showError("Could not load the options (" + err.message + "). Is the server running?");
  }
}

function fillRecords(records) {
  const select = $("record");
  const groups = [
    ["held-out", "Held-out: never used to train the model (honest test)"],
    ["trained-on", "Trained-on: the model studied these (looks better than it is)"],
  ];
  for (const [tag, label] of groups) {
    const group = el("optgroup", { label });
    for (const r of records.filter((x) => x.tag === tag)) {
      const extra = (r.note ? " · " + r.note : "") + (r.seconds === null ? " · not on disk" : "");
      group.append(el("option", { value: r.name, text: r.name + " · " + tag + extra, disabled: r.seconds === null }));
    }
    select.append(group);
  }
}

function fillStays(stays) {
  const select = $("stay");
  const groups = [
    [(s) => s.split !== "train", "Not used to train the risk model (validation / test)"],
    [(s) => s.split === "train", "The risk model was trained on these patients"],
  ];
  for (const [keep, label] of groups) {
    const group = el("optgroup", { label });
    for (const s of stays.filter(keep)) {
      const text = s.id + " · " + s.age + " " + s.sex + " · " + (s.reason || "reason not recorded");
      group.append(el("option", { value: String(s.id), text }));
    }
    if (group.children.length) select.append(group);
  }
}

function showLlm(llm) {
  const chip = $("llm-chip");
  chip.className = "chip " + (llm.ok ? "ok" : "off");
  chip.textContent = llm.ok ? "LLM ● " + llm.model + " ready" : "LLM ○ off: template reports, chat off";
  chip.title = llm.message || "";
}

function recordInfo() {
  const name = $("record").value;
  return state.options && name ? state.options.records.find((r) => r.name === name) : null;
}

function onRecordChange() {
  const rec = recordInfo();
  const hint = $("record-tag");
  if (!rec) { hint.textContent = ""; return; }
  hint.textContent = rec.tag === "held-out"
    ? "Record " + rec.name + " is held-out: the model never saw it, so stage ⑤ is an honest test."
    : "Record " + rec.name + " is trained-on: the model studied it, so stage ⑤ will look better than it really is.";
  if (rec.note) hint.textContent += " (" + rec.note + ")";
  if (rec.seconds) $("start").max = String(Math.max(0, Math.floor(rec.seconds - Number($("seconds").value))));
}

/* ---------- stage cards ---------- */

function buildStages() {
  const root = $("stages");
  for (const [n, title, help] of STAGES) {
    root.append(el("article", { class: "stage card waiting", id: "stage-" + n },
      el("header", {}, el("span", { class: "num" }, CIRCLED[n - 1]), el("h2", {}, title),
        el("span", { class: "state" }, "waiting")),
      el("p", { class: "hint" }, help),
      el("div", { class: "body" })));
  }
}

function card(n) { return $("stage-" + n); }

function setCard(n, cls, label) {
  const c = card(n);
  if (!c) return;
  c.className = "stage card " + cls;
  c.querySelector(".state").textContent = label;
}

function resetStages() {
  for (const [n] of STAGES) {
    setCard(n, "waiting", "waiting");
    card(n).querySelector(".body").replaceChildren();
  }
  state.redraws.clear();
  state.beatHighlight.clear();
  state.knownIds.clear();
  stopTimer();
}

function stopTimer() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

function markWorking(n) {
  const c = card(n);
  if (!c || !c.classList.contains("waiting")) return;
  setCard(n, "working", "working…");
  if (n === 9) {
    const t0 = Date.now();
    stopTimer();
    state.timer = setInterval(() => {
      if (!card(9).classList.contains("working")) return stopTimer();
      card(9).querySelector(".state").textContent = "writing the report… " + Math.round((Date.now() - t0) / 1000) + " s";
    }, 500);
  }
}

/* ---------- drawing ---------- */

function canvasIn(body, cls) {
  const c = el("canvas", { class: cls });
  body.append(c);
  return c;
}

function draw(key, canvas, paint) {
  state.redraws.set(key, () => paint(canvas));
  paint(canvas);
}

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600;
  const h = canvas.clientHeight || 150;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h];
}

function paper(ctx, w, h, seconds) {
  const pxPerS = w / seconds;
  for (const [step, color] of [[0.04, cssVar("--grid")], [0.2, cssVar("--grid-b")], [1, cssVar("--grid-b")]]) {
    const gap = step * pxPerS;
    if (gap < 5) continue;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= w; x += gap) { ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, h); }
    for (let y = h; y >= 0; y -= gap) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(w, Math.round(y) + 0.5); }
    ctx.stroke();
  }
}

function range(arrays) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrays) for (const v of a) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!Number.isFinite(lo)) return [-1, 1];
  if (hi - lo < 1e-6) return [lo - 0.5, hi + 0.5];
  return [lo, hi];
}

function median(a) {
  const s = Float64Array.from(a).sort();
  return s.length ? s[Math.floor(s.length / 2)] : 0;
}

/* One trace; per-pixel min/max when there are more samples than pixels, so
   no spike disappears. */
function trace(ctx, y, w, top, bottom, lo, hi, color, width = 1.2, alpha = 1) {
  const n = y.length;
  if (!n) return;
  const sx = w / n;
  const scale = (bottom - top) / (hi - lo);
  const Y = (v) => bottom - (v - lo) * scale;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = "round";
  ctx.beginPath();
  if (n <= w * 2) {
    for (let i = 0; i < n; i++) { if (i) ctx.lineTo(i * sx, Y(y[i])); else ctx.moveTo(0, Y(y[0])); }
  } else {
    for (let px = 0; px < w; px++) {
      const a = Math.floor(px / sx);
      const b = Math.min(n, Math.floor((px + 1) / sx) + 1);
      let mn = Infinity;
      let mx = -Infinity;
      for (let i = a; i < b; i++) { if (y[i] < mn) mn = y[i]; if (y[i] > mx) mx = y[i]; }
      if (px) ctx.lineTo(px + 0.5, Y(mx)); else ctx.moveTo(0.5, Y(mx));
      ctx.lineTo(px + 0.5, Y(mn));
    }
  }
  ctx.stroke();
  ctx.restore();
}

function stripPainter(signal, fs, extra) {
  return (canvas) => {
    const [ctx, w, h] = setup(canvas);
    const seconds = signal.length / fs;
    paper(ctx, w, h, seconds);
    const top = extra && extra.labels ? 26 : 8;
    const bottom = h - (extra && extra.labels ? 22 : 8);
    const [lo, hi] = range([signal, ...((extra && extra.under) || [])]);
    for (const u of (extra && extra.under) || []) trace(ctx, u, w, top, bottom, lo, hi, cssVar("--ink-3"), 1, 0.45);
    if (extra && extra.highlight) {
      ctx.fillStyle = cssVar("--s1");
      ctx.globalAlpha = 0.12;
      for (const x of extra.highlight(w / signal.length)) ctx.fillRect(x - 9, 0, 18, h);
      ctx.globalAlpha = 1;
    }
    trace(ctx, signal, w, top, bottom, lo, hi, (extra && extra.color) || cssVar("--trace"));
    if (extra && extra.after) extra.after(ctx, w, h, w / signal.length);
  };
}

/* ---------- renderers, one per stage ---------- */

const RENDER = {
  1(ev, body) {
    const r = ev.record;
    body.append(el("p", { class: "note" },
      "Record " + r.name + " (" + r.tag + (r.note ? "; " + r.note : "") + "), lead " + ev.lead + ", " +
      (ev.signal.length / ev.fs).toFixed(1) + " s starting at " + ev.start_s + " s."));
    draw(1, canvasIn(body, "strip"), stripPainter(ev.signal, ev.fs));
  },
  2(ev, body) {
    const raw = state.data[1] ? state.data[1].signal : [];
    const m = median(raw);
    const rawCentered = raw.map((v) => v - m);
    body.append(el("p", { class: "note" }, "Blue: after the filter. Grey: the raw signal, for comparison."));
    draw(2, canvasIn(body, "strip"), stripPainter(ev.signal, state.data[1].fs,
      { under: [rawCentered], color: cssVar("--clean") }));
  },
  3(ev, body) {
    const how = ev.source === "annotation" ? "the cardiologist's marks" : "the automatic detector";
    body.append(el("p", { class: "note" }, ev.r_peaks.length + " beat positions, from " + how + "."));
    const sig = state.data[2].signal;
    draw(3, canvasIn(body, "strip"), stripPainter(sig, state.data[1].fs, {
      color: cssVar("--clean"),
      after(ctx, w, h, sx) {
        ctx.fillStyle = cssVar("--s2");
        for (const p of ev.r_peaks) {
          const x = p * sx;
          ctx.beginPath(); ctx.moveTo(x - 4, 2); ctx.lineTo(x + 4, 2); ctx.lineTo(x, 9); ctx.closePath(); ctx.fill();
        }
      },
    }));
  },
  4(ev, body) {
    body.append(el("p", { class: "note" },
      ev.n_classified + " beats cut out; " + ev.n_dropped + " at the window edges could not be (no full window or no neighbour)." +
      (ev.examples.length ? " The first " + ev.examples.length + ", as the model sees them:" : "")));
    const grid = el("div", { class: "beats" });
    body.append(grid);
    ev.examples.forEach((x, i) => {
      const c = el("canvas", { class: "beat" });
      grid.append(c);
      draw("4-" + i, c, (canvas) => {
        const [ctx, w, h] = setup(canvas);
        const [lo, hi] = range([x]);
        trace(ctx, x, w, 6, h - 6, lo, hi, cssVar("--trace"), 1.1);
      });
    });
  },
  5(ev, body) {
    const shown = ev.display_only;
    const truth = shown.truth;
    body.append(el("div", { class: "legend" },
      ...Object.keys(CLASS_VAR).map((k) => el("span", {}, el("b", { class: "c-" + k }, k), " " + CLASS_NAME[k])),
      el("span", {}, "top: model · bottom: cardiologist · ring: they disagree")));
    const canvas = canvasIn(body, "strip tall");
    const fs = state.data[1].fs;
    const sig = state.data[2].signal;
    const beatX = (sx) => ev.r_peaks.map((p) => p * sx);
    draw(5, canvas, stripPainter(sig, fs, {
      labels: true,
      color: cssVar("--clean"),
      highlight: (sx) => beatX(sx).filter((_, i) => state.beatHighlight.has(i)),
      after(ctx, w, h, sx) {
        ctx.textAlign = "center";
        ctx.font = "600 12px ui-monospace, Menlo, monospace";
        beatX(sx).forEach((x, i) => {
          const call = ev.called[i];
          ctx.fillStyle = cssVar(CLASS_VAR[call] || "--ink");
          ctx.fillText(call, x, 15);
          if (truth[i]) {
            ctx.fillStyle = cssVar("--ink-3");
            ctx.fillText(truth[i], x, h - 6);
            if (truth[i] !== call) {
              ctx.strokeStyle = cssVar("--crit");
              ctx.lineWidth = 1.5;
              ctx.beginPath(); ctx.arc(x, 11, 9, 0, Math.PI * 2); ctx.stroke();
            }
          }
        });
      },
    }));
    canvas.addEventListener("mousemove", (e) => beatTooltip(e, canvas, ev, truth));
    canvas.addEventListener("mouseleave", () => { $("tooltip").hidden = true; });
    body.append(el("p", { class: "note" },
      "Model and cardiologist agree on " + shown.agree + " of " + shown.of +
      " beats. Display only: this comparison is never given to the AI. Scores are uncalibrated."));
  },
  6(ev, body) { findingList(ev.findings, body); },
  7(ev, body) { findingList(ev.findings, body); },
  8(ev, body) {
    if (!ev.matches.length) { body.append(el("p", { class: "note" }, "No rule matched.")); return; }
    for (const m of ev.matches) state.knownIds.add(m.id);
    body.append(el("ul", { class: "findings" }, ev.matches.map((m) => el("li", { "data-id": m.id },
      idTag(m.id), m.statement + " ",
      el("span", { class: "caveat" }, "based on "), ...m.evidence.matched.map(idTag),
      el("span", { class: "badge warn" }, m.evidence.status),
      el("span", { class: "source" }, "Source: " + m.evidence.source)))));
  },
  9(ev, body) {
    setBanner(ev.banner, ev.simulated);
    const r = ev.report;
    const ai = r.kind === "llm";
    body.append(el("p", {},
      el("span", { class: "badge " + (ai ? "good" : "") },
        ai ? "AI-written (" + r.model + "), fact-checked ✓" : "Template report"),
      r.note ? el("span", { class: "caveat" }, " " + r.note) : null,
      ai ? el("span", { class: "caveat" }, " · " + r.seconds + " s, attempt " + r.attempts) : null));
    if (r.problems.length) {
      body.append(el("details", {}, el("summary", {}, "Why the AI draft was rejected"),
        el("ul", {}, r.problems.map((p) => el("li", {}, p)))));
    }
    body.append(el("pre", { class: "report" }, linkify(r.text)));
  },
};

function findingList(findings, body) {
  for (const f of findings) state.knownIds.add(f.id);
  body.append(el("ul", { class: "findings" }, findings.map((f) => el("li", { "data-id": f.id },
    idTag(f.id), f.statement, f.caveat ? el("span", { class: "caveat" }, " (" + f.caveat + ")") : null))));
}

function beatTooltip(e, canvas, ev, truth) {
  const rect = canvas.getBoundingClientRect();
  const sx = rect.width / state.data[2].signal.length;
  const x = e.clientX - rect.left;
  let best = -1;
  let dist = 14;
  ev.r_peaks.forEach((p, i) => { const d = Math.abs(p * sx - x); if (d < dist) { dist = d; best = i; } });
  const tip = $("tooltip");
  if (best < 0) { tip.hidden = true; return; }
  const scores = ev.classes.map((c, k) => c + " " + ev.scores[best][k].toFixed(2)).join("  ");
  tip.textContent = "beat " + (best + 1) + "\nmodel: " + ev.called[best] + "   cardiologist: " +
    (truth[best] || "–") + "\nscores: " + scores;
  tip.hidden = false;
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
  tip.style.top = (e.clientY + 14) + "px";
}

/* ---------- highlighting ---------- */

function select(id) {
  for (const node of document.querySelectorAll(".hl")) node.classList.remove("hl");
  const ids = new Set([id]);
  const rule = state.data[8] && state.data[8].matches.find((m) => m.id === id);
  if (rule) for (const m of rule.evidence.matched) ids.add(m);
  state.beatHighlight.clear();
  const heart = state.data[6] ? state.data[6].findings : [];
  for (const one of ids) {
    for (const node of document.querySelectorAll('[data-id="' + CSS.escape(one) + '"]')) node.classList.add("hl");
    const f = heart.find((x) => x.id === one);
    for (const [a, b] of (f && f.evidence && f.evidence.beat_indices) || []) {
      for (let i = a; i <= b; i++) state.beatHighlight.add(i);
    }
  }
  const redraw = state.redraws.get(5);
  if (redraw) redraw();
  const first = document.querySelector('li[data-id="' + CSS.escape(id) + '"]');
  if (first) first.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

/* ---------- running a case ---------- */

function provisionalBanner(record, stay) {
  if (record && stay) return "SIMULATED CASE: ECG from MIT-BIH record " + record + "; chart from MIMIC-III ICU stay " + stay + ". These are different people. Research demo, not for patient care.";
  if (record) return "ECG from MIT-BIH record " + record + ". Research demo, not for patient care.";
  return "Chart from MIMIC-III ICU stay " + stay + ". Research demo, not for patient care.";
}

function setBanner(text, simulated) {
  const b = $("banner");
  b.textContent = text;
  b.classList.toggle("simulated", Boolean(simulated));
}

function runParams() {
  const p = new URLSearchParams();
  if ($("record").value) {
    p.set("record", $("record").value);
    p.set("start", $("start").value || "0");
    p.set("seconds", $("seconds").value);
    p.set("peaks", $("peaks").value);
  }
  if ($("stay").value) p.set("stay", $("stay").value);
  return p;
}

function start(stepMode) {
  const p = runParams();
  if (!p.has("record") && !p.has("stay")) { showError("Choose an ECG recording, an ICU stay, or both."); return; }
  closeStream();
  resetStages();
  hideError();
  Object.assign(state, { stepMode, credits: stepMode ? 1 : 0, queue: [], running: true,
    caseId: null, stay: p.get("stay"), data: {} });
  setBanner(provisionalBanner(p.get("record"), p.get("stay")), p.has("record") && p.has("stay"));
  setChat(false);
  $("reveal").disabled = true;
  $("outcome").hidden = true;
  markWorking(1);
  updateButtons();

  const source = new EventSource("/api/run?" + p.toString());
  state.source = source;
  source.addEventListener("stage", (e) => {
    const ev = JSON.parse(e.data);
    state.queue.push(ev);
    if (state.stepMode) setCard(ev.stage, "waiting", "ready: press Step ▸");
    markWorking(ev.stage + 1);
    pump();
  });
  source.addEventListener("done", (e) => {
    state.caseId = JSON.parse(e.data).case_id;
    finish(false);
  });
  source.addEventListener("error", (e) => {
    // Either our "event: error" (it has data) or a dropped connection (no data).
    if (state.running) showError(e.data ? JSON.parse(e.data).message : "The connection to the server closed.");
    finish(true);
  });
}

function closeStream() {
  if (state.source) { state.source.close(); state.source = null; }
}

function finish(failed) {
  closeStream();
  state.running = false;
  if (failed) {
    for (const [n] of STAGES) {
      const c = card(n);
      if (c.classList.contains("working") || (c.classList.contains("waiting") && !state.queue.some((q) => q.stage === n))) {
        setCard(n, "skipped", "not run");
      }
    }
    stopTimer();
  }
  if (state.caseId) {
    setChat(true);
    $("reveal").disabled = !(state.stay && state.options && state.options.outcome_available);
  }
  updateButtons();
}

function pump() {
  while (state.queue.length && (!state.stepMode || state.credits > 0)) {
    const ev = state.queue.shift();
    if (state.stepMode) state.credits -= 1;
    reveal(ev);
  }
  if (!state.queue.length && !state.running) state.stepMode = false;
  updateButtons();
}

function reveal(ev) {
  state.data[ev.stage] = ev;
  const c = card(ev.stage);
  const body = c.querySelector(".body");
  body.replaceChildren();
  if (ev.skipped) {
    setCard(ev.stage, "skipped revealed", "skipped");
    body.append(el("p", { class: "note" }, "Skipped: " + ev.skipped + "."));
  } else {
    setCard(ev.stage, "done revealed", "done");
    try {
      RENDER[ev.stage](ev, body);
    } catch (err) {
      body.append(el("p", { class: "error" }, "Could not show this stage: " + err.message));
    }
  }
  if (ev.stage === 9) stopTimer();
}

function updateButtons() {
  const stepping = state.stepMode && (state.running || state.queue.length > 0);
  $("run").textContent = stepping ? "▶ Show all" : "▶ Run";
  $("run").disabled = state.running && !stepping;
  $("step").textContent = stepping && state.queue.length ? "Step ▸ (" + state.queue.length + " ready)" : "Step ▸";
}

/* ---------- chat and outcome ---------- */

function setChat(on) {
  $("question").disabled = !on;
  $("ask").disabled = !on;
  $("question").placeholder = on ? "e.g. What is a V beat? Why is the risk score high?" : "Run a case first";
  if (!on) $("chat-log").replaceChildren();
}

async function ask(question) {
  const item = el("li", {}, el("div", { class: "q" }, question));
  $("chat-log").append(item);
  const thinking = $("thinking");
  const t0 = Date.now();
  thinking.hidden = false;
  thinking.textContent = "thinking…";
  const tick = setInterval(() => { thinking.textContent = "thinking… " + Math.round((Date.now() - t0) / 1000) + " s"; }, 500);
  $("ask").disabled = true;
  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_id: state.caseId, question }),
    });
    const d = await resp.json();
    if (!resp.ok) throw new Error(d.error || "HTTP " + resp.status);
    item.append(el("div", { class: "a" + (d.refused ? " refused" : "") }, linkify(d.answer), " ",
      ...d.cites.map(idTag),
      d.verified && !d.refused ? el("span", { class: "badge good" }, "fact-checked ✓") : null));
  } catch (err) {
    item.append(el("div", { class: "a refused" }, "Could not ask: " + err.message));
  } finally {
    clearInterval(tick);
    thinking.hidden = true;
    $("ask").disabled = false;
    item.scrollIntoView({ block: "nearest" });
  }
}

async function revealOutcome() {
  const out = $("outcome");
  out.hidden = false;
  try {
    const resp = await fetch("/api/outcome?stay=" + encodeURIComponent(state.stay));
    const d = await resp.json();
    out.textContent = resp.ok ? "ICU stay " + d.stay + ": " + d.outcome + "." : (d.error || "Not available.");
  } catch (err) {
    out.textContent = "Could not load the outcome: " + err.message;
  }
}

/* ---------- wiring ---------- */

function init() {
  buildStages();
  loadOptions();
  $("case-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.stepMode && (state.running || state.queue.length)) { state.stepMode = false; pump(); return; }
    start(false);
  });
  $("step").addEventListener("click", () => {
    if (!state.running && !state.queue.length) { start(true); return; }
    state.stepMode = true;
    state.credits += 1;
    pump();
  });
  $("record").addEventListener("change", onRecordChange);
  $("seconds").addEventListener("change", onRecordChange);
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("question").value.trim();
    if (!q || !state.caseId) return;
    $("question").value = "";
    ask(q);
  });
  $("reveal").addEventListener("click", revealOutcome);
  document.addEventListener("click", (e) => {
    const tag = e.target.closest(".fid");
    if (tag) select(tag.dataset.id);
  });
  document.addEventListener("keydown", (e) => {
    const tag = e.target.closest && e.target.closest(".fid");
    if (tag && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); select(tag.dataset.id); }
  });
  let pending = null;
  const redrawAll = () => { for (const fn of state.redraws.values()) fn(); };
  window.addEventListener("resize", () => { clearTimeout(pending); pending = setTimeout(redrawAll, 120); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redrawAll);
  window.addEventListener("beforeunload", closeStream);
}

document.addEventListener("DOMContentLoaded", init);
