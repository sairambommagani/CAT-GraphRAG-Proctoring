// Served by the backend at /ui -> same origin. Opened as a file -> fall back to the dev server.
const API_BASE = location.protocol.startsWith("http") ? location.origin : "http://localhost:8000";

const GAUGE_MIN = -3;
const GAUGE_MAX = 3;

let sessionId = null;
let selectedIndex = null;
let currentQuestion = null;
let answered = false;
let proctor = null;
let integrityAlerts = 0;

const el = (id) => document.getElementById(id);

const viewIntro = el("view-intro");
const viewQuestion = el("view-question");
const viewResult = el("view-result");

el("topbar-meta").textContent = "Adaptive assessment";

el("btn-start").addEventListener("click", startTest);
// ---- account + subject ------------------------------------------------------------
let authInfo = null;
async function showAccount() {
  try {
    const sc = await (await fetch(`${API_BASE}/exam-scope`)).json();     // chosen by the examiner
    el("scope-name").textContent = sc.name;
  } catch (e) { el("scope-name").textContent = "Adaptive assessment"; }
}
showAccount();
el("btn-submit").addEventListener("click", submitAnswer);
el("btn-restart").addEventListener("click", () => location.reload());

async function startTest() {
  el("btn-start").disabled = true;
  el("btn-start").textContent = "Starting…";
  try {
    el("btn-start").textContent = "Preparing your first question…";
    const res = await fetch(`${API_BASE}/start-test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) throw new Error(`start-test failed: ${res.status}`);
    const data = await res.json();
    sessionId = data.session_id;
    el("topbar-meta").textContent = data.subject || "Adaptive assessment";

    // Webcam + mic proctoring: consent -> camera/mic -> 3 s calibration, before the first question is shown.
    el("btn-start").textContent = "Setting up proctoring…";
    proctor = new ProctorClient({
      apiBase: API_BASE,
      examRef: sessionId,
      onAlert: () => {
        integrityAlerts += 1;
        el("topbar-meta").textContent = `Integrity alerts: ${integrityAlerts}`;
        el("topbar-meta").classList.add("alerted");
      },
    });
    const accepted = await proctor.start();
    if (!accepted) {
      el("btn-start").textContent = "Webcam and microphone proctoring is required — begin again";
      el("btn-start").disabled = false;
      return;
    }

    showQuestion(data.question);
    setGauge(0, 1);
    viewIntro.hidden = true;
    viewQuestion.hidden = false;
  } catch (err) {
    el("btn-start").textContent = "Couldn't start (server, camera or microphone) — retry";
    el("btn-start").disabled = false;
    console.error(err);
  }
}

function showQuestion(q) {
  currentQuestion = q;
  selectedIndex = null;
  answered = false;

  el("q-topic").textContent = q.topic;
  if (q.source === "live") {
    const tag = document.createElement("span");
    tag.className = "q-source";
    tag.textContent = `AI-generated for you · ${q.difficulty || "medium"}`;
    el("q-topic").appendChild(tag);
  }
  el("q-count").textContent = `Question ${q.question_number} of ${q.max_questions}`;
  el("q-text").textContent = q.text;

  const optionsEl = el("options");
  optionsEl.innerHTML = "";
  q.options.forEach((opt, idx) => {
    const btn = document.createElement("button");
    btn.className = "option";
    btn.textContent = opt;
    btn.addEventListener("click", () => selectOption(idx, btn));
    optionsEl.appendChild(btn);
  });

  el("feedback").hidden = true;
  el("btn-submit").disabled = true;
  el("btn-submit").textContent = "Submit answer";
}

function selectOption(idx, btnEl) {
  if (answered) return;
  selectedIndex = idx;
  [...el("options").children].forEach((c) => c.classList.remove("selected"));
  btnEl.classList.add("selected");
  el("btn-submit").disabled = false;
}

async function submitAnswer() {
  if (selectedIndex === null || answered) return;
  answered = true;
  el("btn-submit").disabled = true;
  el("btn-submit").textContent = "Preparing your next question…";

  try {
    const res = await fetch(`${API_BASE}/submit-answer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        question_id: currentQuestion.question_id,
        selected_index: selectedIndex,
      }),
    });
    if (res.status === 403) {                     // proctoring not active (server-side check)
      const detail = (await res.json()).detail;
      answered = false;
      el("btn-submit").disabled = false;
      el("feedback").hidden = false;
      el("feedback").textContent = detail;
      return;
    }
    if (!res.ok) throw new Error(`submit-answer failed: ${res.status}`);
    const data = await res.json();

    revealAnswer(data.correct);
    setGauge(data.theta_estimate, data.se);

    if (data.finished) {
      setTimeout(() => finishTest(), 900);
    } else {
      setTimeout(() => showQuestion(data.next_question), 900);
    }
  } catch (err) {
    console.error(err);
    el("feedback").hidden = false;
    el("feedback").textContent = "Something went wrong submitting that answer.";
  }
}

function revealAnswer(correct) {
  const options = [...el("options").children];
  options.forEach((c) => (c.disabled = true));
  const selectedEl = options[selectedIndex];
  selectedEl.classList.remove("selected");
  selectedEl.classList.add(correct ? "correct" : "incorrect");

  const feedback = el("feedback");
  feedback.hidden = false;
  feedback.textContent = correct ? "Correct." : "Not quite.";
}

function setGauge(theta, se) {
  const clamped = Math.max(GAUGE_MIN, Math.min(GAUGE_MAX, theta));
  const fraction = (clamped - GAUGE_MIN) / (GAUGE_MAX - GAUGE_MIN); // 0..1, bottom..top

  el("gauge-fill").style.height = `${fraction * 100}%`;
  el("gauge-needle").style.top = `${(1 - fraction) * 100}%`;
  el("theta-value").textContent = theta.toFixed(2);
  el("theta-se").textContent = `± ${se.toFixed(2)} SE`;
}

async function finishTest() {
  try {
    if (proctor) await proctor.stop();          // flushes pending flags so the result includes them
    const res = await fetch(`${API_BASE}/result/${sessionId}`);
    if (!res.ok) throw new Error(`result failed: ${res.status}`);
    const data = await res.json();
    renderResult(data);
    viewQuestion.hidden = true;
    viewResult.hidden = false;
  } catch (err) {
    console.error(err);
  }
}

function renderResult(data) {
  el("result-proficiency").textContent = data.proficiency;
  el("result-theta-line").textContent =
    `Estimated ability θ = ${data.theta_estimate.toFixed(2)}, based on ${data.questions_administered} adaptively selected questions.`;

  el("result-count").textContent = data.questions_administered;
  el("result-correct").textContent = `${data.correct_count} / ${data.questions_administered}`;
  el("result-theta-val").textContent = data.theta_estimate.toFixed(2);
  el("result-se").textContent = `± ${data.se.toFixed(2)}`;

  renderIntegrity(data.integrity);

  const recs = data.recommendations || [];
  el("study-box").hidden = !recs.length;
  el("study-list").innerHTML = recs.map((r) => `<div class="breakdown-row"><b>${esc(r.topic)}</b>
      <span class="missed"> · ${esc(r.section)}</span>
      <div class="missed">Missed: ${r.concepts.map(esc).join(", ") || "—"}</div>
      ${r.prerequisites.length ? `<div class="missed">Revise first: ${r.prerequisites.map(esc).join(", ")}</div>` : ""}</div>`).join("");
  const hist = data.history || [];
  el("history-box").hidden = hist.length < 1;
  el("history-list").innerHTML = hist.map((h, i) => `<div class="breakdown-row-label"><span>Attempt ${i + 1}${h.finished_at ? " · " + new Date(h.finished_at * 1000).toLocaleDateString() : ""}</span>
      <span>θ ${Number(h.theta).toFixed(2)} · ${h.correct}/${h.total}</span></div>`).join("")
    + (hist.length > 1 ? `<div class="missed">Next attempt starts at your latest level and targets what you missed.</div>` : "");

  const list = el("breakdown-list");
  list.innerHTML = "";
  const rows = (data.sections && data.sections.length) ? data.sections
    : data.topic_breakdown.map((t) => ({ section: t.topic, correct: t.correct, total: t.total }));
  rows.forEach((t) => {
    const pct = t.total ? Math.round((t.correct / t.total) * 100) : 0;
    const row = document.createElement("div");
    row.className = "breakdown-row";
    const badge = t.total ? "" : `<span class="missed">not assessed</span>`;
    const missed = (t.missed_concepts || []).length
      ? `<div class="missed">Review: ${t.missed_concepts.slice(0, 4).map(esc).join(", ")}</div>` : "";
    row.innerHTML = `
      <div class="breakdown-row-label">
        <span>${esc(t.section)} ${badge}</span>
        <span>${t.correct} / ${t.total}</span>
      </div>
      <div class="breakdown-bar-track">
        <div class="breakdown-bar-fill" style="width: ${pct}%"></div>
      </div>${missed}
    `;
    list.appendChild(row);
  });
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderIntegrity(integrity) {
  const labels = {
    clear: "No issues detected",
    under_review: "Flagged for examiner review",
    alerted: "Malpractice alert — under review",
    not_proctored: "Not proctored",
  };
  const cell = el("result-integrity");
  if (!integrity) { cell.textContent = "—"; return; }
  let text = labels[integrity.status] || integrity.status;
  if (integrity.alerts) text += ` (${integrity.alerts} alert${integrity.alerts > 1 ? "s" : ""})`;
  cell.textContent = text;
  cell.className = `integrity-${integrity.status}`;
}

// assessment title from the syllabus knowledge graph
fetch(`${API_BASE}/health`).then((r) => r.json()).then((h) => {
  if (h.assessment && el("cert-title")) el("cert-title").textContent = h.assessment;
}).catch(() => {});
