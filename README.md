# CAT-GraphRAG-Proctoring

An adaptive, AI-proctored exam platform where **every question is generated live for the candidate**.
It connects four engines into one flow:

| Engine | What it does | Code |
|---|---|---|
| **GraphRAG exam knowledge graph** | Turns the syllabus into a graph (Section → Topic → Concept, with prerequisites). The graph decides *what* is asked, grounds every generated question, and explains the result. | `backend/exam_graph/`, `backend/graphrag_core/` |
| **CAT (computerized adaptive testing)** | 2PL IRT with EAP ability estimation. The candidate's current ability decides *how hard* the next question is. | `backend/app/` |
| **Live question generation (NVIDIA NIM)** | Nemotron writes each next question for this candidate, from the graph, at the CAT difficulty, and a second model call verifies the answer key. | `backend/exam_graph/live.py` |
| **AI proctoring (webcam + microphone)** | Gaze, objects, hands, voices and lip-sync. Flagged clips are judged by NVIDIA NIM multimodal models grounded in an exam-rules knowledge graph. Encrypted evidence with data retention. | `backend/proctor/` |
| **Semantic cache (Redis LangCache)** | Examiner Q&A answers are reused for questions with the same meaning: no LLM call, no tokens. | `backend/semantic_cache.py` |

---

## 1. How everything is wired

```
EXAMINER  question bank page: choose Subject / Section / Topic ──► "Use this selection for the exam"
                                     │
                                     ▼
CANDIDATE  exam page shows the selected assessment ──► consent ──► camera + mic calibration (3 s)
                                     │
            ┌────────────────────────┴─────────── for every question ───────────────────────────┐
            │ 1. SECTION     blueprint balancing inside the selected scope (Kingsbury & Zara)      │
            │ 2. TOPIC       GraphRAG: after a wrong answer → prerequisite topic of what was       │
            │                missed; otherwise the least-covered topic of the section              │
            │ 3. DIFFICULTY  CAT: target b = current ability θ  → easy / medium / hard             │
            │ 4. GENERATE    Nemotron writes 1 MCQ from the topic passage + concepts + prerequisites│
            │ 5. CHECK       structure · concept is in the graph · source quote is in the syllabus │
            │                · independent verifier picks the same answer · not a repeat           │
            │ 6. PREFETCH    while the candidate answers, the next question is generated for both  │
            │                outcomes (correct → harder, wrong → easier), so it is ready on submit │
            │ 7. FALLBACK    if generation is slow or fails, the best approved bank question       │
            └───────────────────────────────────────────────────────────────────────────────────────┘
            proctoring runs throughout: gaze · objects · hands · voices → NVIDIA judge → popup only when fraud is confirmed
                                     │
                                     ▼
RESULT     ability θ ± SE · per-section scores · Study next (missed concepts + prerequisite topics
           from the graph) · exam integrity status
                                     │
                                     ▼
FEEDBACK   every answer is logged → item analytics (p-value, discrimination, leak detection) →
           online IRT calibration of difficulty → examiner Q&A over the graph ("which topics are hardest?")
           → answers served through the semantic cache
```

So a question is never "hard-wired": its **scope** comes from the examiner, its **topic** from the syllabus
graph and the candidate's mistakes, its **difficulty** from the candidate's performance, and its **text** is
written live by the model.

---

## 2. Exam scope (examiner)

On **`/ui/exam-admin.html` → Question bank** the examiner filters the bank by **Subject → Section → Topic → Status**
and clicks **Use this selection for the exam**. The selection is saved (`data/exam/exam_scope.json`) and the
candidate's exam page shows it ("Assessment: Machine Learning · Model Evaluation"); the candidate does not choose again.

| Selection | Exam |
|---|---|
| Subject (e.g. Machine Learning) | all its sections, blueprint-balanced, up to 15 questions (5 per section) |
| Subject + Section | that section only, 5 questions |
| Subject + Section + Topic | that topic only, 5 questions |

Subjects are defined in `backend/knowledge/syllabus/subjects.json` (Python Programming, Machine Learning) plus the
full assessment (all sections).

---

## 3. Live question generation (`exam_graph/live.py`)

- **Model:** the NVIDIA NIM text chain (`EXAM_LLM_MODELS`). With the current key that is
  `nvidia/nemotron-3-super-120b-a12b`; if a configured model is retired (404/410) the client discovers the best
  available model in the key's catalogue.
- **Prompt context (from the graph):** section, topic passage, the topic's concepts, prerequisite topics and their
  concepts, concept relations, the candidate's level, the target difficulty, concepts the candidate got wrong,
  and the questions already asked (so it doesn't repeat).
- **Checks before a question is shown:** 4 distinct options and one valid key, the tested concept belongs to the
  syllabus graph, the `source_quote` really appears in the syllabus passage (≥ 80% token overlap), an independent
  verifier call answers the question without seeing the key and must pick the same option, no near-duplicates.
  Answer positions are shuffled.
- **IRT parameters:** a = 1.0, b = the target ability (an LLM-estimated prior). Real responses refine it through
  online calibration, like any new item.
- **Audit:** every generated question is stored in the bank with status **`live`** (filter *Status: live* on the
  question bank page). An examiner can approve good ones into the permanent bank.
- **Latency:** the first question waits at most 30 s, otherwise a bank question is used. Later questions are
  usually ready on submit because they are prefetched for both outcomes.
- The tag **"AI-generated for you · medium"** on the exam page shows which questions were generated live.

---

## 4. GraphRAG exam knowledge graph

**The syllabus is the source of truth:** `backend/knowledge/syllabus/*.md` (sample: *Python & Machine Learning
Assessment*, 5 sections × 4 topics). Format:

```markdown
# Assessment title
exam_length: 15

## Section name
weight: 0.2

### Topic name
requires: Other Topic
Topic text with the key **concepts** in bold.
```

Examiners can edit it on the admin page, or upload a PDF / DOCX / TXT that NIM converts into this format as a draft.

**Graph construction** uses the GraphRAG pipeline from the [GraphRAG project](https://github.com/sairambommagani/GraphRAG)
(`graphrag_core/`): entity resolution, Louvain communities with summaries, local and global search. Entities are
sections, topics, concepts and questions; relations are `contains`, `covers`, `requires` (prerequisite), `tests`
(question → concept) and NIM-extracted concept relations (`contrasts_with`, `prerequisite_of`, …) limited to the
syllabus's own concept names.

**Used for:**
1. **Live generation and bank generation** – grounded, multi-hop questions (Section 3; the *Generate questions* tab
   produces drafts for examiner approval).
2. **Topic choice during the exam** – prerequisites after a wrong answer, coverage otherwise.
3. **Tagging** – every bank question gets section / topic / concepts (94% agreement with hand labels on the seed set).
4. **Blueprint** – section weights for content balancing.
5. **Results** – per-section score, missed concepts, and the **Study next** plan (missed topics + their prerequisites).
6. **Coverage** – which topics/concepts lack approved questions.
7. **Examiner Q&A** – "Which topics do candidates find hardest?", "What does Deep Network Training require?"
   GraphRAG retrieves the facts, Nemotron writes the answer, the semantic cache reuses it.

**Performance analytics** (`exam_graph/analytics.py`): every answer is logged; per question p-value,
point-biserial discrimination and response time; flags for too easy / too hard / low discrimination / **possible
leak** (far more correct answers than IRT predicts); **online calibration** re-estimates difficulty `b` (MAP with a
prior) once a question has ≥ 30 answers.

---

## 5. Adaptive testing (CAT)

- **Response model (2PL IRT):** `p(correct | θ, a, b) = 1 / (1 + exp(-a(θ - b)))`
- **Ability:** EAP over a N(0, 1) prior – stable with few responses and finite when all answers are right or wrong.
- **Difficulty targeting:** the next question is generated at b ≈ θ, where a 2PL item gives maximum Fisher
  information `I(θ) = a²·p·(1-p)`. For bank fallback questions, the most informative unused item in the section.
- **Content balancing:** the section furthest below its blueprint share is asked next.
- **Length:** `exam_length` in the syllabus (15), capped at 5 per section in scope.
- **Result:** θ, SE, proficiency label, per-section results, Study next, integrity.

---

## 6. AI proctoring: webcam + microphone, NVIDIA judge, rules knowledge graph

When the candidate clicks **Begin assessment** they see the consent screen for camera,
microphone and data retention. They then look at a centre dot and stay quiet for 3 seconds.
This records their personal gaze baseline and the room's noise floor. Only then does the
first question appear.

```
exam page (proctor.js) ── 8 fps frames + 16 kHz mic audio over one WebSocket ──▶ backend/proctor
  VISION  MediaPipe face/iris landmarks → head pose, iris, eye openness, lip movement
          MediaPipe object + hand models → phone, book, second device, extra person, foreign hand
  AUDIO   WebRTC VAD + noise-floor gate + pitch and syllable-rhythm checks (typing, fans, hum ignored) → speech episodes
          lip-sync fusion (voice vs the candidate's lip movement) → candidate / ANOTHER PERSON / unattributed
  DETECT  per-candidate baseline, hysteresis, persistence, voting, cooldowns → flag
  CLIP    RAM ring buffer → 16-frame video + audio of the flagged seconds (nothing else is kept)
  CONTEXT faster-whisper transcript of the clip · GraphRAG: exam rules + similar past cases
  JUDGE   NVIDIA NIM chain: Cosmos 3 Reasoner (physical-AI video reasoning)
                          → Nemotron 3 Omni (audio + video; first for speech)
                          → Llama 3.2 Vision (fallback)
fraud popup ◀── one alert, only when the AI judge confirms fraud (confidence ≥ 0.70), citing the rule
```

### What is flagged

| Flag | Signal | Rule |
|---|---|---|
| off-screen gaze | gaze + head pose vs baseline | ≥ 3 s |
| repeated same-direction glances | gaze | ≥ 3 within 30 s |
| looking down (lap / desk) | eyelid drop vs baseline | ≥ 0.6 s sustained, then ≥ 3 s |
| no face / second face | face tracker | ≥ 3 s / ≥ 1 s |
| phone, book, second device | object detector, background-aware | voting over 1.5 s |
| another person (body) / hand | object + hand models | voting over 1.5 s |
| **talking (candidate)** | voice + lips moving | ≥ 1 s of voiced, pitched speech (about one sentence) |
| **another voice in the room** | voice while the candidate's lips are still | ≥ 1 s of voiced, pitched speech |

### AI judge: why these models

| Order | Model | Why |
|---|---|---|
| 1 | `nvidia/cosmos3-nano-reasoner` | NVIDIA **Cosmos 3**, a physical-AI reasoning model. It is trained on how bodies, hands and objects move and interact over time. It watches the clip as a **video** (not stills) and reasons step by step (`<think>`) before answering. That is what "a hand moves to the lap, the head drops, a lit rectangle appears" needs. |
| 2 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA **Nemotron 3 Omni** takes audio and video together. It is routed first for speech events, so the judge actually *hears* the whisper. |
| 3 | `meta/llama-3.2-11b-vision-instruct` | A single-image fallback (2×2 frame grid), so the system never stops judging. |

- **Configured by `NIM_MODELS`.** On the current free key the Cosmos models are not enabled, so the working chain is
  Nemotron 3 Omni (video + audio, ~8–20 s per verdict) → Llama 3.2 Vision. `python -m proctor.check_nim` prints what works.
- **Fallback.** Each model starts with its richest input format
  (video+audio → video → video frames → images → grid) and steps down on a 400.
- **Unavailable models are benched.** A model that isn't enabled for the key (404)
  or keeps failing is benched and the next one answers. What works is saved in
  `data/nim_models.json`.
- **Grounded prompt.** The prompt carries the tracker timeline, the lip-sync result,
  the transcript, the relevant exam rules and similar past cases. The verdict must
  **cite the rule** it applied.
- **Audit trail.** The model's reasoning is stored for the examiner.

### Proctoring rules graph (`proctor/knowledge.py`, `graphrag_core/`)

`graphrag_core/` is the pipeline from the [GraphRAG project](https://github.com/sairambommagani/GraphRAG):
entity resolution, Louvain communities, community summaries, and local and global
search with IDF reranking. It indexes two sources:

1. **Exam rules**: `backend/knowledge/policy/exam_rules.md`, rules R-01 to R-14.
   Examiners can edit them. Each rule lists the flags it governs, its targets,
   its severity and its allowed exceptions.
2. **Incidents**: every judged flag, plus the **examiner's confirm/dismiss decision**.
   These records hold metadata only.

Walking the graph gives the judge:
- the rules for this flag, ranked (e.g. R-10 *No other voices in the room*)
- how examiners decided similar cases ("5 earlier phone flags: 4 confirmed, 1 dismissed")
- this session's history (escalation, R-12)

Examiner decisions therefore feed back into future verdicts. On the review page,
examiners can also **ask the graph** questions such as "Which sessions had repeated
phone use?" or "What does R-10 say?".

### Accuracy evaluation

```bash
python -m proctor.replay recording.mp4 --judge nim        # one recording, second-by-second report (video + mic)
python -m proctor.evaluate eval_clips/ --judge nim        # precision / recall / false-alarm rate on a labelled set
```

`eval_clips/labels.csv` has these columns: `file,label(cheat|clean),expect(flag types),notes`.
The report compares **detector** accuracy (any flag) with **system** accuracy (popup
after the AI judge). It shows how many false flags the judge filtered out and how many
real cases it cleared. It is written to `eval_report.md`.

### Data retention

- **In memory only.** Live video *and audio* exist only in RAM, as a rolling 25 s window.
  Unflagged footage and sound are never stored.
- **Flagged evidence.** The frames and audio of a flagged clip are encrypted on disk
  (Fernet). The transcript is deleted together with them.
- **Retention by verdict.** Benign evidence is deleted immediately, suspicious evidence
  after 7 days, and fraud after 30 days.
- **Metadata.** Metadata and knowledge-graph incidents (no images, audio or text of
  speech) are kept for 90 days.
- **Controls.** There is an hourly purge, a right-to-erasure endpoint (which also
  removes the session's incidents from the graph), and an audit log that records who
  viewed or heard what.

### Other behaviour

- **Enforced on the server.** `/submit-answer` returns 403 unless a live, calibrated
  proctoring session exists. `CAT_REQUIRE_PROCTORING=0` turns this off for development.
- **Result page.** It shows an **Exam integrity** row.
- **Examiner page.** `/ui/review.html` shows:
  - frames, an audio player, the transcript and the lip-sync result
  - the verdict with the cited rule, and the AI reasoning
  - which model answered, and any fallback
  - confirm/dismiss buttons, the model-chain status and a knowledge-graph Q&A box

---

## 7. Semantic cache (Redis LangCache)

Examiners ask the same things in different words. Examiner questions (exam-content **Ask** and the proctoring rules
**Ask**) go through GraphRAG retrieval → NVIDIA LLM answer, and the answer is stored in a **semantic cache**: the
next question with the same *meaning* (embedding similarity ≥ 0.9) is answered from the cache.

```
question ──► semantic cache search (scope + knowledge version)
               hit  ──► cached answer                          (~0.3 s, 0 tokens)
               miss ──► GraphRAG context ──► Nemotron ──► store ──► answer   (~10 s, ~700 tokens)
```

- **Backend:** Redis LangCache (managed, Redis Cloud, REST API) when `LANGCACHE_SERVER_URL`, `LANGCACHE_CACHE_ID`
  and `LANGCACHE_API_KEY` are set; otherwise a local semantic cache (NVIDIA embeddings, in memory). If the cache has
  no attributes configured, entries are scoped by a prompt prefix instead.
- **Freshness:** each entry carries the version of the graph it came from. When questions, analytics, incidents or
  reviews change (including an erasure), old answers stop matching and are purged.
- **Not cached, on purpose:** AI-judge verdicts (each candidate's clip must be judged on its own) and question
  generation (it must produce new questions).
- **Never blocks:** if Redis is down the LLM still answers.
- **Stats:** under every answer in the UI (hit/miss, similarity, tokens saved), `GET /exam/admin/cache` and `/health`.

---

## 8. Setup and running (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then fill in the values (see below)
python -m proctor.check_nim     # tests every NVIDIA model with your key and prints the working chain
python -m uvicorn app.main:app --port 8000 --env-file .env
```

> Always start the server with `--env-file .env`, otherwise the NVIDIA key, admin token and Redis keys are not loaded.
> Run **one** server process (sessions are held in memory).

| Page | URL |
|---|---|
| Candidate exam | http://localhost:8000/ui/ |
| Exam content (GraphRAG, question bank, exam scope, analytics, Ask) | http://localhost:8000/ui/exam-admin.html |
| Proctoring review (evidence, audio, AI reasoning, rules Ask) | http://localhost:8000/ui/review.html |
| Health | http://localhost:8000/health |

The MediaPipe models (a few MB) download on first use; the Whisper `base` speech-to-text model (~150 MB) downloads
in the background at startup. Docker: `docker compose up --build` (reads `backend/.env`; models and evidence on volumes).

**Tests:** `cd backend && python -m pytest` (248 tests: IRT, API flow, live generation with a mocked model, GraphRAG,
proctoring with synthetic faces and voices, audio/lip-sync, judge chain, retention, semantic cache).

### Configuration (`backend/.env`, never committed)

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY` | NVIDIA NIM key (judge, question generation, Q&A, embeddings) |
| `NIM_MODELS` | AI-judge model chain, best first (e.g. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning,meta/llama-3.2-11b-vision-instruct`) |
| `EXAM_LLM_MODELS` | text model chain for generation and Q&A (e.g. `nvidia/nemotron-3-super-120b-a12b`) |
| `NIM_RPM`, `NIM_DAILY_CALLS`, `NIM_DAILY_TOKENS` | rate limit and daily budget for the free tier |
| `PROCTOR_EVIDENCE_KEY` | Fernet key that encrypts evidence (keep it – evidence can't be read without it) |
| `PROCTOR_ADMIN_TOKEN` | token for the examiner pages and admin API (`X-Admin-Token`) |
| `CAT_REQUIRE_PROCTORING` | `1` = answers are rejected unless a calibrated proctoring session is live |
| `CAT_LIVE_QUESTIONS` | `1` (default) = live generation; `0` = bank-only CAT |
| `CAT_REQUIRE_LOGIN` | `0` (default). `1` turns on candidate accounts (see below) |
| `LANGCACHE_SERVER_URL`, `LANGCACHE_CACHE_ID`, `LANGCACHE_API_KEY` | Redis LangCache; empty = local semantic cache |
| `SEMANTIC_CACHE_THRESHOLD` | similarity needed for a cache hit (default 0.9) |
| `PROCTOR_ASR`, `PROCTOR_ASR_MODEL` | speech-to-text on flagged clips (faster-whisper size) |
| `PROCTOR_MOCK_JUDGE` | `1` = offline rule-based judge for development |

**Optional candidate accounts (built, switched off).** With `CAT_REQUIRE_LOGIN=1` candidates register / sign in
(PBKDF2-hashed passwords, `data/accounts.json`), and each attempt is stored (`data/attempts.json`): the next exam on
the same subject starts at the candidate's last ability estimate and targets the concepts they missed last time,
and the result shows their progress across attempts. The exam page currently runs without login.

---

## 9. API

| Endpoint | Method | Purpose |
|---|---|---|
| `/exam-scope` | GET | The assessment selected by the examiner |
| `/start-test` | POST | Creates a session for the selected scope and returns the first (live-generated) question |
| `/submit-answer` | POST | Grades the answer, updates θ/SE, returns the next question (or `finished`) |
| `/result/{session_id}` | GET | θ, SE, per-section results, Study next, integrity |
| `/subjects` | GET | Subjects and their sections |
| `/health` | GET | Models, judge chain, semantic cache and live-generation stats |
| `/auth/register`, `/auth/login` | POST | Candidate accounts (only used when `CAT_REQUIRE_LOGIN=1`) |
| `/exam/admin/scope` | POST | Set the exam scope (subject / section / topic) |
| `/exam/admin/graph` | GET | Knowledge-graph stats and blueprint |
| `/exam/admin/syllabus` | GET/PUT | Read or replace the syllabus (re-indexes the graph, re-tags the bank) |
| `/exam/admin/syllabus/convert` | POST | PDF/DOCX/TXT → syllabus draft (NIM) |
| `/exam/admin/graph/relations` | POST | NIM concept-relation extraction |
| `/exam/admin/coverage` | GET | Topics and concepts that need questions |
| `/exam/admin/generate` | POST | Batch question generation with verification → drafts |
| `/exam/admin/questions[/{id}/review]` | GET/POST | Question bank (approved / live / draft / rejected); approve or reject |
| `/exam/admin/analytics[/calibrate]` | GET/POST | Item statistics, leak flags, apply calibrated difficulties |
| `/exam/admin/ask` | POST | Examiner Q&A over the exam graph (semantic-cached) |
| `/exam/admin/cache` | GET | Semantic cache hits, misses, tokens saved |
| `/proctor/policy`, `/proctor/sessions`, `/proctor/ws/{id}` | – | Consent, proctoring session, webcam + mic stream |
| `/proctor/admin/*` | – | Evidence, frames, audio, review, erasure, purge, usage, audit, rules graph + Ask |

All `/exam/admin/*` and `/proctor/admin/*` endpoints need the `X-Admin-Token` header.

---

## 10. Project structure

```
backend/
  app/
    main.py               FastAPI app: exam scope, start-test / submit-answer / result, health
    irt/engine.py         2PL model, EAP ability estimation, Fisher information
    schemas.py            request/response models
    session_store.py      in-memory exam sessions
    accounts.py           optional candidate accounts (PBKDF2)
    attempts.py           optional per-candidate attempt history
    data/question_bank.json   50 seed questions (imported into the bank on first start)
  exam_graph/
    syllabus.py           syllabus parser
    graph.py              exam knowledge graph (GraphRAG), Q&A retrieval
    live.py               live per-candidate question generation (plan → generate → verify → prefetch)
    generator.py          grounded batch generation + independent verifier
    bank.py               question bank, statuses, dedupe, tagging
    cat_blueprint.py      content-balanced selection, section results
    analytics.py          item statistics, leak detection, online calibration
    nim.py                NVIDIA NIM text client (model chain, catalogue discovery, budgets)
    service.py            ExamService, subjects, exam scope, examiner API
  semantic_cache.py       Redis LangCache / local semantic cache
  graphrag_core/          GraphRAG pipeline (from the GraphRAG project)
  proctor/                features (gaze, lips), calibration, detector, objects, audio (VAD, lip-sync,
                          transcription), clip, judge (NIM chain), knowledge (rules graph), retention,
                          service, api, replay, evaluate, check_nim
  knowledge/syllabus/     syllabus (.md) and subjects.json
  knowledge/policy/       exam rules R-01..R-14
  tests/                  248 automated tests
frontend/
  index.html, app.js, style.css       candidate exam (ability gauge, AI-generated tag, Study next)
  proctor/proctor.js, proctor.css     consent, calibration, frame + mic streaming, alert popup
  exam-admin.html                     examiner: graph, generation, review, bank + exam scope, analytics, syllabus, Ask
  review.html                         examiner: proctoring evidence, audio, AI reasoning, rules Ask
```

---

## 11. Known limitations (honest status)

- **Accuracy numbers** must still be measured on a labelled set of real recordings (≈ 40, half cheating). The
  evaluation tool (`proctor.evaluate`) is ready; thresholds were tuned on the developer's recordings and synthetic tests.
- **Generated-question difficulty** is an LLM-estimated prior until real responses calibrate it.
- **Latency on the free NVIDIA tier:** question generation ~5–15 s (hidden by prefetching), AI judge ~8–20 s per flag,
  examiner Q&A ~10 s on a cache miss.
- **Model availability** depends on the NVIDIA account: on the free key, Cosmos Reason models are listed but not
  enabled (404), so Nemotron 3 Omni is the active judge. The chain switches automatically when access is granted.
- **Lip-sync** needs the face visible; otherwise speech is marked "unattributed" and goes to review.
- **Single process:** sessions, the bank (JSON) and proctoring sessions live in one process. Scaling out needs
  PostgreSQL + Redis for sessions and a queue for judge/generation calls.

## 12. Next steps

Pending review feedback. Natural next items: labelled evaluation set and per-flag precision/recall, turning on
candidate accounts with SSO, PostgreSQL/Redis for multi-worker deployment, self-hosted NIM containers on GPU to cut
latency, exposure control for generated items.
