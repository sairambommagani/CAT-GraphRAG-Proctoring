# CAT-GraphRAG-Proctoring

An adaptive exam platform that integrates three pieces:
**CAT**, **GraphRAG** and **AI proctoring** (webcam + microphone, NVIDIA NIM judge).

| Engine | What it does |
|---|---|
| **GraphRAG exam knowledge graph** (`backend/exam_graph/`, `backend/graphrag_core/`) | Turns the syllabus into a knowledge graph. That graph drives the exam blueprint, tags every question, and grounds the NVIDIA NIM question generator. |
| **CAT, computerized adaptive testing** (`backend/app/`) | 2PL IRT with EAP ability estimation. Questions are chosen by maximum information *within* the graph blueprint, and the result is broken down by section. |
| **AI proctoring** (`backend/proctor/`) | Webcam and microphone. Flagged clips are judged by physics-aware NVIDIA NIM models, grounded in an exam-rules knowledge graph, with encrypted evidence and data retention. |

```
examiner: syllabus (md / pdf / docx) ──▶ GraphRAG graph: Section → Topic → Concept (+ prerequisites)
                                          │                    │
                    NVIDIA NIM generator ◀┘ grounded MCQs,     └▶ exam blueprint (section weights)
                    verified, deduped → examiner approves → question bank (tagged, IRT a/b)
candidate:  consent → camera + mic calibration → 15 adaptive questions (blueprint-balanced CAT)
            ↳ proctoring runs throughout: gaze, objects, voices → NIM judge → one popup only when fraud is confirmed
result:     ability θ, per-section scores + concepts to review, integrity status
analytics:  item statistics, leak detection, online IRT calibration, examiner Q&A over the graph
```

## Exam content: GraphRAG knowledge graph

**The syllabus is the source of truth.** It lives in `backend/knowledge/syllabus/*.md`
(sample: *Python & Machine Learning Assessment*). The format is:
- `## Section` with a `weight:` line (share of the exam)
- `### Topic` with an optional `requires:` line listing prerequisite topics
- **bold** key concepts

Examiners can edit it on the admin page. They can also upload a PDF, DOCX or TXT
file: an NVIDIA NIM model converts it into this format as a draft, and the examiner
applies it.

**Knowledge graph.** It's built with the GraphRAG pipeline: extraction, then graph
construction with entity resolution, then Louvain communities with summaries, then
local and global search. The entities are sections, topics, concepts and questions.
The relations are:
- `contains`, `covers` and `requires` (prerequisite)
- `tests` (question → concept)
- concept relations such as `contrasts_with` and `prerequisite_of`, extracted by NIM.
  These are limited to the syllabus's own concept names, so the model can't invent
  entities.

**What the graph is used for:**

1. **Question generation.** The generator uses `nvidia/llama-3.3-nemotron-super-49b-v1.5`,
   with `meta/llama-3.3-70b-instruct` as the fallback.
   - For each topic, the graph supplies the topic passage, its concepts, its
     prerequisite topics (one hop) and the concept relations. That context gives
     grounded questions, and some of them connect two topics (multi-hop).
   - Every draft must pass four checks:
     - its structure is valid
     - the concept it tests belongs to the graph
     - the quoted source sentence appears in the syllabus
     - a second NIM call answers the question without seeing the key and picks the same option
   - Near-duplicates are then dropped, and the answer positions are shuffled so the key isn't always the first option.
   - Survivors become **drafts**. Candidates only see questions **after an examiner approves them**.
2. **Tagging.** Every question gets a section, topic and concepts. On the 50 seed
   questions, the tagger matches the hand labels for **94%** of them (this is a unit test).
3. **Blueprint.** The section weights come from the graph. CAT uses content balancing
   (Kingsbury & Zara): it picks the section furthest below its target share, then the
   most informative question in that section. With 5 sections × 20% and 15 questions,
   every candidate gets exactly 3 per section.
4. **Section results.** The result shows the score in each section and lists the concepts
   the candidate missed.
5. **Coverage.** Topics and concepts without enough approved questions are ranked, so
   the examiner knows what to generate next.
6. **Performance analytics** (`exam_graph/analytics.py`):
   - Every answer is logged. From the log the system computes each question's p-value,
     point-biserial discrimination and response time.
   - Flags: too easy, too hard, a low-discrimination question (bad key or ambiguous
     wording), and a **possible leak** (far more correct answers than the IRT
     difficulty predicts).
   - **Online calibration.** Once a question has 30 or more answers, its difficulty `b`
     is re-estimated by maximum likelihood. This replaces the difficulty guesses that
     generated questions start with.
7. **Examiner Q&A.** Examiners can ask questions over the graph and the analytics, e.g.
   "Which topics do candidates fail most?", "What does Deep Network Training require?"
   or "Which concepts have no questions?".

**Pages:**
- `/ui/exam-admin.html`: knowledge graph and blueprint, question generation, draft
  review, question bank, analytics, syllabus editor, and Q&A
- `/ui/review.html`: proctoring evidence

## Adaptive testing (CAT)

- **Response model (2PL IRT):** `p(correct | θ, a, b) = 1 / (1 + exp(-a(θ-b)))`.
- **Ability estimation:** EAP over a normal prior. It stays stable with few responses.
- **Question selection:** maximum Fisher information `I(θ) = a²·p·(1-p)` within the
  blueprint section, avoiding two questions in a row from the same topic.
- **Length:** 15 questions, set by `exam_length:` in the syllabus.
- **Difficulty parameters:**
  - Seed questions use expert priors.
  - Generated questions start from easy/medium/hard difficulty guesses (b = −1, 0, +1).
  - Both are replaced by data through the online calibration above.

## AI proctoring: webcam + microphone, physics-aware NVIDIA judge, rules knowledge graph

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

### Known limitations (honest status)

- **Accuracy numbers.** They must be measured on real recordings. The evaluation tool is
  ready, but the thresholds were tuned on the developer's own recordings and synthetic
  tests (the automated test suite). A labelled set of about 40 recordings (half cheating,
  half clean) is the next step.
- **Model availability.** Whether Cosmos 3 and Nemotron Omni are reachable depends on
  the NVIDIA account and tier. `check_nim` shows which models work, and the chain falls
  back automatically. Self-hosting the NIM containers on a GPU removes the free-tier
  limits and cuts latency.
- **Latency.** Reasoning models take roughly 8–20 s per verdict on the free endpoint.
  The popup therefore arrives after that delay. That is acceptable for proctoring,
  because nothing is decided in real time.
- **Lip-sync attribution** needs the face to be visible. With the face turned away,
  speech is marked "unattributed" and goes to review as *suspicious*.
- **Scale.** Detection runs on the server's CPU (about 8 fps per candidate). For many
  simultaneous exams, move detection into the browser.
- **Run one server process for now.** Exam sessions, the question bank (a JSON file) and the
  proctoring sessions are held in the process. Don't start uvicorn with `--workers N` until
  they move to PostgreSQL and Redis (the platform's database stack).

### Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env     # NVIDIA_API_KEY, NIM_MODELS, PROCTOR_EVIDENCE_KEY, PROCTOR_ADMIN_TOKEN
python -m proctor.check_nim          # tests every NVIDIA model (judge chain + question generator) with your key
python -m uvicorn app.main:app --port 8000
```

Open **http://localhost:8000/**. The MediaPipe models (about 3.6 MB for the face model) download
on first use. The Whisper `base` model (about 150 MB) downloads on the first flagged speech.
`PROCTOR_MOCK_JUDGE=1` uses an offline rule-based judge.

To run it with Docker: `docker compose up --build`. It reads `backend/.env`, and the models
and evidence are kept on volumes.

## Project structure

```
backend/
  app/
    main.py              FastAPI app: start-test / submit-answer / result
    irt/engine.py         2PL probability model, EAP ability estimation, Fisher information
    schemas.py            Pydantic request/response models
    session_store.py      In-memory exam sessions (move to Redis/PostgreSQL for multi-worker deployment)
    data/
      question_bank.json          50 seed questions (imported into the exam bank on first start)
      generate_question_bank.py    source script that produced question_bank.json
  exam_graph/              GraphRAG exam content: syllabus parser, knowledge graph, NIM question
                           generator + verifier, question bank + approval, blueprint CAT,
                           analytics / calibration, examiner API
  knowledge/syllabus/      assessment syllabus (source of the exam knowledge graph)
  proctor/                 proctoring: features (gaze, lips), calibration, detector, objects,
                           audio (VAD, lip-sync, transcription), clip, judge (NIM model chain),
                           knowledge (GraphRAG), retention, service, api, replay, evaluate, check_nim
  graphrag_core/           GraphRAG pipeline vendored from the GraphRAG project
  knowledge/policy/        exam rules (R-01..R-14) indexed into the knowledge graph
  run_gaze.py              standalone webcam gaze tool (overlay + CSV logging)
  tests/
    test_irt_engine.py     unit tests: probability model, ability estimation, selection
    test_api_flow.py       integration tests: full adaptive-test runs through the API
    test_cat_proctoring.py CAT + proctoring: answer gating, alert mid-test, integrity report
    proctoring/            proctoring tests: synthetic faces + voices, mocked NIM chain,
                           audio/lip-sync, knowledge graph, evaluation harness
    exam_graph/            exam graph tests: syllabus, tagging accuracy, blueprint CAT, section results,
                           mocked NIM generation + verification, examiner API, calibration
  requirements.txt
frontend/
  index.html / style.css / app.js    vanilla JS client with a live ability gauge
  proctor/proctor.js, proctor.css    consent, calibration, frame + mic streaming (AudioWorklet), alert popup
  review.html                        examiner: proctoring evidence, audio, AI reasoning, rules-graph Q&A
  exam-admin.html                    examiner: exam knowledge graph, question generation + review, analytics
```

## Running it

Backend:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Frontend: open http://localhost:8000/. The backend serves `frontend/` at `/ui/`.

Tests:
```bash
cd backend
python -m pytest
```

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/start-test` | POST | Creates an exam session and returns the first (blueprint-selected) question |
| `/submit-answer` | POST | Records an answer and returns the updated θ/SE and the next question (or `finished: true`) |
| `/result/{session_id}` | GET | Ability, per-section results and concepts missed, integrity status |
| `/health` | GET | Product, assessment, question-bank size, model chains |
| `/exam/admin/graph` | GET | Knowledge-graph stats and exam blueprint |
| `/exam/admin/syllabus` | GET/PUT | Read the syllabus, or replace it (re-indexes the graph and re-tags the bank) |
| `/exam/admin/syllabus/convert` | POST | PDF/DOCX/TXT → syllabus draft (NIM) |
| `/exam/admin/graph/relations` | POST | NIM concept-relation extraction |
| `/exam/admin/coverage` | GET | Topics and concepts that need questions |
| `/exam/admin/generate` | POST | NIM question generation with verification → drafts |
| `/exam/admin/questions[/{id}/review]` | GET/POST | Question bank; approve or reject drafts |
| `/exam/admin/analytics[/calibrate]` | GET/POST | Item statistics and leak flags; apply calibrated difficulties |
| `/exam/admin/ask` | POST | Examiner Q&A over the exam graph |
| `/proctor/policy`, `/proctor/sessions`, `/proctor/ws/{id}` | – | Proctoring consent, session, webcam and mic stream |
| `/proctor/admin/*` | – | Evidence, audio, review, erasure, purge, usage, audit, rules graph |

All admin endpoints need the `X-Admin-Token` header (`PROCTOR_ADMIN_TOKEN`).

## Next steps

This round covers exactly what was requested: the microphone, GraphRAG, and NVIDIA NIM
models in place of Llama. Anything else (question variants, collusion detection, certificates)
waits for review feedback.
