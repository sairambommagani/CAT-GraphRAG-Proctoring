"""Proctoring knowledge graph (GraphRAG over exam rules + incidents)."""
import json

import httpx
import pytest

from proctor.clip import Evidence
from proctor.detector import Event
from proctor.knowledge import NIMTextLLM, ProctorKnowledge, load_rules, parse_policy


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def kg(tmp_path):
    clock = Clock()
    k = ProctorKnowledge(tmp_path / "kg", clock=clock)
    k.clock_obj = clock
    return k


def ev(etype, direction=None, **details):
    return Evidence(Event(etype, 1.0, 4.0, direction, details=details), [], 0.0, 5.0)


def test_policy_parses_every_rule():
    rules = load_rules()
    ids = [r.id for r in rules]
    assert ids == [f"R-{i:02d}" for i in range(1, 15)]
    r3 = next(r for r in rules if r.id == "R-03")
    assert r3.severity == "critical" and "PROHIBITED_OBJECT" in r3.applies_to and "phone" in r3.targets
    assert any("background" in a for a in r3.allowed)


def test_graph_is_built_with_graphrag_pipeline(kg):
    st = kg.stats()
    assert st["rules"] == 14 and st["num_entities"] > 60 and st["num_relationships"] > 150
    assert st["entity_types"]["RULE"] == 14 and st["entity_types"]["EVENT_TYPE"] == 8
    assert st["level0_communities"] >= 3 and st["level1_communities"] >= 1        # Louvain hierarchy
    # multi-hop path: rule -> event type -> signal
    assert kg.graph.has_edge("R-10 No other voices in the room", "VOICE_DETECTED")
    assert kg.graph.has_edge("VOICE_DETECTED", "signal: lip sync")


@pytest.mark.parametrize("etype,direction,first", [
    ("PROHIBITED_OBJECT", "phone", "R-03"),
    ("PROHIBITED_OBJECT", "book", "R-04"),
    ("PROHIBITED_OBJECT", "second device", "R-05"),
    ("OFFSCREEN_SUSTAINED", "down", "R-02"),
    ("REPEATED_GLANCES", "left", "R-01"),
    ("VOICE_DETECTED", "another person", "R-10"),
    ("VOICE_DETECTED", "candidate", "R-09"),
    ("VOICE_DETECTED", "unattributed", "R-11"),
    ("MULTIPLE_FACES", None, "R-06"),
    ("FOREIGN_HAND", None, "R-07"),
    ("NO_FACE", None, "R-08"),
])
def test_judge_gets_the_right_rule_first(kg, etype, direction, first):
    ctx = kg.context_for(ev(etype, direction))
    assert ctx["rules"][0]["id"] == first
    assert len(ctx["rules"]) <= 3 and all(len(r["text"]) < 400 for r in ctx["rules"])


def test_examiner_decisions_become_past_cases(kg):
    phone = ev("PROHIBITED_OBJECT", "phone")
    for i, (verdict, review) in enumerate([("fraud", "confirm"), ("fraud", "confirm"), ("fraud", "dismiss")]):
        kg.record_incident(f"sess{i}".ljust(16, "0"), phone, {"verdict": verdict, "confidence": .9, "rule": "R-03",
                                                              "reason": "phone in hand"}, evidence_id=f"e{i}".ljust(12, "x"))
        kg.record_review(f"e{i}".ljust(12, "x"), review)
    kg.record_incident("sessX".ljust(16, "0"), ev("PROHIBITED_OBJECT", "book"),
                       {"verdict": "benign", "confidence": .8}, evidence_id="book".ljust(12, "x"))
    cases = kg.context_for(phone)["cases"]
    assert "3 earlier PROHIBITED_OBJECT (phone) flags" in cases[0]["summary"]
    assert "2 confirmed by examiner" in cases[0]["summary"] and "1 dismissed by examiner" in cases[0]["summary"]
    assert all("book" not in c["summary"] for c in cases)                        # only similar cases


def test_session_history_supports_escalation(kg):
    sid = "abcdef0123456789"
    kg.record_incident(sid, ev("OFFSCREEN_SUSTAINED", "down"), {"verdict": "fraud", "confidence": .9},
                       evidence_id="a" * 12)
    ctx = kg.context_for(ev("PROHIBITED_OBJECT", "phone"), session_id=sid)
    assert ctx["cases"][0]["summary"].startswith("this session already has 1 fraud incident")
    assert kg.context_for(ev("PROHIBITED_OBJECT", "phone"), session_id="other")["cases"] == []


def test_incidents_persist_and_follow_retention(kg, tmp_path):
    kg.record_incident("s" * 16, ev("NO_FACE"), {"verdict": "fraud", "confidence": .9}, evidence_id="old" + "x" * 9)
    kg.clock_obj.t += 80 * 86400
    kg.record_incident("t" * 16, ev("NO_FACE"), {"verdict": "fraud", "confidence": .9}, evidence_id="new" + "x" * 9)
    again = ProctorKnowledge(tmp_path / "kg")                     # reload from disk
    assert len(again.incidents) == 2
    line = (tmp_path / "kg" / "incidents.jsonl").read_text()
    assert "transcript" not in line and "jpeg" not in line          # metadata only
    kg.clock_obj.t += 15 * 86400                                    # old one is now > 90 days
    assert kg.purge() == 1 and list(kg.incidents) == ["new" + "x" * 9]
    assert kg.forget_session("t" * 16) == 1 and kg.incidents == {}


def test_examiner_questions(kg):
    for i in range(3):
        kg.record_incident("1a2b3c4d".ljust(16, "0"), ev("PROHIBITED_OBJECT", "phone"),
                           {"verdict": "fraud", "confidence": .9, "rule": "R-03"}, evidence_id=f"p{i}".ljust(12, "x"))
    kg.record_incident("9f8e7d6c".ljust(16, "0"), ev("VOICE_DETECTED", "another person"),
                       {"verdict": "fraud", "confidence": .9, "rule": "R-10"}, evidence_id="v".ljust(12, "x"))
    a = kg.ask("Which sessions had repeated phone use?")
    assert "Session 1a2b3c4d: 3 incident(s)" in a["answer"] and "9f8e7d6c" not in a["answer"]
    assert a["method"].startswith("graph traversal")
    a = kg.ask("what does R-06 say?")
    assert a["answer"].startswith("[R-06]") and "alone" in a["answer"].lower()
    a = kg.ask("what happened in session 9f8e7d6c")
    assert "VOICE_DETECTED" in a["answer"] or "incident" in a["answer"].lower()
    a = kg.ask("summarise the policy themes")                       # no anchor -> global search
    assert a["method"].startswith("extractive") and a["communities"]


def test_rules_file_is_editable(tmp_path):
    pol = tmp_path / "policy"
    pol.mkdir()
    (pol / "rules.md").write_text("## R-01 Custom rule\napplies_to: NO_FACE\ntargets: absent\n"
                                  "severity: minor\nallowed: none\nStay visible.\n")
    k = ProctorKnowledge(tmp_path / "kg", policy_dir=pol)
    assert [r.id for r in k.rules] == ["R-01"]
    (pol / "rules.md").write_text((pol / "rules.md").read_text() + "\n## R-02 Another\napplies_to: NO_FACE\nText.\n")
    assert k.reload_rules() == 2
    assert sorted(r["id"] for r in k.context_for(ev("NO_FACE"))["rules"]) == ["R-01", "R-02"]


def test_optional_llm_answer_uses_graph_context(kg):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "Session 1a2b3c4d used a phone (R-03)."}}]})
    kg.record_incident("1a2b3c4d".ljust(16, "0"), ev("PROHIBITED_OBJECT", "phone"),
                       {"verdict": "fraud", "confidence": .9, "rule": "R-03"}, evidence_id="p".ljust(12, "x"))
    kg.llm = NIMTextLLM("meta/llama-3.1-8b-instruct", "k", "https://x/v1",
                        client=httpx.Client(transport=httpx.MockTransport(handler)))
    a = kg.ask("who used a phone?")
    assert a["answer"] == "Session 1a2b3c4d used a phone (R-03)." and "LLM" in a["method"]
    assert "1a2b3c4d" in seen["body"]["messages"][1]["content"]           # grounded in retrieved facts


def test_parse_ignores_non_rule_sections():
    assert parse_policy("# Title\n\n## Notes\nnot a rule\n\n## R-7 Ok\napplies_to: NO_FACE\nx") [0].id == "R-7"
