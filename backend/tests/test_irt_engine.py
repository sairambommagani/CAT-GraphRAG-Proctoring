from app.irt.engine import (
    Response,
    p_correct,
    fisher_information,
    estimate_ability,
    select_next_question,
    proficiency_label,
)


def test_p_correct_at_matching_ability_is_half():
    # when theta == b, the 2PL model predicts exactly 50% regardless of a
    assert abs(p_correct(theta=0.5, a=1.7, b=0.5) - 0.5) < 1e-9


def test_p_correct_monotonic_in_theta():
    p_low = p_correct(theta=-2, a=1.0, b=0.0)
    p_high = p_correct(theta=2, a=1.0, b=0.0)
    assert p_low < p_high


def test_fisher_information_peaks_near_theta_equals_b():
    at_b = fisher_information(theta=0.0, a=1.5, b=0.0)
    far_from_b = fisher_information(theta=3.0, a=1.5, b=0.0)
    assert at_b > far_from_b


def test_estimate_ability_no_responses_returns_prior():
    est = estimate_ability([])
    assert est.theta == 0.0
    assert est.se == 1.0


def test_estimate_ability_moves_up_with_correct_answers():
    responses = [Response(question_id="q1", a=1.5, b=0.0, correct=True)]
    est = estimate_ability(responses)
    assert est.theta > 0.0


def test_estimate_ability_moves_down_with_incorrect_answers():
    responses = [Response(question_id="q1", a=1.5, b=0.0, correct=False)]
    est = estimate_ability(responses)
    assert est.theta < 0.0


def test_estimate_ability_uncertainty_shrinks_with_more_responses():
    one_response = [Response(question_id="q1", a=1.5, b=0.0, correct=True)]
    many_responses = one_response + [
        Response(question_id=f"q{i}", a=1.5, b=0.0, correct=True) for i in range(2, 6)
    ]
    se_one = estimate_ability(one_response).se
    se_many = estimate_ability(many_responses).se
    assert se_many < se_one


def test_select_next_question_picks_highest_information():
    candidates = [
        {"id": "easy", "a": 1.0, "b": -3.0},
        {"id": "match", "a": 1.0, "b": 0.0},
        {"id": "hard", "a": 1.0, "b": 3.0},
    ]
    chosen = select_next_question(theta=0.0, candidates=candidates)
    assert chosen["id"] == "match"


def test_select_next_question_empty_candidates_returns_none():
    assert select_next_question(theta=0.0, candidates=[]) is None


def test_proficiency_label_bands():
    assert proficiency_label(-3.0) == "Novice"
    assert proficiency_label(0.0) == "Intermediate"
    assert proficiency_label(3.0) == "Expert"
