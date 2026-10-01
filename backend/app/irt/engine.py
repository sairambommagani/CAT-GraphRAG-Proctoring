"""
2PL Item Response Theory engine for computerized adaptive testing.

Model:
    p(correct | theta, a, b) = 1 / (1 + exp(-a * (theta - b)))

    theta : latent student ability
    b     : question difficulty
    a     : question discrimination (how sharply p(correct) changes near theta = b)

Ability estimation uses EAP (Expected A Posteriori) over a fixed grid of
theta values with a standard normal prior. EAP is preferred over MLE here
because it stays well-behaved with very few responses (no answers yet,
all-correct, or all-incorrect streaks) and a short adaptive test lives
almost entirely in that low-n regime.

Question selection uses maximum Fisher information at the current theta
estimate, restricted to questions not yet administered to this student.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

THETA_GRID_MIN = -4.0
THETA_GRID_MAX = 4.0
THETA_GRID_STEP = 0.05
PRIOR_MEAN = 0.0
PRIOR_SD = 1.0

_THETA_GRID = [
    THETA_GRID_MIN + i * THETA_GRID_STEP
    for i in range(int((THETA_GRID_MAX - THETA_GRID_MIN) / THETA_GRID_STEP) + 1)
]


def _normal_pdf(x: float, mean: float, sd: float) -> float:
    return math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def p_correct(theta: float, a: float, b: float) -> float:
    """2PL probability of a correct response."""
    z = a * (theta - b)
    # guard against overflow for extreme theta/b combinations
    if z > 35:
        return 1.0
    if z < -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def fisher_information(theta: float, a: float, b: float) -> float:
    """2PL Fisher information: I(theta) = a^2 * p * (1 - p)."""
    p = p_correct(theta, a, b)
    return (a ** 2) * p * (1.0 - p)


@dataclass
class Response:
    question_id: str
    a: float
    b: float
    correct: bool


@dataclass
class AbilityEstimate:
    theta: float
    se: float  # posterior standard deviation, used as standard error


def estimate_ability(responses: list[Response], prior_mean: float = PRIOR_MEAN) -> AbilityEstimate:
    """
    EAP ability estimate given all responses so far.
    With zero responses, returns the prior (theta=prior_mean, se=1). `prior_mean` lets item
    SELECTION start from a returning candidate's last estimate; reported scores use 0.
    """
    if not responses:
        return AbilityEstimate(theta=prior_mean, se=PRIOR_SD)

    weighted_sum = 0.0
    total_weight = 0.0
    weighted_sq_sum = 0.0

    for theta in _THETA_GRID:
        prior = _normal_pdf(theta, prior_mean, PRIOR_SD)
        likelihood = 1.0
        for r in responses:
            p = p_correct(theta, r.a, r.b)
            likelihood *= p if r.correct else (1.0 - p)
        weight = prior * likelihood
        weighted_sum += theta * weight
        weighted_sq_sum += (theta ** 2) * weight
        total_weight += weight

    if total_weight == 0:
        # numerical underflow guard: fall back to prior
        return AbilityEstimate(theta=prior_mean, se=PRIOR_SD)

    mean = weighted_sum / total_weight
    variance = max(weighted_sq_sum / total_weight - mean ** 2, 1e-6)
    return AbilityEstimate(theta=mean, se=math.sqrt(variance))


def select_next_question(theta: float, candidates: list[dict]) -> dict | None:
    """
    Pick the unused question with maximum Fisher information at the
    current theta estimate. `candidates` are question-bank dicts with
    at least 'a' and 'b' keys; returns the chosen dict or None if empty.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda q: fisher_information(theta, q["a"], q["b"]))


def proficiency_label(theta: float) -> str:
    """Human-readable banding of the final ability estimate."""
    if theta < -1.5:
        return "Novice"
    if theta < -0.5:
        return "Beginner"
    if theta < 0.5:
        return "Intermediate"
    if theta < 1.5:
        return "Advanced"
    return "Expert"
