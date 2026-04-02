"""Tests for the Recursive Language Model engine."""

from tracellm.inference.recursive import (
    compute_delta,
    get_pass_category,
    PassCategory,
    DEFAULT_PASS_SCHEDULE,
    REFINEMENT_PROMPTS,
    RecursiveEngine,
    RefinementPass,
)


def test_compute_delta_identical():
    """Identical texts should have delta 0."""
    assert compute_delta("hello world", "hello world") == 0.0


def test_compute_delta_completely_different():
    """Completely different texts should have high delta."""
    delta = compute_delta("aaaaaa", "zzzzzz")
    assert delta > 0.8


def test_compute_delta_similar():
    """Similar texts should have low delta."""
    a = "The quick brown fox jumps over the lazy dog."
    b = "The quick brown fox leaps over the lazy dog."
    delta = compute_delta(a, b)
    assert 0.0 < delta < 0.2


def test_compute_delta_empty_strings():
    assert compute_delta("", "") == 0.0
    assert compute_delta("hello", "") == 1.0
    assert compute_delta("", "hello") == 1.0


def test_get_pass_category_structural():
    """Iterations 1-3 should be structural."""
    assert get_pass_category(1) == PassCategory.STRUCTURAL
    assert get_pass_category(2) == PassCategory.STRUCTURAL
    assert get_pass_category(3) == PassCategory.STRUCTURAL


def test_get_pass_category_factual():
    """Iterations 4-6 should be factual."""
    assert get_pass_category(4) == PassCategory.FACTUAL
    assert get_pass_category(5) == PassCategory.FACTUAL
    assert get_pass_category(6) == PassCategory.FACTUAL


def test_get_pass_category_stylistic():
    """Iterations 7-8 should be stylistic."""
    assert get_pass_category(7) == PassCategory.STYLISTIC
    assert get_pass_category(8) == PassCategory.STYLISTIC


def test_get_pass_category_adversarial():
    assert get_pass_category(9) == PassCategory.ADVERSARIAL


def test_get_pass_category_final_lock():
    assert get_pass_category(10) == PassCategory.FINAL_LOCK


def test_get_pass_category_beyond_schedule():
    """Iterations beyond the schedule should fall back to final_lock."""
    assert get_pass_category(99) == PassCategory.FINAL_LOCK


def test_all_pass_categories_have_prompts():
    """Every PassCategory should have a refinement prompt defined."""
    for category in PassCategory:
        assert category in REFINEMENT_PROMPTS, f"Missing prompt for {category}"


def test_refinement_prompts_are_nonempty():
    for category, prompt in REFINEMENT_PROMPTS.items():
        assert len(prompt) > 50, f"Prompt for {category} is too short"


def test_custom_pass_schedule():
    """Custom schedules should override the default."""
    custom = {
        (1, 5): PassCategory.ADVERSARIAL,
        (6, 10): PassCategory.STRUCTURAL,
    }
    assert get_pass_category(3, custom) == PassCategory.ADVERSARIAL
    assert get_pass_category(8, custom) == PassCategory.STRUCTURAL


def test_refinement_pass_dataclass():
    """RefinementPass should store iteration metadata correctly."""
    rp = RefinementPass(
        iteration=3,
        category=PassCategory.STRUCTURAL,
        input_text="draft v2",
        output_text="draft v3",
        delta_ratio=0.15,
        tokens_generated=200,
        elapsed_s=1.5,
    )
    assert rp.iteration == 3
    assert rp.category == PassCategory.STRUCTURAL
    assert rp.delta_ratio == 0.15
