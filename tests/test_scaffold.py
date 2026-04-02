"""Tests for the Scaffolded Reasoning Pipeline."""

from tracellm.inference.scaffold import (
    PhaseID,
    PhaseResult,
    ScaffoldState,
    PHASE_PROMPTS,
    DEFAULT_PHASES,
)


def test_all_default_phases_exist():
    """All phases in the default pipeline should be valid PhaseIDs."""
    for phase in DEFAULT_PHASES:
        assert isinstance(phase, PhaseID)


def test_default_phase_count():
    """Default pipeline should have 8 phases."""
    assert len(DEFAULT_PHASES) == 8


def test_default_phase_order():
    """Phases should be in the correct order."""
    expected = [
        PhaseID.DECOMPOSE, PhaseID.PLAN, PhaseID.THINK, PhaseID.GENERATE,
        PhaseID.REFINE, PhaseID.QUALITY, PhaseID.VERIFY, PhaseID.EMIT,
    ]
    assert DEFAULT_PHASES == expected


def test_all_phases_have_prompts():
    """Every phase except REFINE should have a system prompt."""
    for phase in PhaseID:
        if phase == PhaseID.REFINE:
            continue  # REFINE delegates to RecursiveEngine
        assert phase in PHASE_PROMPTS, f"Missing prompt for {phase}"


def test_phase_prompts_are_nonempty():
    for phase, prompt in PHASE_PROMPTS.items():
        assert len(prompt) > 50, f"Prompt for {phase} is too short"


def test_scaffold_state_initialization():
    state = ScaffoldState(
        original_prompt="Design a REST API",
        context="For a bookstore",
        constraints=["Must use Python", "No external DBs"],
    )
    assert state.original_prompt == "Design a REST API"
    assert len(state.constraints) == 2
    assert state.decomposition == ""
    assert state.final_output == ""
    assert state.total_tokens == 0


def test_scaffold_state_scratchpad():
    state = ScaffoldState(original_prompt="test")
    state.append_scratchpad("decompose", "Found 3 sub-tasks")
    state.append_scratchpad("plan", "5 action items created")

    text = state.get_scratchpad_text()
    assert "[decompose]" in text
    assert "[plan]" in text
    assert "3 sub-tasks" in text


def test_scaffold_state_phase_results():
    state = ScaffoldState(original_prompt="test")
    pr = PhaseResult(
        phase_id="decompose",
        phase_name="Decompose",
        output="3 sub-tasks found",
        tokens_generated=150,
        elapsed_s=2.3,
    )
    state.phase_results.append(pr)
    state.total_tokens += pr.tokens_generated

    assert len(state.phase_results) == 1
    assert state.total_tokens == 150


def test_phase_result_metadata():
    pr = PhaseResult(
        phase_id="refine",
        phase_name="Recursive Refinement",
        output="refined output",
        tokens_generated=500,
        elapsed_s=8.2,
        metadata={"iterations": 5, "converged": True, "final_delta": 0.03},
    )
    assert pr.metadata["converged"] is True
    assert pr.metadata["iterations"] == 5


def test_confidence_extraction_from_scaffold_state():
    """ScaffoldState should track confidence from verification."""
    state = ScaffoldState(original_prompt="test")
    state.confidence = 0.85
    assert state.confidence == 0.85


def test_scaffold_state_stores_all_phases():
    """State should have fields for every phase output."""
    state = ScaffoldState(original_prompt="test")
    state.decomposition = "tasks"
    state.execution_plan = "plan"
    state.reasoning_trace = "reasoning"
    state.draft = "draft v1"
    state.refined_draft = "draft v2"
    state.quality_report = "PASS"
    state.verification_report = "all checks pass"
    state.final_output = "polished output"

    assert all([
        state.decomposition,
        state.execution_plan,
        state.reasoning_trace,
        state.draft,
        state.refined_draft,
        state.quality_report,
        state.verification_report,
        state.final_output,
    ])


def test_phase_id_values():
    """PhaseID enum should have correct string values."""
    assert PhaseID.DECOMPOSE.value == "decompose"
    assert PhaseID.PLAN.value == "plan"
    assert PhaseID.THINK.value == "think"
    assert PhaseID.GENERATE.value == "generate"
    assert PhaseID.REFINE.value == "refine"
    assert PhaseID.QUALITY.value == "quality"
    assert PhaseID.VERIFY.value == "verify"
    assert PhaseID.EMIT.value == "emit"


def test_phase_id_from_string():
    """PhaseID should be constructible from strings."""
    assert PhaseID("decompose") == PhaseID.DECOMPOSE
    assert PhaseID("refine") == PhaseID.REFINE
    assert PhaseID("emit") == PhaseID.EMIT


def test_custom_phase_subset():
    """Should support running a subset of phases."""
    quick_phases = [PhaseID.DECOMPOSE, PhaseID.GENERATE, PhaseID.VERIFY, PhaseID.EMIT]
    assert len(quick_phases) == 4
    assert PhaseID.REFINE not in quick_phases
    assert PhaseID.THINK not in quick_phases
