"""Scaffolded Reasoning Pipeline — structured multi-phase generation.

Implements the AGL-style 8-phase reasoning pipeline:
  1. DECOMPOSE — break the task into atomic parts
  2. PLAN     — create an execution plan with ordered action items
  3. THINK    — transparent reasoning, surface assumptions
  4. GENERATE — produce the initial output
  5. REFINE   — recursive self-improvement (delegates to RecursiveEngine)
  6. QUALITY  — syntax, security, spelling, formatting checks
  7. VERIFY   — holistic re-evaluation against original request
  8. EMIT     — final formatting and confidence annotation

Each phase receives the accumulated state from prior phases and produces
structured output that feeds into the next. Phases can be skipped,
reordered, or extended with custom phases.

The scaffold is the "bones" of deep reasoning — it forces the model to
think before generating, plan before thinking, and verify before emitting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from tracellm.inference.engine import InferenceEngine
from tracellm.inference.recursive import RecursiveEngine, RecursiveResult
from tracellm.inference.sampling import SamplingParams
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.inference.scaffold")


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseID(str, Enum):
    DECOMPOSE = "decompose"
    PLAN = "plan"
    THINK = "think"
    GENERATE = "generate"
    REFINE = "refine"
    QUALITY = "quality"
    VERIFY = "verify"
    EMIT = "emit"


@dataclass
class PhaseResult:
    """Output of a single scaffold phase."""
    phase_id: str
    phase_name: str
    output: str
    tokens_generated: int
    elapsed_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaffoldState:
    """Accumulated state carried through the scaffold pipeline.

    Every phase can read from and write to this state. The scratchpad
    is the shared memory across phases — phases append their reasoning
    traces here so downstream phases can see HOW conclusions were reached.
    """
    original_prompt: str
    context: str = ""
    constraints: list[str] = field(default_factory=list)

    # Phase outputs — populated as the pipeline progresses
    decomposition: str = ""
    execution_plan: str = ""
    reasoning_trace: str = ""
    draft: str = ""
    refined_draft: str = ""
    quality_report: str = ""
    verification_report: str = ""
    final_output: str = ""

    # Scratchpad — shared memory across all phases
    scratchpad: list[str] = field(default_factory=list)

    # Metadata
    phase_results: list[PhaseResult] = field(default_factory=list)
    total_tokens: int = 0
    total_time_s: float = 0.0
    confidence: float = 0.0

    def append_scratchpad(self, phase: str, content: str) -> None:
        self.scratchpad.append(f"[{phase}] {content}")

    def get_scratchpad_text(self) -> str:
        return "\n\n".join(self.scratchpad)


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_PROMPTS: dict[PhaseID, str] = {
    PhaseID.DECOMPOSE: (
        "You are a task decomposition specialist. Your job is to break down "
        "the user's request into atomic, clearly defined sub-tasks.\n\n"
        "For the given request:\n"
        "1. Identify WHAT is being asked (the deliverable)\n"
        "2. Identify WHY it's being asked (the purpose/context)\n"
        "3. Identify for WHOM (the audience)\n"
        "4. List all EXPLICIT requirements stated in the request\n"
        "5. List all IMPLICIT requirements (things not stated but necessary)\n"
        "6. Identify CONSTRAINTS (limitations, rules, format requirements)\n"
        "7. Flag any AMBIGUITIES that require assumptions\n"
        "8. Break the task into numbered, atomic sub-tasks\n\n"
        "Output a structured decomposition. Be thorough — missed requirements "
        "here will cascade into errors later."
    ),

    PhaseID.PLAN: (
        "You are an execution planner. Given a task decomposition, create a "
        "concrete execution plan.\n\n"
        "For each sub-task from the decomposition:\n"
        "1. Convert it into an ordered ACTION ITEM with a clear deliverable\n"
        "2. Estimate COMPLEXITY (low/medium/high)\n"
        "3. Flag potential FAILURE POINTS or risks\n"
        "4. Define SUCCESS CRITERIA — how will you know this action item is done?\n"
        "5. Note DEPENDENCIES between action items\n\n"
        "Output a numbered execution plan. Each item should be self-contained "
        "enough that it could be executed independently."
    ),

    PhaseID.THINK: (
        "You are performing transparent reasoning. Walk through each action "
        "item in the execution plan and reason through it BEFORE writing.\n\n"
        "For each action item:\n"
        "1. State what you are CONSIDERING\n"
        "2. Identify AMBIGUITIES and how you'll resolve them\n"
        "3. Note the DOMAIN KNOWLEDGE you're applying\n"
        "4. Surface potential ISSUES before they become errors\n"
        "5. State your ASSUMPTIONS explicitly\n\n"
        "Think out loud. This reasoning trace will be used by later phases "
        "to verify the output is grounded."
    ),

    PhaseID.GENERATE: (
        "You are generating the deliverable. Using the decomposition, execution "
        "plan, and reasoning trace provided, produce the actual output.\n\n"
        "Rules:\n"
        "- Follow the execution plan step by step\n"
        "- Apply the domain-appropriate format\n"
        "- Include ALL deliverables identified in the decomposition\n"
        "- Ground every claim in the reasoning trace\n"
        "- Mark anything uncertain with [UNCERTAIN]\n\n"
        "Produce a complete first draft."
    ),

    PhaseID.QUALITY: (
        "You are a quality gate reviewer. Check the draft for:\n\n"
        "1. CODE QUALITY (if applicable):\n"
        "   - Syntax errors\n"
        "   - Missing imports or undefined references\n"
        "   - Security vulnerabilities (injection, XSS, etc.)\n"
        "   - Missing error handling\n\n"
        "2. PROSE QUALITY (if applicable):\n"
        "   - Spelling and grammar\n"
        "   - Consistent terminology\n"
        "   - Logical flow\n\n"
        "3. FORMAT QUALITY:\n"
        "   - Consistent heading levels\n"
        "   - Proper code block formatting\n"
        "   - Complete lists (no trailing items)\n\n"
        "Output a quality report with specific issues found and their severity "
        "(CRITICAL / HIGH / MEDIUM / LOW). If no issues, state 'PASS'."
    ),

    PhaseID.VERIFY: (
        "You are performing dense verification. Re-read the original request "
        "and verify the output satisfies it completely.\n\n"
        "Checks:\n"
        "1. COMPLETENESS — Are all deliverables from the decomposition present?\n"
        "2. ACCURACY — Are all claims grounded in the context or reasoning trace?\n"
        "3. HALLUCINATION — Is anything stated as fact that wasn't provided or derived?\n"
        "4. PLAN ADHERENCE — Were all action items from the execution plan completed?\n"
        "5. CONSTRAINT COMPLIANCE — Are all constraints satisfied?\n"
        "6. CONSISTENCY — Do all sections agree with each other?\n\n"
        "Output a verification report. For each check, state PASS or FAIL with "
        "specifics. Include an overall confidence score from 0.0 to 1.0."
    ),

    PhaseID.EMIT: (
        "You are performing final output preparation. Take the verified draft "
        "and apply final polish.\n\n"
        "Actions:\n"
        "1. Apply consistent formatting throughout\n"
        "2. Add section headers if the output is long\n"
        "3. If any sections were marked [UNCERTAIN], add a note at the end\n"
        "4. Remove any internal markers, reasoning artifacts, or meta-commentary\n"
        "5. Ensure the output reads as a polished, final deliverable\n\n"
        "Output ONLY the final polished result. No preamble, no explanation "
        "of what you changed."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  SCAFFOLD ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

# Default phase ordering
DEFAULT_PHASES = [
    PhaseID.DECOMPOSE,
    PhaseID.PLAN,
    PhaseID.THINK,
    PhaseID.GENERATE,
    PhaseID.REFINE,
    PhaseID.QUALITY,
    PhaseID.VERIFY,
    PhaseID.EMIT,
]


class ScaffoldEngine:
    """Scaffolded reasoning pipeline — 8-phase structured generation.

    Each phase runs the model with a specialized system prompt and accumulated
    context from prior phases. The REFINE phase delegates to the RecursiveEngine
    for iterative self-improvement.

    Usage:
        scaffold = ScaffoldEngine(inference_engine)
        state = scaffold.run("model-name", "Design a REST API for a bookstore")
        print(state.final_output)

    Custom phases:
        scaffold = ScaffoldEngine(inference_engine, phases=[
            PhaseID.DECOMPOSE, PhaseID.GENERATE, PhaseID.VERIFY, PhaseID.EMIT
        ])
    """

    def __init__(
        self,
        engine: InferenceEngine,
        phases: list[PhaseID] | None = None,
        phase_prompts: dict[PhaseID, str] | None = None,
        refine_iterations: int = 5,
        refine_convergence: float = 0.95,
        skip_adversarial: bool = False,
        verify_retries: int = 2,
    ):
        self.engine = engine
        self.phases = phases or DEFAULT_PHASES
        self.phase_prompts = {**PHASE_PROMPTS, **(phase_prompts or {})}
        self.refine_iterations = refine_iterations
        self.refine_convergence = refine_convergence
        self.skip_adversarial = skip_adversarial
        self.verify_retries = verify_retries

        self.recursive_engine = RecursiveEngine(
            engine=engine,
            max_iterations=refine_iterations,
            convergence_threshold=refine_convergence,
            skip_adversarial=skip_adversarial,
        )

    def run(
        self,
        model_name: str,
        prompt: str,
        context: str = "",
        constraints: list[str] | None = None,
        params: SamplingParams | None = None,
        on_phase: Callable[[PhaseResult], None] | None = None,
        show_reasoning: bool = False,
    ) -> ScaffoldState:
        """Execute the full scaffold pipeline.

        Args:
            model_name: Registered model name.
            prompt: The user's request.
            context: Additional context/background.
            constraints: Hard rules the output must follow.
            params: Sampling parameters.
            on_phase: Callback after each phase completes.
            show_reasoning: If True, include reasoning traces in final output.

        Returns:
            ScaffoldState with all phase outputs and the final result.
        """
        params = params or SamplingParams()
        state = ScaffoldState(
            original_prompt=prompt,
            context=context,
            constraints=constraints or [],
        )

        start = time.time()
        log.info(f"Scaffold: starting {len(self.phases)}-phase pipeline for '{prompt[:60]}...'")

        for phase_id in self.phases:
            phase_start = time.time()
            log.info(f"Scaffold: entering phase [{phase_id.value}]")

            if phase_id == PhaseID.REFINE:
                phase_result = self._run_refine_phase(model_name, state, params)
            elif phase_id == PhaseID.VERIFY:
                phase_result = self._run_verify_phase(model_name, state, params)
            else:
                phase_result = self._run_standard_phase(
                    model_name, phase_id, state, params
                )

            # Store result in state
            self._store_phase_output(state, phase_id, phase_result)
            state.phase_results.append(phase_result)
            state.total_tokens += phase_result.tokens_generated
            state.append_scratchpad(phase_id.value, phase_result.output[:500])

            log.info(
                f"Scaffold: [{phase_id.value}] complete — "
                f"{phase_result.tokens_generated} tokens, "
                f"{phase_result.elapsed_s:.1f}s"
            )

            if on_phase:
                on_phase(phase_result)

        state.total_time_s = round(time.time() - start, 3)

        # Set final output
        if show_reasoning:
            state.final_output = self._build_full_reasoning_output(state)
        elif not state.final_output:
            # Use the emit output, or fall back to refined draft
            state.final_output = state.final_output or state.refined_draft or state.draft

        log.info(
            f"Scaffold: pipeline complete — {len(state.phase_results)} phases, "
            f"{state.total_tokens} tokens, {state.total_time_s}s"
        )
        return state

    def stream_run(
        self,
        model_name: str,
        prompt: str,
        context: str = "",
        constraints: list[str] | None = None,
        params: SamplingParams | None = None,
    ):
        """Streaming scaffold — yields phase events as they complete.

        Yields dicts:
            {"type": "phase_start", "phase": "decompose", "index": 0, "total": 8}
            {"type": "phase_complete", "phase": "decompose", "output": "...", "tokens": 234}
            {"type": "refine_pass", "iteration": 2, "category": "factual", "delta": 0.15}
            {"type": "final", "output": "...", "phases_run": 8, "total_tokens": 5432}
        """
        params = params or SamplingParams()
        state = ScaffoldState(
            original_prompt=prompt,
            context=context,
            constraints=constraints or [],
        )
        total_phases = len(self.phases)

        for idx, phase_id in enumerate(self.phases):
            yield {
                "type": "phase_start",
                "phase": phase_id.value,
                "index": idx,
                "total": total_phases,
            }

            if phase_id == PhaseID.REFINE:
                # Stream refinement passes
                for event in self.recursive_engine.stream_refine(
                    model_name,
                    prompt=state.draft,
                    context=self._build_refine_context(state),
                    params=params,
                ):
                    if event["type"] == "pass_complete":
                        yield {
                            "type": "refine_pass",
                            "iteration": event["iteration"],
                            "category": event["category"],
                            "delta": event["delta"],
                        }
                    elif event["type"] == "final":
                        state.refined_draft = event["text"]
                        yield {
                            "type": "phase_complete",
                            "phase": phase_id.value,
                            "output": event["text"][:200] + "...",
                            "tokens": 0,
                            "converged": event["converged"],
                        }
            else:
                phase_result = self._run_standard_phase(
                    model_name, phase_id, state, params
                ) if phase_id != PhaseID.VERIFY else self._run_verify_phase(
                    model_name, state, params
                )

                self._store_phase_output(state, phase_id, phase_result)
                state.phase_results.append(phase_result)
                state.total_tokens += phase_result.tokens_generated

                yield {
                    "type": "phase_complete",
                    "phase": phase_id.value,
                    "output": phase_result.output[:200] + "..." if len(phase_result.output) > 200 else phase_result.output,
                    "tokens": phase_result.tokens_generated,
                }

        yield {
            "type": "final",
            "output": state.final_output or state.refined_draft or state.draft,
            "phases_run": len(self.phases),
            "total_tokens": state.total_tokens,
        }

    def _run_standard_phase(
        self,
        model_name: str,
        phase_id: PhaseID,
        state: ScaffoldState,
        params: SamplingParams,
    ) -> PhaseResult:
        """Run a standard (non-refine, non-verify) phase."""
        prompt = self._build_phase_prompt(phase_id, state)
        start = time.time()
        result = self.engine.generate(model_name, prompt, params)
        elapsed = time.time() - start

        return PhaseResult(
            phase_id=phase_id.value,
            phase_name=phase_id.name.title(),
            output=result.text,
            tokens_generated=result.tokens_generated,
            elapsed_s=round(elapsed, 3),
        )

    def _run_refine_phase(
        self,
        model_name: str,
        state: ScaffoldState,
        params: SamplingParams,
    ) -> PhaseResult:
        """Run the recursive refinement phase."""
        start = time.time()

        refine_context = self._build_refine_context(state)

        recursive_result = self.recursive_engine.refine(
            model_name=model_name,
            prompt=state.draft,
            context=refine_context,
            params=params,
        )

        elapsed = time.time() - start
        state.refined_draft = recursive_result.final_text

        return PhaseResult(
            phase_id=PhaseID.REFINE.value,
            phase_name="Recursive Refinement",
            output=recursive_result.final_text,
            tokens_generated=recursive_result.total_tokens,
            elapsed_s=round(elapsed, 3),
            metadata={
                "iterations": recursive_result.iterations_run,
                "converged": recursive_result.converged,
                "final_delta": recursive_result.convergence_delta,
            },
        )

    def _run_verify_phase(
        self,
        model_name: str,
        state: ScaffoldState,
        params: SamplingParams,
    ) -> PhaseResult:
        """Run verification with retry loop on failure."""
        draft = state.refined_draft or state.draft
        total_tokens = 0
        start = time.time()

        for attempt in range(1, self.verify_retries + 2):  # +1 for initial, +1 for range
            prompt = self._build_phase_prompt(PhaseID.VERIFY, state)
            result = self.engine.generate(model_name, prompt, params)
            total_tokens += result.tokens_generated

            verification_text = result.text
            state.verification_report = verification_text

            # Parse confidence from verification output
            confidence = self._extract_confidence(verification_text)
            state.confidence = confidence

            if confidence >= 0.7 or attempt > self.verify_retries:
                break

            # Verification failed — feed back into a quick refinement pass
            log.warning(
                f"Scaffold: verification confidence {confidence:.2f} < 0.7, "
                f"retry {attempt}/{self.verify_retries}"
            )
            fix_prompt = (
                f"<|system|>\nThe verification found issues. Fix them.\n\n"
                f"<|verification_report|>\n{verification_text}\n\n"
                f"<|current_draft|>\n{draft}\n\n"
                f"<|user|>\nFix all FAIL items from the verification report. "
                f"Output the complete corrected version.\n"
                f"<|assistant|>\n"
            )
            fix_result = self.engine.generate(model_name, fix_prompt, params)
            total_tokens += fix_result.tokens_generated

            # Update the draft for next verification pass
            draft = fix_result.text
            state.refined_draft = draft

        elapsed = time.time() - start
        return PhaseResult(
            phase_id=PhaseID.VERIFY.value,
            phase_name="Dense Verification",
            output=verification_text,
            tokens_generated=total_tokens,
            elapsed_s=round(elapsed, 3),
            metadata={
                "confidence": confidence,
                "attempts": attempt,
            },
        )

    def _build_phase_prompt(self, phase_id: PhaseID, state: ScaffoldState) -> str:
        """Build the full prompt for a phase, including accumulated state."""
        system = self.phase_prompts.get(phase_id, "")
        parts = [f"<|system|>\n{system}\n"]

        # Always include original request
        parts.append(f"<|original_request|>\n{state.original_prompt}\n")

        # Include context if present
        if state.context:
            parts.append(f"<|context|>\n{state.context}\n")

        # Include constraints if present
        if state.constraints:
            constraints_text = "\n".join(f"- {c}" for c in state.constraints)
            parts.append(f"<|constraints|>\n{constraints_text}\n")

        # Include prior phase outputs based on what this phase needs
        if phase_id == PhaseID.PLAN and state.decomposition:
            parts.append(f"<|decomposition|>\n{state.decomposition}\n")

        elif phase_id == PhaseID.THINK:
            if state.decomposition:
                parts.append(f"<|decomposition|>\n{state.decomposition}\n")
            if state.execution_plan:
                parts.append(f"<|execution_plan|>\n{state.execution_plan}\n")

        elif phase_id == PhaseID.GENERATE:
            if state.decomposition:
                parts.append(f"<|decomposition|>\n{state.decomposition}\n")
            if state.execution_plan:
                parts.append(f"<|execution_plan|>\n{state.execution_plan}\n")
            if state.reasoning_trace:
                parts.append(f"<|reasoning_trace|>\n{state.reasoning_trace}\n")

        elif phase_id == PhaseID.QUALITY:
            draft = state.refined_draft or state.draft
            parts.append(f"<|draft|>\n{draft}\n")

        elif phase_id == PhaseID.VERIFY:
            draft = state.refined_draft or state.draft
            parts.append(f"<|draft|>\n{draft}\n")
            if state.decomposition:
                parts.append(f"<|decomposition|>\n{state.decomposition}\n")
            if state.execution_plan:
                parts.append(f"<|execution_plan|>\n{state.execution_plan}\n")
            if state.quality_report:
                parts.append(f"<|quality_report|>\n{state.quality_report}\n")

        elif phase_id == PhaseID.EMIT:
            draft = state.refined_draft or state.draft
            parts.append(f"<|draft|>\n{draft}\n")
            if state.verification_report:
                parts.append(f"<|verification_report|>\n{state.verification_report}\n")

        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _build_refine_context(self, state: ScaffoldState) -> str:
        """Build context string for the refinement phase."""
        parts = []
        if state.context:
            parts.append(f"Context: {state.context}")
        if state.decomposition:
            parts.append(f"Decomposition:\n{state.decomposition}")
        if state.execution_plan:
            parts.append(f"Execution Plan:\n{state.execution_plan}")
        if state.reasoning_trace:
            parts.append(f"Reasoning:\n{state.reasoning_trace}")
        return "\n\n".join(parts)

    def _store_phase_output(
        self, state: ScaffoldState, phase_id: PhaseID, result: PhaseResult
    ) -> None:
        """Store a phase's output in the appropriate state field."""
        output = result.output
        if phase_id == PhaseID.DECOMPOSE:
            state.decomposition = output
        elif phase_id == PhaseID.PLAN:
            state.execution_plan = output
        elif phase_id == PhaseID.THINK:
            state.reasoning_trace = output
        elif phase_id == PhaseID.GENERATE:
            state.draft = output
        elif phase_id == PhaseID.REFINE:
            state.refined_draft = output
        elif phase_id == PhaseID.QUALITY:
            state.quality_report = output
        elif phase_id == PhaseID.VERIFY:
            state.verification_report = output
        elif phase_id == PhaseID.EMIT:
            state.final_output = output

    def _extract_confidence(self, verification_text: str) -> float:
        """Extract a confidence score from verification output."""
        text = verification_text.lower()
        # Look for explicit confidence scores
        for marker in ["confidence:", "confidence score:", "overall confidence:", "score:"]:
            idx = text.find(marker)
            if idx >= 0:
                after = text[idx + len(marker):idx + len(marker) + 20].strip()
                # Try to parse a float
                for token in after.split():
                    token = token.strip("()[]%,")
                    try:
                        val = float(token)
                        if val > 1.0:
                            val = val / 100.0
                        return min(max(val, 0.0), 1.0)
                    except ValueError:
                        continue

        # Heuristic: count PASS vs FAIL
        passes = text.count("pass")
        fails = text.count("fail")
        total = passes + fails
        if total > 0:
            return round(passes / total, 2)

        return 0.5  # default uncertainty

    def _build_full_reasoning_output(self, state: ScaffoldState) -> str:
        """Build output that includes all reasoning phases."""
        parts = []

        if state.decomposition:
            parts.append(f"## Decomposition\n\n{state.decomposition}")
        if state.execution_plan:
            parts.append(f"## Execution Plan\n\n{state.execution_plan}")
        if state.reasoning_trace:
            parts.append(f"## Reasoning Trace\n\n{state.reasoning_trace}")

        # Main output
        draft = state.refined_draft or state.draft
        if draft:
            parts.append(f"## Output\n\n{draft}")

        if state.quality_report:
            parts.append(f"## Quality Report\n\n{state.quality_report}")
        if state.verification_report:
            parts.append(f"## Verification\n\n{state.verification_report}")
        if state.final_output and state.final_output != draft:
            parts.append(f"## Final Output\n\n{state.final_output}")

        parts.append(f"\n---\n*Confidence: {state.confidence:.0%} | "
                     f"Phases: {len(state.phase_results)} | "
                     f"Tokens: {state.total_tokens:,}*")

        return "\n\n".join(parts)
