"""Recursive Language Model (RLM) — iterative self-refinement engine.

The RLM engine wraps the base inference engine and feeds outputs back as
inputs for multiple refinement passes. Each pass targets a different quality
dimension (structural, factual, stylistic, adversarial). Convergence is
detected by measuring the semantic delta between consecutive passes — when
the output stabilizes, iteration stops early.

This is the "thinking" layer — it makes any model reason more deeply by
giving it structured opportunities to criticize and improve its own output.
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator

from tracellm.inference.engine import InferenceEngine, GenerationResult
from tracellm.inference.sampling import SamplingParams
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.inference.recursive")


# ═══════════════════════════════════════════════════════════════════════════════
#  REFINEMENT PASS CATEGORIES
# ═══════════════════════════════════════════════════════════════════════════════

class PassCategory(str, Enum):
    STRUCTURAL = "structural"
    FACTUAL = "factual"
    STYLISTIC = "stylistic"
    ADVERSARIAL = "adversarial"
    FINAL_LOCK = "final_lock"


# Maps iteration ranges to pass categories (1-indexed)
DEFAULT_PASS_SCHEDULE = {
    (1, 3): PassCategory.STRUCTURAL,
    (4, 6): PassCategory.FACTUAL,
    (7, 8): PassCategory.STYLISTIC,
    (9, 9): PassCategory.ADVERSARIAL,
    (10, 10): PassCategory.FINAL_LOCK,
}

# System prompts that direct the model's focus for each pass category
REFINEMENT_PROMPTS = {
    PassCategory.STRUCTURAL: (
        "You are a structural editor. Review the following draft for:\n"
        "- Logical gaps or missing steps\n"
        "- Incorrect ordering of ideas\n"
        "- Incoherent transitions between sections\n"
        "- Missing conclusions or incomplete arguments\n"
        "Rewrite the draft with structural improvements. Preserve all good content.\n"
    ),
    PassCategory.FACTUAL: (
        "You are a fact-checker. Review the following draft for:\n"
        "- Claims that are not supported by the original request or context\n"
        "- Incorrect technical details or calculations\n"
        "- Hallucinated facts, names, dates, or statistics\n"
        "- Unsupported generalizations\n"
        "Rewrite the draft correcting any factual issues. Mark anything uncertain with [UNCERTAIN].\n"
    ),
    PassCategory.STYLISTIC: (
        "You are a copy editor. Review the following draft for:\n"
        "- Tone and voice consistency\n"
        "- Grammar, spelling, and punctuation\n"
        "- Formatting consistency (headers, lists, code blocks)\n"
        "- Conciseness — cut anything that doesn't earn its place\n"
        "Rewrite the draft with stylistic polish. Do not change the substance.\n"
    ),
    PassCategory.ADVERSARIAL: (
        "You are a devil's advocate. Review the following draft by:\n"
        "- Steelmanning the strongest counterargument to each major claim\n"
        "- Identifying the weakest points that an expert would challenge\n"
        "- Finding edge cases or scenarios where the advice would fail\n"
        "- Checking for bias or one-sided reasoning\n"
        "Rewrite the draft addressing these weaknesses. Strengthen, don't weaken.\n"
    ),
    PassCategory.FINAL_LOCK: (
        "You are performing a final quality review. Check:\n"
        "- Does this fully answer the original request?\n"
        "- Is every claim grounded in the provided context?\n"
        "- Is the formatting clean and professional?\n"
        "- Would an expert in this domain find this credible?\n"
        "If the draft is already strong, output it unchanged. Only fix genuine issues.\n"
    ),
}


@dataclass
class RefinementPass:
    """Record of a single refinement iteration."""
    iteration: int
    category: PassCategory
    input_text: str
    output_text: str
    delta_ratio: float             # 0.0 = identical, 1.0 = completely different
    tokens_generated: int
    elapsed_s: float


@dataclass
class RecursiveResult:
    """Result of recursive refinement."""
    final_text: str
    iterations_run: int
    converged: bool
    convergence_delta: float
    passes: list[RefinementPass] = field(default_factory=list)
    total_tokens: int = 0
    total_time_s: float = 0.0
    original_prompt: str = ""


def compute_delta(text_a: str, text_b: str) -> float:
    """Compute the normalized edit distance between two texts.

    Returns 0.0 if identical, approaches 1.0 for completely different texts.
    Uses SequenceMatcher for a reasonable approximation of semantic delta.
    """
    if text_a == text_b:
        return 0.0
    if not text_a or not text_b:
        return 1.0

    ratio = difflib.SequenceMatcher(None, text_a, text_b).ratio()
    return round(1.0 - ratio, 4)


def get_pass_category(iteration: int, schedule: dict | None = None) -> PassCategory:
    """Determine the refinement focus for a given iteration number."""
    schedule = schedule or DEFAULT_PASS_SCHEDULE
    for (lo, hi), category in schedule.items():
        if lo <= iteration <= hi:
            return category
    return PassCategory.FINAL_LOCK


class RecursiveEngine:
    """Recursive Language Model engine — iterative self-refinement.

    Wraps InferenceEngine and adds a refinement loop:
    1. Generate initial draft from the user prompt
    2. Feed draft + refinement instructions back to the model
    3. Compare output to previous pass — measure delta
    4. Repeat with different focus areas (structural → factual → style → adversarial)
    5. Stop when converged (delta below threshold) or max iterations reached

    Usage:
        engine = RecursiveEngine(inference_engine, max_iterations=10)
        result = engine.refine("model-name", "Write a REST API design doc")
        print(result.final_text)
        print(f"Converged in {result.iterations_run} passes")
    """

    def __init__(
        self,
        engine: InferenceEngine,
        max_iterations: int = 10,
        convergence_threshold: float = 0.95,
        skip_adversarial: bool = False,
        pass_schedule: dict | None = None,
        refinement_prompts: dict | None = None,
    ):
        self.engine = engine
        self.max_iterations = max_iterations
        # Convergence: stop when similarity >= threshold (i.e. delta <= 1 - threshold)
        self.convergence_threshold = convergence_threshold
        self.delta_threshold = 1.0 - convergence_threshold
        self.skip_adversarial = skip_adversarial
        self.pass_schedule = pass_schedule or DEFAULT_PASS_SCHEDULE
        self.prompts = refinement_prompts or REFINEMENT_PROMPTS

    def refine(
        self,
        model_name: str,
        prompt: str,
        context: str = "",
        params: SamplingParams | None = None,
        on_pass: Callable[[RefinementPass], None] | None = None,
    ) -> RecursiveResult:
        """Run the full recursive refinement loop.

        Args:
            model_name: Registered model to use.
            prompt: The original user request.
            context: Optional context to include in every pass.
            params: Sampling parameters for generation.
            on_pass: Callback invoked after each refinement pass (for progress tracking).

        Returns:
            RecursiveResult with the final refined text and iteration metadata.
        """
        params = params or SamplingParams()
        start = time.time()
        passes: list[RefinementPass] = []
        total_tokens = 0

        # ── Phase 0: Initial generation ──────────────────────────────────
        initial_prompt = self._build_initial_prompt(prompt, context)
        log.info(f"RLM: generating initial draft for '{prompt[:60]}...'")

        initial_result = self.engine.generate(model_name, initial_prompt, params)
        current_text = initial_result.text
        total_tokens += initial_result.tokens_generated

        log.info(f"RLM: initial draft — {initial_result.tokens_generated} tokens")

        # ── Refinement loop ──────────────────────────────────────────────
        converged = False
        final_delta = 1.0

        for iteration in range(1, self.max_iterations + 1):
            category = get_pass_category(iteration, self.pass_schedule)

            # Skip adversarial if disabled
            if self.skip_adversarial and category == PassCategory.ADVERSARIAL:
                log.debug(f"RLM: skipping adversarial pass (iteration {iteration})")
                continue

            # Build the refinement prompt
            refinement_prompt = self._build_refinement_prompt(
                original_request=prompt,
                context=context,
                current_draft=current_text,
                category=category,
            )

            pass_start = time.time()
            result = self.engine.generate(model_name, refinement_prompt, params)
            pass_elapsed = time.time() - pass_start

            refined_text = result.text
            delta = compute_delta(current_text, refined_text)
            total_tokens += result.tokens_generated

            rp = RefinementPass(
                iteration=iteration,
                category=category,
                input_text=current_text,
                output_text=refined_text,
                delta_ratio=delta,
                tokens_generated=result.tokens_generated,
                elapsed_s=round(pass_elapsed, 3),
            )
            passes.append(rp)

            log.info(
                f"RLM: pass {iteration}/{self.max_iterations} "
                f"[{category.value}] delta={delta:.4f} "
                f"({result.tokens_generated} tok, {pass_elapsed:.1f}s)"
            )

            if on_pass:
                on_pass(rp)

            current_text = refined_text
            final_delta = delta

            # Convergence check
            if delta <= self.delta_threshold:
                converged = True
                log.info(
                    f"RLM: converged at iteration {iteration} "
                    f"(delta {delta:.4f} <= threshold {self.delta_threshold:.4f})"
                )
                break

        total_time = time.time() - start

        return RecursiveResult(
            final_text=current_text,
            iterations_run=len(passes),
            converged=converged,
            convergence_delta=final_delta,
            passes=passes,
            total_tokens=total_tokens,
            total_time_s=round(total_time, 3),
            original_prompt=prompt,
        )

    def stream_refine(
        self,
        model_name: str,
        prompt: str,
        context: str = "",
        params: SamplingParams | None = None,
    ) -> Iterator[dict]:
        """Streaming refinement — yields progress events as dicts.

        Events:
            {"type": "draft", "text": "...", "iteration": 0}
            {"type": "pass_start", "iteration": 1, "category": "structural"}
            {"type": "pass_complete", "iteration": 1, "delta": 0.23, "text": "..."}
            {"type": "converged", "iteration": 3, "delta": 0.02}
            {"type": "final", "text": "...", "iterations": 3, "converged": true}
        """
        params = params or SamplingParams()

        # Initial generation
        initial_prompt = self._build_initial_prompt(prompt, context)
        initial_result = self.engine.generate(model_name, initial_prompt, params)
        current_text = initial_result.text

        yield {
            "type": "draft",
            "text": current_text,
            "iteration": 0,
            "tokens": initial_result.tokens_generated,
        }

        converged = False
        final_delta = 1.0
        iterations_run = 0

        for iteration in range(1, self.max_iterations + 1):
            category = get_pass_category(iteration, self.pass_schedule)

            if self.skip_adversarial and category == PassCategory.ADVERSARIAL:
                continue

            yield {
                "type": "pass_start",
                "iteration": iteration,
                "category": category.value,
            }

            refinement_prompt = self._build_refinement_prompt(
                original_request=prompt,
                context=context,
                current_draft=current_text,
                category=category,
            )

            result = self.engine.generate(model_name, refinement_prompt, params)
            refined_text = result.text
            delta = compute_delta(current_text, refined_text)

            yield {
                "type": "pass_complete",
                "iteration": iteration,
                "category": category.value,
                "delta": delta,
                "text": refined_text,
                "tokens": result.tokens_generated,
            }

            current_text = refined_text
            final_delta = delta
            iterations_run = iteration

            if delta <= self.delta_threshold:
                converged = True
                yield {
                    "type": "converged",
                    "iteration": iteration,
                    "delta": delta,
                }
                break

        yield {
            "type": "final",
            "text": current_text,
            "iterations": iterations_run,
            "converged": converged,
            "delta": final_delta,
        }

    def _build_initial_prompt(self, prompt: str, context: str) -> str:
        """Build the prompt for the initial draft generation."""
        parts = []
        parts.append(
            "<|system|>\n"
            "You are an expert assistant. Provide a thorough, well-structured "
            "response to the user's request. This is a first draft — be "
            "comprehensive rather than concise.\n"
        )
        if context:
            parts.append(f"<|context|>\n{context}\n")
        parts.append(f"<|user|>\n{prompt}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _build_refinement_prompt(
        self,
        original_request: str,
        context: str,
        current_draft: str,
        category: PassCategory,
    ) -> str:
        """Build the prompt for a refinement pass."""
        system_prompt = self.prompts.get(category, REFINEMENT_PROMPTS[PassCategory.FINAL_LOCK])

        parts = [f"<|system|>\n{system_prompt}\n"]

        parts.append(f"<|original_request|>\n{original_request}\n")

        if context:
            parts.append(f"<|context|>\n{context}\n")

        parts.append(f"<|current_draft|>\n{current_draft}\n")

        parts.append(
            "<|user|>\n"
            "Apply the review criteria above to improve this draft. "
            "Output the complete improved version — not just the changes.\n"
        )
        parts.append("<|assistant|>\n")
        return "".join(parts)
