"""API routes — OpenAI-compatible chat/completion + TraceLLM management endpoints."""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from tracellm.api.schemas import (
    BatchItemRequest,
    BatchItemResponse,
    BatchRequest,
    BatchResponse,
    BenchmarkPromptResult,
    BenchmarkRequest,
    BenchmarkResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DPORequest,
    HealthResponse,
    MemoryInfo,
    ModelInfo,
    PPORequest,
    PhaseInfo,
    PullRequest,
    RecursiveRequest,
    RecursiveResponse,
    RefinementPassInfo,
    RewardModelRequest,
    RLHFStatusResponse,
    ScaffoldRequest,
    ScaffoldResponse,
    TrainRequest,
    TrainStatusResponse,
    UsageInfo,
)
from tracellm.inference.engine import InferenceEngine, GenerationResult
from tracellm.inference.sampling import SamplingParams
from tracellm.utils.hardware import detect_hardware
from tracellm import __version__

router = APIRouter()

# These get set by server.py at startup
_engine: InferenceEngine | None = None
_trainer = None  # TrainingManager, set at startup
_rlhf_manager = None  # RLHFManager, set at startup


def set_engine(engine: InferenceEngine) -> None:
    global _engine
    _engine = engine


def set_trainer(trainer) -> None:
    global _trainer
    _trainer = trainer


def set_rlhf_manager(manager) -> None:
    global _rlhf_manager
    _rlhf_manager = manager


def _get_engine() -> InferenceEngine:
    if _engine is None:
        raise HTTPException(503, "Engine not initialized")
    return _engine


def _build_sampling_params(
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 4096,
    stop: list[str] | None = None,
    repetition_penalty: float = 1.1,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    seed: int | None = None,
) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        stop_sequences=stop,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
    )


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string."""
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"<|system|>\n{msg.content}\n")
        elif msg.role == "user":
            parts.append(f"<|user|>\n{msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"<|assistant|>\n{msg.content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
@router.get("/v1/health", response_model=HealthResponse)
async def health():
    engine = _get_engine()
    hw = detect_hardware()
    return HealthResponse(
        version=__version__,
        models_loaded=engine.loader.loaded_models,
        gpu_count=hw.gpu_count,
        device=hw.best_device,
    )


# ── Models ───────────────────────────────────────────────────────────────────

@router.get("/v1/models")
async def list_models():
    engine = _get_engine()
    models = engine.registry.list_models()
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "owned_by": "tracellm",
                "format": m.format,
                "architecture": m.architecture,
                "size_gb": m.size_gb,
                "parameters": m.parameters,
            }
            for m in models
        ],
    }


@router.get("/v1/models/{model_name}")
async def get_model(model_name: str):
    engine = _get_engine()
    card = engine.registry.get(model_name)
    if not card:
        raise HTTPException(404, f"Model '{model_name}' not found")
    return ModelInfo(
        name=card.name,
        format=card.format,
        architecture=card.architecture,
        size_gb=card.size_gb,
        parameters=card.parameters,
        quantization=card.quantization,
        source=card.source,
        tags=card.tags,
    )


@router.post("/v1/models/pull")
async def pull_model(req: PullRequest):
    engine = _get_engine()
    card = engine.registry.pull(
        repo_id=req.repo_id,
        name=req.name,
        revision=req.revision,
        quantization=req.quantization,
    )
    return {"status": "ok", "model": card.to_dict()}


@router.delete("/v1/models/{model_name}")
async def delete_model(model_name: str, delete_files: bool = False):
    engine = _get_engine()
    engine.loader.unload(model_name)
    removed = engine.registry.remove(model_name, delete_files=delete_files)
    if not removed:
        raise HTTPException(404, f"Model '{model_name}' not found")
    return {"status": "ok", "deleted": model_name}


@router.post("/v1/models/{model_name}/load")
async def load_model(model_name: str):
    engine = _get_engine()
    engine.load_model(model_name)
    return {"status": "ok", "loaded": model_name}


@router.post("/v1/models/{model_name}/unload")
async def unload_model(model_name: str):
    engine = _get_engine()
    engine.loader.unload(model_name)
    return {"status": "ok", "unloaded": model_name}


# ── Chat completions (OpenAI-compatible) ─────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    engine = _get_engine()
    prompt = _messages_to_prompt(req.messages)
    params = _build_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
        stop=req.stop,
        repetition_penalty=req.repetition_penalty,
        presence_penalty=req.presence_penalty,
        frequency_penalty=req.frequency_penalty,
        seed=req.seed,
    )

    if req.stream:
        return StreamingResponse(
            _stream_chat(engine, req.model, prompt, params),
            media_type="text/event-stream",
        )

    result = engine.generate(req.model, prompt, params)
    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.tokens_generated,
            total_tokens=result.prompt_tokens + result.tokens_generated,
        ),
    )


async def _stream_chat(
    engine: InferenceEngine,
    model: str,
    prompt: str,
    params: SamplingParams,
) -> AsyncIterator[str]:
    """SSE streaming for chat completions."""
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Initial role chunk
    initial = ChatCompletionStreamResponse(
        id=stream_id,
        model=model,
        choices=[ChatCompletionStreamChoice(delta=ChatCompletionStreamDelta(role="assistant"))],
    )
    yield f"data: {initial.model_dump_json()}\n\n"

    # Content chunks
    for token in engine.stream(model, prompt, params):
        chunk = ChatCompletionStreamResponse(
            id=stream_id,
            model=model,
            choices=[ChatCompletionStreamChoice(delta=ChatCompletionStreamDelta(content=token))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk
    final = ChatCompletionStreamResponse(
        id=stream_id,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                delta=ChatCompletionStreamDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ── Text completions ─────────────────────────────────────────────────────────

@router.post("/v1/completions")
async def completions(req: CompletionRequest):
    engine = _get_engine()
    params = _build_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
        stop=req.stop,
        repetition_penalty=req.repetition_penalty,
        seed=req.seed,
    )

    if req.stream:
        return StreamingResponse(
            _stream_completion(engine, req.model, req.prompt, params),
            media_type="text/event-stream",
        )

    result = engine.generate(req.model, req.prompt, params)
    return CompletionResponse(
        model=req.model,
        choices=[CompletionChoice(text=result.text, finish_reason=result.finish_reason)],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.tokens_generated,
            total_tokens=result.prompt_tokens + result.tokens_generated,
        ),
    )


async def _stream_completion(
    engine: InferenceEngine,
    model: str,
    prompt: str,
    params: SamplingParams,
) -> AsyncIterator[str]:
    for token in engine.stream(model, prompt, params):
        chunk = {"choices": [{"text": token, "index": 0, "finish_reason": None}]}
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ── Training ─────────────────────────────────────────────────────────────────

@router.post("/v1/training/start")
async def start_training(req: TrainRequest):
    if _trainer is None:
        raise HTTPException(503, "Training manager not initialized")
    job_id = _trainer.start_training(req)
    return {"status": "started", "job_id": job_id}


@router.get("/v1/training/{job_id}")
async def training_status(job_id: str):
    if _trainer is None:
        raise HTTPException(503, "Training manager not initialized")
    status = _trainer.get_status(job_id)
    if not status:
        raise HTTPException(404, f"Training job '{job_id}' not found")
    return status


@router.post("/v1/training/{job_id}/stop")
async def stop_training(job_id: str):
    if _trainer is None:
        raise HTTPException(503, "Training manager not initialized")
    _trainer.stop(job_id)
    return {"status": "stopping", "job_id": job_id}


# ── Recursive Language Model ─────────────────────────────────────────────────

@router.post("/v1/recursive", response_model=RecursiveResponse)
async def recursive_generate(req: RecursiveRequest):
    """Run recursive self-refinement — model iteratively improves its own output."""
    engine = _get_engine()
    from tracellm.inference.recursive import RecursiveEngine

    params = _build_sampling_params(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    recursive = RecursiveEngine(
        engine=engine,
        max_iterations=req.max_iterations,
        convergence_threshold=req.convergence_threshold,
        skip_adversarial=req.skip_adversarial,
    )

    if req.stream:
        return StreamingResponse(
            _stream_recursive(recursive, req.model, req.prompt, req.context, params),
            media_type="text/event-stream",
        )

    result = recursive.refine(req.model, req.prompt, context=req.context, params=params)

    return RecursiveResponse(
        model=req.model,
        final_text=result.final_text,
        iterations_run=result.iterations_run,
        converged=result.converged,
        convergence_delta=result.convergence_delta,
        passes=[
            RefinementPassInfo(
                iteration=p.iteration,
                category=p.category.value,
                delta_ratio=p.delta_ratio,
                tokens_generated=p.tokens_generated,
                elapsed_s=p.elapsed_s,
            )
            for p in result.passes
        ],
        total_tokens=result.total_tokens,
        total_time_s=result.total_time_s,
    )


async def _stream_recursive(recursive, model, prompt, context, params) -> AsyncIterator[str]:
    """SSE streaming for recursive refinement."""
    for event in recursive.stream_refine(model, prompt, context=context, params=params):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


# ── Scaffolded Reasoning ─────────────────────────────────────────────────────

SCAFFOLD_PROFILES = {
    "full": ["decompose", "plan", "think", "generate", "refine", "quality", "verify", "emit"],
    "standard": ["decompose", "plan", "generate", "refine", "verify", "emit"],
    "quick": ["decompose", "generate", "verify", "emit"],
    "code": ["decompose", "plan", "think", "generate", "refine", "quality", "verify", "emit"],
    "creative": ["decompose", "generate", "refine", "emit"],
    "research": ["decompose", "plan", "think", "generate", "refine", "quality", "verify", "emit"],
}


@router.post("/v1/scaffold", response_model=ScaffoldResponse)
async def scaffold_generate(req: ScaffoldRequest):
    """Run scaffolded multi-phase reasoning pipeline."""
    engine = _get_engine()
    from tracellm.inference.scaffold import ScaffoldEngine, PhaseID

    # Resolve phases from profile or explicit list
    if req.phases:
        phase_ids = [PhaseID(p) for p in req.phases]
    else:
        profile_phases = SCAFFOLD_PROFILES.get(req.profile, SCAFFOLD_PROFILES["standard"])
        phase_ids = [PhaseID(p) for p in profile_phases]

    params = _build_sampling_params(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    scaffold = ScaffoldEngine(
        engine=engine,
        phases=phase_ids,
        refine_iterations=req.refine_iterations,
        refine_convergence=req.refine_convergence,
        skip_adversarial=req.skip_adversarial,
    )

    if req.stream:
        return StreamingResponse(
            _stream_scaffold(scaffold, req, params),
            media_type="text/event-stream",
        )

    state = scaffold.run(
        model_name=req.model,
        prompt=req.prompt,
        context=req.context,
        constraints=req.constraints,
        params=params,
        show_reasoning=req.show_reasoning,
    )

    return ScaffoldResponse(
        model=req.model,
        final_output=state.final_output,
        confidence=state.confidence,
        phases_run=[
            PhaseInfo(
                phase_id=pr.phase_id,
                phase_name=pr.phase_name,
                tokens_generated=pr.tokens_generated,
                elapsed_s=pr.elapsed_s,
                metadata=pr.metadata,
            )
            for pr in state.phase_results
        ],
        total_tokens=state.total_tokens,
        total_time_s=state.total_time_s,
        decomposition=state.decomposition if req.show_reasoning else "",
        execution_plan=state.execution_plan if req.show_reasoning else "",
        reasoning_trace=state.reasoning_trace if req.show_reasoning else "",
    )


async def _stream_scaffold(scaffold, req, params) -> AsyncIterator[str]:
    """SSE streaming for scaffolded reasoning."""
    for event in scaffold.stream_run(
        model_name=req.model,
        prompt=req.prompt,
        context=req.context,
        constraints=req.constraints,
        params=params,
    ):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


@router.get("/v1/scaffold/profiles")
async def list_scaffold_profiles():
    """List available scaffold profiles."""
    return {
        "profiles": {
            name: {"phases": phases, "description": desc}
            for name, phases, desc in [
                ("full", SCAFFOLD_PROFILES["full"], "All 8 phases — maximum rigor"),
                ("standard", SCAFFOLD_PROFILES["standard"], "Core reasoning without deep refinement"),
                ("quick", SCAFFOLD_PROFILES["quick"], "Fast — decompose, generate, verify"),
                ("code", SCAFFOLD_PROFILES["code"], "Optimized for code generation with quality gate"),
                ("creative", SCAFFOLD_PROFILES["creative"], "Creative work — lighter verification"),
                ("research", SCAFFOLD_PROFILES["research"], "Heavy on reasoning and factual verification"),
            ]
        }
    }


# ── Hub search ───────────────────────────────────────────────────────────────

@router.get("/v1/hub/search")
async def search_hub(q: str, limit: int = 10):
    engine = _get_engine()
    results = engine.registry.search_hub(q, limit=limit)
    return {"results": results}


# ── Batch Inference ─────────────────────────────────────────────────────────

@router.post("/v1/batch", response_model=BatchResponse)
async def batch_generate(req: BatchRequest):
    """Run batch inference on multiple prompts concurrently."""
    engine = _get_engine()
    from tracellm.inference.batch import BatchEngine, BatchItem

    default_params = _build_sampling_params(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )

    items = []
    for item in req.items:
        per_item_params = None
        if any(v is not None for v in [item.temperature, item.max_tokens, item.top_p, item.top_k]):
            per_item_params = _build_sampling_params(
                temperature=item.temperature or req.temperature,
                top_p=item.top_p or req.top_p,
                top_k=item.top_k or req.top_k,
                max_tokens=item.max_tokens or req.max_tokens,
            )
        items.append(BatchItem(id=item.id, prompt=item.prompt, params=per_item_params))

    batch_engine = BatchEngine(engine=engine, max_concurrency=req.max_concurrency)

    if req.stream:
        return StreamingResponse(
            _stream_batch(batch_engine, req.model, items, default_params),
            media_type="text/event-stream",
        )

    result = batch_engine.run(req.model, items, default_params)

    return BatchResponse(
        id=result.batch_id,
        model=req.model,
        results=[
            BatchItemResponse(
                id=r.id,
                text=r.result.text if r.result else "",
                tokens_generated=r.result.tokens_generated if r.result else 0,
                prompt_tokens=r.result.prompt_tokens if r.result else 0,
                finish_reason=r.result.finish_reason if r.result else "",
                tokens_per_second=r.result.tokens_per_second if r.result else 0.0,
                error=r.error,
            )
            for r in result.results
        ],
        total_time_s=result.total_time_s,
        total_tokens=result.total_tokens,
        total_prompt_tokens=result.total_prompt_tokens,
        avg_tokens_per_second=result.avg_tokens_per_second,
        items_succeeded=result.items_succeeded,
        items_failed=result.items_failed,
    )


async def _stream_batch(batch_engine, model, items, params) -> AsyncIterator[str]:
    """SSE streaming for batch progress."""
    for event in batch_engine.stream_progress(model, items, params):
        yield f"data: {json.dumps(event)}\n\n"
    yield "data: [DONE]\n\n"


# ── Model Benchmarking ──────────────────────────────────────────────────────

@router.post("/v1/benchmark", response_model=BenchmarkResponse)
async def benchmark_model(req: BenchmarkRequest):
    """Run a benchmark suite against a model — measures latency, throughput, memory."""
    engine = _get_engine()
    from tracellm.inference.benchmark import BenchmarkEngine

    bench = BenchmarkEngine(engine=engine)
    result = bench.run(
        model_name=req.model,
        max_tokens=req.max_tokens,
        num_runs=req.num_runs,
        warmup_runs=req.warmup_runs,
        prompt_categories=req.prompt_categories,
        custom_prompt=req.custom_prompt,
    )

    return BenchmarkResponse(
        id=result.benchmark_id,
        model=result.model_name,
        device=result.device,
        dtype=result.dtype,
        prompt_results={
            name: BenchmarkPromptResult(
                category=pr.category,
                prompt_tokens=pr.prompt_tokens,
                avg_completion_tokens=pr.avg_completion_tokens,
                avg_time_to_first_token_s=pr.avg_time_to_first_token_s,
                avg_total_time_s=pr.avg_total_time_s,
                avg_tokens_per_second=pr.avg_tokens_per_second,
                p50_tokens_per_second=pr.p50_tokens_per_second,
                p90_tokens_per_second=pr.p90_tokens_per_second,
                p99_tokens_per_second=pr.p99_tokens_per_second,
                min_tokens_per_second=pr.min_tokens_per_second,
                max_tokens_per_second=pr.max_tokens_per_second,
            )
            for name, pr in result.prompt_results.items()
        },
        memory_baseline=MemoryInfo(
            gpu_allocated_gb=result.memory_baseline.gpu_allocated_gb,
            gpu_reserved_gb=result.memory_baseline.gpu_reserved_gb,
            gpu_peak_gb=result.memory_baseline.gpu_peak_gb,
            ram_used_gb=result.memory_baseline.ram_used_gb,
        ),
        memory_loaded=MemoryInfo(
            gpu_allocated_gb=result.memory_loaded.gpu_allocated_gb,
            gpu_reserved_gb=result.memory_loaded.gpu_reserved_gb,
            gpu_peak_gb=result.memory_loaded.gpu_peak_gb,
            ram_used_gb=result.memory_loaded.ram_used_gb,
        ),
        memory_peak=MemoryInfo(
            gpu_allocated_gb=result.memory_peak.gpu_allocated_gb,
            gpu_reserved_gb=result.memory_peak.gpu_reserved_gb,
            gpu_peak_gb=result.memory_peak.gpu_peak_gb,
            ram_used_gb=result.memory_peak.ram_used_gb,
        ),
        total_time_s=result.total_time_s,
        total_tokens_generated=result.total_tokens_generated,
        overall_tokens_per_second=result.overall_tokens_per_second,
        hardware_summary=result.hardware_summary,
    )


# ── RLHF Training ──────────────────────────────────────────────────────────

def _get_rlhf():
    if _rlhf_manager is None:
        raise HTTPException(503, "RLHF manager not initialized")
    return _rlhf_manager


@router.post("/v1/rlhf/dpo")
async def start_dpo(req: DPORequest):
    """Start DPO (Direct Preference Optimization) alignment training."""
    manager = _get_rlhf()
    job_id = manager.start_dpo(
        model=req.model,
        dataset=req.dataset,
        ref_model=req.ref_model,
        beta=req.beta,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        max_seq_length=req.max_seq_length,
        max_prompt_length=req.max_prompt_length,
        output_name=req.output_name,
        lora_r=req.lora_r,
        lora_alpha=req.lora_alpha,
        use_lora=req.use_lora,
        prompt_field=req.prompt_field,
        chosen_field=req.chosen_field,
        rejected_field=req.rejected_field,
        loss_type=req.loss_type,
    )
    return {"status": "started", "job_id": job_id, "method": "dpo"}


@router.post("/v1/rlhf/reward")
async def start_reward_model(req: RewardModelRequest):
    """Start reward model training."""
    manager = _get_rlhf()
    job_id = manager.start_reward_model(
        model=req.model,
        dataset=req.dataset,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        max_seq_length=req.max_seq_length,
        output_name=req.output_name,
        num_labels=req.num_labels,
    )
    return {"status": "started", "job_id": job_id, "method": "reward"}


@router.post("/v1/rlhf/ppo")
async def start_ppo(req: PPORequest):
    """Start PPO alignment training with a reward model."""
    manager = _get_rlhf()
    job_id = manager.start_ppo(
        model=req.model,
        reward_model=req.reward_model,
        dataset=req.dataset,
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        max_seq_length=req.max_seq_length,
        output_name=req.output_name,
        kl_penalty=req.kl_penalty,
        use_lora=req.use_lora,
        lora_r=req.lora_r,
    )
    return {"status": "started", "job_id": job_id, "method": "ppo"}


@router.get("/v1/rlhf/{job_id}")
async def rlhf_status(job_id: str):
    """Get status of an RLHF training job."""
    manager = _get_rlhf()
    status = manager.get_status(job_id)
    if not status:
        raise HTTPException(404, f"RLHF job '{job_id}' not found")
    return status


@router.post("/v1/rlhf/{job_id}/stop")
async def stop_rlhf(job_id: str):
    """Stop an RLHF training job."""
    manager = _get_rlhf()
    manager.stop(job_id)
    return {"status": "stopping", "job_id": job_id}


@router.get("/v1/rlhf")
async def list_rlhf_jobs():
    """List all RLHF training jobs."""
    manager = _get_rlhf()
    return {"jobs": manager.list_jobs()}
