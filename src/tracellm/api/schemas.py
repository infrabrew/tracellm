"""Pydantic request/response schemas — OpenAI-compatible + TraceLLM extensions."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field


# ── Chat completions (OpenAI-compatible) ─────────────────────────────────────

class ChatMessage(BaseModel):
    role: str                    # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    stream: bool = False
    stop: list[str] | None = None
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionStreamDelta(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionStreamDelta
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionStreamChoice]


# ── Text completions ─────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    stream: bool = False
    stop: list[str] | None = None
    repetition_penalty: float = 1.1
    seed: int | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


# ── Model management ─────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    format: str
    architecture: str
    size_gb: float
    parameters: str
    quantization: str
    source: str
    tags: list[str]


class PullRequest(BaseModel):
    repo_id: str
    name: str | None = None
    revision: str = "main"
    quantization: str = ""


class TrainRequest(BaseModel):
    model: str
    dataset: str                              # HuggingFace dataset ID or local path
    method: str = "lora"                      # lora | qlora | full
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    optimizer: str = "adamw"                  # adamw | muon | deepspeed
    scheduler: str = "cosine"
    output_name: str | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    dataset_text_field: str = "text"
    dataset_split: str = "train"
    code_paths: list[str] | None = None       # local code folders for training


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str                               # "running" | "completed" | "failed"
    epoch: float
    loss: float | None
    learning_rate: float | None
    elapsed_seconds: float
    output_path: str | None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    models_loaded: list[str]
    gpu_count: int
    device: str


# ── Recursive Language Model ─────────────────────────────────────────────────

class RecursiveRequest(BaseModel):
    model: str
    prompt: str
    context: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    max_iterations: int = 10
    convergence_threshold: float = 0.95
    skip_adversarial: bool = False
    stream: bool = False


class RefinementPassInfo(BaseModel):
    iteration: int
    category: str
    delta_ratio: float
    tokens_generated: int
    elapsed_s: float


class RecursiveResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"rlm-{uuid.uuid4().hex[:8]}")
    model: str
    final_text: str
    iterations_run: int
    converged: bool
    convergence_delta: float
    passes: list[RefinementPassInfo]
    total_tokens: int
    total_time_s: float


# ── Scaffolded Reasoning ─────────────────────────────────────────────────────

class ScaffoldRequest(BaseModel):
    model: str
    prompt: str
    context: str = ""
    constraints: list[str] = []
    profile: str = "standard"             # full | standard | quick | code | creative | research
    phases: list[str] | None = None       # override phases list (e.g. ["decompose","generate","emit"])
    temperature: float = 0.7
    max_tokens: int = 4096
    refine_iterations: int = 5
    refine_convergence: float = 0.90
    show_reasoning: bool = False
    skip_adversarial: bool = False
    stream: bool = False


class PhaseInfo(BaseModel):
    phase_id: str
    phase_name: str
    tokens_generated: int
    elapsed_s: float
    metadata: dict = {}


class ScaffoldResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"scaffold-{uuid.uuid4().hex[:8]}")
    model: str
    final_output: str
    confidence: float
    phases_run: list[PhaseInfo]
    total_tokens: int
    total_time_s: float
    decomposition: str = ""
    execution_plan: str = ""
    reasoning_trace: str = ""


# ── Batch Inference ─────────────────────────────────────────────────────────

class BatchItemRequest(BaseModel):
    id: str = Field(default_factory=lambda: f"item-{uuid.uuid4().hex[:6]}")
    prompt: str
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None


class BatchRequest(BaseModel):
    model: str
    items: list[BatchItemRequest]
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 4096
    max_concurrency: int = 4
    stream: bool = False


class BatchItemResponse(BaseModel):
    id: str
    text: str = ""
    tokens_generated: int = 0
    prompt_tokens: int = 0
    finish_reason: str = ""
    tokens_per_second: float = 0.0
    error: str | None = None


class BatchResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"batch-{uuid.uuid4().hex[:8]}")
    model: str
    results: list[BatchItemResponse]
    total_time_s: float
    total_tokens: int
    total_prompt_tokens: int
    avg_tokens_per_second: float
    items_succeeded: int
    items_failed: int


# ── Model Benchmarking ──────────────────────────────────────────────────────

class BenchmarkRequest(BaseModel):
    model: str
    max_tokens: int = 128
    num_runs: int = 3
    warmup_runs: int = 1
    prompt_categories: list[str] | None = None     # short | medium | long | code
    custom_prompt: str | None = None


class BenchmarkPromptResult(BaseModel):
    category: str
    prompt_tokens: int
    avg_completion_tokens: float
    avg_time_to_first_token_s: float
    avg_total_time_s: float
    avg_tokens_per_second: float
    p50_tokens_per_second: float
    p90_tokens_per_second: float
    p99_tokens_per_second: float
    min_tokens_per_second: float
    max_tokens_per_second: float


class MemoryInfo(BaseModel):
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_peak_gb: float = 0.0
    ram_used_gb: float = 0.0


class BenchmarkResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"bench-{uuid.uuid4().hex[:8]}")
    model: str
    device: str
    dtype: str
    prompt_results: dict[str, BenchmarkPromptResult]
    memory_baseline: MemoryInfo
    memory_loaded: MemoryInfo
    memory_peak: MemoryInfo
    total_time_s: float
    total_tokens_generated: int
    overall_tokens_per_second: float
    hardware_summary: dict


# ── RLHF Training ──────────────────────────────────────────────────────────

class DPORequest(BaseModel):
    model: str
    dataset: str                         # HuggingFace dataset or local path
    ref_model: str | None = None         # reference model for KL (default: copy of model)
    beta: float = 0.1                    # DPO temperature
    epochs: int = 1
    batch_size: int = 2
    learning_rate: float = 5e-7
    max_seq_length: int = 1024
    max_prompt_length: int = 512
    output_name: str | None = None
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    loss_type: str = "sigmoid"           # sigmoid | hinge | ipo
    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"


class RewardModelRequest(BaseModel):
    model: str
    dataset: str
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_seq_length: int = 1024
    output_name: str | None = None
    num_labels: int = 1


class PPORequest(BaseModel):
    model: str
    reward_model: str                    # name of trained reward model
    dataset: str                         # prompts dataset
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-6
    max_seq_length: int = 1024
    output_name: str | None = None
    kl_penalty: float = 0.2
    use_lora: bool = True
    lora_r: int = 16


class RLHFStatusResponse(BaseModel):
    job_id: str
    method: str                          # dpo | reward | ppo
    model: str
    status: str
    epoch: float
    loss: float | None
    reward_accuracy: float | None
    learning_rate: float | None
    elapsed_seconds: float
    output_path: str | None
    error: str | None = None
