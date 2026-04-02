# TraceLLM

LLM inference, training, and fine-tuning platform — like ollama meets vLLM with built-in training, RLHF alignment, and advanced reasoning pipelines.

TraceLLM provides a unified CLI and API for pulling models from HuggingFace, running inference (single, batch, streaming), benchmarking performance, fine-tuning with LoRA/QLoRA, aligning with DPO/PPO/RLHF, and advanced multi-phase reasoning — all from one tool.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
  - [Server](#server)
  - [Model Management](#model-management)
  - [Text Generation](#text-generation)
  - [Interactive Chat](#interactive-chat)
  - [Batch Inference](#batch-inference)
  - [Benchmarking](#benchmarking)
  - [Recursive Refinement (RLM)](#recursive-refinement-rlm)
  - [Scaffolded Reasoning](#scaffolded-reasoning)
  - [Fine-Tuning](#fine-tuning)
  - [RLHF Alignment](#rlhf-alignment)
  - [Quantization](#quantization)
  - [Hardware Info](#hardware-info)
- [API Reference](#api-reference)
  - [Health & Models](#health--models)
  - [Chat Completions (OpenAI-compatible)](#chat-completions-openai-compatible)
  - [Text Completions](#text-completions)
  - [Batch Inference API](#batch-inference-api)
  - [Benchmark API](#benchmark-api)
  - [Recursive Refinement API](#recursive-refinement-api)
  - [Scaffolded Reasoning API](#scaffolded-reasoning-api)
  - [Training API](#training-api)
  - [RLHF API](#rlhf-api)
  - [Hub Search API](#hub-search-api)
- [Configuration](#configuration)
- [Prompt Templates](#prompt-templates)
- [Docker](#docker)
- [Development](#development)

---

## Features

- **Model Management** — Pull, list, inspect, and remove models from HuggingFace Hub
- **Multi-Format Support** — Transformers, GGUF (llama.cpp), Mamba (SSM), Safetensors
- **OpenAI-Compatible API** — Drop-in replacement for `/v1/chat/completions` and `/v1/completions`
- **Streaming** — Server-Sent Events (SSE) for all generation endpoints
- **Batch Inference** — Process multiple prompts concurrently with configurable parallelism
- **Model Benchmarking** — Measure TTFT, throughput percentiles (p50/p90/p99), and memory usage
- **Recursive Language Model (RLM)** — Iterative self-refinement with convergence detection
- **Scaffolded Reasoning** — 8-phase pipeline (decompose, plan, think, generate, refine, quality, verify, emit)
- **Fine-Tuning** — LoRA, QLoRA, and full fine-tuning with HuggingFace Trainer
- **RLHF Alignment** — DPO, reward model training, and PPO via TRL
- **Custom Optimizers** — Muon (Newton-Schulz), DeepSpeed ZeRO Stage 2/3, FlashInfer attention
- **Quantization** — GPTQ, AWQ, and GGUF export
- **Dataset Loading** — HuggingFace Hub, local files (JSONL/CSV/JSON/TXT), code folder ingestion (40+ languages)
- **KV Cache** — LRU-evicted key-value cache for efficient multi-turn generation
- **Docker** — Multi-stage CPU and GPU (CUDA 12.4) images

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/tracellm/tracellm.git
cd tracellm
pip install -e .
```

### With optional extras

```bash
# GPU quantization support (GPTQ, AWQ, bitsandbytes)
pip install -e ".[quantize]"

# GGUF / llama.cpp support
pip install -e ".[gguf]"

# Mamba (state-space models)
pip install -e ".[mamba]"

# DeepSpeed distributed training
pip install -e ".[deepspeed]"

# FlashInfer attention kernels
pip install -e ".[flashinfer]"

# RLHF alignment (DPO, PPO via TRL)
pip install -e ".[rlhf]"

# Everything
pip install -e ".[all]"

# Development tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.2.0
- CUDA 12.x (optional, for GPU acceleration)

---

## Quick Start

```bash
# Pull a model from HuggingFace
tracellm pull microsoft/phi-2

# Generate text
tracellm run phi-2 "Explain quantum computing in simple terms"

# Start the API server
tracellm serve

# Interactive chat
tracellm chat phi-2

# Fine-tune with LoRA
tracellm train phi-2 tatsu-lab/alpaca --method lora --epochs 3

# Align with DPO
tracellm align phi-2 Anthropic/hh-rlhf --method dpo

# Benchmark performance
tracellm benchmark phi-2
```

---

## CLI Reference

### Server

Start the TraceLLM API server:

```bash
tracellm serve [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port`, `-p` | `8400` | Bind port |
| `--workers`, `-w` | `1` | Worker processes |
| `--config`, `-c` | auto-detect | Path to config YAML |

```bash
# Start on custom port
tracellm serve --port 9000

# With custom config
tracellm serve -c my-config.yml -w 4
```

---

### Model Management

#### Pull a model

```bash
tracellm pull <repo_id> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name`, `-n` | repo name | Local name for the model |
| `--revision` | `main` | Branch, tag, or commit hash |
| `--quantization`, `-q` | none | Quantization label |

```bash
tracellm pull meta-llama/Meta-Llama-3-8B
tracellm pull TheBloke/Mistral-7B-GGUF -n mistral-gguf
tracellm pull microsoft/phi-2 --revision main
```

#### List local models

```bash
tracellm list
```

Displays a table with name, format, architecture, size, quantization, and source.

#### Model details

```bash
tracellm info <model>
```

```bash
tracellm info phi-2
```

#### Remove a model

```bash
tracellm remove <model> [--delete-files]
```

```bash
tracellm remove phi-2
tracellm remove phi-2 --delete-files  # also deletes from disk
```

#### Search HuggingFace Hub

```bash
tracellm search <query> [--limit 10]
```

```bash
tracellm search "llama 3 instruct" --limit 5
```

---

### Text Generation

```bash
tracellm run <model> [prompt] [OPTIONS]
```

If no prompt is given, enters interactive mode (type `/bye` to exit).

| Option | Default | Description |
|--------|---------|-------------|
| `--temperature`, `-t` | `0.7` | Sampling temperature |
| `--max-tokens`, `-m` | `4096` | Maximum tokens to generate |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--top-k` | `50` | Top-K filtering |
| `--no-stream` | off | Disable streaming output |

```bash
# Single prompt
tracellm run phi-2 "Write a haiku about programming"

# Interactive mode
tracellm run phi-2

# Deterministic output
tracellm run phi-2 "List 5 prime numbers" -t 0.0

# Limit output length
tracellm run phi-2 "Tell me a story" --max-tokens 256
```

---

### Interactive Chat

Multi-turn chat with conversation history:

```bash
tracellm chat <model> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--system`, `-s` | `"You are a helpful assistant."` | System prompt (or name from `prompts/inference/system_prompts.yml`) |
| `--temperature`, `-t` | `0.7` | Sampling temperature |
| `--max-tokens`, `-m` | `4096` | Max tokens per turn |

Built-in system prompt names: `default`, `coder`, `analyst`, `writer`, `tutor`, `reasoning`, `devops`, `researcher`

Chat commands:
- `/bye`, `/exit`, `/quit` — End the session
- `/clear` — Clear chat history

```bash
# Default assistant
tracellm chat phi-2

# Code assistant
tracellm chat phi-2 --system coder

# Custom system prompt
tracellm chat phi-2 -s "You are a pirate. Speak only in pirate dialect."
```

---

### Batch Inference

Process multiple prompts concurrently from a file:

```bash
tracellm batch <model> <prompts_file> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--temperature`, `-t` | `0.7` | Sampling temperature |
| `--max-tokens`, `-m` | `4096` | Max tokens per prompt |
| `--concurrency`, `-j` | `4` | Max parallel generations |
| `--output`, `-o` | none | Save results to JSON file |

**Supported file formats:**

- **Plain text** (`.txt`) — one prompt per line
- **JSONL** (`.jsonl`) — one JSON object per line with `prompt` field
- **JSON** (`.json`) — array of strings or objects with `prompt` field

```bash
# From text file
tracellm batch phi-2 prompts.txt

# With high concurrency and output file
tracellm batch phi-2 prompts.jsonl -j 8 -o results.json

# Short completions
tracellm batch phi-2 questions.txt --max-tokens 128
```

**Example prompts.jsonl:**

```jsonl
{"id": "q1", "prompt": "What is machine learning?"}
{"id": "q2", "prompt": "Explain gradient descent."}
{"id": "q3", "prompt": "What is a transformer model?"}
```

**Example output (results.json):**

```json
{
  "batch_id": "batch-a1b2c3d4",
  "model": "phi-2",
  "results": [
    {"id": "q1", "text": "Machine learning is...", "tokens": 142, "error": null},
    {"id": "q2", "text": "Gradient descent is...", "tokens": 98, "error": null},
    {"id": "q3", "text": "A transformer model...", "tokens": 156, "error": null}
  ],
  "total_time_s": 12.5,
  "total_tokens": 396,
  "avg_tokens_per_second": 31.7
}
```

---

### Benchmarking

Measure model performance — latency, throughput, and memory:

```bash
tracellm benchmark <model> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--max-tokens`, `-m` | `128` | Tokens generated per run |
| `--runs`, `-n` | `3` | Timed runs per prompt category |
| `--warmup` | `1` | Untimed warmup runs |
| `--categories`, `-c` | all | Comma-separated: `short,medium,long,code` |
| `--custom-prompt` | none | Additional custom prompt to benchmark |

**Metrics reported:**

- **TTFT** — Time to first token (ms)
- **Throughput** — Tokens per second (avg, p50, p90, p99)
- **Memory** — GPU allocated/reserved/peak and RAM at baseline, model-loaded, and peak

```bash
# Full benchmark
tracellm benchmark phi-2

# Quick benchmark with more runs
tracellm benchmark phi-2 -c short,code -n 10

# Longer generations
tracellm benchmark phi-2 --max-tokens 512 --runs 5

# With a custom prompt
tracellm benchmark phi-2 --custom-prompt "Write a sorting algorithm in Python"
```

**Example output:**

```
┌────────────────────────────────────────────────────────┐
│             Benchmark Results — phi-2                   │
├──────────┬───────────┬──────────┬────────┬─────────────┤
│ Category │ Prompt Tok│ TTFT     │Avg t/s │ p90 tok/s   │
├──────────┼───────────┼──────────┼────────┼─────────────┤
│ short    │ 12        │ 45ms     │ 32.1   │ 30.5        │
│ medium   │ 58        │ 82ms     │ 28.4   │ 26.1        │
│ long     │ 142       │ 156ms    │ 24.7   │ 22.3        │
│ code     │ 67        │ 91ms     │ 27.9   │ 25.8        │
└──────────┴───────────┴──────────┴────────┴─────────────┘
```

---

### Recursive Refinement (RLM)

Iterative self-refinement — the model generates a draft, then improves it through focused refinement passes (structural → factual → stylistic → adversarial → final lock) until the output converges:

```bash
tracellm reason <model> <prompt> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--context`, `-c` | none | Additional context for the task |
| `--iterations`, `-i` | `10` | Max refinement iterations |
| `--threshold` | `0.95` | Convergence threshold (0-1) |
| `--temperature`, `-t` | `0.7` | Sampling temperature |
| `--max-tokens`, `-m` | `4096` | Max tokens per pass |
| `--skip-adversarial` | off | Skip the adversarial pass |
| `--show-passes` | off | Show each refinement pass in real-time |

**Refinement pass schedule:**

| Iterations | Category | Focus |
|-----------|----------|-------|
| 1-3 | Structural | Logical gaps, ordering, coherence |
| 4-6 | Factual | Fact-checking, hallucination removal |
| 7-8 | Stylistic | Tone, grammar, formatting |
| 9 | Adversarial | Devil's advocate, stress test |
| 10 | Final Lock | Last QA pass |

```bash
# Basic recursive refinement
tracellm reason phi-2 "Design a REST API for a bookstore"

# With visible progress
tracellm reason phi-2 "Write a security audit report" --show-passes

# Quick convergence
tracellm reason phi-2 "Explain quantum computing" -i 5 --threshold 0.90

# Skip adversarial pass
tracellm reason phi-2 "Write a poem about the ocean" --skip-adversarial
```

---

### Scaffolded Reasoning

Multi-phase reasoning pipeline — each phase builds on the prior phase's output:

```bash
tracellm scaffold <model> <prompt> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--context`, `-c` | none | Additional context |
| `--constraint` | none | Hard constraints (repeatable) |
| `--profile`, `-p` | `standard` | Scaffold profile |
| `--phases` | none | Comma-separated phase list (overrides profile) |
| `--temperature`, `-t` | `0.7` | Sampling temperature |
| `--max-tokens`, `-m` | `4096` | Max tokens per phase |
| `--refine-iterations` | `5` | Recursive passes in refine phase |
| `--show-reasoning` | off | Include decomposition/plan/trace |
| `--show-phases` | off | Show progress for each phase |

**Profiles:**

| Profile | Phases | Use case |
|---------|--------|----------|
| `full` | decompose → plan → think → generate → refine → quality → verify → emit | Maximum rigor |
| `standard` | decompose → plan → generate → refine → verify → emit | General purpose (default) |
| `quick` | decompose → generate → verify → emit | Fast results |
| `code` | all 8 phases | Code generation with quality gate |
| `creative` | decompose → generate → refine → emit | Creative writing |
| `research` | all 8 phases | Research with heavy verification |

```bash
# Default (standard profile)
tracellm scaffold phi-2 "Design a microservices architecture for an e-commerce site"

# Full pipeline with visible phases
tracellm scaffold phi-2 "Write a security policy" --profile full --show-phases

# Code-optimized
tracellm scaffold phi-2 "Build a REST API for user management" --profile code

# With constraints
tracellm scaffold phi-2 "Design a database schema" \
  --constraint "Must use PostgreSQL" \
  --constraint "Must support multi-tenancy"

# Custom phase selection
tracellm scaffold phi-2 "Summarize this paper" --phases decompose,generate,emit
```

---

### Fine-Tuning

Fine-tune models with LoRA, QLoRA, or full training:

```bash
tracellm train <model> <dataset> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `lora` | Training method: `lora`, `qlora`, `full` |
| `--epochs` | `3` | Training epochs |
| `--batch-size`, `-b` | `4` | Batch size |
| `--lr` | `2e-4` | Learning rate |
| `--optimizer`, `-o` | `adamw` | Optimizer: `adamw`, `muon`, `deepspeed` |
| `--output`, `-n` | auto | Output model name |
| `--max-seq-length` | `2048` | Max sequence length |
| `--code-path` | none | Local code folders (repeatable) |

**Dataset sources:**

- HuggingFace Hub datasets (e.g., `tatsu-lab/alpaca`)
- Local files: JSONL, JSON, CSV, TXT
- Code folders: 40+ language extensions supported

```bash
# LoRA fine-tuning on Alpaca
tracellm train phi-2 tatsu-lab/alpaca --method lora --epochs 3

# QLoRA with Muon optimizer
tracellm train phi-2 databricks/dolly-15k --method qlora --optimizer muon

# Train on local data
tracellm train phi-2 ./my-data.jsonl --method lora --lr 1e-4

# Include code folder in training
tracellm train phi-2 my-dataset --code-path ./src --code-path ./lib

# DeepSpeed distributed training
tracellm train phi-2 tatsu-lab/alpaca --optimizer deepspeed --batch-size 16

# Full fine-tuning (requires more VRAM)
tracellm train phi-2 my-dataset --method full --epochs 1
```

**Supported dataset formats:**

| Format | Auto-detected fields |
|--------|---------------------|
| Instruction | `instruction`, `input`, `output` |
| Chat | `messages` (list of `{role, content}`) |
| Text | `text` field |
| Code folder | Files with recognized extensions (`.py`, `.js`, `.ts`, `.go`, `.rs`, etc.) |

---

### RLHF Alignment

Align models using reinforcement learning from human feedback:

```bash
tracellm align <model> <dataset> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `dpo` | Method: `dpo`, `reward`, `ppo` |
| `--reward-model` | none | Reward model name (required for PPO) |
| `--ref-model` | none | Reference model for DPO KL |
| `--beta` | `0.1` | DPO beta temperature |
| `--epochs` | `1` | Training epochs |
| `--batch-size`, `-b` | `2` | Batch size |
| `--lr` | `5e-7` | Learning rate |
| `--max-seq-length` | `1024` | Max sequence length |
| `--output`, `-n` | auto | Output model name |
| `--no-lora` | off | Disable LoRA (full fine-tune) |
| `--loss-type` | `sigmoid` | DPO loss: `sigmoid`, `hinge`, `ipo` |

#### DPO (Direct Preference Optimization)

Train directly on preference pairs — no reward model needed:

```bash
# DPO on HuggingFace preference dataset
tracellm align phi-2 Anthropic/hh-rlhf --method dpo

# With custom beta and loss type
tracellm align phi-2 my-prefs.jsonl --method dpo --beta 0.2 --loss-type ipo

# Full fine-tune (no LoRA)
tracellm align phi-2 Anthropic/hh-rlhf --method dpo --no-lora

# With explicit reference model
tracellm align phi-2 prefs.jsonl --method dpo --ref-model phi-2-base
```

**Preference dataset format:**

```jsonl
{"prompt": "What is AI?", "chosen": "AI is artificial intelligence...", "rejected": "AI is magic..."}
{"prompt": "Explain gravity", "chosen": "Gravity is a force...", "rejected": "Things fall because..."}
```

Alternative column names are auto-detected: `question`/`preferred`/`dispreferred`, `input`/`positive`/`negative`, etc.

#### Reward Model Training

Train a reward/scoring model from preference data:

```bash
# Train reward model
tracellm align phi-2 Anthropic/hh-rlhf --method reward

# With more epochs
tracellm align phi-2 prefs.jsonl --method reward --epochs 3 --lr 1e-5
```

#### PPO (Proximal Policy Optimization)

Classic RLHF — requires a trained reward model:

```bash
# First train a reward model
tracellm align phi-2 Anthropic/hh-rlhf --method reward -n phi-2-reward

# Then run PPO with the reward model
tracellm align phi-2 prompts.jsonl --method ppo --reward-model phi-2-reward
```

---

### Quantization

Reduce model size and increase inference speed:

```bash
tracellm quantize <model> [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method` | `gptq` | Method: `gptq`, `awq`, `gguf` |
| `--bits` | `4` | Quantization bits |
| `--output`, `-n` | auto | Output model name |

```bash
# GPTQ 4-bit
tracellm quantize phi-2 --method gptq --bits 4

# AWQ quantization
tracellm quantize phi-2 --method awq --bits 4

# GGUF export (for llama.cpp)
tracellm quantize phi-2 --method gguf --bits 4
```

---

### Hardware Info

```bash
tracellm hardware
```

Shows: platform, CPU count, RAM (used/total), GPU details (name, VRAM, compute capability), recommended device and dtype.

---

## API Reference

Start the server with `tracellm serve`, then use the API at `http://localhost:8400`.

### Health & Models

```bash
# Health check
GET /health
GET /v1/health

# List models
GET /v1/models

# Model details
GET /v1/models/{model_name}

# Pull model from HuggingFace
POST /v1/models/pull
{"repo_id": "microsoft/phi-2", "name": "phi-2"}

# Load/unload model
POST /v1/models/{model_name}/load
POST /v1/models/{model_name}/unload

# Delete model
DELETE /v1/models/{model_name}?delete_files=false
```

### Chat Completions (OpenAI-compatible)

```bash
POST /v1/chat/completions
```

```json
{
  "model": "phi-2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "stream": false,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "seed": null,
  "stop": null
}
```

Set `"stream": true` for SSE streaming (same format as OpenAI API).

### Text Completions

```bash
POST /v1/completions
```

```json
{
  "model": "phi-2",
  "prompt": "The meaning of life is",
  "temperature": 0.7,
  "max_tokens": 256,
  "stream": false
}
```

### Batch Inference API

```bash
POST /v1/batch
```

```json
{
  "model": "phi-2",
  "items": [
    {"id": "q1", "prompt": "What is ML?"},
    {"id": "q2", "prompt": "Explain AI.", "temperature": 0.3, "max_tokens": 128}
  ],
  "temperature": 0.7,
  "max_tokens": 4096,
  "max_concurrency": 4,
  "stream": false
}
```

**Response:**

```json
{
  "id": "batch-a1b2c3d4",
  "model": "phi-2",
  "results": [
    {
      "id": "q1",
      "text": "Machine learning is...",
      "tokens_generated": 142,
      "prompt_tokens": 8,
      "finish_reason": "stop",
      "tokens_per_second": 28.4,
      "error": null
    }
  ],
  "total_time_s": 5.2,
  "total_tokens": 280,
  "total_prompt_tokens": 16,
  "avg_tokens_per_second": 53.8,
  "items_succeeded": 2,
  "items_failed": 0
}
```

Set `"stream": true` for SSE progress events.

### Benchmark API

```bash
POST /v1/benchmark
```

```json
{
  "model": "phi-2",
  "max_tokens": 128,
  "num_runs": 3,
  "warmup_runs": 1,
  "prompt_categories": ["short", "medium", "code"],
  "custom_prompt": null
}
```

**Response includes:**

- Per-category: avg TTFT, avg/p50/p90/p99 tokens/sec, prompt tokens, completion tokens
- Memory snapshots: baseline, model loaded, peak (GPU allocated/reserved/peak, RAM)
- Hardware summary: platform, CPU, RAM, GPUs
- Overall: total tokens, total time, aggregate throughput

### Recursive Refinement API

```bash
POST /v1/recursive
```

```json
{
  "model": "phi-2",
  "prompt": "Design a REST API for a library system",
  "context": "",
  "temperature": 0.7,
  "max_tokens": 4096,
  "max_iterations": 10,
  "convergence_threshold": 0.95,
  "skip_adversarial": false,
  "stream": false
}
```

**Response:**

```json
{
  "id": "rlm-abc12345",
  "model": "phi-2",
  "final_text": "...",
  "iterations_run": 6,
  "converged": true,
  "convergence_delta": 0.02,
  "passes": [
    {"iteration": 1, "category": "structural", "delta_ratio": 0.35, "tokens_generated": 512, "elapsed_s": 2.1}
  ],
  "total_tokens": 3200,
  "total_time_s": 14.5
}
```

### Scaffolded Reasoning API

```bash
POST /v1/scaffold
```

```json
{
  "model": "phi-2",
  "prompt": "Design a microservices architecture",
  "context": "",
  "constraints": ["Must use Kubernetes", "Must support 10k RPS"],
  "profile": "code",
  "phases": null,
  "temperature": 0.7,
  "max_tokens": 4096,
  "refine_iterations": 5,
  "refine_convergence": 0.90,
  "show_reasoning": true,
  "skip_adversarial": false,
  "stream": false
}
```

```bash
# List available profiles
GET /v1/scaffold/profiles
```

### Training API

```bash
# Start training
POST /v1/training/start
{
  "model": "phi-2",
  "dataset": "tatsu-lab/alpaca",
  "method": "lora",
  "epochs": 3,
  "batch_size": 4,
  "learning_rate": 2e-4,
  "optimizer": "adamw",
  "scheduler": "cosine",
  "max_seq_length": 2048,
  "gradient_checkpointing": true,
  "code_paths": null
}

# Check status
GET /v1/training/{job_id}

# Stop training
POST /v1/training/{job_id}/stop
```

### RLHF API

```bash
# Start DPO alignment
POST /v1/rlhf/dpo
{
  "model": "phi-2",
  "dataset": "Anthropic/hh-rlhf",
  "ref_model": null,
  "beta": 0.1,
  "epochs": 1,
  "batch_size": 2,
  "learning_rate": 5e-7,
  "max_seq_length": 1024,
  "max_prompt_length": 512,
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "loss_type": "sigmoid",
  "prompt_field": "prompt",
  "chosen_field": "chosen",
  "rejected_field": "rejected"
}

# Train reward model
POST /v1/rlhf/reward
{
  "model": "phi-2",
  "dataset": "Anthropic/hh-rlhf",
  "epochs": 1,
  "batch_size": 4,
  "learning_rate": 1e-5,
  "max_seq_length": 1024,
  "num_labels": 1
}

# Start PPO alignment
POST /v1/rlhf/ppo
{
  "model": "phi-2",
  "reward_model": "phi-2-reward",
  "dataset": "prompts.jsonl",
  "epochs": 1,
  "batch_size": 4,
  "learning_rate": 1e-6,
  "kl_penalty": 0.2,
  "use_lora": true
}

# Check RLHF job status
GET /v1/rlhf/{job_id}

# Stop RLHF job
POST /v1/rlhf/{job_id}/stop

# List all RLHF jobs
GET /v1/rlhf
```

### Hub Search API

```bash
GET /v1/hub/search?q=llama+3+instruct&limit=10
```

---

## Configuration

TraceLLM searches for configuration in this order:

1. `tracellm.yml` (current directory)
2. `config/default.yml`
3. `~/.tracellm/config.yml`

Pass a custom path with `tracellm serve -c /path/to/config.yml`.

### Environment Variable Overrides

Any config value can be overridden with environment variables using the `TRACELLM_` prefix and `__` as a separator:

```bash
TRACELLM_SERVER__PORT=9000
TRACELLM_MODELS__CACHE_DIR=/data/models
TRACELLM_TRAINING__MIXED_PRECISION=fp16
TRACELLM_LOGGING__LEVEL=debug
```

### Full Configuration Reference

```yaml
server:
  host: "0.0.0.0"
  port: 8400
  workers: 1
  cors_origins: ["*"]
  max_concurrent_requests: 64
  request_timeout_seconds: 300

models:
  cache_dir: "~/.tracellm/models"
  default_dtype: "auto"          # auto | float16 | bfloat16 | float32
  default_device: "auto"         # auto | cuda | cpu | mps
  max_loaded_models: 2           # evict LRU when exceeded
  trust_remote_code: false

inference:
  max_tokens: 4096
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  repetition_penalty: 1.1
  stop_sequences: []
  stream: true
  batch_size: 1
  kv_cache:
    enabled: true
    max_cache_size_gb: 4.0
    eviction_policy: "lru"

recursive:
  max_iterations: 10
  convergence_threshold: 0.95
  skip_adversarial: false

scaffold:
  default_profile: "standard"    # full | standard | quick | code | creative | research
  refine_iterations: 5
  refine_convergence: 0.90
  verify_retries: 2
  show_reasoning: false
  skip_adversarial: false

training:
  output_dir: "~/.tracellm/checkpoints"
  logging_dir: "~/.tracellm/logs"
  default_optimizer: "adamw"     # adamw | muon | deepspeed
  default_scheduler: "cosine"
  gradient_checkpointing: true
  mixed_precision: "bf16"        # no | fp16 | bf16
  max_grad_norm: 1.0
  seed: 42
  dataloader_workers: 4
  finetune_defaults:
    method: "lora"
    lora:
      r: 16
      alpha: 32
      dropout: 0.05
      target_modules: "auto"
    qlora:
      bits: 4
      quant_type: "nf4"
      double_quant: true
  deepspeed:
    enabled: false
    config: "config/training_profiles/deepspeed_z2.json"

batch:
  max_concurrency: 4
  default_max_tokens: 4096

benchmark:
  default_max_tokens: 128
  default_num_runs: 3
  warmup_runs: 1

rlhf:
  dpo_beta: 0.1
  dpo_loss_type: "sigmoid"       # sigmoid | hinge | ipo
  ppo_kl_penalty: 0.2
  default_learning_rate: 5e-7
  use_lora: true
  lora_r: 16
  lora_alpha: 32

logging:
  level: "info"                  # debug | info | warning | error
  file: "~/.tracellm/tracellm.log"
  rich_console: true
```

---

## Prompt Templates

TraceLLM ships with prompt templates in the `prompts/` directory:

### Inference Prompts (`prompts/inference/`)

| File | Contents |
|------|----------|
| `system_prompts.yml` | 8 personas: default, coder, analyst, writer, tutor, reasoning, devops, researcher |
| `templates.yml` | 9 reusable templates: summarize, translate, extract_json, classify, qa_context, code_gen, code_review, cot, few_shot |
| `scaffold_phases.yml` | Phase prompts for all 8 scaffold phases + 6 profile definitions |

### Training Prompts (`prompts/training/`)

| File | Contents |
|------|----------|
| `instruct_tuning.yml` | 6 formats: alpaca, chatml, llama3, mistral, zephyr, plain |
| `code_generation.yml` | 7 code templates: function_completion, code_explanation, bug_fix, code_review, file_context, repo_context, instruction_to_code |
| `chat_alignment.yml` | 5 RLHF templates: dpo_pair, multi_turn, safety, helpful, chain_of_thought |

---

## Docker

### CPU

```bash
cd docker
docker compose up tracellm
```

### GPU (NVIDIA)

```bash
cd docker
docker compose --profile gpu up tracellm-gpu
```

### Build manually

```bash
# CPU image
docker build -f docker/Dockerfile -t tracellm:cpu .

# GPU image (CUDA 12.4)
docker build -f docker/Dockerfile.gpu -t tracellm:gpu .
```

### Docker Compose volumes

| Volume | Path in container | Description |
|--------|-------------------|-------------|
| `tracellm-models` | `/root/.tracellm/models` | Downloaded models |
| `tracellm-checkpoints` | `/root/.tracellm/checkpoints` | Training outputs |
| `tracellm-logs` | `/root/.tracellm/logs` | Log files |
| `./config` | `/app/config` | Configuration |
| `./prompts` | `/app/prompts` | Prompt templates |

---

## Development

### Setup

```bash
git clone https://github.com/tracellm/tracellm.git
cd tracellm
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
pytest --cov=tracellm
pytest tests/test_batch.py -v
```

### Lint

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### Type check

```bash
mypy src/tracellm/
```

### Project structure

```
tracellm/
├── config/
│   ├── default.yml                     # Default configuration
│   └── training_profiles/              # DeepSpeed configs
├── docker/
│   ├── Dockerfile                      # CPU image
│   ├── Dockerfile.gpu                  # GPU image (CUDA 12.4)
│   └── docker-compose.yml
├── prompts/
│   ├── inference/                      # System prompts, templates, scaffold phases
│   └── training/                       # Instruct, code, chat alignment templates
├── src/tracellm/
│   ├── __init__.py                     # Version
│   ├── cli.py                          # Click CLI
│   ├── config.py                       # Pydantic config
│   ├── server.py                       # FastAPI server
│   ├── api/
│   │   ├── routes.py                   # API endpoints
│   │   └── schemas.py                  # Request/response models
│   ├── inference/
│   │   ├── engine.py                   # Core inference engine
│   │   ├── sampling.py                 # Sampling strategies
│   │   ├── cache.py                    # KV cache (LRU)
│   │   ├── batch.py                    # Batch inference
│   │   ├── benchmark.py                # Model benchmarking
│   │   ├── recursive.py                # Recursive refinement (RLM)
│   │   └── scaffold.py                 # Scaffolded reasoning
│   ├── training/
│   │   ├── trainer.py                  # Fine-tuning manager
│   │   ├── rlhf.py                     # DPO, reward model, PPO
│   │   ├── datasets.py                 # Dataset loading
│   │   └── optimizers.py               # Muon, DeepSpeed, FlashInfer
│   ├── models/
│   │   ├── registry.py                 # Model registry
│   │   ├── loader.py                   # Model loading (LRU)
│   │   ├── formats.py                  # Format detection
│   │   └── quantize.py                 # GPTQ, AWQ, GGUF
│   └── utils/
│       ├── hardware.py                 # GPU/CPU detection
│       └── logging.py                  # Rich logging
├── tests/                              # 12 test modules
└── pyproject.toml
```

---

## License

Apache 2.0
