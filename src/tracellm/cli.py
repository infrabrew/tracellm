"""TraceLLM CLI — ollama-style command interface for LLM management.

Usage:
    tracellm serve                    Start the inference server
    tracellm pull <repo_id>           Download a model from HuggingFace
    tracellm list                     List local models
    tracellm run <model> [prompt]     Interactive generation / chat
    tracellm chat <model>             Interactive chat session
    tracellm batch <model> <file>     Batch inference on multiple prompts
    tracellm benchmark <model>        Benchmark model performance
    tracellm reason <model> <prompt>  Recursive self-refinement (RLM)
    tracellm scaffold <model> <prompt> Scaffolded multi-phase reasoning
    tracellm train <model> <dataset>  Fine-tune a model
    tracellm align <model> <dataset>  RLHF alignment (DPO/PPO/reward)
    tracellm remove <model>           Remove a model
    tracellm quantize <model>         Quantize a model
    tracellm search <query>           Search HuggingFace Hub
    tracellm info <model>             Show model details
    tracellm hardware                 Show hardware info
"""

from __future__ import annotations

import sys
import time

import click
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from tracellm import __version__
from tracellm.config import load_config

console = Console()


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="tracellm")
@click.pass_context
def main(ctx):
    """TraceLLM — inference, training, and fine-tuning platform."""
    if ctx.invoked_subcommand is None:
        console.print(Panel(
            "[bold]TraceLLM[/bold] — inference, training, and fine-tuning platform\n\n"
            "Commands:\n"
            "  [cyan]tracellm serve[/cyan]                    Start the API server\n"
            "  [cyan]tracellm pull[/cyan] <repo_id>           Download a model\n"
            "  [cyan]tracellm list[/cyan]                     List local models\n"
            "  [cyan]tracellm run[/cyan] <model> [prompt]     Generate text\n"
            "  [cyan]tracellm chat[/cyan] <model>             Interactive chat\n"
            "  [cyan]tracellm batch[/cyan] <model> <file>     Batch inference on multiple prompts\n"
            "  [cyan]tracellm benchmark[/cyan] <model>        Benchmark model performance\n"
            "  [cyan]tracellm reason[/cyan] <model> <prompt>  Recursive self-refinement (RLM)\n"
            "  [cyan]tracellm scaffold[/cyan] <model> <prompt> Multi-phase reasoning pipeline\n"
            "  [cyan]tracellm train[/cyan] <model> <dataset>  Fine-tune a model\n"
            "  [cyan]tracellm align[/cyan] <model> <dataset>  RLHF alignment (DPO/PPO/reward)\n"
            "  [cyan]tracellm remove[/cyan] <model>           Remove a model\n"
            "  [cyan]tracellm quantize[/cyan] <model>         Quantize a model\n"
            "  [cyan]tracellm search[/cyan] <query>           Search HuggingFace Hub\n"
            "  [cyan]tracellm info[/cyan] <model>             Show model details\n"
            "  [cyan]tracellm hardware[/cyan]                 Show hardware info\n"
            f"\nVersion: {__version__}",
            title="tracellm",
            border_style="blue",
        ))


# ── serve ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--host", default=None, help="Bind host (default: 0.0.0.0)")
@click.option("--port", "-p", default=None, type=int, help="Bind port (default: 8400)")
@click.option("--workers", "-w", default=None, type=int, help="Worker processes")
@click.option("--config", "-c", default=None, help="Config file path")
def serve(host, port, workers, config):
    """Start the TraceLLM inference server."""
    from tracellm.server import run_server
    console.print(f"[bold blue]TraceLLM[/bold blue] v{__version__} — starting server...")
    run_server(config_path=config, host=host, port=port, workers=workers)


# ── pull ─────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("repo_id")
@click.option("--name", "-n", default=None, help="Local name for the model")
@click.option("--revision", default="main", help="Branch/tag/commit")
@click.option("--quantization", "-q", default="", help="Quantization label")
def pull(repo_id, name, revision, quantization):
    """Download a model from HuggingFace Hub."""
    config = load_config()
    from tracellm.models.registry import ModelRegistry

    registry = ModelRegistry(cache_dir=config.models.cache_dir)

    console.print(f"Pulling [bold]{repo_id}[/bold]...")
    with console.status("Downloading..."):
        card = registry.pull(repo_id, name=name, revision=revision, quantization=quantization)

    console.print(f"[green]Done![/green] Model '{card.name}' ready ({card.size_gb:.1f} GB)")


# ── list ─────────────────────────────────────────────────────────────────────

@main.command(name="list")
def list_models():
    """List all local models."""
    config = load_config()
    from tracellm.models.registry import ModelRegistry

    registry = ModelRegistry(cache_dir=config.models.cache_dir)
    models = registry.list_models()

    if not models:
        console.print("[dim]No models found. Run [cyan]tracellm pull <repo_id>[/cyan] to get started.[/dim]")
        return

    table = Table(title="Local Models")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Format", style="magenta")
    table.add_column("Architecture", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Quantization")
    table.add_column("Source", style="dim")

    for m in models:
        table.add_row(
            m.name,
            m.format,
            m.architecture,
            f"{m.size_gb:.1f} GB",
            m.quantization or "—",
            m.source[:40],
        )

    console.print(table)


# ── run ──────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.argument("prompt", required=False)
@click.option("--temperature", "-t", default=0.7, type=float)
@click.option("--max-tokens", "-m", default=4096, type=int)
@click.option("--top-p", default=0.9, type=float)
@click.option("--top-k", default=50, type=int)
@click.option("--no-stream", is_flag=True, help="Disable streaming output")
def run(model, prompt, temperature, max_tokens, top_p, top_k, no_stream):
    """Generate text from a model. Interactive if no prompt given."""
    config = load_config()
    from tracellm.inference.engine import InferenceEngine
    from tracellm.inference.sampling import SamplingParams

    engine = InferenceEngine(config)

    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    if prompt:
        _generate_once(engine, model, prompt, params, stream=not no_stream)
    else:
        # Interactive mode
        console.print(f"[bold blue]TraceLLM[/bold blue] — {model} (type /bye to exit)\n")
        while True:
            try:
                user_input = console.input("[bold green]>>> [/bold green]")
            except (EOFError, KeyboardInterrupt):
                break
            if user_input.strip().lower() in ("/bye", "/exit", "/quit"):
                break
            if not user_input.strip():
                continue
            _generate_once(engine, model, user_input, params, stream=not no_stream)
            console.print()


def _generate_once(engine, model, prompt, params, stream=True):
    """Generate and print output for a single prompt."""
    if stream:
        for token in engine.stream(model, prompt, params):
            console.print(token, end="", highlight=False)
        console.print()
    else:
        with console.status("Generating..."):
            result = engine.generate(model, prompt, params)
        console.print(result.text)
        console.print(
            f"\n[dim]{result.tokens_generated} tokens in {result.generation_time_s}s "
            f"({result.tokens_per_second} tok/s)[/dim]"
        )


# ── chat ─────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.option("--system", "-s", default=None, help="System prompt (or name from prompts/)")
@click.option("--temperature", "-t", default=0.7, type=float)
@click.option("--max-tokens", "-m", default=4096, type=int)
def chat(model, system, temperature, max_tokens):
    """Interactive multi-turn chat session."""
    config = load_config()
    from tracellm.inference.engine import InferenceEngine
    from tracellm.inference.sampling import SamplingParams

    engine = InferenceEngine(config)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    # Load system prompt
    sys_prompt = system or "You are a helpful assistant."
    if not system or len(system) < 50:
        # Try loading from prompts file
        try:
            import yaml
            from pathlib import Path
            prompts_file = Path("prompts/inference/system_prompts.yml")
            if prompts_file.exists():
                data = yaml.safe_load(prompts_file.read_text())
                key = system or "default"
                if key in data.get("prompts", {}):
                    sys_prompt = data["prompts"][key]["content"]
        except Exception:
            pass

    messages = [f"<|system|>\n{sys_prompt}\n"]
    console.print(f"[bold blue]TraceLLM Chat[/bold blue] — {model}")
    console.print(f"[dim]System: {sys_prompt[:80]}...[/dim]\n" if len(sys_prompt) > 80
                  else f"[dim]System: {sys_prompt}[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("/bye", "/exit", "/quit"):
            break
        if user_input.strip() == "/clear":
            messages = [f"<|system|>\n{sys_prompt}\n"]
            console.print("[dim]Chat history cleared.[/dim]")
            continue
        if not user_input.strip():
            continue

        messages.append(f"<|user|>\n{user_input}\n")
        full_prompt = "".join(messages) + "<|assistant|>\n"

        console.print("[bold blue]TraceLLM:[/bold blue] ", end="")
        response_text = ""
        for token in engine.stream(model, full_prompt, params):
            console.print(token, end="", highlight=False)
            response_text += token
        console.print("\n")

        messages.append(f"<|assistant|>\n{response_text}\n")


# ── reason (Recursive Language Model) ────────────────────────────────────────

@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--context", "-c", default="", help="Additional context for the task")
@click.option("--iterations", "-i", default=10, type=int, help="Max refinement iterations")
@click.option("--threshold", default=0.95, type=float, help="Convergence threshold (0-1)")
@click.option("--temperature", "-t", default=0.7, type=float)
@click.option("--max-tokens", "-m", default=4096, type=int)
@click.option("--skip-adversarial", is_flag=True, help="Skip the adversarial pass")
@click.option("--show-passes", is_flag=True, help="Show each refinement pass")
def reason(model, prompt, context, iterations, threshold, temperature, max_tokens, skip_adversarial, show_passes):
    """Recursive self-refinement — model iteratively improves its own output.

    The RLM engine generates an initial draft, then feeds it back through
    focused refinement passes (structural → factual → stylistic → adversarial)
    until the output converges or max iterations are reached.

    Examples:
        tracellm reason llama3 "Design a REST API for a bookstore"
        tracellm reason llama3 "Explain quantum computing" --iterations 5
        tracellm reason llama3 "Write a security audit" --show-passes
    """
    config = load_config()
    from tracellm.inference.engine import InferenceEngine
    from tracellm.inference.recursive import RecursiveEngine
    from tracellm.inference.sampling import SamplingParams

    engine = InferenceEngine(config)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    recursive = RecursiveEngine(
        engine=engine,
        max_iterations=iterations,
        convergence_threshold=threshold,
        skip_adversarial=skip_adversarial,
    )

    console.print(
        f"[bold blue]TraceLLM RLM[/bold blue] — recursive refinement\n"
        f"  Model: {model} | Max iterations: {iterations} | Threshold: {threshold}\n"
    )

    def on_pass(rp):
        if show_passes:
            status = "[green]▼" if rp.delta_ratio < (1 - threshold) else "[yellow]►"
            console.print(
                f"  {status} Pass {rp.iteration}[/] [{rp.category.value}] "
                f"delta={rp.delta_ratio:.4f} | {rp.tokens_generated} tok | {rp.elapsed_s:.1f}s"
            )

    with console.status("Generating initial draft...") if not show_passes else _noop_context():
        result = recursive.refine(
            model_name=model,
            prompt=prompt,
            context=context,
            params=params,
            on_pass=on_pass,
        )

    # Print result
    console.print()
    console.print(Panel(result.final_text, title="Output", border_style="green"))
    console.print(
        f"\n[dim]Iterations: {result.iterations_run} | "
        f"Converged: {'yes' if result.converged else 'no'} | "
        f"Final delta: {result.convergence_delta:.4f} | "
        f"Tokens: {result.total_tokens:,} | "
        f"Time: {result.total_time_s:.1f}s[/dim]"
    )


# ── scaffold (Multi-phase reasoning) ─────────────────────────────────────────

PROFILE_PHASES = {
    "full": ["decompose", "plan", "think", "generate", "refine", "quality", "verify", "emit"],
    "standard": ["decompose", "plan", "generate", "refine", "verify", "emit"],
    "quick": ["decompose", "generate", "verify", "emit"],
    "code": ["decompose", "plan", "think", "generate", "refine", "quality", "verify", "emit"],
    "creative": ["decompose", "generate", "refine", "emit"],
    "research": ["decompose", "plan", "think", "generate", "refine", "quality", "verify", "emit"],
}


@main.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--context", "-c", default="", help="Additional context")
@click.option("--constraint", multiple=True, help="Hard constraints (repeatable)")
@click.option("--profile", "-p", default="standard",
              type=click.Choice(["full", "standard", "quick", "code", "creative", "research"]),
              help="Scaffold profile")
@click.option("--phases", default=None, help="Comma-separated phase list (overrides profile)")
@click.option("--temperature", "-t", default=0.7, type=float)
@click.option("--max-tokens", "-m", default=4096, type=int)
@click.option("--refine-iterations", default=5, type=int, help="Recursive passes in refine phase")
@click.option("--show-reasoning", is_flag=True, help="Include decomposition/plan/trace in output")
@click.option("--show-phases", is_flag=True, help="Show progress for each phase")
def scaffold(model, prompt, context, constraint, profile, phases, temperature, max_tokens,
             refine_iterations, show_reasoning, show_phases):
    """Scaffolded multi-phase reasoning pipeline.

    Runs the AGL-style pipeline: decompose → plan → think → generate →
    refine → quality → verify → emit. Each phase builds on the prior
    phase's output. The refine phase uses the recursive engine internally.

    Profiles control which phases run:
        full      — All 8 phases, maximum rigor
        standard  — Core reasoning (default)
        quick     — Decompose, generate, verify
        code      — Full pipeline with quality gate
        creative  — Lighter verification, more refinement
        research  — Heavy reasoning and factual checks

    Examples:
        tracellm scaffold llama3 "Design a microservices architecture"
        tracellm scaffold llama3 "Write a security policy" --profile full
        tracellm scaffold llama3 "Build a REST API" --profile code --show-phases
        tracellm scaffold llama3 "Write a blog post" --profile creative
    """
    config = load_config()
    from tracellm.inference.engine import InferenceEngine
    from tracellm.inference.scaffold import ScaffoldEngine, PhaseID
    from tracellm.inference.sampling import SamplingParams

    engine = InferenceEngine(config)
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    # Resolve phases
    if phases:
        phase_ids = [PhaseID(p.strip()) for p in phases.split(",")]
    else:
        phase_ids = [PhaseID(p) for p in PROFILE_PHASES[profile]]

    scaffold_engine = ScaffoldEngine(
        engine=engine,
        phases=phase_ids,
        refine_iterations=refine_iterations,
        refine_convergence=config.scaffold.refine_convergence,
        skip_adversarial=config.scaffold.skip_adversarial,
        verify_retries=config.scaffold.verify_retries,
    )

    phase_names = [p.value for p in phase_ids]
    console.print(
        f"[bold blue]TraceLLM Scaffold[/bold blue] — {profile} profile\n"
        f"  Model: {model}\n"
        f"  Phases: {' → '.join(phase_names)}\n"
    )

    def on_phase(pr):
        if show_phases:
            meta = ""
            if pr.metadata:
                if "converged" in pr.metadata:
                    meta = f" (converged={pr.metadata['converged']}, iters={pr.metadata.get('iterations', '?')})"
                elif "confidence" in pr.metadata:
                    meta = f" (confidence={pr.metadata['confidence']:.0%})"
            console.print(
                f"  [green]✓[/green] [{pr.phase_id}] {pr.phase_name} — "
                f"{pr.tokens_generated} tok, {pr.elapsed_s:.1f}s{meta}"
            )

    with console.status("Running scaffold pipeline...") if not show_phases else _noop_context():
        state = scaffold_engine.run(
            model_name=model,
            prompt=prompt,
            context=context,
            constraints=list(constraint),
            params=params,
            on_phase=on_phase,
            show_reasoning=show_reasoning,
        )

    # Print result
    console.print()
    console.print(Panel(state.final_output, title="Output", border_style="green"))
    console.print(
        f"\n[dim]Phases: {len(state.phase_results)} | "
        f"Confidence: {state.confidence:.0%} | "
        f"Tokens: {state.total_tokens:,} | "
        f"Time: {state.total_time_s:.1f}s[/dim]"
    )

    if show_reasoning and state.decomposition:
        console.print(Panel(state.decomposition, title="Decomposition", border_style="dim"))
    if show_reasoning and state.execution_plan:
        console.print(Panel(state.execution_plan, title="Execution Plan", border_style="dim"))


class _noop_context:
    """No-op context manager for when we don't want a spinner."""
    def __enter__(self): return self
    def __exit__(self, *args): pass


# ── batch ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.argument("prompts_file")
@click.option("--temperature", "-t", default=0.7, type=float)
@click.option("--max-tokens", "-m", default=4096, type=int)
@click.option("--concurrency", "-j", default=4, type=int, help="Max concurrent generations")
@click.option("--output", "-o", default=None, help="Save results to JSON file")
def batch(model, prompts_file, temperature, max_tokens, concurrency, output):
    """Run batch inference on multiple prompts from a file.

    The prompts file should contain one prompt per line, or be a JSON/JSONL
    file with a "prompt" field per entry.

    Examples:
        tracellm batch llama3 prompts.txt
        tracellm batch llama3 prompts.jsonl -j 8 -o results.json
        tracellm batch llama3 questions.txt --max-tokens 256
    """
    import json as json_module
    from pathlib import Path

    config = load_config()
    from tracellm.inference.engine import InferenceEngine
    from tracellm.inference.batch import BatchEngine, BatchItem
    from tracellm.inference.sampling import SamplingParams

    engine = InferenceEngine(config)
    batch_engine = BatchEngine(engine=engine, max_concurrency=concurrency)

    # Load prompts from file
    prompts_path = Path(prompts_file)
    if not prompts_path.exists():
        console.print(f"[red]File not found: {prompts_file}[/red]")
        return

    items = []
    content = prompts_path.read_text()

    if prompts_path.suffix.lower() in (".jsonl",):
        for i, line in enumerate(content.strip().split("\n")):
            data = json_module.loads(line)
            prompt = data.get("prompt", data.get("text", data.get("input", "")))
            items.append(BatchItem(id=data.get("id", f"item-{i}"), prompt=prompt))
    elif prompts_path.suffix.lower() == ".json":
        data = json_module.loads(content)
        if isinstance(data, list):
            for i, entry in enumerate(data):
                if isinstance(entry, str):
                    items.append(BatchItem(id=f"item-{i}", prompt=entry))
                else:
                    prompt = entry.get("prompt", entry.get("text", ""))
                    items.append(BatchItem(id=entry.get("id", f"item-{i}"), prompt=prompt))
    else:
        # Plain text — one prompt per line
        for i, line in enumerate(content.strip().split("\n")):
            line = line.strip()
            if line:
                items.append(BatchItem(id=f"item-{i}", prompt=line))

    if not items:
        console.print("[red]No prompts found in file.[/red]")
        return

    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    console.print(
        f"[bold blue]TraceLLM Batch[/bold blue] — {len(items)} prompts, "
        f"concurrency={concurrency}\n"
    )

    completed = 0

    def on_complete(item_result):
        nonlocal completed
        completed += 1
        status = "[green]OK[/green]" if not item_result.error else f"[red]ERR: {item_result.error}[/red]"
        console.print(f"  [{completed}/{len(items)}] {item_result.id}: {status}")

    result = batch_engine.run(model, items, params, on_item_complete=on_complete)

    console.print(
        f"\n[bold]Done![/bold] {result.items_succeeded} succeeded, "
        f"{result.items_failed} failed — "
        f"{result.total_tokens:,} tokens in {result.total_time_s}s "
        f"({result.avg_tokens_per_second} tok/s)"
    )

    if output:
        output_data = {
            "batch_id": result.batch_id,
            "model": model,
            "results": [
                {
                    "id": r.id,
                    "text": r.result.text if r.result else "",
                    "tokens": r.result.tokens_generated if r.result else 0,
                    "error": r.error,
                }
                for r in result.results
            ],
            "total_time_s": result.total_time_s,
            "total_tokens": result.total_tokens,
            "avg_tokens_per_second": result.avg_tokens_per_second,
        }
        Path(output).write_text(json_module.dumps(output_data, indent=2))
        console.print(f"Results saved to [cyan]{output}[/cyan]")


# ── benchmark ───────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.option("--max-tokens", "-m", default=128, type=int, help="Max tokens per run")
@click.option("--runs", "-n", default=3, type=int, help="Timed runs per category")
@click.option("--warmup", default=1, type=int, help="Warmup runs (untimed)")
@click.option("--categories", "-c", default=None, help="Comma-separated: short,medium,long,code")
@click.option("--custom-prompt", default=None, help="Additional custom prompt to benchmark")
def benchmark(model, max_tokens, runs, warmup, categories, custom_prompt):
    """Benchmark a model — measure latency, throughput, and memory.

    Runs standardized prompts (short/medium/long/code) multiple times
    and reports TTFT, tokens/sec percentiles, and memory usage.

    Examples:
        tracellm benchmark llama3
        tracellm benchmark llama3 --runs 5 --max-tokens 256
        tracellm benchmark llama3 -c short,code -n 10
    """
    config = load_config()
    from tracellm.inference.engine import InferenceEngine
    from tracellm.inference.benchmark import BenchmarkEngine

    engine = InferenceEngine(config)
    bench = BenchmarkEngine(engine=engine)

    cats = categories.split(",") if categories else None

    console.print(
        f"[bold blue]TraceLLM Benchmark[/bold blue] — {model}\n"
        f"  Max tokens: {max_tokens} | Runs: {runs} | Warmup: {warmup}\n"
    )

    def on_progress(category, run_idx, total):
        console.print(f"  [dim]{category}[/dim] run {run_idx}/{total}", end="\r")

    with console.status("Running benchmark..."):
        result = bench.run(
            model_name=model,
            max_tokens=max_tokens,
            num_runs=runs,
            warmup_runs=warmup,
            prompt_categories=cats,
            custom_prompt=custom_prompt,
            on_progress=on_progress,
        )

    # Results table
    table = Table(title=f"Benchmark Results — {model}")
    table.add_column("Category", style="cyan")
    table.add_column("Prompt Tok", justify="right")
    table.add_column("Avg Gen Tok", justify="right")
    table.add_column("TTFT", justify="right", style="yellow")
    table.add_column("Avg tok/s", justify="right", style="green")
    table.add_column("p50 tok/s", justify="right")
    table.add_column("p90 tok/s", justify="right")
    table.add_column("p99 tok/s", justify="right")

    for name, pr in result.prompt_results.items():
        table.add_row(
            name,
            str(pr.prompt_tokens),
            f"{pr.avg_completion_tokens:.0f}",
            f"{pr.avg_time_to_first_token_s*1000:.0f}ms",
            f"{pr.avg_tokens_per_second:.1f}",
            f"{pr.p50_tokens_per_second:.1f}",
            f"{pr.p90_tokens_per_second:.1f}",
            f"{pr.p99_tokens_per_second:.1f}",
        )

    console.print(table)

    # Memory table
    mem_table = Table(title="Memory Usage")
    mem_table.add_column("Phase", style="cyan")
    mem_table.add_column("GPU Alloc", justify="right")
    mem_table.add_column("GPU Reserved", justify="right")
    mem_table.add_column("GPU Peak", justify="right")
    mem_table.add_column("RAM", justify="right")

    for label, snap in [
        ("Baseline", result.memory_baseline),
        ("Model Loaded", result.memory_loaded),
        ("Peak", result.memory_peak),
    ]:
        mem_table.add_row(
            label,
            f"{snap.gpu_allocated_gb:.2f} GB",
            f"{snap.gpu_reserved_gb:.2f} GB",
            f"{snap.gpu_peak_gb:.2f} GB",
            f"{snap.ram_used_gb:.2f} GB",
        )

    console.print(mem_table)
    console.print(
        f"\n[dim]Device: {result.device} | Dtype: {result.dtype} | "
        f"Total: {result.total_tokens_generated:,} tokens in {result.total_time_s:.1f}s "
        f"({result.overall_tokens_per_second} tok/s)[/dim]"
    )


# ── align (RLHF/DPO) ──────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.argument("dataset")
@click.option("--method", default="dpo", type=click.Choice(["dpo", "reward", "ppo"]),
              help="RLHF method")
@click.option("--reward-model", default=None, help="Reward model name (required for PPO)")
@click.option("--ref-model", default=None, help="Reference model for DPO KL divergence")
@click.option("--beta", default=0.1, type=float, help="DPO beta (temperature)")
@click.option("--epochs", default=1, type=int)
@click.option("--batch-size", "-b", default=2, type=int)
@click.option("--lr", default=5e-7, type=float, help="Learning rate")
@click.option("--max-seq-length", default=1024, type=int)
@click.option("--output", "-n", default=None, help="Output model name")
@click.option("--no-lora", is_flag=True, help="Disable LoRA (full fine-tune)")
@click.option("--loss-type", default="sigmoid", type=click.Choice(["sigmoid", "hinge", "ipo"]),
              help="DPO loss type")
def align(model, dataset, method, reward_model, ref_model, beta, epochs, batch_size,
          lr, max_seq_length, output, no_lora, loss_type):
    """Align a model using RLHF methods (DPO, reward model, PPO).

    DPO (Direct Preference Optimization):
        Train directly on preference pairs (chosen vs rejected).
        Dataset needs: prompt, chosen, rejected columns.

    Reward Model:
        Train a reward/scoring model from preference data.
        Can use preference pairs or text+label format.

    PPO (Proximal Policy Optimization):
        Classic RLHF — requires a trained reward model.
        Uses RL to maximize reward while staying close to reference.

    Examples:
        tracellm align llama3 Anthropic/hh-rlhf --method dpo
        tracellm align llama3 prefs.jsonl --method dpo --beta 0.2
        tracellm align llama3 Anthropic/hh-rlhf --method reward
        tracellm align llama3 prompts.jsonl --method ppo --reward-model llama3-reward
    """
    config = load_config()
    from tracellm.models.registry import ModelRegistry
    from tracellm.training.rlhf import RLHFManager

    registry = ModelRegistry(cache_dir=config.models.cache_dir)
    manager = RLHFManager(config, registry)

    method_labels = {"dpo": "DPO", "reward": "Reward Model", "ppo": "PPO"}
    console.print(f"[bold blue]RLHF Alignment[/bold blue] — {method_labels[method]}")
    console.print(f"  Model: {model} | Dataset: {dataset}")
    console.print(f"  Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")

    if method == "dpo":
        console.print(f"  Beta: {beta} | Loss: {loss_type} | LoRA: {not no_lora}")
        job_id = manager.start_dpo(
            model=model,
            dataset=dataset,
            ref_model=ref_model,
            beta=beta,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            max_seq_length=max_seq_length,
            output_name=output,
            use_lora=not no_lora,
            loss_type=loss_type,
        )
    elif method == "reward":
        job_id = manager.start_reward_model(
            model=model,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            max_seq_length=max_seq_length,
            output_name=output,
        )
    elif method == "ppo":
        if not reward_model:
            console.print("[red]PPO requires --reward-model[/red]")
            return
        console.print(f"  Reward model: {reward_model}")
        job_id = manager.start_ppo(
            model=model,
            reward_model=reward_model,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            max_seq_length=max_seq_length,
            output_name=output,
            use_lora=not no_lora,
        )

    console.print(f"\n  Job ID: [cyan]{job_id}[/cyan]\n")

    # Poll for progress
    try:
        while True:
            status = manager.get_status(job_id)
            if not status:
                break

            loss_str = f"{status['loss']:.4f}" if status["loss"] else "-"
            lr_str = f"{status['learning_rate']:.2e}" if status["learning_rate"] else "-"
            reward_str = ""
            if status["reward_accuracy"] is not None:
                reward_str = f" | reward_acc {status['reward_accuracy']:.2%}"
            elapsed = f"{status['elapsed_seconds']:.0f}s"

            console.print(
                f"\r  [{status['status']}] epoch {status['epoch']:.2f} | "
                f"loss {loss_str} | lr {lr_str}{reward_str} | {elapsed}",
                end="",
            )

            if status["status"] in ("completed", "failed"):
                console.print()
                if status["status"] == "completed":
                    console.print(f"\n[green]Alignment complete![/green] Model saved to: {status['output_path']}")
                else:
                    console.print(f"\n[red]Alignment failed:[/red] {status.get('error', 'unknown')}")
                break

            time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping alignment...[/yellow]")
        manager.stop(job_id)


# ── train ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.argument("dataset")
@click.option("--method", default="lora", type=click.Choice(["lora", "qlora", "full"]))
@click.option("--epochs", default=3, type=int)
@click.option("--batch-size", "-b", default=4, type=int)
@click.option("--lr", default=2e-4, type=float, help="Learning rate")
@click.option("--optimizer", "-o", default="adamw", type=click.Choice(["adamw", "muon", "deepspeed"]))
@click.option("--output", "-n", default=None, help="Output model name")
@click.option("--max-seq-length", default=2048, type=int)
@click.option("--code-path", multiple=True, help="Local code folders to include in training data")
def train(model, dataset, method, epochs, batch_size, lr, optimizer, output, max_seq_length, code_path):
    """Fine-tune a model on a dataset."""
    config = load_config()
    from tracellm.models.registry import ModelRegistry
    from tracellm.training.trainer import TrainingManager
    from tracellm.api.schemas import TrainRequest

    registry = ModelRegistry(cache_dir=config.models.cache_dir)
    manager = TrainingManager(config, registry)

    req = TrainRequest(
        model=model,
        dataset=dataset,
        method=method,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer=optimizer,
        output_name=output,
        max_seq_length=max_seq_length,
        code_paths=list(code_path) if code_path else None,
    )

    console.print(f"[bold blue]Training[/bold blue] '{model}' on '{dataset}'")
    console.print(f"  Method: {method} | Optimizer: {optimizer} | Epochs: {epochs} | LR: {lr}")

    job_id = manager.start_training(req)
    console.print(f"  Job ID: [cyan]{job_id}[/cyan]\n")

    # Poll for progress
    try:
        while True:
            status = manager.get_status(job_id)
            if not status:
                break

            loss_str = f"{status.loss:.4f}" if status.loss else "—"
            lr_str = f"{status.learning_rate:.2e}" if status.learning_rate else "—"
            elapsed = f"{status.elapsed_seconds:.0f}s"

            console.print(
                f"\r  [{status.status}] epoch {status.epoch:.2f} | "
                f"loss {loss_str} | lr {lr_str} | {elapsed}",
                end="",
            )

            if status.status in ("completed", "failed"):
                console.print()
                if status.status == "completed":
                    console.print(f"\n[green]Training complete![/green] Model saved to: {status.output_path}")
                else:
                    console.print(f"\n[red]Training failed.[/red]")
                break

            time.sleep(2)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping training...[/yellow]")
        manager.stop(job_id)


# ── remove ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.option("--delete-files", is_flag=True, help="Also delete model files from disk")
def remove(model, delete_files):
    """Remove a model from the registry."""
    config = load_config()
    from tracellm.models.registry import ModelRegistry

    registry = ModelRegistry(cache_dir=config.models.cache_dir)

    if delete_files:
        if not click.confirm(f"Delete model files for '{model}' from disk?"):
            return

    removed = registry.remove(model, delete_files=delete_files)
    if removed:
        console.print(f"[green]Removed[/green] '{model}'")
    else:
        console.print(f"[red]Model '{model}' not found[/red]")


# ── quantize ─────────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
@click.option("--method", default="gptq", type=click.Choice(["gptq", "awq", "gguf"]))
@click.option("--bits", default=4, type=int)
@click.option("--output", "-n", default=None, help="Output model name")
def quantize(model, method, bits, output):
    """Quantize a model to reduce size and increase speed."""
    config = load_config()
    from tracellm.models.registry import ModelRegistry
    from tracellm.models import quantize as quant_module

    registry = ModelRegistry(cache_dir=config.models.cache_dir)
    card = registry.get(model)
    if not card:
        console.print(f"[red]Model '{model}' not found[/red]")
        return

    output_name = output or f"{model}-{method}-{bits}bit"
    output_path = str(registry.cache_dir / output_name)

    console.print(f"Quantizing [bold]{model}[/bold] → {method} {bits}-bit...")

    with console.status("Quantizing..."):
        if method == "gptq":
            quant_module.quantize_gptq(card.path, output_path, bits=bits)
        elif method == "awq":
            quant_module.quantize_awq(card.path, output_path, bits=bits)
        elif method == "gguf":
            quant_module.export_gguf(card.path, output_path, quantization=f"q{bits}_k_m")

    registry.register(
        name=output_name,
        path=output_path,
        source=f"quantized:{model}",
        quantization=f"{method}-{bits}bit",
    )
    console.print(f"[green]Done![/green] Quantized model registered as '{output_name}'")


# ── search ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("query")
@click.option("--limit", "-l", default=10, type=int)
def search(query, limit):
    """Search HuggingFace Hub for models."""
    config = load_config()
    from tracellm.models.registry import ModelRegistry

    registry = ModelRegistry(cache_dir=config.models.cache_dir)

    with console.status("Searching HuggingFace Hub..."):
        results = registry.search_hub(query, limit=limit)

    if not results:
        console.print("[dim]No results found.[/dim]")
        return

    table = Table(title=f"HuggingFace Hub — '{query}'")
    table.add_column("Model ID", style="cyan", no_wrap=True)
    table.add_column("Downloads", justify="right", style="green")
    table.add_column("Likes", justify="right")
    table.add_column("Tags", style="dim")

    for r in results:
        table.add_row(
            r["id"],
            f"{r['downloads']:,}",
            str(r["likes"]),
            ", ".join(r["tags"][:3]),
        )

    console.print(table)


# ── info ─────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("model")
def info(model):
    """Show detailed info about a local model."""
    config = load_config()
    from tracellm.models.registry import ModelRegistry

    registry = ModelRegistry(cache_dir=config.models.cache_dir)
    card = registry.get(model)
    if not card:
        console.print(f"[red]Model '{model}' not found[/red]")
        return

    console.print(Panel(
        f"[bold]Name:[/bold]          {card.name}\n"
        f"[bold]Path:[/bold]          {card.path}\n"
        f"[bold]Source:[/bold]        {card.source}\n"
        f"[bold]Format:[/bold]        {card.format}\n"
        f"[bold]Architecture:[/bold]  {card.architecture}\n"
        f"[bold]Size:[/bold]          {card.size_gb:.2f} GB\n"
        f"[bold]Parameters:[/bold]    {card.parameters or '—'}\n"
        f"[bold]Quantization:[/bold]  {card.quantization or '—'}\n"
        f"[bold]Tags:[/bold]          {', '.join(card.tags) or '—'}",
        title=f"Model: {model}",
        border_style="blue",
    ))


# ── hardware ─────────────────────────────────────────────────────────────────

@main.command()
def hardware():
    """Show detected hardware information."""
    from tracellm.utils.hardware import detect_hardware
    from rich.panel import Panel

    info = detect_hardware()

    gpu_lines = ""
    if info.gpus:
        for gpu in info.gpus:
            gpu_lines += (
                f"  GPU {gpu.index}: {gpu.name}\n"
                f"    VRAM:    {gpu.memory_free_gb:.1f} / {gpu.memory_total_gb:.1f} GB\n"
                f"    Compute: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}\n"
            )
    elif info.mps_available:
        gpu_lines = "  Apple MPS (unified memory)\n"
    else:
        gpu_lines = "  [dim]No GPU detected — CPU-only mode[/dim]\n"

    console.print(Panel(
        f"[bold]Platform:[/bold]    {info.platform}\n"
        f"[bold]CPUs:[/bold]        {info.cpu_count}\n"
        f"[bold]RAM:[/bold]         {info.ram_available_gb:.1f} / {info.ram_total_gb:.1f} GB\n"
        f"[bold]GPUs:[/bold]\n{gpu_lines}"
        f"[bold]Best device:[/bold] {info.best_device}\n"
        f"[bold]Best dtype:[/bold]  {info.best_dtype}",
        title="Hardware",
        border_style="blue",
    ))


if __name__ == "__main__":
    main()
