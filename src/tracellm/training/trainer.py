"""Training orchestrator — manages fine-tuning jobs with LoRA/QLoRA/full."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from tracellm.api.schemas import TrainRequest, TrainStatusResponse
from tracellm.config import TraceConfig
from tracellm.models.registry import ModelRegistry
from tracellm.training.datasets import prepare_dataset
from tracellm.training.optimizers import (
    create_optimizer,
    apply_flashinfer_attention,
    build_deepspeed_config,
)
from tracellm.utils.hardware import resolve_device, resolve_dtype
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.training.trainer")


@dataclass
class TrainingJob:
    job_id: str
    model_name: str
    status: str = "initializing"     # initializing | running | completed | failed | stopping
    epoch: float = 0.0
    loss: float | None = None
    learning_rate: float | None = None
    started_at: float = field(default_factory=time.time)
    output_path: str | None = None
    error: str | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_flag: bool = field(default=False, repr=False)


class TrainingManager:
    """Manages concurrent training/fine-tuning jobs."""

    def __init__(self, config: TraceConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self._jobs: dict[str, TrainingJob] = {}

    def start_training(self, req: TrainRequest) -> str:
        """Launch a training job in a background thread."""
        job_id = f"train-{uuid.uuid4().hex[:8]}"
        job = TrainingJob(job_id=job_id, model_name=req.model)
        self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_training,
            args=(job, req),
            daemon=True,
        )
        job._thread = thread
        thread.start()

        log.info(f"Training job {job_id} started for model '{req.model}'")
        return job_id

    def get_status(self, job_id: str) -> TrainStatusResponse | None:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return TrainStatusResponse(
            job_id=job.job_id,
            status=job.status,
            epoch=job.epoch,
            loss=job.loss,
            learning_rate=job.learning_rate,
            elapsed_seconds=round(time.time() - job.started_at, 1),
            output_path=job.output_path,
        )

    def stop(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job._stop_flag = True
            job.status = "stopping"

    def _run_training(self, job: TrainingJob, req: TrainRequest) -> None:
        """Execute the training pipeline."""
        try:
            job.status = "initializing"
            cfg = self.config.training

            # Resolve model path
            card = self.registry.get(req.model)
            if not card:
                raise ValueError(f"Model '{req.model}' not found in registry")

            model_path = card.path
            device = resolve_device(self.config.models.default_device)
            dtype = resolve_dtype(self.config.models.default_dtype)

            # Output directory
            output_name = req.output_name or f"{req.model}-ft-{job.job_id}"
            output_dir = Path(cfg.output_dir).expanduser() / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            job.output_path = str(output_dir)

            log.info(f"[{job.job_id}] Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load dataset
            log.info(f"[{job.job_id}] Preparing dataset: {req.dataset}")
            dataset = prepare_dataset(
                source=req.dataset,
                text_field=req.dataset_text_field,
                split=req.dataset_split,
                code_paths=req.code_paths,
            )

            # Tokenize
            def tokenize_fn(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=req.max_seq_length,
                    padding="max_length",
                )

            tokenized = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing",
            )

            # Load model based on method
            log.info(f"[{job.job_id}] Loading model ({req.method})")
            model = self._load_model_for_training(
                model_path, req.method, device, dtype
            )

            # Apply FlashInfer if available
            if device == "cuda":
                model = apply_flashinfer_attention(model)

            # Gradient checkpointing
            if req.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()

            # Build training arguments
            training_args = self._build_training_args(req, cfg, output_dir, job)

            # DeepSpeed config
            ds_config = None
            if req.optimizer == "deepspeed":
                ds_config = build_deepspeed_config(
                    stage=2,
                    batch_size=req.batch_size,
                    lr=req.learning_rate,
                    bf16=(cfg.mixed_precision == "bf16"),
                    fp16=(cfg.mixed_precision == "fp16"),
                )
                training_args.deepspeed = ds_config

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

            # Custom optimizer (Muon, etc.) — only if not using DeepSpeed
            optimizers = (None, None)
            if req.optimizer not in ("deepspeed", "adamw") and req.optimizer != "":
                optimizer = create_optimizer(
                    name=req.optimizer,
                    model_params=model.parameters(),
                    lr=req.learning_rate,
                    weight_decay=0.01,
                )
                optimizers = (optimizer, None)

            # Custom callback for progress tracking
            class ProgressCallback(torch.nn.Module):
                """Not a real module — just a training callback interface."""
                pass

            from transformers import TrainerCallback

            class TraceCallback(TrainerCallback):
                def __init__(self, training_job):
                    self.job = training_job

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        self.job.loss = logs.get("loss", self.job.loss)
                        self.job.learning_rate = logs.get("learning_rate", self.job.learning_rate)
                        self.job.epoch = state.epoch or 0.0

                def on_step_end(self, args, state, control, **kwargs):
                    if self.job._stop_flag:
                        control.should_training_stop = True

            # Train
            job.status = "running"
            log.info(f"[{job.job_id}] Starting training — {req.epochs} epochs, "
                      f"bs={req.batch_size}, lr={req.learning_rate}, opt={req.optimizer}")

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
                data_collator=data_collator,
                tokenizer=tokenizer,
                optimizers=optimizers,
                callbacks=[TraceCallback(job)],
            )

            trainer.train()

            # Save final model
            log.info(f"[{job.job_id}] Saving model to {output_dir}")
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            # Register the fine-tuned model
            self.registry.register(
                name=output_name,
                path=str(output_dir),
                source=f"finetune:{req.model}",
                tags=["finetuned", req.method],
            )

            job.status = "completed"
            log.info(f"[{job.job_id}] Training completed successfully")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            log.error(f"[{job.job_id}] Training failed: {e}")

    def _load_model_for_training(
        self,
        model_path: str,
        method: str,
        device: str,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        """Load model with appropriate fine-tuning setup."""

        if method == "qlora":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError("bitsandbytes required for QLoRA: pip install 'tracellm[quantize]'")

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.training.finetune_defaults.qlora.quant_type,
                bnb_4bit_use_double_quant=self.config.training.finetune_defaults.qlora.double_quant,
                bnb_4bit_compute_dtype=dtype,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device if device == "auto" else {"": device},
                trust_remote_code=True,
            )

        # Apply LoRA / QLoRA adapter
        if method in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

            if method == "qlora":
                model = prepare_model_for_kbit_training(model)

            lora_cfg = self.config.training.finetune_defaults.lora
            target_modules = None
            if lora_cfg.target_modules != "auto":
                target_modules = lora_cfg.target_modules.split(",")

            peft_config = LoraConfig(
                r=lora_cfg.r,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
            )
            model = get_peft_model(model, peft_config)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            log.info(f"LoRA trainable params: {trainable:,} / {total:,} "
                      f"({100 * trainable / total:.2f}%)")

        return model

    def _build_training_args(
        self,
        req: TrainRequest,
        cfg: Any,
        output_dir: Path,
        job: TrainingJob,
    ) -> TrainingArguments:
        """Build HuggingFace TrainingArguments."""
        mixed_precision = cfg.mixed_precision

        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=req.epochs,
            per_device_train_batch_size=req.batch_size,
            learning_rate=req.learning_rate,
            lr_scheduler_type=req.scheduler,
            warmup_ratio=0.05,
            max_grad_norm=cfg.max_grad_norm,
            gradient_checkpointing=req.gradient_checkpointing,
            gradient_accumulation_steps=max(1, 32 // req.batch_size),
            fp16=(mixed_precision == "fp16"),
            bf16=(mixed_precision == "bf16"),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            seed=cfg.seed,
            dataloader_num_workers=cfg.dataloader_workers,
            report_to="none",
            remove_unused_columns=False,
        )
