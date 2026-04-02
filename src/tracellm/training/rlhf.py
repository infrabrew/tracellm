"""RLHF training — DPO, reward modeling, and PPO alignment."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)

from tracellm.config import TraceConfig
from tracellm.models.registry import ModelRegistry
from tracellm.utils.hardware import resolve_device, resolve_dtype
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.training.rlhf")


# ── Data types ──────────────────────────────────────────────────────────────


@dataclass
class RLHFJob:
    """Tracks the state of an RLHF training job."""
    job_id: str
    model_name: str
    method: str                         # "dpo" | "reward" | "ppo"
    status: str = "initializing"        # initializing | running | completed | failed | stopping
    epoch: float = 0.0
    loss: float | None = None
    reward_accuracy: float | None = None
    learning_rate: float | None = None
    started_at: float = field(default_factory=time.time)
    output_path: str | None = None
    error: str | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_flag: bool = field(default=False, repr=False)


# ── Dataset helpers ─────────────────────────────────────────────────────────


def load_preference_dataset(
    source: str,
    split: str = "train",
    prompt_field: str = "prompt",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
):
    """Load a preference dataset for DPO/reward model training.

    Expected format: each row has prompt, chosen response, rejected response.
    Supports HuggingFace datasets (e.g., 'Anthropic/hh-rlhf') or local JSONL.
    """
    from datasets import load_dataset

    if Path(source).exists():
        ext = Path(source).suffix.lower()
        if ext in (".jsonl", ".json"):
            ds = load_dataset("json", data_files=source, split=split)
        elif ext == ".csv":
            ds = load_dataset("csv", data_files=source, split=split)
        elif ext == ".parquet":
            ds = load_dataset("parquet", data_files=source, split=split)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        ds = load_dataset(source, split=split)

    # Validate required columns
    cols = ds.column_names
    for required in [prompt_field, chosen_field, rejected_field]:
        if required not in cols:
            # Try common alternative names
            alt_map = {
                "prompt": ["question", "input", "instruction", "context"],
                "chosen": ["preferred", "positive", "chosen_response", "winner"],
                "rejected": ["dispreferred", "negative", "rejected_response", "loser"],
            }
            found = False
            for alt in alt_map.get(required, []):
                if alt in cols:
                    ds = ds.rename_column(alt, required)
                    found = True
                    break
            if not found:
                raise ValueError(
                    f"Dataset missing '{required}' column. "
                    f"Available columns: {cols}"
                )

    return ds


def load_reward_dataset(
    source: str,
    split: str = "train",
    text_field: str = "text",
    label_field: str = "label",
):
    """Load a dataset for reward model training.

    Expected format: text + numeric label (reward score).
    """
    from datasets import load_dataset

    if Path(source).exists():
        ext = Path(source).suffix.lower()
        if ext in (".jsonl", ".json"):
            ds = load_dataset("json", data_files=source, split=split)
        elif ext == ".csv":
            ds = load_dataset("csv", data_files=source, split=split)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    else:
        ds = load_dataset(source, split=split)

    # If it's preference data, convert to reward format
    if "chosen" in ds.column_names and "rejected" in ds.column_names:
        prompt_col = "prompt" if "prompt" in ds.column_names else "instruction"

        def to_reward_pairs(examples):
            texts = []
            labels = []
            for i in range(len(examples[prompt_col])):
                prompt = examples[prompt_col][i]
                texts.append(f"{prompt}\n{examples['chosen'][i]}")
                labels.append(1.0)
                texts.append(f"{prompt}\n{examples['rejected'][i]}")
                labels.append(0.0)
            return {"text": texts, "label": labels}

        ds = ds.map(to_reward_pairs, batched=True, remove_columns=ds.column_names)
        text_field = "text"
        label_field = "label"

    return ds, text_field, label_field


# ── RLHF Manager ────────────────────────────────────────────────────────────


class RLHFManager:
    """Manages RLHF alignment training jobs (DPO, reward model, PPO)."""

    def __init__(self, config: TraceConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self._jobs: dict[str, RLHFJob] = {}

    def start_dpo(
        self,
        model: str,
        dataset: str,
        ref_model: str | None = None,
        beta: float = 0.1,
        epochs: int = 1,
        batch_size: int = 2,
        learning_rate: float = 5e-7,
        max_seq_length: int = 1024,
        max_prompt_length: int = 512,
        output_name: str | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        use_lora: bool = True,
        prompt_field: str = "prompt",
        chosen_field: str = "chosen",
        rejected_field: str = "rejected",
        loss_type: str = "sigmoid",
    ) -> str:
        """Start a DPO (Direct Preference Optimization) training job."""
        job_id = f"dpo-{uuid.uuid4().hex[:8]}"
        job = RLHFJob(job_id=job_id, model_name=model, method="dpo")
        self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_dpo,
            args=(job,),
            kwargs=dict(
                dataset=dataset,
                ref_model=ref_model,
                beta=beta,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                max_prompt_length=max_prompt_length,
                output_name=output_name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                use_lora=use_lora,
                prompt_field=prompt_field,
                chosen_field=chosen_field,
                rejected_field=rejected_field,
                loss_type=loss_type,
            ),
            daemon=True,
        )
        job._thread = thread
        thread.start()
        log.info(f"DPO job {job_id} started for model '{model}'")
        return job_id

    def start_reward_model(
        self,
        model: str,
        dataset: str,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        max_seq_length: int = 1024,
        output_name: str | None = None,
        num_labels: int = 1,
    ) -> str:
        """Start reward model training."""
        job_id = f"reward-{uuid.uuid4().hex[:8]}"
        job = RLHFJob(job_id=job_id, model_name=model, method="reward")
        self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_reward_model,
            args=(job,),
            kwargs=dict(
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                output_name=output_name,
                num_labels=num_labels,
            ),
            daemon=True,
        )
        job._thread = thread
        thread.start()
        log.info(f"Reward model job {job_id} started for model '{model}'")
        return job_id

    def start_ppo(
        self,
        model: str,
        reward_model: str,
        dataset: str,
        epochs: int = 1,
        batch_size: int = 4,
        learning_rate: float = 1e-6,
        max_seq_length: int = 1024,
        output_name: str | None = None,
        kl_penalty: float = 0.2,
        use_lora: bool = True,
        lora_r: int = 16,
    ) -> str:
        """Start PPO alignment training."""
        job_id = f"ppo-{uuid.uuid4().hex[:8]}"
        job = RLHFJob(job_id=job_id, model_name=model, method="ppo")
        self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_ppo,
            args=(job,),
            kwargs=dict(
                reward_model=reward_model,
                dataset=dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_seq_length=max_seq_length,
                output_name=output_name,
                kl_penalty=kl_penalty,
                use_lora=use_lora,
                lora_r=lora_r,
            ),
            daemon=True,
        )
        job._thread = thread
        thread.start()
        log.info(f"PPO job {job_id} started for model '{model}'")
        return job_id

    def get_status(self, job_id: str) -> dict | None:
        """Get current status of an RLHF job."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "method": job.method,
            "model": job.model_name,
            "status": job.status,
            "epoch": job.epoch,
            "loss": job.loss,
            "reward_accuracy": job.reward_accuracy,
            "learning_rate": job.learning_rate,
            "elapsed_seconds": round(time.time() - job.started_at, 1),
            "output_path": job.output_path,
            "error": job.error,
        }

    def stop(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job._stop_flag = True
            job.status = "stopping"

    def list_jobs(self) -> list[dict]:
        return [self.get_status(jid) for jid in self._jobs]

    # ── DPO implementation ──────────────────────────────────────────────────

    def _run_dpo(self, job: RLHFJob, **kwargs) -> None:
        try:
            job.status = "initializing"
            cfg = self.config.training

            card = self.registry.get(job.model_name)
            if not card:
                raise ValueError(f"Model '{job.model_name}' not found in registry")

            model_path = card.path
            device = resolve_device(self.config.models.default_device)
            dtype = resolve_dtype(self.config.models.default_dtype)

            output_name = kwargs.get("output_name") or f"{job.model_name}-dpo-{job.job_id}"
            output_dir = Path(cfg.output_dir).expanduser() / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            job.output_path = str(output_dir)

            log.info(f"[{job.job_id}] Loading model and tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load preference dataset
            log.info(f"[{job.job_id}] Loading preference dataset: {kwargs['dataset']}")
            dataset = load_preference_dataset(
                source=kwargs["dataset"],
                prompt_field=kwargs.get("prompt_field", "prompt"),
                chosen_field=kwargs.get("chosen_field", "chosen"),
                rejected_field=kwargs.get("rejected_field", "rejected"),
            )

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device if device == "auto" else {"": device},
                trust_remote_code=True,
            )

            # Reference model (for KL divergence)
            ref_model_path = kwargs.get("ref_model")
            ref_model = None
            if ref_model_path:
                ref_card = self.registry.get(ref_model_path)
                if ref_card:
                    ref_model = AutoModelForCausalLM.from_pretrained(
                        ref_card.path,
                        torch_dtype=dtype,
                        device_map=device if device == "auto" else {"": device},
                        trust_remote_code=True,
                    )

            # Apply LoRA if requested
            peft_config = None
            if kwargs.get("use_lora", True):
                from peft import LoraConfig, TaskType
                peft_config = LoraConfig(
                    r=kwargs.get("lora_r", 16),
                    lora_alpha=kwargs.get("lora_alpha", 32),
                    lora_dropout=0.05,
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=None,  # auto-detect
                    bias="none",
                )

            # DPO training via TRL
            try:
                from trl import DPOTrainer, DPOConfig
            except ImportError:
                raise ImportError(
                    "TRL library required for DPO training: pip install trl>=0.9.0"
                )

            training_args = DPOConfig(
                output_dir=str(output_dir),
                num_train_epochs=kwargs.get("epochs", 1),
                per_device_train_batch_size=kwargs.get("batch_size", 2),
                learning_rate=kwargs.get("learning_rate", 5e-7),
                beta=kwargs.get("beta", 0.1),
                loss_type=kwargs.get("loss_type", "sigmoid"),
                max_length=kwargs.get("max_seq_length", 1024),
                max_prompt_length=kwargs.get("max_prompt_length", 512),
                gradient_checkpointing=True,
                gradient_accumulation_steps=max(1, 16 // kwargs.get("batch_size", 2)),
                bf16=(cfg.mixed_precision == "bf16"),
                fp16=(cfg.mixed_precision == "fp16"),
                logging_steps=5,
                save_strategy="epoch",
                save_total_limit=2,
                seed=cfg.seed,
                report_to="none",
                remove_unused_columns=False,
            )

            callback = self._make_callback(job)

            job.status = "running"
            log.info(f"[{job.job_id}] Starting DPO training — beta={kwargs.get('beta', 0.1)}, "
                     f"epochs={kwargs.get('epochs', 1)}")

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
                callbacks=[callback],
            )

            trainer.train()

            # Save
            log.info(f"[{job.job_id}] Saving DPO model to {output_dir}")
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            self.registry.register(
                name=output_name,
                path=str(output_dir),
                source=f"dpo:{job.model_name}",
                tags=["dpo", "rlhf", "aligned"],
            )

            job.status = "completed"
            log.info(f"[{job.job_id}] DPO training completed successfully")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            log.error(f"[{job.job_id}] DPO training failed: {e}")

    # ── Reward model implementation ─────────────────────────────────────────

    def _run_reward_model(self, job: RLHFJob, **kwargs) -> None:
        try:
            job.status = "initializing"
            cfg = self.config.training

            card = self.registry.get(job.model_name)
            if not card:
                raise ValueError(f"Model '{job.model_name}' not found in registry")

            model_path = card.path
            device = resolve_device(self.config.models.default_device)
            dtype = resolve_dtype(self.config.models.default_dtype)

            output_name = kwargs.get("output_name") or f"{job.model_name}-reward-{job.job_id}"
            output_dir = Path(cfg.output_dir).expanduser() / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            job.output_path = str(output_dir)

            log.info(f"[{job.job_id}] Loading tokenizer from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load dataset
            log.info(f"[{job.job_id}] Loading reward dataset: {kwargs['dataset']}")
            dataset, text_field, label_field = load_reward_dataset(
                source=kwargs["dataset"],
            )

            # Tokenize
            def tokenize_fn(examples):
                tokens = tokenizer(
                    examples[text_field],
                    truncation=True,
                    max_length=kwargs.get("max_seq_length", 1024),
                    padding="max_length",
                )
                tokens["labels"] = [float(l) for l in examples[label_field]]
                return tokens

            tokenized = dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing reward data",
            )

            # Load as sequence classification model
            num_labels = kwargs.get("num_labels", 1)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=num_labels,
                torch_dtype=dtype,
                device_map=device if device == "auto" else {"": device},
                trust_remote_code=True,
            )

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=kwargs.get("epochs", 1),
                per_device_train_batch_size=kwargs.get("batch_size", 4),
                learning_rate=kwargs.get("learning_rate", 1e-5),
                gradient_checkpointing=True,
                bf16=(cfg.mixed_precision == "bf16"),
                fp16=(cfg.mixed_precision == "fp16"),
                logging_steps=10,
                save_strategy="epoch",
                save_total_limit=2,
                seed=cfg.seed,
                report_to="none",
                remove_unused_columns=False,
            )

            from transformers import Trainer, DataCollatorWithPadding

            callback = self._make_callback(job)

            job.status = "running"
            log.info(f"[{job.job_id}] Starting reward model training — "
                     f"epochs={kwargs.get('epochs', 1)}")

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized,
                tokenizer=tokenizer,
                callbacks=[callback],
            )

            trainer.train()

            log.info(f"[{job.job_id}] Saving reward model to {output_dir}")
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            self.registry.register(
                name=output_name,
                path=str(output_dir),
                source=f"reward:{job.model_name}",
                tags=["reward_model", "rlhf"],
            )

            job.status = "completed"
            log.info(f"[{job.job_id}] Reward model training completed")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            log.error(f"[{job.job_id}] Reward model training failed: {e}")

    # ── PPO implementation ──────────────────────────────────────────────────

    def _run_ppo(self, job: RLHFJob, **kwargs) -> None:
        try:
            job.status = "initializing"
            cfg = self.config.training

            card = self.registry.get(job.model_name)
            if not card:
                raise ValueError(f"Model '{job.model_name}' not found in registry")

            reward_card = self.registry.get(kwargs["reward_model"])
            if not reward_card:
                raise ValueError(
                    f"Reward model '{kwargs['reward_model']}' not found in registry"
                )

            model_path = card.path
            device = resolve_device(self.config.models.default_device)
            dtype = resolve_dtype(self.config.models.default_dtype)

            output_name = kwargs.get("output_name") or f"{job.model_name}-ppo-{job.job_id}"
            output_dir = Path(cfg.output_dir).expanduser() / output_name
            output_dir.mkdir(parents=True, exist_ok=True)
            job.output_path = str(output_dir)

            try:
                from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
            except ImportError:
                raise ImportError(
                    "TRL library required for PPO training: pip install trl>=0.9.0"
                )

            log.info(f"[{job.job_id}] Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load policy model with value head
            peft_config = None
            if kwargs.get("use_lora", True):
                from peft import LoraConfig, TaskType
                peft_config = LoraConfig(
                    r=kwargs.get("lora_r", 16),
                    lora_alpha=32,
                    lora_dropout=0.05,
                    task_type=TaskType.CAUSAL_LM,
                    bias="none",
                )

            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device if device == "auto" else {"": device},
                trust_remote_code=True,
                peft_config=peft_config,
            )

            # Load reward model
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_card.path,
                torch_dtype=dtype,
                device_map=device if device == "auto" else {"": device},
                trust_remote_code=True,
            )
            reward_tokenizer = AutoTokenizer.from_pretrained(
                reward_card.path, trust_remote_code=True
            )

            # Load prompts dataset
            from datasets import load_dataset
            if Path(kwargs["dataset"]).exists():
                ds = load_dataset("json", data_files=kwargs["dataset"], split="train")
            else:
                ds = load_dataset(kwargs["dataset"], split="train")

            # Extract prompt column
            prompt_col = None
            for col in ["prompt", "question", "input", "instruction"]:
                if col in ds.column_names:
                    prompt_col = col
                    break
            if not prompt_col:
                prompt_col = ds.column_names[0]

            # Tokenize prompts
            def tokenize_prompts(examples):
                return tokenizer(
                    examples[prompt_col],
                    truncation=True,
                    max_length=kwargs.get("max_seq_length", 1024) // 2,
                    padding=False,
                )

            ds = ds.map(tokenize_prompts, batched=True)

            # PPO config
            ppo_config = PPOConfig(
                learning_rate=kwargs.get("learning_rate", 1e-6),
                batch_size=kwargs.get("batch_size", 4),
                mini_batch_size=min(2, kwargs.get("batch_size", 4)),
                ppo_epochs=kwargs.get("epochs", 1),
                kl_penalty="kl",
                init_kl_coef=kwargs.get("kl_penalty", 0.2),
                log_with=None,
                seed=cfg.seed,
            )

            ppo_trainer = PPOTrainer(
                config=ppo_config,
                model=model,
                tokenizer=tokenizer,
            )

            job.status = "running"
            log.info(f"[{job.job_id}] Starting PPO training — "
                     f"kl_coef={kwargs.get('kl_penalty', 0.2)}")

            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_seq_length", 1024) // 2,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }

            total_steps = 0
            for epoch in range(kwargs.get("epochs", 1)):
                if job._stop_flag:
                    break

                for batch_idx in range(0, len(ds), kwargs.get("batch_size", 4)):
                    if job._stop_flag:
                        break

                    batch_end = min(batch_idx + kwargs.get("batch_size", 4), len(ds))
                    batch = ds[batch_idx:batch_end]

                    # Encode prompts
                    query_tensors = [
                        torch.tensor(ids).to(model.pretrained_model.device)
                        for ids in batch["input_ids"]
                    ]

                    # Generate responses
                    response_tensors = ppo_trainer.generate(
                        query_tensors, **gen_kwargs
                    )

                    # Compute rewards
                    rewards = []
                    for q, r in zip(query_tensors, response_tensors):
                        full_text = tokenizer.decode(
                            torch.cat([q, r]), skip_special_tokens=True
                        )
                        reward_inputs = reward_tokenizer(
                            full_text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=kwargs.get("max_seq_length", 1024),
                        ).to(reward_model.device)

                        with torch.no_grad():
                            reward_output = reward_model(**reward_inputs)
                            score = reward_output.logits[0, 0].float()
                        rewards.append(score)

                    # PPO step
                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

                    total_steps += 1
                    job.epoch = epoch + (batch_idx / len(ds))
                    job.loss = stats.get("ppo/loss/total", job.loss)
                    job.reward_accuracy = (
                        sum(r.item() > 0 for r in rewards) / len(rewards)
                        if rewards else None
                    )
                    job.learning_rate = kwargs.get("learning_rate", 1e-6)

            # Save model
            log.info(f"[{job.job_id}] Saving PPO model to {output_dir}")
            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            self.registry.register(
                name=output_name,
                path=str(output_dir),
                source=f"ppo:{job.model_name}",
                tags=["ppo", "rlhf", "aligned"],
            )

            job.status = "completed"
            log.info(f"[{job.job_id}] PPO training completed")

        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            log.error(f"[{job.job_id}] PPO training failed: {e}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _make_callback(self, job: RLHFJob) -> TrainerCallback:
        """Create a TrainerCallback that updates job state."""

        class _RLHFCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    job.loss = logs.get("loss", job.loss)
                    job.learning_rate = logs.get("learning_rate", job.learning_rate)
                    job.epoch = state.epoch or 0.0
                    if "eval_accuracy" in logs:
                        job.reward_accuracy = logs["eval_accuracy"]

            def on_step_end(self, args, state, control, **kwargs):
                if job._stop_flag:
                    control.should_training_stop = True

        return _RLHFCallback()
