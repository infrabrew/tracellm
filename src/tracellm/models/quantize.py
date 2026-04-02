"""Model quantization utilities — GPTQ, AWQ, bitsandbytes, GGUF export."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from tracellm.models.registry import ModelCard
from tracellm.utils.logging import get_logger

log = get_logger("tracellm.models.quantize")


def quantize_gptq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    dataset: str = "c4",
) -> Path:
    """Quantize a model using GPTQ."""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install auto-gptq: pip install 'tracellm[quantize]'")

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"GPTQ quantization: {bits}-bit, group_size={group_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    quant_config = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=False)

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quant_config)
    model.quantize(tokenizer, dataset=dataset)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))

    log.info(f"GPTQ quantized model saved to {out}")
    return out


def quantize_awq(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
) -> Path:
    """Quantize a model using AWQ."""
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install autoawq: pip install 'tracellm[quantize]'")

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"AWQ quantization: {bits}-bit, group_size={group_size}")

    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.quantize(
        tokenizer,
        quant_config={"zero_point": True, "q_group_size": group_size, "w_bit": bits},
    )
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))

    log.info(f"AWQ quantized model saved to {out}")
    return out


def export_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
) -> Path:
    """Export a model to GGUF format for llama.cpp inference.

    Requires llama.cpp's convert scripts to be available.
    """
    import subprocess

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    output_file = out / f"model-{quantization}.gguf"

    log.info(f"GGUF export: {quantization}")

    # Step 1: Convert to fp16 GGUF
    fp16_file = out / "model-f16.gguf"
    subprocess.run(
        ["python", "-m", "llama_cpp.convert", model_path, "--outfile", str(fp16_file)],
        check=True,
    )

    # Step 2: Quantize
    if quantization != "f16":
        subprocess.run(
            ["python", "-m", "llama_cpp.quantize", str(fp16_file), str(output_file), quantization],
            check=True,
        )
        fp16_file.unlink(missing_ok=True)
    else:
        output_file = fp16_file

    log.info(f"GGUF model saved to {output_file}")
    return output_file
