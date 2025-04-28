# qwen_omni.py
"""
Run Qwen2.5-Omni-7B on Modal ☁️🚀

Usage examples
--------------
Simple text Q&A (no audio output):
    modal run qwen_omni.py --ask "Describe quantum entanglement in one tweet"

Multimodal (video URL) with spoken answer in Ethan's voice:
    modal run qwen_omni.py \
        --conversation examples/video_prompt.json \
        --speaker Ethan \
        --return-audio

Inside another Modal script:
    from modal import Stub
    stub = Stub.from_name("qwen-omni-runner", namespace="your-username")
    result = stub.generate.remote(convo_json, use_audio_in_video=True)
    print(result["text"])
"""

import base64
import json
import os
import tempfile
from pathlib import Path

import modal
# Insert global caching variables after imports
# Global model and processor to persist across invocations
_GLOBAL_MODEL = None
_GLOBAL_PROCESSOR = None


# ---------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------
APP_NAME = "qwen-omni-runner"
MODEL_NAME = "Qwen/Qwen2.5-Omni-7B"
GPU_TYPE = "A100-40GB"                        # needs ~40 GB for full-featured Omni
CACHE_DIR = modal.Volume.from_name("qwen-omni-cache", create_if_missing=True)  # persisted volume between runs
# Need PyTorch 2.6+ for security fix (CVE-2025-32434)
PYTORCH_VERSION = "2.6.0"
TRANSFORMERS_COMMIT = "main"                  # build with Omni patches
FLASH_ATTN_VERSION = "2.7.4.post1"  # upgrade to latest prebuilt Flash-Attn wheel

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------
# IMAGE DEFINITION
# ---------------------------------------------------------------------
# Use NVIDIA's CUDA image as base to ensure proper CUDA toolkit installation
cuda_version = "12.8.0"  # Should be ≤ host-driver's CUDA version
flavor = "devel"         # Includes full CUDA toolkit needed for flash-attention
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    # Ensure unbuffered output
    .env({
        "PYTHONUNBUFFERED": "1"
    })
    # Install ffmpeg for audio/video, git and clang for building native extensions
    .apt_install("ffmpeg", "git", "clang")
    # Install build helpers and PyTorch
    .pip_install(
        "ninja",
        "packaging",
        "wheel",
        f"torch=={PYTORCH_VERSION}",
        "torchvision",
        "torchaudio",
    )
    # Install Flash-Attention wheel (prebuilt) from PyPI
    .pip_install(
        f"flash-attn=={FLASH_ATTN_VERSION}",
        extra_options="--no-build-isolation"
    )
    # Install transformers and other runtime dependencies
    .pip_install(
        f"git+https://github.com/huggingface/transformers@{TRANSFORMERS_COMMIT}",
        "accelerate",
        "qwen-omni-utils[decord]",
        "soundfile",
    )
)

# ---------------------------------------------------------------------
# MODEL PRE-DOWNLOAD STEP
# ---------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/cache": CACHE_DIR}
)
def download_weights():
    """Downloads model & processor into persistent volume between runs."""
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    import torch

    os.makedirs("/cache", exist_ok=True)
    print("🔻  Downloading model and processor…")
    print(f"Using PyTorch version: {torch.__version__}")
    
    # Use safetensors format when available to avoid torch.load security issues
    Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        cache_dir="/cache", 
        torch_dtype="auto",
        use_safetensors=True
    )
    Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME, cache_dir="/cache")
    CACHE_DIR.commit()  # Ensure changes are persisted
    print("✅  Pre-download complete")


# ---------------------------------------------------------------------
# TORCH TEST FUNCTION
# ---------------------------------------------------------------------
@app.function(gpu="any", image=image)
def run_torch():
    import torch
    has_cuda = torch.cuda.is_available()
    print(f"It is {has_cuda} that torch can access CUDA")
    
    if has_cuda:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"CUDA Current Device: {torch.cuda.current_device()}")
    
    return has_cuda

# ---------------------------------------------------------------------
# WHISPER TRANSCRIPTION FUNCTION
# ---------------------------------------------------------------------
@app.function(gpu="any", image=image)
def run_whisper():
    from transformers import pipeline
    transcriber = pipeline(model="openai/whisper-tiny.en", device="cuda")
    result = transcriber("https://modal-cdn.com/mlk.flac")
    print(result["text"])
    return result["text"]
 
# ---------------------------------------------------------------------
# FLASH-ATTENTION TEST FUNCTION
# ---------------------------------------------------------------------
@app.function(gpu=GPU_TYPE, image=image)
def run_flash_attn_test():
    """Sanity-check Flash-Attention integration by running a small attention kernel."""
    import torch
    from flash_attn import flash_attn_func

    B, S, H, D = 2, 4, 3, 16
    q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

    out = flash_attn_func(q, k, v)
    return {"output_shape": list(out.shape)}

# ---------------------------------------------------------------------
# RUNTIME FUNCTION
# ---------------------------------------------------------------------
@app.function(
    gpu=GPU_TYPE,
    image=image,
    volumes={"/cache": CACHE_DIR},
    timeout=60 * 30,          # 30 min so you can run long video prompts
    max_containers=1,         # Omni model is heavy – keep one per GPU (renamed from concurrency_limit)
    retries=2                 # Retry on failure
)
def generate(
    conversation: list[dict],
    *,
    use_audio_in_video: bool = True,
    speaker: str | None = None,          # "Chelsie" (default) or "Ethan"
    return_audio: bool = False,
):
    """
    Run the model on an arbitrary multimodal *conversation* (same schema as HF Omni docs).

    Parameters
    ----------
    conversation : list[dict]
        List of {"role": ..., "content": [...]} messages following Qwen chat template.
    use_audio_in_video : bool
        Whether to let the model "hear" embedded audio tracks inside user-supplied videos.
    speaker : str | None
        Optional voice id. If None -> default "Chelsie". Ignored when `return_audio` is False.
    return_audio : bool
        If True, returns a base64-encoded WAV in the JSON result.

    Returns
    -------
    dict
        {"text": str}  or  {"text": str, "audio_b64": str}
    """
    import soundfile as sf
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    # ------------------------- LOAD MODEL + PROCESSOR -------------------------
    # Reload the volume to ensure we have the latest model weights
    CACHE_DIR.reload()
    
    print(f"Using PyTorch version: {torch.__version__}")

    # Use global model and processor to avoid reloading weights on each invocation
    global _GLOBAL_MODEL, _GLOBAL_PROCESSOR
    if _GLOBAL_MODEL is None or _GLOBAL_PROCESSOR is None:
        print("Loading model and processor into memory...")
        try:
            _GLOBAL_MODEL = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,      # ↳ bfloat16 keeps memory low, FA2 supported
                device_map="auto",
                attn_implementation="flash_attention_2",
                cache_dir="/cache",
                use_safetensors=True             # Use safetensors format when available
            )
            _GLOBAL_PROCESSOR = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME, cache_dir="/cache")
        except Exception as e:
            print(f"Error loading model from cache: {e}")
            print("Falling back to direct load of model and processor...")
            _GLOBAL_MODEL = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                use_safetensors=True             # Use safetensors format when available
            )
            _GLOBAL_PROCESSOR = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME)
    else:
        print("Using cached model and processor")

    model = _GLOBAL_MODEL
    processor = _GLOBAL_PROCESSOR

    # ------------------------- PREPARE INPUTS ---------------------------------
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    ).to(model.device).to(model.dtype)

    # ------------------------- GENERATE ---------------------------------------
    if return_audio:
        text_ids, audio = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            speaker=speaker,
        )
    else:
        text_ids = model.generate(
            **inputs,
            use_audio_in_video=use_audio_in_video,
            speaker=speaker,
            return_audio=False,
        )
        audio = None

    out_text: str = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # ------------------------- PACKAGE RESULT ---------------------------------
    if audio is not None:
        # serialize WAV → base64 so result stays JSON-serializable
        wav_path = Path(tempfile.mkstemp(suffix=".wav")[1])
        sf.write(
            wav_path,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24_000,
        )
        audio_b64 = base64.b64encode(wav_path.read_bytes()).decode()
        wav_path.unlink(missing_ok=True)
        return {"text": out_text, "audio_b64": audio_b64}

    return {"text": out_text}


# ---------------------------------------------------------------------
# OPTIONAL LOCAL CLI  (modal run qwen_omni.py …)
# ---------------------------------------------------------------------
@app.local_entrypoint()
def main(ask: str = None, conversation: str = None, speaker: str = None, 
         return_audio: bool = False, use_audio_in_video: bool = False):
    """
    CLI entrypoint for running the Qwen model.
    
    Parameters:
    -----------
    ask : str, optional
        Quick single-question prompt
    conversation : str, optional
        Path to JSON file with full conversation payload
    speaker : str, optional
        Voice type ("Chelsie" or "Ethan")
    return_audio : bool, default=False
        Whether to return synthesized audio
    use_audio_in_video : bool, default=False
        Whether to use audio in videos
    """
    import argparse
    from datetime import datetime

    # If called via CLI with no parameters, parse arguments
    if ask is None and conversation is None:
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--ask", type=str, help="Quick single-question prompt")
        group.add_argument(
            "--conversation",
            type=str,
            help="Path to JSON file with full conversation payload",
        )
        parser.add_argument(
            "--speaker", type=str, choices=["Chelsie", "Ethan"], default=None
        )
        parser.add_argument("--return-audio", action="store_true")
        parser.add_argument("--use-audio-in-video", action="store_true", default=False)

        args = parser.parse_args()
        ask = args.ask
        conversation = args.conversation
        speaker = args.speaker
        return_audio = args.return_audio
        use_audio_in_video = args.use_audio_in_video

    if ask:
        convo = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": ask}]},
        ]
    else:
        with open(conversation) as f:
            convo = json.load(f)

    print(f"⏳  Submitting job at {datetime.now().isoformat(timespec='seconds')}")
    
    try:
        # Download the model weights to the persistent volume first
        print("Downloading model weights to persistent volume...")
        download_weights.remote()
        print("Model weights downloaded successfully")
        
        # Generate the response
        result = generate.remote(
            convo,
            use_audio_in_video=use_audio_in_video,
            speaker=speaker,
            return_audio=return_audio,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    print("📝  TEXT RESPONSE\n" + "-" * 72)
    print(result["text"])

    if return_audio:
        out_wav = "qwen_output.wav"
        with open(out_wav, "wb") as f:
            f.write(base64.b64decode(result["audio_b64"]))
        print(f"\n🔊  Saved synthesized speech to {out_wav}")
