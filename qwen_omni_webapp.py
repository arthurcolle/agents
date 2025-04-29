"""qwen_omni_webapp.py

Light-weight Modal web application that lets you **talk** to the multimodal
`qwen_omni.py` runner.  It exposes a small FastAPI HTTP API together with a
vanilla-JS chat UI so you can speak into your microphone and have Qwen reply
with both text *and* synthesized speech.

How to run locally (with hot-reload on file edits):

    modal serve qwen_omni_webapp.py

Deploy permanently to a public URL (requires a paid Modal plan):

    modal deploy qwen_omni_webapp.py --name qwen-omni-web

Once the server is up, open the displayed URL in a browser and start
chatting!  When you click the mic button the browser records a short audio
clip (Opus/WEBM).  The backend transcodes it to WAV, feeds the transcribed
text to the *qwen-omni-runner* GPU endpoint, and finally streams the textual
answer together with a Base64-encoded WAV that will automatically play back
in the page.

The code purposefully keeps state in the browser – every request contains the
full conversation history so the backend stays completely stateless.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
from pathlib import Path
from typing import List

import modal
# NOTE: The `Stub` helper class was renamed to `App` in recent versions of
# Modal. Import the new name directly to silence the deprecation warning and
# use the modern helper methods (`lookup`, etc.) when referencing a remote
# application.
from modal import App
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------------------------------
# Modal plumbing
# ---------------------------------------------------------------------------

# The tiny FastAPI image is good enough for a lightweight web head; we call
# the heavy model from a separate GPU app so we don't need CUDA here.
image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi[standard]",  # fastapi + uvicorn + h11 etc.
        "jinja2",
        "soundfile",         # used to transcode recorded audio to WAV
        "ffmpeg-python",     # thin wrapper – requires ffmpeg binary, see apt
        "numpy",
    )
    .apt_install("ffmpeg")
)

# Create volumes for templates and static files
template_volume = modal.Volume.from_name("qwen-omni-templates", create_if_missing=True)
static_volume = modal.Volume.from_name("qwen-omni-static", create_if_missing=True)

# Upload local template files to the volume if they exist
templates_dir = Path(__file__).parent / "templates"
if templates_dir.exists():
    # Use force=True to overwrite existing files
    with template_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(templates_dir), "/")

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)

# Upload static files to the volume if directory exists
if static_dir.exists():
    # Use force=True to overwrite existing files
    with static_volume.batch_upload(force=True) as batch:
        batch.put_directory(str(static_dir), "/")

# ---------------------------------------------------------------------------
# Reference the already-deployed GPU runner.
#
# Using `App.lookup` ensures we attach to the existing Modal application
# (created in `qwen_omni.py`) **with all its remote functions bound**, so
# calling `qwen_app.generate.remote(...)` works as expected.
# ---------------------------------------------------------------------------

qwen_app = App.lookup("qwen-omni-runner")
# Defer function lookup until inside the endpoint to avoid InvalidError in local dev

# This app only handles HTTP so CPU is fine.
app = modal.App("qwen-omni-web", image=image, volumes={
    "/app/templates": template_volume,
    "/app/static": static_volume
})


# ---------------------------------------------------------------------------
# FastAPI set-up
# ---------------------------------------------------------------------------

web_app = FastAPI(title="Qwen-Omni Voice Chat")

# Set the templates and static directories to the local folders
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# Mount static files for FastAPI
web_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory="/app/templates")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _webm_to_wav_bytes(webm_bytes: bytes) -> bytes:
    """Convert the WebM/Opus blob coming from the browser into 24 kHz WAV.

    FFmpeg is available inside the image, so we spin up a subprocess.  We keep
    everything in-memory to avoid writing large files to disk.
    """

    import subprocess, shlex

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.webm"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(webm_bytes)

        cmd = f"ffmpeg -hide_banner -loglevel error -y -i {src} -ar 24000 -ac 1 {dst}"
        subprocess.check_call(shlex.split(cmd))
        return dst.read_bytes()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


from fastapi import Request


@web_app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the single-page chat UI."""
    logger.info("Home page requested: %s", request.url.path)
    # Serve the new unified chat UI
    return templates.TemplateResponse("index.html", {"request": request, "assistant_voice": "Chelsie"})


@web_app.post("/api/chat")
def chat_endpoint(
    # One of these must be provided by the browser ↓
    user_text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    # The entire conversation so far (list[dict] serialised as JSON str)
    conversation_json: str = Form(...),
    # Optional voice id for TTS (Chelsie default)
    speaker: str | None = Form(None),
    # Output mode: "text", "audio", "both"
    output_mode: str = Form("both"),
):
    """Main inference endpoint.

    The **browser** keeps the chat history and sends it along every request so
    the backend is entirely stateless.  We simply append the new user
    message, call the GPU runner, and ship the answer back.
    """

    import json
    import soundfile as sf

    # Log request details
    logger.info(
        "Chat request received: user_text=%r, audio_provided=%s, conversation_json length=%d, speaker=%r",
        user_text,
        bool(audio),
        len(conversation_json),
        speaker,
    )

    # ------------------------------------------------------------------
    # Reconstruct conversation list from JSON string sent by the client.
    # ------------------------------------------------------------------
    try:
        conversation: List[dict] = json.loads(conversation_json)
    except json.JSONDecodeError:
        logger.error("Failed to parse conversation JSON: %s", conversation_json)
        return JSONResponse(status_code=400, content={"error": "Invalid conversation JSON"})
    # Log conversation length
    logger.info("Parsed conversation: %d messages", len(conversation))

    # ------------------------------------------------------------------
    # Figure out what the user sent (plain text, audio, or image)
    # ------------------------------------------------------------------
    if user_text is None and audio is None and image is None:
        return JSONResponse(status_code=400, content={"error": "Provide either text, audio, or image"})

    content = []
    if audio is not None:
        # FastAPI gives us a SpooledTemporaryFile; read raw bytes.
        webm_bytes = audio.file.read()
        logger.info("Audio uploaded: bytes=%d, filename=%s", len(webm_bytes), getattr(audio, "filename", None))
        try:
            wav_bytes = _webm_to_wav_bytes(webm_bytes)
        except Exception as e:
            logger.exception("Audio conversion failed")
            return JSONResponse(status_code=500, content={"error": f"Audio conversion failed: {e}"})
        # Log successful audio conversion
        logger.info("Audio conversion to WAV successful: bytes=%d", len(wav_bytes))

        # Save the WAV to a temporary file and use the file path in the conversation
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(wav_bytes)
            tmp_wav_path = tmp_wav.name
        audio_item = {"type": "audio", "audio": tmp_wav_path}
        content.append(audio_item)
    if image is not None:
        # Read image bytes and save to a temporary file, use the file path in the conversation
        image_bytes = image.file.read()
        logger.info("Image uploaded: bytes=%d, filename=%s", len(image_bytes), getattr(image, "filename", None))
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            tmp_img.write(image_bytes)
            tmp_img.flush()
            tmp_img_path = tmp_img.name
        # Ensure the file is flushed and exists before passing to model
        if Path(tmp_img_path).exists():
            image_item = {"type": "image", "image": tmp_img_path}
            content.append(image_item)
        else:
            logger.error("Temporary image file not found after write: %s", tmp_img_path)
    if user_text is not None:
        logger.info("User text provided: %s", user_text)
        content.append({"type": "text", "text": user_text})
    if not content:
        return JSONResponse(status_code=400, content={"error": "No valid content provided"})

    # ------------------------------------------------------------------
    # Append user message to history and call the GPU function.
    # ------------------------------------------------------------------
    conversation.append({"role": "user", "content": content})
    # Log before invoking model
    logger.info("Invoking GPU runner with conversation length=%d", len(conversation))

    # Determine system prompt based on task (from frontend)
    # Default: "chat"
    task = None
    sys_prompt = None
    if len(conversation) > 0 and conversation[0]["role"] == "system":
        sys_prompt = conversation[0]["content"][0].get("text", "")
        # Try to infer task from system prompt
        if "speech recognition" in sys_prompt.lower():
            task = "asr"
        elif "translation" in sys_prompt.lower():
            task = "translation"
        elif "classification" in sys_prompt.lower():
            task = "classification"
        else:
            task = "chat"
    else:
        task = "chat"

    # Output mode logic
    return_audio = output_mode in ("audio", "both")
    return_text = output_mode in ("text", "both")

    # Use Modal's new explicit handle-lookup helper to get the remote function.
    import modal
    qwen_generate = modal.Function.from_name(
        "qwen-omni-runner",          # the App name you passed to modal.App(...)
        "generate"                   # the function’s decorator name
    )

    try:
        # Call the remote function with additional parameters required by the API
        # Add use_audio_in_video parameter to match the API requirements
        result = qwen_generate.remote(
            conversation,
            speaker=speaker,
            return_audio=return_audio,
            use_audio_in_video=True,
        )
    except Exception as e:
        logger.exception("Model invocation failed")
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(
            status_code=500, 
            content={
                "error": f"Model invocation failed: {e}",
                "details": error_details
            }
        )

    # The GPU app already returns base64 encoded WAV so we can forward as-is.
    logger.info("Model result: text length=%d, audio present=%s",
                len(result.get("text", "")), bool(result.get("audio_b64")))
    assistant_message = {
        "role": "assistant",
        "content": [],
    }
    if return_text and result.get("text"):
        assistant_message["content"].append({"type": "text", "text": result["text"]})
    if return_audio and result.get("audio_b64"):
        assistant_message["content"].append({"type": "audio", "audio": "(omitted)"})
    conversation.append(assistant_message)

    # Don't send huge waveforms or file paths back to the browser – replace them with a stub.
    def _strip_audio(payload: List[dict]):
        cleaned: List[dict] = []
        for msg in payload:
            new_msg = {"role": msg["role"], "content": []}
            for item in msg["content"]:
                if item["type"] == "audio":
                    new_msg["content"].append({"type": "audio", "audio": "(omitted)"})
                elif item["type"] == "image" and isinstance(item.get("image"), str):
                    # If the image is a temp file path, replace with stub
                    if item["image"].startswith("/tmp/") or item["image"].startswith("/var/"):
                        new_msg["content"].append({"type": "image", "image": "(omitted)"})
                    else:
                        new_msg["content"].append(item)
                else:
                    new_msg["content"].append(item)
            cleaned.append(new_msg)
        return cleaned

    cleaned_history = _strip_audio(conversation)
    # Log response details
    logger.info("Returning response: conversation length=%d messages", len(cleaned_history))

    return {
        "reply_text": result["text"] if return_text else "",
        "reply_audio_b64": result.get("audio_b64") if return_audio else None,
        "conversation": cleaned_history,
    }


# ---------------------------------------------------------------------------
# Auxiliary API routes
# ---------------------------------------------------------------------------

@web_app.get("/api/system-goal")
def system_goal():
    """Return the current high-level objective of the agent (for UI banner)."""
    goal = os.getenv(
        "SYSTEM_GOAL",
        (
            "Assist users with multimodal interactions using Qwen2.5-Omni. "
            "Capabilities: text Q&A, math, image understanding, audio transcription, "
            "speech synthesis, video analysis, file Q&A, translation, and more."
        )
    )
    return {"goal": goal}

@web_app.get("/api/capabilities")
def capabilities():
    """Return a list of supported capabilities for the UI to display."""
    return {
        "capabilities": [
            {"name": "Text Q&A", "description": "Answer questions and chat in natural language."},
            {"name": "Math", "description": "Solve math problems and explain solutions."},
            {"name": "Image Understanding", "description": "Describe, analyze, and answer questions about images."},
            {"name": "Audio Transcription", "description": "Transcribe speech and classify sounds from audio files."},
            {"name": "Speech Synthesis", "description": "Generate spoken answers in different voices."},
            {"name": "Video Analysis", "description": "Extract information and answer questions about video content."},
            {"name": "File Q&A", "description": "Answer questions about uploaded files (images, audio, video)."},
            {"name": "Translation", "description": "Translate speech or text between languages."},
            {"name": "Multimodal Reasoning", "description": "Combine information from text, images, audio, and video."},
        ]
    }


# ---------------------------------------------------------------------------
# Bind ASGI to Modal
# ---------------------------------------------------------------------------


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app


# ---------------------------------------------------------------------------
# Local development entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(port: int = 8000):
    """Run the FastAPI application locally with live-reload.

    Allows quick iteration without deploying to Modal.  Once the server is
    running, open http://localhost:{port} in your browser.
    """
    import uvicorn

    logger.info("Starting local FastAPI server at http://127.0.0.1:%d", port)
    uvicorn.run("qwen_omni_webapp:web_app", host="0.0.0.0", port=port, reload=True)
