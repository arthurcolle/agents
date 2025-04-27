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
from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

# Upload local template files to the volume
with template_volume.batch_upload() as batch:
    templates_dir = Path(__file__).parent / "templates"
    if templates_dir.exists():
        batch.put_directory(str(templates_dir), "/")

# Create static directory if it doesn't exist
static_dir = Path(__file__).parent / "static"
if not static_dir.exists():
    static_dir.mkdir(exist_ok=True)

# Upload static files to the volume
with static_volume.batch_upload() as batch:
    batch.put_directory(str(static_dir), "/")

# Reference the already-defined GPU runner so we can call `.generate.remote`.
qwen_app = modal.App("qwen-omni-runner")

# This app only handles HTTP so CPU is fine.
app = modal.App("qwen-omni-web", image=image, volumes={
    "/app/templates": template_volume,
    "/app/static": static_volume
})


# ---------------------------------------------------------------------------
# FastAPI set-up
# ---------------------------------------------------------------------------

web_app = FastAPI(title="Qwen-Omni Voice Chat")

# Set the templates directory to the mounted volume path
TEMPLATES_DIR = Path("/app/templates")
STATIC_DIR = Path("/app/static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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
    return templates.TemplateResponse("voice_chat.html", {"request": request})


@web_app.post("/api/chat")
def chat_endpoint(
    # One of these two must be provided by the browser ↓
    user_text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    # The entire conversation so far (list[dict] serialised as JSON str)
    conversation_json: str = Form(...),
    # Optional voice id for TTS (Chelsie default)
    speaker: str | None = Form(None),
):
    """Main inference endpoint.

    The **browser** keeps the chat history and sends it along every request so
    the backend is entirely stateless.  We simply append the new user
    message, call the GPU runner, and ship the answer back.
    """

    import json
    import soundfile as sf

    # ------------------------------------------------------------------
    # Reconstruct conversation list from JSON string sent by the client.
    # ------------------------------------------------------------------
    try:
        conversation: List[dict] = json.loads(conversation_json)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid conversation JSON"})

    # ------------------------------------------------------------------
    # Figure out what the user sent (plain text vs. recorded audio)
    # ------------------------------------------------------------------
    if user_text is None and audio is None:
        return JSONResponse(status_code=400, content={"error": "Provide either text or audio"})

    if audio is not None:
        # FastAPI gives us a SpooledTemporaryFile; read raw bytes.
        webm_bytes = audio.file.read()
        try:
            wav_bytes = _webm_to_wav_bytes(webm_bytes)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Audio conversion failed: {e}"})

        # Decode WAV to numpy float32 so it can go straight into the model.
        waveform, sr = sf.read(io.BytesIO(wav_bytes))
        if sr != 24_000:
            # Resample with soundfile if needed; Omni wants 24 kHz.
            import numpy as np
            import librosa  # heavy but already in qwen runner; cheap here because CPU

            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=24_000)

        # Make sure mono + float32 shape (T,)
        import numpy as np

        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)
        audio_item = {"type": "audio", "audio": waveform.tolist()}  # list so it's JSON-serialisable
        content = [audio_item]
    else:
        content = [{"type": "text", "text": user_text}]

    # ------------------------------------------------------------------
    # Append user message to history and call the GPU function.
    # ------------------------------------------------------------------
    conversation.append({"role": "user", "content": content})

    try:
        result = qwen_app.generate.remote(
            conversation,
            speaker=speaker,
            return_audio=True,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Model invocation failed: {e}"})

    # The GPU app already returns base64 encoded WAV so we can forward as-is.
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": result["text"]}],
    }
    conversation.append(assistant_message)

    # Don't send huge waveforms back to the browser – replace them with a stub.
    def _strip_audio(payload: List[dict]):
        cleaned: List[dict] = []
        for msg in payload:
            new_msg = {"role": msg["role"], "content": []}
            for item in msg["content"]:
                if item["type"] == "audio":
                    new_msg["content"].append({"type": "audio", "audio": "(omitted)"})
                else:
                    new_msg["content"].append(item)
            cleaned.append(new_msg)
        return cleaned

    cleaned_history = _strip_audio(conversation)

    return {
        "reply_text": result["text"],
        "reply_audio_b64": result.get("audio_b64"),
        "conversation": cleaned_history,
    }


# ---------------------------------------------------------------------------
# Bind ASGI to Modal
# ---------------------------------------------------------------------------


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
