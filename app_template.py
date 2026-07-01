"""
AI Media Server — Template
Build your own self-hosted AI media processing backend.

This template strips out all model-specific logic and replaces it with
clear TODO placeholders. Wire up your own models and you're good to go.

See README.md for full setup instructions.

Search for "TODO" in this file to find where to add your own logic.
"""

import os
import sys
import json
import time
import random
import shutil
import glob
import uuid
import threading
import subprocess
import zipfile

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "outputs")
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

_API_KEY = "REPLACE_WITH_YOUR_OWN_RANDOM_STRING"

# ---------------------------------------------------------
# TORCH / DIFFUSERS SETUP
# ---------------------------------------------------------
import torch

try:
    import diffusers
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Diffusers failed to import: {e}")
    DIFFUSERS_AVAILABLE = False

_sd_pipe_txt = None
_sd_pipe_img = None
_sd_pipe_ipa = None
_current_model_id = None

# ---------------------------------------------------------
# MODEL CONFIGS — TODO: Add your own models here
# ---------------------------------------------------------
MODEL_CONFIGS = {
    # TODO: Define your models. Example:
    #
    # "my-sd-model": {
    #     "name": "My SD Model (Display Name)",
    #     "resolution": 1024,
    #     "steps_range": (10, 50),
    #     "default_steps": 25,
    #     "guidance_scale": 7.5,
    #     "variant": "fp16",
    #     "needs_offloading": True,
    # },
    #
    # Keys here must match the values in your HTML <select id="genModel"> dropdown.
    "stabilityai/sd-turbo": {
        "name": "SD-Turbo (Fast, Low Quality)",
        "resolution": 512,
        "steps_range": (1, 10),
        "default_steps": 4,
        "guidance_scale": 0.0,
        "variant": "fp16",
        "needs_offloading": False,
    },
}

_gen_history = []


def get_history_html():
    if not _gen_history:
        return "<p style='color:#999; font-size:11px;'>No generations yet.</p>"
    html = ""
    for entry in reversed(_gen_history[-10:]):
        html += f"<div style='margin-bottom:4px; font-size:10px;'><b>{entry['time']}</b> — {entry['model']}<br/><i>{entry['prompt'][:80]}</i></div>"
    return html


# ---------------------------------------------------------
# HISTORY PERSISTENCE
# ---------------------------------------------------------
HISTORY_FILE = os.path.join(OUTPUT_DIR, "_gen_history.json")


def load_history():
    global _gen_history
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                _gen_history = json.load(f)
        except Exception:
            _gen_history = []


def save_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(_gen_history[-100:], f)
    except Exception:
        pass


load_history()

# ---------------------------------------------------------
# OLLAMA PROMPT REFINEMENT
# ---------------------------------------------------------


def refine_prompt_with_ollama(raw_prompt, model_name, progress=gr.Progress()):
    """Send the user prompt to local Ollama for refinement."""
    import requests
    try:
        ollama_url = "http://127.0.0.1:11434/api/chat"
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an image generation prompt optimizer. "
                        "Given a user prompt, output ONLY the refined prompt. "
                        "No explanations, no analysis, no thinking tags, no markdown. "
                        "Just the refined prompt text."
                    ),
                },
                {"role": "user", "content": raw_prompt},
            ],
            "stream": False,
        }
        resp = requests.post(ollama_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        refined = data.get("message", {}).get("content", raw_prompt).strip()
        if refined and len(refined) > 5:
            return refined
        return raw_prompt
    except Exception as e:
        print(f"Ollama refinement failed: {e}")
        return raw_prompt


# ---------------------------------------------------------
# UNLOAD MODELS
# ---------------------------------------------------------
def unload_current_model():
    global _sd_pipe_txt, _sd_pipe_img, _sd_pipe_ipa, _current_model_id
    if _sd_pipe_txt is not None:
        del _sd_pipe_txt
        _sd_pipe_txt = None
    if _sd_pipe_img is not None:
        del _sd_pipe_img
        _sd_pipe_img = None
    if _sd_pipe_ipa is not None:
        del _sd_pipe_ipa
        _sd_pipe_ipa = None
    _current_model_id = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------
# FILE METADATA
# ---------------------------------------------------------
def get_file_metadata_and_stems(file_path):
    if not file_path or not os.path.exists(file_path):
        return (None, None, None)
    file_url = f"outputs/{os.path.basename(file_path)}"
    file_size = os.path.getsize(file_path)
    return (file_url, file_size, None)


# ---------------------------------------------------------
# BG REMOVAL — TODO: Implement your own model
# ---------------------------------------------------------
def process_bg(input_path, fast_mode=False, threshold=0.5, output_type="rgba"):
    """
    Remove background from an image.

    TODO: Replace this with your own model inference. Example:

        from transparent_background import Remover
        remover = Remover(mode="base")
        img = Image.open(input_path)
        out = remover.process(img, type=...)
        out_path = os.path.join(OUTPUT_DIR, "result.png")
        out.save(out_path)
        return out_path
    """
    raise NotImplementedError("TODO: Implement process_bg() with your own model")


# ---------------------------------------------------------
# UPSCALER — TODO: Implement your own model
# ---------------------------------------------------------
def process_upscale(input_path, scale="4", model_choice="realesrgan-x4plus"):
    """
    Upscale an image using Real-ESRGAN or similar.

    TODO: Replace this with your own model inference. Example:

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        model = RRDBNet(...)
        upsampler = RealESRGANer(model=model, ...)
        output, _ = upsampler.enhance(img, outscale=int(scale))
        out_path = os.path.join(OUTPUT_DIR, "upscaled.png")
        output.save(out_path)
        return out_path
    """
    raise NotImplementedError("TODO: Implement process_upscale() with your own model")


# ---------------------------------------------------------
# TEXT-TO-IMAGE / IMAGE-TO-IMAGE — TODO: Implement your own model
# ---------------------------------------------------------
def _try_load_sd_model(pipeline_cls, model_id, device_hint, variant, progress):
    """Load a Stable Diffusion pipeline with MPS/CPU fallback."""
    if device_hint == "mps":
        try:
            progress(0, desc=f"Loading {model_id} on MPS...")
            pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch.float16, variant=variant)
            pipe = pipe.to("mps")
            print(f"Loaded {model_id} on MPS")
            return pipe, "mps"
        except Exception as e:
            print(f"MPS load failed for {model_id}: {e}")
    progress(0, desc=f"Loading {model_id} on CPU...")
    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch.float32, variant=variant)
    pipe = pipe.to("cpu")
    print(f"Loaded {model_id} on CPU")
    return pipe, "cpu"


def process_txt2img(prompt, steps, ref_image, strength, model_id, progress=gr.Progress()):
    """
    Generate an image from text (or image-to-image if ref_image is provided).

    TODO: Replace this with your own model inference if you want custom behavior.
    The default implementation uses diffusers AutoPipeline which works with most
    SD/SDXL models out of the box.
    """
    global _sd_pipe_txt, _sd_pipe_img, _current_model_id
    if not DIFFUSERS_AVAILABLE:
        return None

    config = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS.get(list(MODEL_CONFIGS.keys())[0]))
    total_steps = int(steps)
    resolution = config["resolution"]
    guidance_scale = config["guidance_scale"]
    variant = config.get("variant")

    if _current_model_id != model_id:
        progress(0, desc=f"Unloading previous model...")
        unload_current_model()

        use_mps = torch.backends.mps.is_available() and config["resolution"] <= 512
        device_hint = "mps" if use_mps else "cpu"

        if ref_image is not None:
            _sd_pipe_img, actual_device = _try_load_sd_model(AutoPipelineForImage2Image, model_id, device_hint, variant, progress)
            _sd_pipe_txt = None
        else:
            _sd_pipe_txt, actual_device = _try_load_sd_model(AutoPipelineForText2Image, model_id, device_hint, variant, progress)
            _sd_pipe_img = None

        _current_model_id = model_id

    if ref_image is not None:
        params = list(_sd_pipe_img.text_encoder.parameters()) if hasattr(_sd_pipe_img, 'text_encoder') else list(_sd_pipe_img.unet.parameters())
    else:
        params = list(_sd_pipe_txt.text_encoder.parameters()) if hasattr(_sd_pipe_txt, 'text_encoder') else list(_sd_pipe_txt.unet.parameters())
    device = str(params[0].device) if params else "cpu"

    generator = torch.Generator(device=device).manual_seed(random.randint(0, 2**32))

    def step_callback(step_idx, timestep, latents):
        progress((step_idx + 1) / total_steps, desc=f"Generating step {step_idx + 1}/{total_steps}...")

    if ref_image is not None:
        init_image = Image.open(ref_image).convert("RGB")
        orig_w, orig_h = init_image.size
        ratio = min(resolution / orig_w, resolution / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        init_image = init_image.resize((new_w, new_h), Image.LANCZOS)
        padded = Image.new("RGB", (resolution, resolution), (128, 128, 128))
        paste_x = (resolution - new_w) // 2
        paste_y = (resolution - new_h) // 2
        padded.paste(init_image, (paste_x, paste_y))

        progress(0, desc=f"Generating image-to-image ({total_steps} steps)...")
        result = _sd_pipe_img(prompt=prompt, image=padded, strength=float(strength), num_inference_steps=total_steps, guidance_scale=guidance_scale, callback=step_callback, callback_steps=1, generator=generator).images[0]
    else:
        progress(0, desc=f"Generating text-to-image ({total_steps} steps)...")
        result = _sd_pipe_txt(prompt=prompt, num_inference_steps=total_steps, guidance_scale=guidance_scale, callback=step_callback, callback_steps=1, generator=generator).images[0]

    progress(1, desc="Saving result...")
    out_path = os.path.join(OUTPUT_DIR, f"ai_gen_{abs(hash(prompt)) % 100000}.png")
    result.save(out_path)

    if device == "mps":
        torch.mps.empty_cache()

    return out_path


def process_ip_adapter(prompt, steps, ref_image, ipa_scale, progress=gr.Progress()):
    """
    Process image using IP-Adapter (face/style transfer from reference image).

    TODO: Implement your own IP-Adapter logic or remove if not needed.
    """
    global _sd_pipe_ipa, _current_model_id

    if not DIFFUSERS_AVAILABLE or not ref_image:
        return None

    if _current_model_id != "IP-Adapter XL" or _sd_pipe_ipa is None:
        progress(0, desc="Loading IP-Adapter pipeline...")
        unload_current_model()
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base_model_id, torch_dtype=torch.float16, variant="fp16"
            )
            progress(0.3, desc="Loading IP-Adapter weights...")
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter_sdxl_vit-h.safetensors",
            )
            pipe.set_ip_adapter_scale(float(ipa_scale))
            _sd_pipe_ipa = pipe.to("cpu")
            _current_model_id = "IP-Adapter XL"
            progress(0.6, desc="IP-Adapter loaded on CPU")
        except Exception as e:
            print(f"IP-Adapter load failed: {e}")
            import traceback; traceback.print_exc()
            return None

    pipe = _sd_pipe_ipa
    pipe.set_ip_adapter_scale(float(ipa_scale))

    params = list(pipe.image_encoder.parameters()) if hasattr(pipe, 'image_encoder') else list(pipe.unet.parameters())
    device = str(params[0].device) if params else "cpu"

    init_image = Image.open(ref_image).convert("RGB")
    orig_w, orig_h = init_image.size
    resolution = 1024
    ratio = min(resolution / orig_w, resolution / orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    init_image = init_image.resize((new_w, new_h), Image.LANCZOS)
    padded = Image.new("RGB", (resolution, resolution), (128, 128, 128))
    paste_x = (resolution - new_w) // 2
    paste_y = (resolution - new_h) // 2
    padded.paste(init_image, (paste_x, paste_y))

    generator = torch.Generator(device=device).manual_seed(random.randint(0, 2**32))

    def step_callback(step_idx, timestep, latents):
        progress((step_idx + 1) / steps, desc=f"Generating step {step_idx + 1}/{steps}...")

    progress(0, desc=f"Generating with IP-Adapter ({steps} steps)...")
    result = pipe(
        prompt=prompt if prompt else "",
        image=padded,
        num_inference_steps=steps,
        strength=float(ipa_scale),
        guidance_scale=7.5,
        callback=step_callback,
        callback_steps=1,
        generator=generator,
    ).images[0]

    progress(1, desc="Saving result...")
    out_path = os.path.join(OUTPUT_DIR, f"ai_gen_{abs(hash(prompt or 'ipa')) % 100000}.png")
    result.save(out_path)

    if device == "mps":
        torch.mps.empty_cache()

    return out_path


def handle_txt2img_click(prompt, steps, ref_image, strength, use_refine, ollama_model, model_id, progress=gr.Progress()):
    if use_refine and prompt:
        prompt = refine_prompt_with_ollama(prompt, ollama_model, progress)
    res_file = process_txt2img(prompt, steps, ref_image, strength, model_id, progress)
    meta = get_file_metadata_and_stems(res_file)
    if res_file:
        config = MODEL_CONFIGS.get(model_id, MODEL_CONFIGS.get(list(MODEL_CONFIGS.keys())[0]))
        entry = {
            "prompt": prompt,
            "time": time.strftime("%H:%M:%S"),
            "file": res_file,
            "steps": steps,
            "model": config["name"],
        }
        _gen_history.append(entry)
        save_history()
    return (
        meta[0],
        meta[1],
        meta[2],
    )


# ---------------------------------------------------------
# AUDIO STEM SEPARATION — TODO: Implement your own model
# ---------------------------------------------------------
def process_audio(file_path, model_choice, shifts, aud_format):
    """
    Separate audio into stems.

    TODO: Replace with your own model. Example:

        cmd = ["demucs", "-n", model_choice, "--shifts", str(shifts), "-o", OUTPUT_DIR, file_path]
        subprocess.run(cmd, check=True)
        ...
    """
    raise NotImplementedError("TODO: Implement process_audio() with your own model")


def handle_audio_click(file_path, model_choice, shifts, aud_format, progress=gr.Progress()):
    progress(0, desc="Starting stem separation...")
    res_files = process_audio(file_path, model_choice, shifts, aud_format)
    progress(1, desc="Done!")
    return get_file_metadata_and_stems(res_files)


# ---------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------
with gr.Blocks(title="AI Media Server", theme=gr.themes.Soft()) as app:
    gr.Markdown("# AI Media Server — Build Your Own")

    with gr.Tabs():
        with gr.Tab("BG Removal"):
            bg_input = gr.Image(label="Upload Image", type="filepath")
            bg_thresh = gr.Slider(0, 100, value=50, label="Threshold %")
            bg_btn = gr.Button("Remove Background")
            bg_output = gr.Image(label="Result")
            bg_btn.click(fn=handle_bg_click, inputs=[bg_input, bg_thresh], outputs=bg_output)

        with gr.Tab("Upscaler"):
            up_input = gr.Image(label="Upload Image", type="filepath")
            up_scale = gr.Dropdown(["2", "4"], value="4", label="Scale")
            up_btn = gr.Button("Upscale")
            up_output = gr.Image(label="Result")
            up_btn.click(fn=handle_upscale_click, inputs=[up_input, up_scale], outputs=up_output)

        with gr.Tab("Generate"):
            gen_model = gr.Dropdown(
                choices=[(v["name"], k) for k, v in MODEL_CONFIGS.items()],
                label="Model",
            )
            gen_prompt = gr.Textbox(label="Prompt", rows=3)
            gen_ref = gr.Image(label="Reference Image (optional)", type="filepath")
            gen_strength = gr.Slider(0.0, 1.0, value=0.5, label="Strength")
            gen_steps = gr.Slider(1, 50, value=25, label="Steps")
            gen_btn = gr.Button("Generate")
            gen_output = gr.Image(label="Result")
            gen_btn.click(
                fn=handle_txt2img_click,
                inputs=[gen_prompt, gen_steps, gen_ref, gen_strength, gen_model],
                outputs=gen_output,
            )

        with gr.Tab("Audio Separation"):
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            audio_model = gr.Dropdown(["htdemucs"], value="htdemucs", label="Model")
            audio_btn = gr.Button("Separate")
            audio_output = gr.File(label="Stems")
            audio_btn.click(fn=handle_audio_click, inputs=[audio_input, audio_model], outputs=audio_output)


# ---------------------------------------------------------
# REST API
# ---------------------------------------------------------
import mimetypes
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware


def _verify_api(request):
    key = request.headers.get("x-api-key", "")
    if key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.middleware("http")
async def _log_requests(request: Request, call_next):
    response = await call_next(request)
    origin = request.headers.get("origin", "none")
    host = request.headers.get("host", "none")
    print(f"[API] {request.method} {request.url.path} -> {response.status_code} | origin={origin} | host={host}")
    return response


@fastapi_app.get("/outputs/{file_path:path}")
def serve_output(file_path: str):
    abs_path = os.path.normpath(os.path.join(OUTPUT_DIR, file_path))
    if not abs_path.startswith(os.path.normpath(OUTPUT_DIR)):
        raise HTTPException(status_code=403, detail="Forbidden")
    if not os.path.exists(abs_path) or os.path.isdir(abs_path):
        raise HTTPException(status_code=404, detail="Not found")
    media_type, _ = mimetypes.guess_type(abs_path)
    return FileResponse(abs_path, media_type=media_type)


# TODO: Implement each endpoint with your own models.

@fastapi_app.post("/api/bg-remove")
def api_bg_remove(
    request: Request,
    file: UploadFile = File(...),
    fast_mode: str = Form("false"),
    threshold: str = Form("0.5"),
    output_type: str = Form("rgba"),
):
    _verify_api(request)
    temp_path = os.path.join(TEMP_DIR, file.filename or "upload")
    try:
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
        result = process_bg(temp_path, fast_mode == "true", float(threshold), output_type)
        if result and os.path.exists(result):
            _increment_processed()
            return {"url": f"outputs/{os.path.basename(result)}"}
        return JSONResponse({"error": "Processing failed"}, status_code=500)
    except NotImplementedError as e:
        return JSONResponse({"error": str(e)}, status_code=501)
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@fastapi_app.post("/api/upscale")
def api_upscale(
    request: Request,
    file: UploadFile = File(...),
    scale: str = Form("4"),
    model: str = Form("realesrgan-x4plus"),
):
    _verify_api(request)
    temp_path = os.path.join(TEMP_DIR, file.filename or "upload")
    try:
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
        result = process_upscale(temp_path, scale, model)
        if result and os.path.exists(result):
            _increment_processed()
            return {"url": f"outputs/{os.path.basename(result)}"}
        return JSONResponse({"error": "Processing failed"}, status_code=500)
    except NotImplementedError as e:
        return JSONResponse({"error": str(e)}, status_code=501)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@fastapi_app.post("/api/generate")
def api_generate(
    request: Request,
    prompt: str = Form(...),
    steps: str = Form("4"),
    model: str = Form("stabilityai/sd-turbo"),
    ref_image: UploadFile = File(None),
    strength: str = Form("0.5"),
    refine: str = Form("false"),
    ollama_model: str = Form("qwen-3.5:2b"),
):
    _verify_api(request)
    if not DIFFUSERS_AVAILABLE:
        return JSONResponse({"error": "Diffusers not available"}, status_code=500)
    try:
        final_prompt = prompt
        if refine == "true" and final_prompt:
            final_prompt = refine_prompt_with_ollama(final_prompt, ollama_model, gr.Progress())
        ref_path = None
        if ref_image and ref_image.filename:
            ref_path = os.path.join(TEMP_DIR, ref_image.filename)
            with open(ref_path, "wb") as f:
                f.write(ref_image.file.read())
        if model == "IP-Adapter XL":
            if not ref_path:
                return JSONResponse({"error": "IP-Adapter requires a reference image"}, status_code=400)
            result = process_ip_adapter(final_prompt, int(steps), ref_path, float(strength), gr.Progress())
        else:
            result = process_txt2img(final_prompt, int(steps), ref_path, float(strength), model, gr.Progress())
        if result and os.path.exists(result):
            _increment_processed()
            return {"url": f"outputs/{os.path.basename(result)}"}
        return JSONResponse({"error": "Generation failed"}, status_code=500)
    except NotImplementedError as e:
        return JSONResponse({"error": str(e)}, status_code=501)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)


# ---------------------------------------------------------
# TASK STORE & COUNTER
# ---------------------------------------------------------
_demucs_tasks: dict[str, dict] = {}
_task_lock = threading.Lock()
COUNTER_FILE = os.path.join(OUTPUT_DIR, "_visit_counter.json")


def _get_counter():
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE) as f:
                return json.load(f).get("count", 0)
        except Exception:
            return 0
    return 0


def _save_counter(count):
    try:
        with open(COUNTER_FILE, "w") as f:
            json.dump({"count": count}, f)
    except Exception:
        pass


PROCESSED_FILE = os.path.join(OUTPUT_DIR, "_processed_counter.json")


def _get_processed_count():
    if os.path.exists(PROCESSED_FILE):
        try:
            with open(PROCESSED_FILE) as f:
                return json.load(f).get("count", 0)
        except Exception:
            return 0
    return 0


def _save_processed_count(count):
    try:
        with open(PROCESSED_FILE, "w") as f:
            json.dump({"count": count}, f)
    except Exception:
        pass


def _increment_processed():
    count = _get_processed_count() + 1
    _save_processed_count(count)


@fastapi_app.post("/counter")
def api_counter():
    count = _get_counter() + 1
    _save_counter(count)
    return {"count": count}


@fastapi_app.get("/api/stats")
def api_stats():
    visit_count = _get_counter()
    processed_count = _get_processed_count()
    queue_count = 0
    with _task_lock:
        queue_count = sum(1 for t in _demucs_tasks.values() if t["status"] in ("queued", "processing"))
    return {
        "visits": visit_count,
        "files_processed": processed_count,
        "queue_length": queue_count,
    }


# TODO: Implement audio separation task.

def _run_demucs_task(task_id, file_path, filename):
    """Run audio separation in a background thread.

    TODO: Replace the demucs subprocess call with your own model.
    """
    global _demucs_tasks
    name, _ = os.path.splitext(filename)

    with _task_lock:
        _demucs_tasks[task_id]["status"] = "processing"
        _demucs_tasks[task_id]["progress"] = 0
        _demucs_tasks[task_id]["message"] = "Initializing..."

    try:
        # TODO: Replace with your own model inference
        raise NotImplementedError("TODO: Implement _run_demucs_task() with your own model")
    except Exception as e:
        with _task_lock:
            _demucs_tasks[task_id]["status"] = "error"
            _demucs_tasks[task_id]["message"] = str(e)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@fastapi_app.post("/separate")
def api_demucs_separate(
    request: Request,
    audioFile: UploadFile = File(...),
    cf_turnstile_response: str = Form(""),
):
    _verify_api(request)
    task_id = uuid.uuid4().hex
    filename = audioFile.filename or "audio"
    temp_path = os.path.join(TEMP_DIR, filename)
    with open(temp_path, "wb") as f:
        f.write(audioFile.file.read())
    with _task_lock:
        _demucs_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Queued",
            "data": None,
        }
    thread = threading.Thread(target=_run_demucs_task, args=(task_id, temp_path, filename), daemon=True)
    thread.start()
    return {"task_id": task_id}


@fastapi_app.get("/status/{task_id}")
def api_demucs_status(task_id: str):
    with _task_lock:
        task = _demucs_tasks.get(task_id)
    if task is None:
        return JSONResponse({"error": "Task not found"}, status_code=404)
    resp = {
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
    }
    if task["status"] == "complete" and task.get("data"):
        resp["data"] = task["data"]
    return resp


# ---------------------------------------------------------
# GRADIO ↔ FASTAPI BRIDGE
# ---------------------------------------------------------
from gradio.routes import App as GradioApp
_create_app_original = GradioApp.create_app


def _patched_create_app(blocks, *args, **kwargs):
    fastapi_app = _create_app_original(blocks, *args, **kwargs)
    # Mount custom routes onto Gradio's FastAPI app
    for route in fastapi_app.routes:
        pass  # Routes already registered on fastapi_app above
    return fastapi_app


GradioApp.create_app = _patched_create_app


def main():
    app.launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[OUTPUT_DIR])


if __name__ == "__main__":
    main()
