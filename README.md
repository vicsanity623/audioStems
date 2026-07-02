# AI - Local Media Server — Background Removal, Upscaling, AI img2img Generation & Audio Stem Separation

[![Website](https://img.shields.io/badge/Website-Live-brightgreen)](https://vicsanity623.github.io)
![Files Processed](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fvics-imac-1.tail37b4f2.ts.net%3A10000%2Fapi%2Fstats&query=%24.files_processed&label=Files%20Processed&color=cyan)
[![PWA](https://img.shields.io/badge/PWA-Installable-blue)](https://vicsanity623.github.io)
A **100% free**, self-hosted AI media processing suite — no paywalls, no credit cards, no usage caps. Runs entirely on your own hardware through a secure Tailscale tunnel.

**Try the live frontend:** https://vicsanity623.github.io

---

## ✨ What Can It Do?

This isn't a one-trick pony. The frontend you'll find on GitHub Pages gives you four parallel tools, each powered by its own dedicated AI model running on the backend:

| Tab | What It Does | Model |
|-----|-------------|-------|
| 🖼️ **BG Removal** | Removes backgrounds from images with adjustable threshold | `transparent-background` (PyTorch) |
| 🔍 **Upscaler** | Upscales images 2x–4x using real-world super-resolution | `Real-ESRGAN` |
| 🎨 **Generate** | Text-to-image / image-to-image / face & style transfer with AI prompt refinement | `Stable Diffusion` (SD-Turbo, SDXL, RealVisXL, Juggernaut, IP-Adapter) |
| 🎵 **Separate** | Splits audio into Vocals, Drums, Bass, Other stems | `Meta Demucs` |

All four share a single Python backend behind a Tailscale Funnel — meaning you can run every model on one machine and access it from anywhere on the planet.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              GitHub Pages                   │
│  (Static HTML/JS — no backend required)     │
│  https://vicsanity623.github.io             │
│                                             │
│  Contains: 4-tab interface + Firebase Auth  │
│  + Cloudflare Turnstile + Wavesurfer player │
└────────────────────┬────────────────────────┘
                     │
                     ▼
            Tailscale Funnel
   (https://your-machine.tailXXXXX.ts.net)
                     │
                     ▼
┌─────────────────────────────────────────────┐
│         Your Local Machine (Backend)        │
│                                             │
│  Python Gradio/FastAPI app on :7860         │
│  ┌──────────────────────────────────────┐   │
│  │  Gradio 5 UI (used for testing)      │   │
│  │  ───────────────────────────────     │   │
│  │  Custom REST API endpoints:          │   │
│  │  POST /api/bg-remove   (bg removal)  │   │
│  │  POST /api/upscale     (upscaling)   │   │
│  │  POST /api/generate    (AI gen)      │   │
│  │  POST /separate        (demucs)      │   │
│  │  GET  /status/{id}     (task poll)   │   │
│  │  POST /counter         (analytics)   │   │
│  │  GET  /api/stats       (live stats)  │   │
│  │  GET  /outputs/{path}  (file serve)  │   │
│  └──────────────────────────────────────┘   │
│                                             │
│  Models on disk (~10–30 GB total):          │
│  ├── transparent-background (PyTorch)       │
│  ├── realesrgan-x4plus / -x2                │
│  ├── sd-turbo /sd-xl /realvisxl /juggernaut │
│  ├── IP-Adapter XL (face/style transfer)    │
│  └── demucs htdemucs (Meta)                 │
└─────────────────────────────────────────────┘
```

---

# 🧑‍💻 Full macOS Setup Guide

Below is a _you-are-there_, step-by-step walkthrough of setting this entire backend on macOS (Darwin). Every command, every gotcha, every file we created along the way.

## Prerequisites

- **macOS** (tested on Intel Mac with macOS Tahoe v26.5.1)
- **Homebrew** (`/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
- **Python 3.10** (not 3.11+, because PyTorch + certain wheels have issues on newer Pythons on macOS)
- **~30 GB free disk space** for model weights
- **Tailscale account** (free tier works) if you want remote access
- **Firebase project** (free Spark plan) if you want auth + analytics
- **Cloudflare Turnstile** (free) if you want bot protection
- **A GitHub account** to fork and host the frontend

### Step 1: Install Python 3.10

You need Python 3.10 specifically. Python 3.11+ breaks several dependencies on macOS.

```bash
brew install python@3.10
```

Verify:
```bash
python3.10 --version
# → Python 3.10.x
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/vicsanity623/BGRemover.git
cd BGRemover
```

### Step 3: Create the Virtual Environment

Always use a venv — never install these packages globally (they pin very specific torch versions).

```bash
python3.10 -m venv .venvBGR
source .venvBGR/bin/activate
```

Your prompt should now show `(.venvBGR)`.

### Step 4: Install PyTorch (MPS or CPU)

This is the trickiest part. On macOS you have two choices:

**Option A — MPS (Metal Performance Shaders)** — for Apple Silicon Macs:
```bash
pip install torch torchvision torchaudio
```

**Option B — CPU** — for Intel Macs without a powerful GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Either way, verify:
```bash
python3 -c "import torch; print(torch.__version__); print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else 'N/A')"
```

### Step 5: Install Core Dependencies

```bash
pip install \
  gradio==5.16.0 \
  fastapi \
  uvicorn \
  python-multipart \
  Pillow \
  numpy \
  opencv-python-headless \
  pydub \
  ffmpeg-python
```

**Important:** You also need `ffmpeg` on your system (not just the Python wrapper):

```bash
brew install ffmpeg
```

### Step 6: Install AI Models

Each model is a separate install. Do them one at a time so you can see if anything breaks.

#### 6a — Background Removal (`transparent-background`)

```bash
pip install transparent-background
```

On first run it downloads the model checkpoint (`~/.cache/transparent-background/...`). This is ~500 MB.

#### 6b — Upscaler (`Real-ESRGAN`)

```bash
pip install realesrgan
```

Models download on first use to `~/.cache/realesrgan/...`. Approx 200 MB.

#### 6c — Audio Separation (`demucs`)

```bash
pip install demucs
```

The `htdemucs` model downloads on first run to `~/Library/Caches/demucs/...`. Approx 1.5 GB.

**Troubleshooting:** If `demucs` installs but the `demucs` command isn't found, add your Python bin to PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```
Or symlink it:
```bash
ln -s $(which demucs) ~/.local/bin/demucs 2>/dev/null || true
```

**Vital:** Install `soundfile` so demucs can write WAV stems:
```bash
pip install soundfile
```

#### 6d — Stable Diffusion Models (Text-to-Image / Image-to-Image)

This is the biggest install. It pulls in `diffusers`, `transformers`, `accelerate`, and the model weights (~5–15 GB).

```bash
pip install diffusers transformers accelerate
```

The app supports six diffusion models:

| Model ID | Size | Notes |
|----------|------|-------|
| `stabilityai/sd-turbo` | ~3 GB | Default; fast (1–4 steps) |
| `Lykon/dreamshaper-8` | ~2 GB | SD 1.5, good general purpose |
| `stabilityai/sdxl-turbo` | ~7 GB | SDXL, fast (1–4 steps) |
| `stabilityai/stable-diffusion-xl-base-1.0` | ~7 GB | SDXL base, highest quality |
| `SG161222/RealVisXL_V4.0` | ~7 GB | SDXL, photorealistic |
| `RunDiffusion/Juggernaut-XL-v9` | ~7 GB | SDXL, versatile |
| `IP-Adapter XL` | ~7 GB (base) + ~600 MB (adapter) | Face & style transfer from reference images |

These download to `~/.cache/huggingface/` on first use. On a 50 Mbps connection, expect each to take 10–30 minutes.

**MPS gotcha:** SDXL and FLUX use `float64` operations that MPS doesn't support. The app automatically patches these:
```python
xpu_module = types.ModuleType("xpu")
xpu_module.empty_cache = lambda: None
xpu_module.is_available = lambda: False
torch.xpu = xpu_module
```
This is already in `app.py` — you don't need to do anything.

### Step 7: Set Up the Directory Structure

The app expects these directories relative to the project root:

```bash
mkdir -p outputs temp .cache/huggingface .cache/torch
```

### Step 8: Configure the Backend URL & API Key

The frontend on GitHub Pages needs to know where to find your backend, and both sides need a matching secret key.

#### 8a — Set your Tailscale Funnel URL

Open `index.html` (or your deployed `.html`) and find the `getBackendUrl` function:

```javascript
const getBackendUrl = () => "https://your-machine.tailXXXXX.ts.net:10000/";
```

Replace `your-machine.tailXXXXX.ts.net:10000` with your actual Tailscale Funnel URL and port. You can find this by running `tailscale status` — look for your machine name.

#### 8b — Generate a unique API key

The API key is a shared secret between the frontend and backend. **Do not use the key from this repo** — generate your own:

```bash
openssl rand -hex 32
```

This outputs a 64-character random hex string. Copy it.

Now update **both** files with your new key:

**In `src/media_server/app.py`:**
```python
_API_KEY = "paste-your-generated-key-here"
```

**In `index.html`:**
```javascript
const API_KEY = "paste-your-generated-key-here";
```

**Both must match exactly.** If you change one, change the other.

### Step 9: Run the Backend Locally

```bash
cd /path/to/BGRemover
source .venvBGR/bin/activate
python3 src/media_server/app.py
```

You should see:
```
* Running on local URL:  http://127.0.0.1:7860
```

At this point you can test it locally by visiting `http://127.0.0.1:7860` in your browser. You'll see the Gradio UI with all four tabs working.

Test the REST API too:
```bash
curl -X POST http://127.0.0.1:7860/counter \
  -H "x-api-key: jshdgcjhBSDCHLAsrdhvsdvhjasbdvlha34hbvb234kjv"
# → {"count":1}
```

### Step 10: Expose via Tailscale Funnel (for Remote Access)

This is what makes the whole thing work from anywhere — no static IP, no port forwarding, no cloud server.

1. **Install Tailscale** on your Mac:
   ```bash
   brew install tailscale
   ```

2. **Sign in:**
   ```bash
   tailscale up
   ```
   This opens a browser to authenticate.

3. **Enable Funnel:**
   ```bash
   tailscale funnel --bg 10000
   ```
   This tells Tailscale to accept HTTPS traffic on port 10000 and forward it to `http://127.0.0.1:7860`.

4. **Find your funnel URL:**
   ```bash
   tailscale status
   ```
   Look for your machine name. Your funnel URL will be:
   ```
   https://your-machine.tailXXXXX.ts.net:10000/
   ```

5. **Test the funnel** from your phone (disconnect from WiFi):
   ```
   https://your-machine.tailXXXXX.ts.net:10000/counter
   ```
   If you get `{"detail":"Method Not Allowed"}`, the funnel is working (405 means the server received it, it's just a GET on a POST-only route).

### Step 11: Deploy the Frontend to GitHub Pages

1. Create a **new repository** on GitHub (e.g., `vicsanity623/vicsanity623.github.io` for a user site, or any name for a project site).

2. Copy the frontend files into it:
   ```bash
   # From your BGRemover project
   cp index.html /path/to/your-github-pages-repo/index.html
   # Also copy any assets, audio files, etc.
   ```

3. **Update the `getBackendUrl`** in the HTML to point to your Tailscale Funnel URL.

4. **Push to GitHub:**
   ```bash
   cd /path/to/your-github-pages-repo
   git add .
   git commit -m "Deploy AI media processing suite"
   git push
   ```

5. **Enable GitHub Pages** in the repo Settings → Pages → select `main` branch → Save.

6. Wait ~2 minutes, then visit `https://your-username.github.io`.

### Step 12: Firebase Auth (Optional but Recommended)

The frontend uses Firebase Authentication (Google Sign-In) to track users and prevent abuse.

1. Go to [Firebase Console](https://console.firebase.google.com) → Create a project (Spark free tier).
2. Enable **Authentication** → **Google provider**.
3. Enable **Cloud Firestore** (start in test mode).
4. Get your Firebase config object from Project Settings → General → Your apps → Web app.
5. Put it in the HTML file:
   ```javascript
   const firebaseConfig = {
     apiKey: "AIzaSy...",
     authDomain: "your-project.firebaseapp.com",
     projectId: "your-project",
     storageBucket: "your-project.appspot.com",
     messagingSenderId: "...",
     appId: "..."
   };
   ```

### Step 13: Cloudflare Turnstile (Optional)

The audio separation tab uses Turnstile CAPTCHA to prevent bots.

1. Go to [Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/) → Add a site.
2. Get your **site key** and **secret key**.
3. Replace the `cf-turnstile-response` verification in the backend (right now it's a placeholder that always passes).

---

## 🔄 How to Update

When you pull new changes:

```bash
git pull
source .venvBGR/bin/activate
pip install -r requirements.txt  # if one exists; otherwise install individually
pkill -f "python3 src/media_server/app.py"
python3 src/media_server/app.py
```

---

## ⚙️ Model Cache Locations

All models are cached locally. To keep your home drive from filling up, the app uses a custom cache directory on the same drive as the project:

| Cache | Location |
|-------|----------|
| HuggingFace models | `BGRemover/.cache/huggingface/` |
| Torch hub | `BGRemover/.cache/torch/` |
| Transparent Background | `BGRemover/.transparent-background/` |
| Demucs models | `~/Library/Caches/demucs/` |

To move the HuggingFace cache to an external drive:

```bash
export HF_HOME=/Volumes/YourExternalDrive/.cache/huggingface
```

---

## 🩺 Troubleshooting

### "MPS not available" or XPU errors

This is normal on Intel Macs. The app monkey-patches the missing XPU attributes so diffusers loads without crashing. If you see `torch.xpu` errors, make sure the monkey-patch near the top of `app.py` is present.

### "demucs: command not found"

```bash
python3 -m demucs --help
```
If that works, use `python3 -m demucs` instead of `demucs` in the code. Or install demucs with pip's `--user` flag and add `~/.local/bin` to your PATH.

### "soundfile not installed" or demucs errors

```bash
pip install soundfile
```

### "ModuleNotFoundError: No module named 'torch'"

You skipped step 4. Run the PyTorch install command.

### Frontend shows "NODE OFFLINE OR FAILED"

This means either:
1. Your Tailscale Funnel is down — run `tailscale funnel --bg 10000` again.
2. The backend isn't running — restart it.
3. The API key in the HTML doesn't match `_API_KEY` in `app.py`.
4. The `getBackendUrl` in the HTML points to the wrong address.

### The REST API returns 500 during bg-remove

The `transparent-background` library saves output with the original file extension (e.g., `.jpeg` for JPEG inputs) for green-screen mode, but always uses `.png` for RGBA mode. If you change the code, make sure `out_ext` is set correctly:

```python
# In process_bg():
if use_rgba:
    out_ext = ".png"     # RGBA always outputs PNG
else:
    out_ext = ext         # Green screen preserves original extension
```

---

## 🛠️ Build Your Own Version

A **template version** of `app.py` is included as `app_template.py`. It strips out all model-specific logic and replaces it with clear placeholders so you can wire up your own models.

### What's in the template

- `MODEL_CONFIGS` dict → replace with your own model definitions
- Empty `process_bg()`, `process_upscale()`, `process_txt2img()`, `process_ip_adapter()` functions → fill in your own model loading/inference
- Empty `_run_demucs_task()` → wire up your own audio separation
- API endpoints already wired up and ready to go
- Auth, file serving, counter, stats — all pre-built

### How to use it

1. **Copy the template:**
   ```bash
   cp app_template.py src/media_server/app.py
   ```

2. **Open `src/media_server/app.py`** and search for `TODO` comments. Each one marks where you need to add your own logic.

3. **Define your models** in `MODEL_CONFIGS`:
   ```python
   MODEL_CONFIGS = {
       "my-model-id": {
           "name": "My Model Display Name",
           "resolution": 1024,
           "steps_range": (10, 50),
           "default_steps": 25,
           "guidance_scale": 7.5,
           "variant": "fp16",
           "needs_offloading": True,
       },
   }
   ```

4. **Implement the processing functions** — each one receives the input and should return the output file path:
   ```python
   def process_bg(input_path, fast_mode, threshold, output_type):
       # TODO: Load your model, process the image, return output path
       output_path = os.path.join(OUTPUT_DIR, "result.png")
       # ... your inference code here ...
       return output_path
   ```

5. **Update the HTML** to match your models — the `<select id="genModel">` dropdown values must match your `MODEL_CONFIGS` keys.

6. **Run it:**
   ```bash
   source .venvBGR/bin/activate
   python3 src/media_server/app.py
   ```

### Template API endpoints (pre-built)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/bg-remove` | Background removal |
| POST | `/api/upscale` | Image upscaling |
| POST | `/api/generate` | Text-to-image / image-to-image |
| POST | `/separate` | Audio stem separation |
| GET | `/status/{task_id}` | Poll async task status |
| POST | `/counter` | Visit counter (public) |
| GET | `/api/stats` | Live stats (visits, files processed, queue) |
| GET | `/outputs/{path}` | Serve output files |

---

## 📜 License & Usage

This project is for **educational and personal use only**. You are responsible for ensuring you have the rights to any media you process. The AI models included have their own licenses (MIT, Apache 2.0, CC-BY-NC, etc.) — check each project's terms before commercial use.

- **Demucs**: MIT License (Meta Research)
- **Real-ESRGAN**: BSD 3-Clause
- **transparent-background**: MIT License
- **Stable Diffusion**: Stability AI CreativeML Open RAIL-M
- **FLUX**: black-forest-labs terms
