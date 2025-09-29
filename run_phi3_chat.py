
#!/usr/bin/env python3
"""
run_phi3_chat.py
Polished terminal chat REPL for GGUF models (Phi-3 example).

- Suppresses verbose vendor logs and llama-cpp perf prints by default
- Use --show-stats to enable the internal performance / metadata output
- Auto-detects GPU VRAM and picks safe n_gpu_layers (with fallback)
- Slim conversation history trimming
- Colored clean Q/A output
- Saves transcripts under transcripts/
- CLI flags for quick tweaking
"""

import os
# suppress verbose vendor logs early by default
os.environ.setdefault("LLAMA_VERBOSE", "0")

import argparse
import contextlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# color output
try:
    from colorama import Fore, Style, init as colorama_init
except Exception:
    print("Installing colorama automatically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "colorama"])
    from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

try:
    from llama_cpp import Llama
except Exception as e:
    print(Fore.RED + "ERROR: llama_cpp (llama-cpp-python) not available.")
    print("Install it in your venv: pip install llama-cpp-python")
    raise

# -------- helpers --------
def get_gpu_vram_mib():
    """Return integer VRAM (MiB) of first NVIDIA GPU or 0 if none found."""
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        out = p.stdout.strip().splitlines()
        if not out:
            return 0
        return int(out[0].strip())
    except Exception:
        return 0

def choose_n_gpu_layers(total_vram_mib, model_layers=32):
    """Conservative heuristic to choose number of layers to put on GPU."""
    if total_vram_mib >= 24576:
        return -1
    if total_vram_mib >= 12288:
        return model_layers
    if total_vram_mib >= 8192:
        return max(1, model_layers - 2)
    if total_vram_mib >= 6144:
        return max(1, model_layers - 4)
    if total_vram_mib >= 4096:
        return max(1, model_layers - 12)
    return 0

def try_load_model_with_fallback(model_path, n_gpu_layers, n_threads, suppress_stderr=True, max_reduce_step=4):
    """
    Try to instantiate Llama with requested n_gpu_layers, reduce if OOM/failed.
    If suppress_stderr is True, stderr is redirected to /dev/null during attempts
    (so initial model metadata and perf prints are hidden).
    """
    attempt = 0
    cur_layers = n_gpu_layers
    while True:
        attempt += 1
        try:
            if suppress_stderr:
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stderr(devnull):
                        llm = Llama(model_path=model_path, n_gpu_layers=cur_layers if cur_layers != 0 else 0, n_threads=n_threads)
            else:
                llm = Llama(model_path=model_path, n_gpu_layers=cur_layers if cur_layers != 0 else 0, n_threads=n_threads)
            return llm, cur_layers
        except Exception as e:
            # If no GPU or already CPU-only, bubble up
            if cur_layers is None or cur_layers == 0:
                raise
            print(Fore.YELLOW + f"Warning: failed to load with n_gpu_layers={cur_layers} (attempt {attempt}). Error: {e}")
            if cur_layers == -1:
                cur_layers = max(1, 16)
            else:
                cur_layers = max(0, cur_layers - max_reduce_step)
            if cur_layers == 0:
                print(Fore.YELLOW + "Falling back to CPU-only (n_gpu_layers=0).")
                try:
                    if suppress_stderr:
                        with open(os.devnull, "w") as devnull:
                            with contextlib.redirect_stderr(devnull):
                                llm = Llama(model_path=model_path, n_gpu_layers=0, n_threads=n_threads)
                    else:
                        llm = Llama(model_path=model_path, n_gpu_layers=0, n_threads=n_threads)
                    return llm, 0
                except Exception as e2:
                    raise RuntimeError(f"Failed to load model on CPU as well: {e2}") from e

# -------- argument parsing --------
parser = argparse.ArgumentParser(description="Run a slim, nice terminal chat for a GGUF model")
parser.add_argument("--model", "-m", default="/home/leon/localai/phi3-chat/models/Phi-3-mini-128k-instruct.Q4_K_M.gguf",
                    help="Path to model GGUF file")
parser.add_argument("--max-history", "-H", type=int, default=6, help="Number of recent messages to keep (user+assistant count)")
parser.add_argument("--max-tokens", "-k", type=int, default=200, help="Max tokens per response")
parser.add_argument("--temp", "-t", type=float, default=0.7, help="Temperature")
parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
parser.add_argument("--threads", type=int, default=max(1, os.cpu_count()//2), help="CPU threads to use")
parser.add_argument("--no-gpu", action="store_true", help="Force CPU-only (do not use GPU)")
parser.add_argument("--transcripts-dir", default="./transcripts", help="Directory to save conversation transcripts")
parser.add_argument("--quiet", action="store_true", help="Minimize printed status messages")
parser.add_argument("--show-stats", action="store_true", help="Show llama-cpp internal performance / metadata")
args = parser.parse_args()

# verify model exists
model_path = Path(args.model).expanduser()
if not model_path.exists():
    print(Fore.RED + f"Model file not found: {model_path}")
    sys.exit(1)

# header
if not args.quiet:
    print(Fore.MAGENTA + "Phi-3 Chat — starting up")
    print(Fore.MAGENTA + f"Model: {model_path}")
    print(Fore.MAGENTA + f"Threads: {args.threads}  Max-history: {args.max_history}  Max-tokens: {args.max_tokens}")

# prepare transcripts dir
transcripts_dir = Path(args.transcripts_dir)
transcripts_dir.mkdir(parents=True, exist_ok=True)
transcript_file = transcripts_dir / f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# decide GPU usage
gpu_vram = 0
n_gpu_layers = 0
if not args.no_gpu:
    gpu_vram = get_gpu_vram_mib()
    if gpu_vram > 0:
        suggested = choose_n_gpu_layers(gpu_vram, model_layers=32)
        n_gpu_layers = suggested if suggested != 0 else 0
        if not args.quiet:
            print(Fore.MAGENTA + f"Detected GPU VRAM: {gpu_vram} MiB -> suggested n_gpu_layers = {n_gpu_layers}")
    else:
        if not args.quiet:
            print(Fore.YELLOW + "No NVIDIA GPU detected (or nvidia-smi not found). Using CPU-only.")

# load model (suppress stderr unless user wants stats)
suppress = not args.show_stats
try:
    llm, used_gpu_layers = try_load_model_with_fallback(str(model_path), n_gpu_layers if n_gpu_layers != 0 else 0, args.threads, suppress_stderr=suppress)
except Exception as e:
    print(Fore.RED + "Failed to load model: " + str(e))
    sys.exit(1)

if not args.quiet:
    if used_gpu_layers and used_gpu_layers != 0:
        print(Fore.GREEN + f"Model loaded with GPU support (n_gpu_layers={used_gpu_layers}).")
    else:
        print(Fore.GREEN + "Model loaded (CPU-only).")

# chat state
history = []

def save_to_transcript(user_msg, ai_msg):
    t = datetime.now().isoformat(sep=" ", timespec="seconds")
    with open(transcript_file, "a", encoding="utf-8") as fh:
        fh.write(f"[{t}] You: {user_msg}\n")
        fh.write(f"[{t}] AI: {ai_msg}\n\n")

# nice REPL
print(Fore.CYAN + "\n✅ Phi-3 chat ready. Type messages (Ctrl+C to exit).\n")
try:
    while True:
        user_input = input(Fore.GREEN + Style.BRIGHT + "You> " + Style.RESET_ALL).strip()
        if not user_input:
            continue

        # append and trim history
        history.append({"role": "user", "content": user_input})
        if len(history) > args.max_history:
            history = history[-args.max_history:]

        # build chat prompt using GGUF chat template tags
        prompt_parts = []
        for m in history:
            if m["role"] == "user":
                prompt_parts.append(f"<|user|>\n{m['content']}<|end|>\n<|assistant|>\n")
            else:
                prompt_parts.append(f"{m['content']}<|end|>\n")
        prompt = "".join(prompt_parts)

        # call model: silence stderr if --show-stats is NOT set
        if args.show_stats:
            resp = llm(prompt, max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p)
        else:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stderr(devnull):
                    resp = llm(prompt, max_tokens=args.max_tokens, temperature=args.temp, top_p=args.top_p)

        # extract text robustly
        try:
            text = resp["choices"][0]["text"].strip()
        except Exception:
            text = str(resp).strip()

        # print nicely
        print(Fore.CYAN + Style.BRIGHT + "AI> " + Style.RESET_ALL + text + "\n")

        # append and save
        history.append({"role": "assistant", "content": text})
        save_to_transcript(user_input, text)

except (KeyboardInterrupt, EOFError):
    print(Fore.MAGENTA + "\nExiting. Transcript saved to: " + str(transcript_file))
    sys.exit(0)
