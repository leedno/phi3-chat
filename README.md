
# phi3-chat

**Lightweight terminal chat for a local Phi-3 language model.**

## Overview

`phi3-chat` is a Python-based REPL (Read-Eval-Print Loop) for interacting with a local GGUF-format Phi-3 language model.
It supports GPU acceleration (if available), slim conversation history, colored terminal output, and transcript logging.

## Features

* Simple terminal interface for chatting with a local language model
* Automatic GPU detection and layer allocation for faster inference
* Slim conversation history (configurable)
* Colored Q/A output for readability
* Saves transcripts for later review
* Fully configurable via CLI arguments

## Requirements

* Python 3.12+
* [llama-cpp-python](https://pypi.org/project/llama-cpp-python/)
* [colorama](https://pypi.org/project/colorama/)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup / Download Model

This project does **not include the model file** due to its large size. The model is ignored by Git (`.gitignore`) and will not be committed.

To run the chat, download the Phi-3 model (`Phi-3-mini-128k-instruct.Q4_K_M.gguf`) from the official source and place it in the `models/` directory:

```
localai/phi3-chat/models/Phi-3-mini-128k-instruct.Q4_K_M.gguf
```

## Usage

Activate your virtual environment and run:

```bash
python run_phi3_chat.py --model ./models/Phi-3-mini-128k-instruct.Q4_K_M.gguf
```

**Note:** Update the hardcoded model path in `run_phi3_chat.py`.


### Optional Arguments

* `--max-history`: Number of recent messages to keep
* `--max-tokens`: Maximum tokens per response
* `--threads`: CPU threads to use
* `--no-gpu`: Force CPU-only
* `--transcripts-dir`: Directory to save conversation logs
* `--quiet`: Minimize printed status messages

## File Structure

```
phi3-chat/
├─ models/                   # GGUF model files (ignored by Git)
├─ transcripts/              # Saved conversation logs
├─ run_phi3_chat.py          # Main chat REPL script
├─ requirements.txt          # Python dependencies
├─ README.md
└─ .gitignore
```

## License

[MIT License](LICENSE)
