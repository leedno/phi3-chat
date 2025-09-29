#!/usr/bin/env bash
# -----------------------------
# start_phi3.sh
# Lightweight wrapper to launch Phi-3 chat in the terminal
# -----------------------------

# move to the project folder
cd ~/localai/phi3-chat || {
  echo "Folder not found"
  exit 1
}

# activate the virtual environment
if [ -f ./venv/bin/activate ]; then
  source ./venv/bin/activate
else
  echo "Virtual environment not found. Create it first with:"
  echo "python3 -m venv venv && source venv/bin/activate && pip install llama-cpp-python colorama"
  exit 1
fi

# run the chat script, passing any extra flags through
python3 run_phi3_chat.py "$@"
