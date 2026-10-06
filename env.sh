export LD_LIBRARY_PATH=$HOME/workspace/local-voice-bench/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:$HOME/workspace/local-voice-bench/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1  # whisper is cached; without this, loading it checks Hugging Face
