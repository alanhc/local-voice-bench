# local-voice-bench

Stage 0 for a local voice interviewer in CodeTrial: before building a
`BidiGenerateContent` shim (VAD, STT, LLM, TTS behind Gemini Live's protocol),
measure whether the parts are fast enough on one RTX 5070 Ti.

## Results (2026-10-06)

RTX 5070 Ti 16 GB, driver 580, everything on the one GPU. gemma-4-12b Q4_K_M on
llama.cpp (`llama-server`, thinking off), faster-whisper 1.2.1 large-v3-turbo
int8_float16, kokoro-onnx 0.6.1 on onnxruntime-gpu 1.30, Silero VAD on CPU.
Five runs each; medians, p90 in `out-*.txt`.

| Stage | Measured |
|---|---|
| VAD (Silero, CPU) | 62 us per 32 ms frame |
| STT, 1.7 to 8.4 s of speech | 82 to 122 ms |
| STT term accuracy | 5/6 greedy, 6/6 with hotwords ("O of 1" becomes "O of one") |
| LLM, warm turn | first token 213 ms, first sentence 334 ms |
| LLM, hint request (one tool round trip) | first token 545 ms, first sentence 909 ms |
| LLM, cold (4,759-token instructions uncached) | first sentence 2,066 ms |
| TTS, 1.7 to 8.6 s of speech | 35 to 114 ms to the whole reply |
| VRAM, whisper + Kokoro | 2.4 GB (gemma 8.2 GB alongside) |

End of speech to first audio, after the endpointing wait (3,000 ms by default
in CodeTrial, which Gemini Live waits too):

- ordinary turn: about 120 (STT) + 330 (LLM) + 40 (TTS) = **~0.5 s**
- hint turn: about 120 + 910 + 40 = **~1.1 s**
- the session's first turn: ~2.2 s cold, avoidable by prefilling the
  instructions when the session opens, before the candidate speaks

## Caveats

- The candidate speech is Kokoro-synthesized: clean, no accent, no room noise.
  STT latency should hold for real speech; accuracy will not.
- Gemini Live was not measured here (no key), so there is no side-by-side.
- No barge-in, no LiveKit hop, no concurrent load beyond what was already on
  the GPU.
- Kokoro is English (and a few other languages); no Mandarin.

## Run

```bash
uv sync
uv pip uninstall onnxruntime && uv pip install --reinstall onnxruntime-gpu==1.30.0
source env.sh            # puts pip's cuBLAS and cuDNN on LD_LIBRARY_PATH
uv run --no-sync python bench_speech.py   # TTS, STT, VAD
LLM_BASE=http://127.0.0.1:8180 uv run --no-sync python bench_llm.py
```

`uv run` without `--no-sync` puts the CPU `onnxruntime`, which `kokoro-onnx`
depends on, back over the GPU build. Models: `models/kokoro-v1.0.onnx` and
`voices-v1.0.bin` from kokoro-onnx's `model-files-v1.0` release; whisper
downloads on first use (about six minutes here).
