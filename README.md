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

## Stage 1: a whole interview without Google (2026-10-06)

`live_shim.py` answers Gemini Live's `BidiGenerateContent` socket from Silero
VAD, faster-whisper, llama-server and Kokoro. With CodeTrial's
`CODETRIAL_GEMINI_LIVE_URL` (local branch `local-voice` in the
`codetrial-voice` worktree, on top of #110) and `CODETRIAL_GEMINI_REST_BASE`,
and LiveKit's own `livekit-server --dev`, nothing in an interview reaches
Google or LiveKit Cloud.

Checked with CodeTrial's own `scripts/browser-check.sh`, `BROWSER_CHECK_AGENT=dispatch`:

| Flow | Result |
|---|---|
| `report` | Greeting, test run, the interviewer reacting to the failed tests, end, report from gemma (0/0 NO_HIRE, summary true to an empty session) |
| `barge`, candidate audio from a WAV | Candidate speech through LiveKit, transcribed in 167 ms |
| `soak`, 11 minutes, "I'm stuck, can I get a hint?" every 35 s | 18 of 18 requests heard, 18 `log_hint` calls, 32 interviewer turns, none failed; first audio a median 1.63 s and at most 2.91 s after the input was complete. The flow itself fails at the end, on "no transport restart", which only Gemini's ten-minute socket cap causes |

Found on the way:

- Silero v5 through ONNX needs the 64 samples before each frame prepended;
  without them it scores clear speech under 0.05 and nothing is ever heard.
  Stage 0 measured its speed and missed this.
- gemma-4-12b repeats itself: the same hint sentence, nearly word for word,
  several times in the soak, against the prompt's "never say the same thing
  twice".
- A silence-nudge event and the candidate's request often land together, and
  the interviewer answers both, back to back.
- Hint turns take 1.9 to 2.9 s, slower than stage 0's 1.1 s estimate: two LLM
  calls, whole-sentence TTS before sending, and one GPU lock for all three
  models.

Not done in stage 1: barge-in (stage 2, below), latency work, any real
human speech.

### Run one yourself

```bash
~/.local/livekit/livekit-server --dev --config livekit-local.yaml --bind 127.0.0.1 --node-ip 127.0.0.1 &
source env.sh && .venv/bin/python live_shim.py --listen 127.0.0.1:8191 --llm http://127.0.0.1:8180 &
# in the codetrial-voice worktree, config/codetrial.env.local holding
# LIVEKIT_URL=ws://127.0.0.1:7880, LIVEKIT_API_KEY=devkey,
# LIVEKIT_API_SECRET=secret, GOOGLE_API_KEY=local
CODETRIAL_GEMINI_LIVE_URL=ws://127.0.0.1:8191/live \
CODETRIAL_GEMINI_REST_BASE=http://127.0.0.1:8190 \
CXX=$HOME/.local/clang-webrtc/bin/clang++ cargo run -- web
```

The report also needs CodeTrial's `scripts/gemini-shim.py --thinking off` on
8190 in front of the same llama-server.

## Stage 2: barge-in and merged inputs (2026-10-06)

Speech lasting about 190 ms while the interviewer has the floor (generating,
or its audio still playing out) stops the turn, sends `interrupted`, and keeps
in the history only the sentences that had started playing. A platform event
arriving while the candidate is mid-sentence waits for their words and is
answered with them as one turn. Whisper always gets common complexity phrases
as hotwords.

- `smoke_barge.py`, straight to the shim: `interrupted` 384 ms after the
  candidate started over the greeting, 2 sentences kept and 2 cut, then a reply
  to their request; a nudge sent one second into a 9.6 s answer was answered
  with it as one turn, first audio 0.71 s.
- `soak` through CodeTrial, the candidate speaking every 12 s for 11 minutes:
  55 utterances heard, 55 interruptions (54 over the last sentence still
  playing, one mid-reply cutting two unplayed sentences), 56 turns, none
  failed, first audio a median 1.08 s and at most 2.23 s. What the shim cannot
  see is the browser actually stopping playback; what it can see is that every
  interruption was followed by a normal turn, so CodeTrial did not stall on one.
- Two fixes on the way: the merge wait was eight seconds and a nine-second
  answer outlasted it, so it is now a 30 s bound on a stuck microphone rather
  than a guess at a sentence; and with no hotwords "O of n squared" came back
  as "O ,".

## Stage 3: replies prepared during the silence (2026-10-06)

Once the candidate has been quiet for 600 ms the shim transcribes, asks the
model and synthesizes the reply in the background, sending nothing; when the
configured silence (3,000 ms in CodeTrial) runs out it sends what is ready. If
they speak again first, the work is thrown away. A tool call is only decided
in advance, never made, since tools have effects. Sentences that repeat earlier
ones are dropped when the reply has something new; a reply made only of
repeats is said anyway.

Eleven minutes each through CodeTrial (`soak`, `analyze_log.py`):

| | Stage 2 | Stage 3, same request every 12 s | Stage 3, five different lines |
|---|---|---|---|
| turns prepared during the silence | 0 of 56 | 55 of 56 | 51 of 52 |
| first audio after the silence ends, median | 1,078 ms | **0 ms** | **0 ms** |
| p90 / max | 1,424 / 2,232 ms | 0 / 1,609 ms | 1,140 / 1,589 ms |
| sentences that nearly repeat an earlier one | 88% | 88% | 43% |
| failures | 0 | 0 | 0 |

What is left is the configured silence itself, which Gemini waits too; with
the reply ready when it starts, an operator could now lower
`GEMINI_SILENCE_MS` without the candidate waiting on the pipeline.

Repetition is not fixed. gemma-4-12b answers a repeated request with nearly
the same words each time, and asked to regenerate with a note listing what it
had said, it read the note aloud ("[Platform note: you already said ...]"),
so that was taken out. Dropping repeats outright made the interviewer go
silent on 26 of 55 requests, which is worse, so a reply that is all repeats is
now said. The remaining fix is a model that follows "never say the same thing
twice"; there is no VRAM left beside gemma to try one alongside it.

## Other models for the repetition (2026-10-07)

`eval_model.py` feeds candidate lines to the shim as text (no speech, no
real-time playback) and answers tools the way CodeTrial does, hint ladder
included: two rungs, then "withheld". Two scripts: a 20-line Two Sum interview,
and "I'm stuck. Can I get a hint?" 30 times. Results in `eval-results.jsonl`.

| Model (llama-server, thinking off) | realistic: repeats / hints logged / first sentence | 30x same request: repeats / hints logged |
|---|---|---|
| gemma-4-12b Q4_K_M | 0% / 3 of 5 / 488 ms | 72% / 3 of 31 |
| gemma-4-12b, DRY 0.8 (`--sampling`) | 0% / 3 of 5 / 653 ms | 47% / 1 of 31 |
| Qwen3.5-9B Q6_K | 2% / 1 of 5 / 479 ms | 41% / 1 of 31 |
| Qwen3-14B Q4_K_M, 36 of 41 layers on GPU | 0% / 4 of 5 / 2,280 ms | 73% / 3 of 31 |

- In an ordinary interview none of them repeats; the repetition is the
  candidate asking the same thing over and over, after the ladder is spent.
- Qwen3.5-9B is worse at the job: it named the hash map unasked, said "I can
  only provide one hint per session", and closed the interview early.
- Qwen3-14B follows the hint rules best, saying each rung as given, but does
  not fit beside whisper, Kokoro and the other 4 GB process: partly on the CPU
  it takes 2.3 s to the first sentence, and on the repeated request it said
  "What would you try first, even if it's a slow approach?" 27 times word for
  word.
- DRY turns the verbatim repeats into paraphrases (72% to 47%) but they say
  the same thing.
- Every model stops calling `log_hint` once the ladder is spent and answers
  the request itself, so CodeTrial no longer counts it.

So a different model that fits does not fix it. gemma stays.

## The shim logs hint requests itself (2026-10-07)

When the candidate's words ask for a hint (`HINT_REQUEST`, with "don't need a
hint" and the platform's bracketed events left out), the shim makes the
`log_hint` call with `requested: true` before the model speaks, and the model
answers from CodeTrial's result. A requested `log_hint` the model makes itself
in that turn is answered by the shim and not sent on, so one request counts
once. A hint request is not prepared during the silence, since its reply
depends on the call's result.

`eval_model.py`, gemma-4-12b (counts below leave out the greeting, which the
first version of the script counted as a request because it mentions hints):

| | before | after |
|---|---|---|
| ordinary interview, requests logged | 3 of 4 | **4 of 4** |
| 30x same request, requests logged | 3 of 30 | **30 of 30** |
| counted twice / counted unasked | not measured | 0 / 0 |
| 30x same request, near-repeats | 72% | 49% |

Repeats fell because CodeTrial's answer now arrives on every request and the
model words around it, but the withheld answer is the same text each time, so
what it says is still much the same. Varying that is CodeTrial's to do.

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
