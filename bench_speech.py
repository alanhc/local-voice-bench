"""Stage 0: how long the speech halves of a local voice interviewer take.

TTS: Kokoro (ONNX, CUDA) speaking interviewer replies, timed to the first
audio chunk and to the end. It also speaks the candidate lines in another
voice and saves them, so the STT half has real speech to transcribe.

STT: faster-whisper large-v3-turbo on CUDA transcribing those lines, timed
per utterance, with the transcript held against the line that was spoken.

Run with `uv run --no-sync` after `source env.sh`: a plain `uv run` would
reinstall the CPU onnxruntime that kokoro-onnx depends on over the GPU one.
"""

import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
OUT = HERE / "out"
RUNS = 5

INTERVIEWER = [
    "Gotcha, Python it is.",
    "Makes sense. Before you code, walk me through one ordinary example and one edge case.",
    "If you're looking at a transaction of four and the target is zero, what amount would its partner need to be?",
    "Hmm, your loop on line seven checks every pair. What does that cost as the statement grows, and could you remember what you've already seen?",
]
CANDIDATE = [
    "I'd like to use Python for this one.",
    "I'm stuck. Can I get a hint?",
    "So the input is a list of transaction amounts and a target, and I return the indices of the two that add up to it.",
    "Could I use a hash map to store each value's index as I go, so each lookup is O of one?",
    "The brute force is O of n squared, but with a dictionary it becomes linear time and linear space, and I only need one pass over the list.",
]
TERMS = ["hash map", "o of one", "o of n squared", "dictionary", "linear", "indices"]


def stats(values):
    values = sorted(values)
    p90 = values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]
    return f"median {statistics.median(values) * 1000:7.0f} ms   p90 {p90 * 1000:7.0f} ms"


def norm(text):
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in text).split())


async def tts(kokoro, text, voice):
    """Time to the first chunk and to the whole reply, streaming."""
    started = time.perf_counter()
    first = None
    chunks = []
    rate = 24000
    async for samples, rate in kokoro.create_stream(text, voice=voice, speed=1.0, lang="en-us"):
        if first is None:
            first = time.perf_counter() - started
        chunks.append(samples)
    total = time.perf_counter() - started
    audio = np.concatenate(chunks)
    return first, total, audio, rate


async def bench_tts():
    from kokoro_onnx import Kokoro
    import onnxruntime as ort

    started = time.perf_counter()
    session = ort.InferenceSession(
        str(HERE / "models/kokoro-v1.0.onnx"),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    kokoro = Kokoro.from_session(session, str(HERE / "models/voices-v1.0.bin"))
    print(f"kokoro load {time.perf_counter() - started:.1f}s on {session.get_providers()[0]}")

    await tts(kokoro, "Warming up.", "am_michael")  # first call builds CUDA graphs
    print("\nTTS, interviewer voice (am_michael), streamed:")
    for text in INTERVIEWER:
        firsts, totals, length = [], [], 0.0
        for _ in range(RUNS):
            first, total, audio, rate = await tts(kokoro, text, "am_michael")
            firsts.append(first)
            totals.append(total)
            length = len(audio) / rate
        print(f"  {len(text):3d} chars, {length:4.1f}s of speech")
        print(f"     first chunk  {stats(firsts)}")
        print(f"     whole reply  {stats(totals)}   (real-time factor {statistics.median(totals) / length:.2f})")

    OUT.mkdir(exist_ok=True)
    for index, text in enumerate(CANDIDATE):
        _, _, audio, rate = await tts(kokoro, text, "af_heart")
        # Whisper wants 16 kHz mono, which is what LiveKit hands the agent.
        resampled = np.interp(
            np.linspace(0, len(audio), int(len(audio) * 16000 / rate), endpoint=False),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
        sf.write(OUT / f"candidate-{index}.wav", resampled, 16000)


def bench_stt():
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    print(f"\nwhisper load {time.perf_counter() - started:.1f}s")
    hotwords = "hash map, O of n, O of one, two pointers, dictionary, indices"

    clips = sorted(OUT.glob("candidate-*.wav"))
    audio, _ = sf.read(clips[0], dtype="float32")
    list(model.transcribe(audio, language="en", beam_size=1)[0])  # warm up

    print("\nSTT, whisper large-v3-turbo int8_float16:")
    for mode, options in [
        ("greedy", {"beam_size": 1}),
        ("greedy + hotwords", {"beam_size": 1, "hotwords": hotwords}),
    ]:
        print(f"  [{mode}]")
        term_hits = 0
        for clip, reference in zip(clips, CANDIDATE):
            audio, _ = sf.read(clip, dtype="float32")
            seconds = len(audio) / 16000
            times, text = [], ""
            for _ in range(RUNS):
                started = time.perf_counter()
                segments, _ = model.transcribe(
                    audio, language="en", vad_filter=False, condition_on_previous_text=False, **options
                )
                text = "".join(segment.text for segment in segments).strip()
                times.append(time.perf_counter() - started)
            heard = norm(text)
            terms = [term for term in TERMS if term in norm(reference)]
            hits = [term for term in terms if term in heard]
            term_hits += len(hits)
            exact = "exact" if heard == norm(reference) else "differs"
            print(f"    {seconds:4.1f}s clip  {stats(times)}  {exact}")
            if heard != norm(reference):
                print(f"        said : {reference}")
                print(f"        heard: {text}")
        total_terms = sum(1 for reference in CANDIDATE for term in TERMS if term in norm(reference))
        print(f"    technical terms heard: {term_hits}/{total_terms}")


def bench_vad():
    """Silero VAD per 32 ms frame, on the CPU the way it would run."""
    import onnxruntime as ort
    import urllib.request

    path = HERE / "models/silero_vad.onnx"
    if not path.exists():
        urllib.request.urlretrieve(
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx", path
        )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    audio, _ = sf.read(sorted(OUT.glob("candidate-*.wav"))[-1], dtype="float32")
    state = np.zeros((2, 1, 128), dtype=np.float32)
    rate = np.array(16000, dtype=np.int64)
    frames = [audio[i : i + 512] for i in range(0, len(audio) - 512, 512)]
    times = []
    for frame in frames:
        started = time.perf_counter()
        _, state = session.run(None, {"input": frame[None, :], "state": state, "sr": rate})
        times.append(time.perf_counter() - started)
    print(f"\nSilero VAD on CPU: {len(frames)} frames of 32 ms, median {statistics.median(times) * 1e6:.0f} us per frame")


if __name__ == "__main__":
    which = sys.argv[1:] or ["tts", "stt", "vad"]
    if "tts" in which:
        asyncio.run(bench_tts())
    if "stt" in which:
        bench_stt()
    if "vad" in which:
        bench_vad()
