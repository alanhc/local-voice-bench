"""A local stand-in for Gemini Live's BidiGenerateContent socket.

CodeTrial's interviewer speaks Live's protocol over one WebSocket: audio in,
audio and transcripts out, tool calls in between. This answers that protocol
from parts that run on one GPU:

    candidate audio -> Silero VAD -> faster-whisper -> llama-server -> Kokoro -> audio

Stage 1 of the plan in README.md: enough for a whole interview, nothing more.
Speech that lasts while the interviewer has the floor interrupts it, as
Live's START_OF_ACTIVITY_INTERRUPTS does: generation stops, `interrupted` goes
out, and the history keeps only what the candidate actually heard. Point
CodeTrial at it with

    CODETRIAL_GEMINI_LIVE_URL=ws://127.0.0.1:8191/live

Run with `uv run --no-sync` after `source env.sh` (see README.md).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import onnxruntime as ort
import websockets
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

HERE = Path(__file__).parent
IN_RATE = 16000
OUT_RATE = 24000
FRAME = 512  # Silero's frame at 16 kHz, 32 ms
# Silero v5 reads each frame with the 64 samples before it. Its Python wrapper
# prepends them; called through ONNX directly, a frame without them scores
# under 0.05 even on clear speech, so nothing is ever heard.
VAD_CONTEXT = 64
SPEECH_PROBABILITY = 0.5
PRE_ROLL_FRAMES = 10  # keep ~320 ms before speech starts, so the first word is whole
ECHO_GUARD = 0.3  # seconds after the interviewer's audio ends before listening again
# While the interviewer has the floor, speech has to last this many frames
# (~190 ms) before it takes the floor: a cough or the tail of the interviewer's
# own voice leaking past echo cancellation should not cut a reply off.
BARGE_FRAMES = 6
# How long a turn waits for a candidate who is mid-sentence when it is due, so
# that their words and a platform event that arrived meanwhile are answered as
# one turn instead of two back to back. A bound against a stuck microphone, not
# a measure of a sentence: eight seconds cut off a nine-second answer.
MERGE_WAIT = 30.0
# Once the candidate has been silent this long, the reply is prepared in the
# background while the configured silence runs out, so it can go out the moment
# it does. Thinking pauses are longer than this, which is what the configured
# silence is for; this only decides when to start working, never when to speak.
SPECULATE_MS = 600
# A sentence sharing this much of its words with one already said in the
# interview is held back. gemma-4-12b said the same two sentences fifty times
# each in an eleven-minute run, against a prompt that forbids it.
DUPLICATE_OVERLAP = 0.8
RECENT_SAID = 60
# Always given to whisper beside the problem's own vocabulary, which is its
# title and the starter's identifiers. Without them "O of n squared" came back
# as "O ,".
INTERVIEW_HOTWORDS = [
    "O of n", "O of n squared", "O of one", "O of log n", "O of n log n",
    "hash map", "dictionary", "two pointers", "binary search", "edge case",
]
# A candidate asking for a hint. CodeTrial counts hints through `log_hint`, which
# the model is told to call before every requested clue; once the ladder is
# spent every model tried stopped calling it and answered the request itself,
# so the request went uncounted. The shim makes the call instead, before the
# model speaks, and the model answers from its result.
HINT_REQUEST = re.compile(
    r"\b(?:(?:can|could|may) (?:i|you|we) (?:get|have|give(?: me)?|use)"
    r"|(?:give|show) me|i(?:'d| would)? (?:like|want|need)|i could use|is there|got|any(?:thing)?(?: like)?)"
    r"(?: (?:a|an|another|one|one more|some|any|a little|a small|more|other))*"
    r" (?:hint|clue|nudge)s?\b"
    r"|\b(?:another|one more|a) (?:hint|clue)(?:,)? please\b|^\s*(?:a |another )?hint\??\s*$",
    re.I,
)
HINT_DECLINED = re.compile(r"\b(?:don't|do not|no|without|not)\b[^.?!]{0,20}\b(?:hint|clue|nudge)", re.I)
AUDIO_CHUNK = OUT_RATE // 5  # 200 ms of output audio per message
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Gemma 4's thought-channel markup, handled as scripts/gemini-shim.py in
# CodeTrial handles it: with thinking off, a model with nothing to say opens
# empty channels until the limit, so generation stops at the fourth opener in a
# row and any markup left in the text is stripped before it is spoken.
THOUGHT_BLOCK = "<|channel>thought\n<channel|>"
THOUGHT_LOOP = "<channel|>" + THOUGHT_BLOCK * 2 + "<|channel>"
THOUGHT_MARKUP = re.compile(r"<\|channel>.*?(?:<channel\|>|$)|<channel\|>", re.S)


def asks_for_hint(text: str) -> bool:
    """The candidate's own words ask for a hint. Lines in square brackets are
    CodeTrial's platform events, which may mention hints themselves."""
    said = [line for line in text.splitlines() if not line.lstrip().startswith("[")]
    return any(HINT_REQUEST.search(line) and not HINT_DECLINED.search(line) for line in said)


def log(session: str, message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{session}] {message}", flush=True)


def convert_schema(node):
    """Gemini's OpenAPI subset to JSON Schema: lower-case types, no nullable."""
    if isinstance(node, list):
        return [convert_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in ("propertyOrdering", "nullable"):
            continue
        out[key] = value.lower() if key == "type" and isinstance(value, str) else convert_schema(value)
    if node.get("nullable"):
        return {"anyOf": [out, {"type": "null"}]}
    return out


class Models:
    """Loaded once and shared; one lock each, since one GPU runs them."""

    def __init__(self, voice: str):
        started = time.perf_counter()
        self.whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
        session = ort.InferenceSession(
            str(HERE / "models/kokoro-v1.0.onnx"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.kokoro = Kokoro.from_session(session, str(HERE / "models/voices-v1.0.bin"))
        self.vad = ort.InferenceSession(str(HERE / "models/silero_vad.onnx"), providers=["CPUExecutionProvider"])
        self.voice = voice
        self.gpu = asyncio.Lock()
        print(f"models loaded in {time.perf_counter() - started:.1f}s", flush=True)

    def transcribe(self, audio: np.ndarray, hotwords: str | None) -> str:
        segments, _ = self.whisper.transcribe(
            audio, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, hotwords=hotwords or None,
        )
        return "".join(segment.text for segment in segments).strip()

    def speak(self, text: str) -> np.ndarray:
        samples, _ = self.kokoro.create(text, voice=self.voice, speed=1.0, lang="en-us")
        return (np.clip(samples, -1, 1) * 32767).astype("<i2")


class Session:
    def __init__(self, socket, models: Models, args, sessions: dict):
        self.socket = socket
        self.models = models
        self.args = args
        self.sessions = sessions
        self.id = uuid.uuid4().hex[:8]
        self.messages: list[dict] = []
        self.tools: list[dict] = []
        self.silence_ms = 3000
        self.hotwords = None
        self.inputs: asyncio.Queue = asyncio.Queue()
        self.tool_waiters: dict[str, asyncio.Future] = {}
        self.speaking_until = 0.0
        self.busy = False
        self.turn_task: asyncio.Task | None = None
        # This turn's sentences as the room plays them: (text, start, end,
        # index of their assistant message, or None while still generating).
        self.played: list[list] = []
        self.barge_frames = 0
        self.transcribing = 0
        self.spec: asyncio.Task | None = None
        self.said: list[set[str]] = []
        # VAD
        self.vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.vad_context = np.zeros(VAD_CONTEXT, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.pre_roll: list[np.ndarray] = []
        self.utterance: list[np.ndarray] = []
        self.silent_frames = 0
        self.http = httpx.AsyncClient(timeout=120)

    async def send(self, message: dict) -> None:
        await self.socket.send(json.dumps(message))

    # --- setup --------------------------------------------------------------

    async def setup(self, setup: dict) -> None:
        system = "".join(part.get("text", "") for part in setup.get("systemInstruction", {}).get("parts", []))
        for group in setup.get("tools", []):
            for declaration in group.get("functionDeclarations", []):
                self.tools.append({"type": "function", "function": {
                    "name": declaration["name"],
                    "description": declaration.get("description", ""),
                    "parameters": convert_schema(declaration.get("parameters") or {"type": "OBJECT", "properties": {}}),
                }})
        detection = setup.get("realtimeInputConfig", {}).get("automaticActivityDetection", {})
        self.silence_ms = int(detection.get("silenceDurationMs", self.silence_ms))
        vocabulary = setup.get("inputAudioTranscription", {}).get("customVocabulary") or []
        self.hotwords = ", ".join([*vocabulary, *INTERVIEW_HOTWORDS])

        handle = (setup.get("sessionResumption") or {}).get("handle")
        if handle and handle in self.sessions:
            self.messages = self.sessions[handle]
            self.id = handle[:8]
            log(self.id, f"resumed with {len(self.messages)} messages")
        else:
            self.messages = [{"role": "system", "content": system}]
            log(self.id, f"new session: {len(system)} chars of instructions, {len(self.tools)} tools, silence {self.silence_ms} ms")
        await self.send({"setupComplete": {}})
        asyncio.create_task(self.prefill())

    async def prefill(self) -> None:
        """Put the instructions in llama-server's cache before anyone speaks."""
        started = time.perf_counter()
        try:
            await self.http.post(f"{self.args.llm}/v1/chat/completions", json={
                "messages": self.messages[:1] + [{"role": "user", "content": "."}],
                "tools": self.tools, "max_tokens": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            log(self.id, f"instructions prefilled in {time.perf_counter() - started:.1f}s")
        except Exception as error:
            log(self.id, f"prefill failed: {error}")

    # --- input --------------------------------------------------------------

    async def on_audio(self, data: bytes) -> None:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            await self.on_frame(frame)

    async def on_frame(self, frame: np.ndarray) -> None:
        probability, self.vad_state = self.models.vad.run(None, {
            "input": np.concatenate([self.vad_context, frame])[None, :],
            "state": self.vad_state,
            "sr": np.array(IN_RATE, dtype=np.int64),
        })
        self.vad_context = frame[-VAD_CONTEXT:]
        speech = float(probability[0][0]) > SPEECH_PROBABILITY
        floor = self.busy or time.monotonic() < self.speaking_until + ECHO_GUARD
        if not self.utterance:
            self.pre_roll = (self.pre_roll + [frame])[-PRE_ROLL_FRAMES:]
            if not speech:
                self.barge_frames = 0
                return
            if floor:
                # The interviewer has the floor: only sustained speech takes it.
                self.barge_frames += 1
                if self.barge_frames < BARGE_FRAMES:
                    return
                self.barge_frames = 0
                await self.interrupt()
            self.utterance = list(self.pre_roll)
            self.silent_frames = 0
            return
        self.utterance.append(frame)
        if speech:
            if self.spec is not None:
                # They went on: what was prepared answers the wrong sentence.
                self.spec.cancel()
                self.spec = None
            self.silent_frames = 0
        else:
            self.silent_frames += 1
        silent_ms = self.silent_frames * FRAME * 1000 / IN_RATE
        if self.spec is None and not speech and SPECULATE_MS <= silent_ms < self.silence_ms:
            self.spec = asyncio.create_task(self.speculate(np.concatenate(self.utterance)))
        if silent_ms >= self.silence_ms:
            audio = np.concatenate(self.utterance)
            self.utterance, self.pre_roll = [], []
            spec, self.spec = self.spec, None
            asyncio.create_task(self.on_utterance(audio, spec))

    async def speculate(self, audio: np.ndarray) -> dict:
        """Everything a reply needs, worked out while the configured silence
        runs: the transcript, the model's answer, and its audio. Nothing is
        sent. A tool call is only decided here, never made, since tools have
        effects (a hint is counted) that a sentence the candidate goes on with
        must not leave behind."""
        started = time.perf_counter()
        async with self.models.gpu:
            transcript = await asyncio.to_thread(self.models.transcribe, audio, self.hotwords)
        result = {"transcript": transcript, "stt_ms": 1000 * (time.perf_counter() - started)}
        if not transcript or self.busy or not self.inputs.empty():
            return result
        if self.counts_hints() and asks_for_hint(transcript):
            # Its reply comes from log_hint's result, which only the call
            # itself can give, and the call cannot be made ahead.
            return result
        base = len(self.messages)
        content, calls = await self.complete(self.messages + [{"role": "user", "content": transcript}])
        sentences = self.trim_repeats(self.sentences(content))
        prepared = []
        for sentence in sentences:
            async with self.models.gpu:
                prepared.append((sentence, await asyncio.to_thread(self.models.speak, sentence)))
        result.update(base=base, sentences=prepared, calls=calls, ready_ms=1000 * (time.perf_counter() - started))
        return result

    async def on_utterance(self, audio: np.ndarray, spec: asyncio.Task | None = None) -> None:
        heard_at = time.perf_counter()
        prepared = None
        self.transcribing += 1
        try:
            if spec is not None:
                try:
                    prepared = await spec
                except Exception as error:
                    log(self.id, f"speculation failed: {type(error).__name__}: {error}")
            text = prepared["transcript"] if prepared else None
            if text is None:
                async with self.models.gpu:
                    text = await asyncio.to_thread(self.models.transcribe, audio, self.hotwords)
            if not text:
                return
            ahead = " (prepared during the silence)" if prepared and "sentences" in prepared else ""
            log(self.id, f"heard ({len(audio) / IN_RATE:.1f}s, {1000 * (time.perf_counter() - heard_at):.0f} ms){ahead}: {text}")
            await self.send({"serverContent": {"inputTranscription": {"text": text}}})
            await self.inputs.put((text, heard_at, prepared if ahead else None))
        finally:
            # Only once the words are queued: a turn waiting to merge them
            # reads this count, and dropping it first let the turn start
            # without them.
            self.transcribing -= 1

    # --- turns --------------------------------------------------------------

    async def turns(self) -> None:
        while True:
            text, started, prepared = await self.inputs.get()
            # A candidate mid-sentence is finishing what this turn should
            # answer: wait for their words and take everything waiting as one
            # input, the way Live reads text and audio as one stream.
            deadline = time.monotonic() + MERGE_WAIT
            while (self.utterance or self.transcribing) and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            merged = 1
            while not self.inputs.empty():
                more, more_started, _ = self.inputs.get_nowait()
                text, started, merged = f"{text}\n{more}", max(started, more_started), merged + 1
            if merged > 1:
                prepared = None  # it answered only part of what is now asked
                log(self.id, f"answering {merged} inputs as one turn")
            self.busy = True
            self.played = []
            self.turn_task = asyncio.create_task(self.turn(text, started, prepared))
            await asyncio.wait({self.turn_task})
            if not self.turn_task.cancelled() and self.turn_task.exception():
                error = self.turn_task.exception()
                log(self.id, f"turn failed: {type(error).__name__}: {error}")
            self.busy = False

    async def interrupt(self) -> None:
        """The candidate spoke over the interviewer: stop, say so, and keep in
        the history only the sentences the room had started playing."""
        now = time.monotonic()
        task = self.turn_task
        generating = task is not None and not task.done()
        if generating:
            task.cancel()
            await asyncio.wait({task})
        heard = [entry for entry in self.played if entry[1] <= now]
        unheard = [entry for entry in self.played if entry[1] > now]
        if not generating and not unheard and now >= self.speaking_until:
            return  # nothing of the interviewer's was still going
        self.repair_history(heard, unheard, generating)
        self.played = []
        self.speaking_until = now
        await self.send({"serverContent": {"interrupted": True}})
        if generating:
            await self.send({"serverContent": {"turnComplete": True}})
        log(self.id, f"interrupted: {len(heard)} sentence(s) heard, {len(unheard)} cut")

    def repair_history(self, heard, unheard, generating) -> None:
        # Sentences already in an assistant message that never played come out
        # of it; sentences that played but whose generation was cancelled
        # before it was recorded go in as one message.
        for index in {entry[3] for entry in unheard if entry[3] is not None}:
            kept = [entry[0] for entry in heard if entry[3] == index]
            self.messages[index]["content"] = " ".join(kept)
        pending = [entry[0] for entry in heard if entry[3] is None]
        if generating and pending:
            self.messages.append({"role": "assistant", "content": " ".join(pending)})
        # A tool call left without its answer would make the next request
        # invalid, so it gets one.
        last_calls = next(
            (i for i in range(len(self.messages) - 1, -1, -1) if self.messages[i].get("tool_calls")), None
        )
        if last_calls is not None:
            answered = {m.get("tool_call_id") for m in self.messages[last_calls + 1:] if m["role"] == "tool"}
            for call in self.messages[last_calls]["tool_calls"]:
                if call["id"] not in answered:
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "content": json.dumps({"result": "interrupted before it was answered"})})

    def counts_hints(self) -> bool:
        return any(tool["function"]["name"] == "log_hint" for tool in self.tools)

    async def log_requested_hint(self) -> None:
        """Make the log_hint call the request needs, as if the model had,
        so the model answers from the result."""
        call = {"id": f"call-{uuid.uuid4().hex[:8]}", "name": "log_hint", "arguments": json.dumps({"requested": True})}
        self.record([], [call])
        await self.call_tools([call])

    async def turn(self, text: str, started: float, prepared: dict | None = None) -> None:
        self.messages.append({"role": "user", "content": text})
        first_audio = None
        more = True
        logged = self.counts_hints() and asks_for_hint(text)
        if logged:
            log(self.id, "hint requested: logging it")
            await self.log_requested_hint()
        usable = prepared is not None and prepared.get("base") == len(self.messages) - 1
        if usable:
            # Worked out during the silence against this same history: send it.
            for sentence, audio in prepared["sentences"]:
                at = await self.emit(sentence, audio)
                first_audio = first_audio or at
            calls = prepared["calls"]
            self.record([sentence for sentence, _ in prepared["sentences"]], calls)
            more = bool(calls)
            if calls:
                await self.call_tools(calls, logged)
        for _ in range(4):
            if not more:
                break
            spoken, calls = await self.generate()
            first_audio = first_audio or spoken
            if not calls:
                break
            await self.call_tools(calls, logged)
        await self.send({"serverContent": {"turnComplete": True}})
        handle = self.id + uuid.uuid4().hex
        self.sessions[handle] = self.messages
        await self.send({"sessionResumptionUpdate": {"newHandle": handle, "resumable": True}})
        if first_audio:
            how = "prepared" if usable else "generated"
            log(self.id, f"turn done ({how}), first audio {1000 * (first_audio - started):.0f} ms after the input was complete")

    def request(self, messages: list[dict], stream: bool = False) -> dict:
        return {"messages": messages, "tools": self.tools, "temperature": 0.7, "max_tokens": 400,
                "stream": stream, "stop": [THOUGHT_LOOP], "chat_template_kwargs": {"enable_thinking": False},
                **self.args.sampling}

    async def complete(self, messages: list[dict]) -> tuple[str, list[dict]]:
        response = await self.http.post(f"{self.args.llm}/v1/chat/completions", json=self.request(messages))
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        calls = [
            {"id": call.get("id") or f"call-{uuid.uuid4().hex[:8]}", "name": call["function"]["name"],
             "arguments": call["function"].get("arguments") or "{}"}
            for call in message.get("tool_calls") or []
        ]
        return THOUGHT_MARKUP.sub("", message.get("content") or "").strip(), calls

    @staticmethod
    def sentences(text: str) -> list[str]:
        return [part.strip() for part in SENTENCE_END.split(THOUGHT_MARKUP.sub("", text)) if part.strip()]

    @staticmethod
    def words(sentence: str) -> set[str]:
        return set(re.findall(r"[a-z0-9']+", sentence.lower()))

    def is_repeat(self, sentence: str) -> bool:
        """Close to something already said in this interview. Short sentences
        ("Gotcha.", "Take your time.") are left alone: saying them again is
        how people talk."""
        words = self.words(sentence)
        if len(words) < 5:
            return False
        return any(len(words & said) / len(words | said) >= DUPLICATE_OVERLAP for said in self.said[-RECENT_SAID:])

    def trim_repeats(self, sentences: list[str]) -> list[str]:
        """Drop the sentences that repeat earlier ones, but only when something
        new is left to say. A reply made only of repeats is said anyway: asked
        to regenerate with a note, gemma read the note aloud, and answering a
        request with silence is worse than answering it again."""
        fresh = [sentence for sentence in sentences if not self.is_repeat(sentence)]
        held = len(sentences) - len(fresh)
        if held and fresh:
            log(self.id, f"held back {held} repeated sentence(s)")
        return fresh or sentences

    def record(self, spoken: list[str], calls: list[dict]) -> None:
        message = {"role": "assistant", "content": " ".join(spoken)}
        for call in calls:
            call["id"] = call["id"] or f"call-{uuid.uuid4().hex[:8]}"
        if calls:
            message["tool_calls"] = [
                {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"] or "{}"}}
                for call in calls
            ]
        self.messages.append(message)
        for entry in self.played:
            if entry[3] is None:
                entry[3] = len(self.messages) - 1

    async def generate(self):
        """Stream one completion, speaking each new sentence as it completes.
        A sentence repeating an earlier one waits: if anything new follows it
        is dropped, and if nothing does it is said after all (see
        `trim_repeats`). Returns when the first audio went out, and any tool
        calls."""
        spoken_at, calls, buffer = None, {}, ""
        spoken: list[str] = []
        held: list[str] = []

        async def say(sentence: str) -> None:
            nonlocal spoken_at
            async with self.models.gpu:
                audio = await asyncio.to_thread(self.models.speak, sentence)
            at = await self.emit(sentence, audio)
            spoken.append(sentence)
            spoken_at = spoken_at or at

        async def take(sentence: str) -> None:
            if self.is_repeat(sentence):
                held.append(sentence)
            else:
                await say(sentence)

        async with self.http.stream("POST", f"{self.args.llm}/v1/chat/completions",
                                    json=self.request(self.messages, stream=True)) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0].get("delta") or {}
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(call.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    slot["id"] = call.get("id") or slot["id"]
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
                if delta.get("content"):
                    buffer += delta["content"]
                    *done, buffer = SENTENCE_END.split(THOUGHT_MARKUP.sub("", buffer)) or [""]
                    for sentence in done:
                        if sentence.strip():
                            await take(sentence.strip())
        tail = THOUGHT_MARKUP.sub("", buffer).strip()
        if tail:
            await take(tail)
        calls = [call for call in calls.values() if call["name"]]
        if held and not spoken and not calls:
            for sentence in held:
                await say(sentence)
        elif held:
            log(self.id, f"held back {len(held)} repeated sentence(s)")
        self.record(spoken, calls)
        return spoken_at, calls

    async def emit(self, sentence: str, audio: np.ndarray) -> float:
        sent_at = time.perf_counter()
        await self.send({"serverContent": {"outputTranscription": {"text": sentence + " "}}})
        for start in range(0, len(audio), AUDIO_CHUNK):
            chunk = audio[start:start + AUDIO_CHUNK].tobytes()
            await self.send({"serverContent": {"modelTurn": {"parts": [{"inlineData": {
                "mimeType": f"audio/pcm;rate={OUT_RATE}", "data": base64.b64encode(chunk).decode(),
            }}]}}})
        # The room plays it out in real time; until then this is the
        # interviewer's turn, and the microphone hears the interviewer.
        start = max(self.speaking_until, time.monotonic())
        self.speaking_until = start + len(audio) / OUT_RATE
        self.played.append([sentence, start, self.speaking_until, None])
        self.said.append(self.words(sentence))
        log(self.id, f"said: {sentence}")
        return sent_at

    async def call_tools(self, calls: list[dict], logged: bool = False) -> None:
        """Send the calls to CodeTrial and wait for its answers. With the
        request already logged by the shim, the model's own requested log_hint
        is answered here: sent on, the one request would count twice."""
        if logged:
            for call in [call for call in calls if self.is_requested_hint(call)]:
                calls.remove(call)
                log(self.id, "tool call: log_hint already made for this request, not sent")
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(
                    {"result": "Already recorded for this request; answer from the result above."})})
            if not calls:
                return
        loop = asyncio.get_running_loop()
        for call in calls:
            self.tool_waiters[call["id"]] = loop.create_future()
        await self.send({"toolCall": {"functionCalls": [
            {"id": call["id"], "name": call["name"], "args": json.loads(call["arguments"] or "{}")}
            for call in calls
        ]}})
        log(self.id, f"tool call: {', '.join(call['name'] for call in calls)}")
        for call in calls:
            # Awaited where it sits and removed only after: popped first, the
            # answer arriving meanwhile would find no waiter and be dropped.
            try:
                response = await asyncio.wait_for(self.tool_waiters[call["id"]], timeout=30)
            finally:
                self.tool_waiters.pop(call["id"], None)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(response)})

    @staticmethod
    def is_requested_hint(call: dict) -> bool:
        if call["name"] != "log_hint":
            return False
        try:
            return bool(json.loads(call["arguments"] or "{}").get("requested"))
        except json.JSONDecodeError:
            return False

    def on_tool_response(self, message: dict) -> None:
        for response in message.get("functionResponses", []):
            waiter = self.tool_waiters.get(response.get("id"))
            if waiter and not waiter.done():
                waiter.set_result(response.get("response", {}))

    # --- socket -------------------------------------------------------------

    async def run(self) -> None:
        worker = asyncio.create_task(self.turns())
        try:
            async for raw in self.socket:
                message = json.loads(raw)
                if "setup" in message:
                    await self.setup(message["setup"])
                elif "realtimeInput" in message:
                    realtime = message["realtimeInput"]
                    if "audio" in realtime:
                        await self.on_audio(base64.b64decode(realtime["audio"]["data"]))
                    if realtime.get("text"):
                        log(self.id, f"event: {realtime['text'][:100]!r}")
                        await self.inputs.put((realtime["text"], time.perf_counter(), None))
                elif "toolResponse" in message:
                    self.on_tool_response(message["toolResponse"])
        except websockets.ConnectionClosed:
            pass
        finally:
            worker.cancel()
            await self.http.aclose()
            log(self.id, "closed")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listen", default="127.0.0.1:8191")
    parser.add_argument("--llm", default="http://127.0.0.1:8180", help="llama-server base URL")
    parser.add_argument("--voice", default="am_michael", help="Kokoro voice")
    parser.add_argument("--sampling", type=json.loads, default={},
                        help='extra llama-server sampling fields, e.g. \'{"dry_multiplier": 0.8}\'')
    args = parser.parse_args()
    models = Models(args.voice)
    sessions: dict[str, list] = {}
    host, port = args.listen.rsplit(":", 1)
    async with websockets.serve(
        lambda socket: Session(socket, models, args, sessions).run(), host, int(port), max_size=None,
    ):
        print(f"listening on ws://{args.listen}/live, LLM at {args.llm}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
