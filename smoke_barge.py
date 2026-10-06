"""Barge-in and input merging against live_shim.py, without CodeTrial.

1. The candidate speaks while the greeting is still playing: expect
   `interrupted` within a few hundred ms of the speech starting, then the
   transcript and a reply to what they said.
2. A platform event arrives while the candidate is mid-sentence: expect one
   turn answering both, not two.
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

HERE = Path(__file__).parent
URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8191/live"
PROMPTS = json.loads((HERE / "golden-prompts.json").read_text())
TOOLS = [
    {"name": "log_hint", "description": "Record a hint. Before a hint the candidate asked for, call with requested true and give the clue it returns.", "parameters": {"type": "OBJECT", "properties": {"requested": {"type": "BOOLEAN"}}, "required": ["requested"]}},
]
NUDGE = "[SYSTEM EVENT] The candidate has been quiet for a while. Check in briefly."


def pcm(path, rate=16000):
    audio, _ = sf.read(path, dtype="float32")
    return (audio * 32767).astype("<i2")


async def main():
    started = time.perf_counter()
    events = []

    def log(text):
        events.append((time.perf_counter() - started, text))
        print(f"{time.perf_counter() - started:6.2f}s  {text}", flush=True)

    async with websockets.connect(f"{URL}?key=local", max_size=None) as socket:
        await socket.send(json.dumps({"setup": {
            "systemInstruction": {"parts": [{"text": PROMPTS["instructions"]}]},
            "tools": [{"functionDeclarations": TOOLS}],
            "realtimeInputConfig": {"automaticActivityDetection": {"silenceDurationMs": 1000}},
        }}))
        await socket.recv()

        async def reader():
            async for raw in socket:
                message = json.loads(raw)
                content = message.get("serverContent", {})
                if content.get("interrupted"):
                    log("<< interrupted")
                if "inputTranscription" in content:
                    log(f"<< heard: {content['inputTranscription']['text']}")
                if "outputTranscription" in content:
                    log(f"<< Jim: {content['outputTranscription']['text'].strip()}")
                if content.get("turnComplete"):
                    log("<< turnComplete")
                if "toolCall" in message:
                    calls = message["toolCall"]["functionCalls"]
                    log(f"<< tool call {[c['name'] for c in calls]}")
                    await socket.send(json.dumps({"toolResponse": {"functionResponses": [
                        {"name": c["name"], "id": c["id"], "response": {"result": "Recorded. Hint rung 1: what amount would the partner of 4 need, with a target of 0?"}}
                        for c in calls]}}))

        reading = asyncio.create_task(reader())

        async def speak(samples, then_silence=1.6, during=None):
            stream = np.concatenate([samples, np.zeros(int(then_silence * 16000), dtype="<i2")])
            for i, start in enumerate(range(0, len(stream), 320)):
                if during and i == during[0]:
                    await during[1]()
                await socket.send(json.dumps({"realtimeInput": {"audio": {
                    "data": base64.b64encode(stream[start:start + 320].tobytes()).decode(),
                    "mimeType": "audio/pcm;rate=16000"}}}))
                await asyncio.sleep(0.02)

        # 1. Speak over the greeting, which plays for well over ten seconds.
        await socket.send(json.dumps({"realtimeInput": {"text": PROMPTS["greeting"]}}))
        while not any("Jim:" in text for _, text in events):
            await asyncio.sleep(0.05)
        await asyncio.sleep(2.0)  # two seconds into the greeting's playback
        spoke_at = time.perf_counter() - started
        log(">> candidate speaks over the greeting: 'I'm stuck. Can I get a hint?'")
        await speak(pcm(HERE / "out/candidate-1.wav"))
        await asyncio.sleep(6)
        interrupted = [t for t, text in events if text == "<< interrupted"]
        if interrupted:
            log(f"== interrupted {1000 * (interrupted[0] - spoke_at):.0f} ms after the candidate started")
        else:
            log("== NO interruption")

        # Let any reply finish playing before the next part.
        await asyncio.sleep(12)
        before = sum(1 for _, text in events if text == "<< turnComplete")

        # 2. A nudge lands while the candidate is mid-sentence.
        async def nudge():
            log(">> platform event mid-sentence: silence nudge")
            await socket.send(json.dumps({"realtimeInput": {"text": NUDGE}}))

        log(">> candidate speaks a long line, nudge one second in")
        await speak(pcm(HERE / "out/candidate-4.wav"), during=(50, nudge))
        await asyncio.sleep(10)
        after = sum(1 for _, text in events if text == "<< turnComplete")
        log(f"== turns answering the line and the nudge: {after - before}")
        reading.cancel()


asyncio.run(main())
