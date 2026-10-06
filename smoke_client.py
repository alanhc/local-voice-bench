"""Talk to live_shim.py the way CodeTrial's agent does, without CodeTrial.

Sends a setup with the real interviewer instructions and tools, the greeting
event, then a recorded candidate line in real time followed by silence, and
answers whatever tool calls come back the way the agent would.
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
INSTRUCTIONS = json.loads((HERE / "golden-prompts.json").read_text())["instructions"]
GREETING = json.loads((HERE / "golden-prompts.json").read_text())["greeting"]
TOOLS = [
    {"name": "read_editor", "description": "The editor's language and numbered code, latest test run and minutes left.", "parameters": {"type": "OBJECT", "properties": {"fromLine": {"type": "INTEGER"}}}},
    {"name": "log_hint", "description": "Record a hint. Before a hint the candidate asked for, call with requested true and give the clue it returns.", "parameters": {"type": "OBJECT", "properties": {"requested": {"type": "BOOLEAN"}}, "required": ["requested"]}},
    {"name": "end_interview", "description": "Close the interview because it is genuinely finished."},
]
ANSWERS = {
    "read_editor": {"result": "Editor language: python\n(the editor is currently empty)\nTIMER: about 41 minutes remain."},
    "log_hint": {"result": "Recorded. Total hints so far: 1. Hint rung 1, the only clue to give now: Take one transaction of amount 4 with a target of 0. What exact amount would its partner need to have? Say it as one question in your own words and stop."},
}


async def listen(socket, until_turns, log):
    """Collect frames until `until_turns` turnComplete frames have arrived.
    Returns the seconds of interviewer audio received."""
    turns = 0
    audio_seconds = 0.0
    while turns < until_turns:
        message = json.loads(await asyncio.wait_for(socket.recv(), timeout=60))
        content = message.get("serverContent", {})
        for part in content.get("modelTurn", {}).get("parts", []):
            audio_seconds += len(base64.b64decode(part["inlineData"]["data"])) / 2 / 24000
        if "outputTranscription" in content:
            log(f"Jim: {content['outputTranscription']['text'].strip()}")
        if "inputTranscription" in content:
            log(f"heard: {content['inputTranscription']['text']}")
        if "toolCall" in message:
            calls = message["toolCall"]["functionCalls"]
            log(f"tool call: {[(c['name'], c['args']) for c in calls]}")
            await socket.send(json.dumps({"toolResponse": {"functionResponses": [
                {"name": c["name"], "id": c["id"], "response": ANSWERS.get(c["name"], {"result": "OK"})} for c in calls
            ]}}))
        if content.get("turnComplete"):
            turns += 1
            log(f"turnComplete ({audio_seconds:.1f}s of audio so far)")
        if "sessionResumptionUpdate" in message:
            log(f"resumption handle {message['sessionResumptionUpdate']['newHandle'][:12]}...")
    return audio_seconds


async def main():
    started = time.perf_counter()
    log = lambda text: print(f"{time.perf_counter() - started:6.2f}s  {text}", flush=True)
    async with websockets.connect(f"{URL}?key=local", max_size=None) as socket:
        await socket.send(json.dumps({"setup": {
            "model": "models/local",
            "systemInstruction": {"parts": [{"text": INSTRUCTIONS}]},
            "tools": [{"functionDeclarations": TOOLS}],
            "inputAudioTranscription": {"customVocabulary": ["Chargeback Pair Match", "amounts", "target"]},
            "realtimeInputConfig": {"automaticActivityDetection": {"silenceDurationMs": 1000}},
        }}))
        reply = json.loads(await socket.recv())
        log(f"setup reply: {reply}")

        await socket.send(json.dumps({"realtimeInput": {"text": GREETING}}))
        greeting_seconds = await listen(socket, 1, log)

        # Wait out the greeting's playback, as a candidate would, then speak.
        log(f"waiting {greeting_seconds:.1f}s for the greeting to play out")
        await asyncio.sleep(greeting_seconds + 0.5)
        audio, rate = sf.read(HERE / "out/candidate-1.wav", dtype="float32")
        pcm = (audio * 32767).astype("<i2")
        silence = np.zeros(int(1.5 * rate), dtype="<i2")
        sender_started = time.perf_counter()

        async def stream():
            for chunk_start in range(0, len(pcm) + len(silence), 320):  # 20 ms frames
                block = np.concatenate([pcm, silence])[chunk_start:chunk_start + 320]
                await socket.send(json.dumps({"realtimeInput": {"audio": {
                    "data": base64.b64encode(block.tobytes()).decode(), "mimeType": "audio/pcm;rate=16000"}}}))
                await asyncio.sleep(0.02)

        log("candidate speaks: 'I'm stuck. Can I get a hint?' then 1.5 s of silence")
        await asyncio.gather(stream(), listen(socket, 1, log))
        log(f"done; speech ended at {len(pcm) / rate:.1f}s into the stream")


asyncio.run(main())
