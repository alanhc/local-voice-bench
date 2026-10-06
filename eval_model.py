"""How a model behaves as the interviewer, fast: candidate lines go to
live_shim.py as text, skipping speech recognition and real-time playback, and
tool calls are answered the way CodeTrial answers them, hint ladder included.

    python eval_model.py LABEL [ws://127.0.0.1:8191/live]

Two scripts: a realistic Two Sum interview in twenty lines, and the same hint
request thirty times. Reported per script: how many of the interviewer's
sentences nearly repeat an earlier one, whether hint requests were logged,
markup leaking into speech, empty replies, and the time to the first sentence.
"""

import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path

import websockets

HERE = Path(__file__).parent
LABEL = sys.argv[1] if len(sys.argv) > 1 else "model"
URL = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8191/live"
PROMPTS = json.loads((HERE / "golden-prompts.json").read_text())
TOOLS = [
    {"name": "read_editor", "description": "The recorded framework phases, editor's language and numbered code, latest test run and minutes left.", "parameters": {"type": "OBJECT", "properties": {"fromLine": {"type": "INTEGER"}}}},
    {"name": "log_hint", "description": "Record a hint. Before a hint the candidate asked for, call with requested true and give the clue it returns; after any other hint, call with requested false.", "parameters": {"type": "OBJECT", "properties": {"requested": {"type": "BOOLEAN"}}, "required": ["requested"]}},
    {"name": "record_framework_evidence", "description": "Record trusted REACTO or STAR evidence only after it is present in candidate speech, an editor snapshot, or a test event.", "parameters": {"type": "OBJECT", "properties": {"phase": {"type": "STRING"}, "source": {"type": "STRING"}, "kind": {"type": "STRING"}, "confidence": {"type": "INTEGER"}, "summary": {"type": "STRING"}}, "required": ["phase", "source", "kind", "confidence", "summary"]}},
    {"name": "end_interview", "description": "Close the interview because it is genuinely finished and there is nothing further to ask."},
]
RUNGS = [
    "Take one transaction of amount 4 with a target of 0. What exact amount would its partner need to have?",
    "As you move through the statement, what could you remember about earlier transactions so you can tell immediately whether the partner you need already went by?",
]
WITHHELD = ("Not counted as a hint; total hints so far: 2. The next rung names the key step and stays withheld "
            "until the candidate has put an approach of their own into words or code. Give no clue this turn: in one "
            "short sentence, ask what they would try first, even a slow version, and wait. Do not restate an earlier "
            "clue, and name no technique, data structure, ordering, or step.")
REALISTIC = [
    "I'd like to use Python for this one.",
    "So I get a list of transaction amounts and a target, and I return the indices of the two that add up to it.",
    "Can the amounts be negative, like refunds?",
    "I'm stuck. Can I get a hint?",
    "Hmm, so the partner of four would be minus four.",
    "I still don't see how to find it quickly. Could I get another hint?",
    "I guess I could check every pair with two nested loops.",
    "That would be O of n squared.",
    "Can I get one more hint?",
    "Maybe I could remember the amounts I've already seen as I go?",
    "I'd use a dictionary from amount to its index.",
    "Okay, I've written the loop. Could you take a look?",
    "I ran the tests and one of them failed.",
    "Oh, I was inserting before checking. I'll check the dictionary first.",
    "All the tests pass now.",
    "Time is O of n and space is O of n.",
    "I'm not sure what else I could optimize. Any hint?",
    "Could we sort and use two pointers instead?",
    "That would lose the original indices, though.",
    "I think I'm done with this one.",
]
REPEATED = ["I'm stuck. Can I get a hint?"] * 30
EDITOR = ("Editor language: python\n 1 def two_sum(amounts, target):\n 2     seen = {}\n 3     for i, a in enumerate(amounts):\n"
          " 4         if target - a in seen:\n 5             return [seen[target - a], i]\n 6         seen[a] = i\n\n"
          "Latest test run: 5/5 passed.\nTIMER: about 30 minutes remain.")


def words(sentence):
    return set(re.findall(r"[a-z0-9']+", sentence.lower()))


async def run(lines, name):
    hints_given = 0
    sentences, firsts, logged, empties, leaks = [], [], 0, 0, 0
    hint_requests, doubled, unasked = 0, 0, 0
    async with websockets.connect(f"{URL}?key=local", max_size=None) as socket:
        await socket.send(json.dumps({"setup": {
            "systemInstruction": {"parts": [{"text": PROMPTS["instructions"]}]},
            "tools": [{"functionDeclarations": TOOLS}],
            "realtimeInputConfig": {"automaticActivityDetection": {"silenceDurationMs": 3000}},
        }}))
        await socket.recv()
        for line in [PROMPTS["greeting"], *lines]:
            asked = "hint" in line.lower() and not line.startswith("[")
            hint_requests += asked
            called_hint, requested_calls = False, 0
            started, first, said = time.perf_counter(), None, []
            await socket.send(json.dumps({"realtimeInput": {"text": line}}))
            while True:
                message = json.loads(await asyncio.wait_for(socket.recv(), timeout=120))
                content = message.get("serverContent", {})
                if "outputTranscription" in content:
                    text = content["outputTranscription"]["text"].strip()
                    first = first or time.perf_counter() - started
                    said.append(text)
                if "toolCall" in message:
                    answers = []
                    for call in message["toolCall"]["functionCalls"]:
                        result = "Recorded."
                        if call["name"] == "read_editor":
                            result = EDITOR
                        elif call["name"] == "log_hint":
                            called_hint = called_hint or bool(call["args"].get("requested"))
                            requested_calls += bool(call["args"].get("requested"))
                            if call["args"].get("requested"):
                                if hints_given < len(RUNGS):
                                    hints_given += 1
                                    result = f"Recorded. Total hints so far: {hints_given}. Hint rung {hints_given}, the only clue to give now: {RUNGS[hints_given - 1]} Say it as one question or nudge in your own words and stop."
                                else:
                                    result = WITHHELD
                            else:
                                result = f"Recorded. Total hints so far: {hints_given}."
                        answers.append({"name": call["name"], "id": call["id"], "response": {"result": result}})
                    await socket.send(json.dumps({"toolResponse": {"functionResponses": answers}}))
                if content.get("turnComplete"):
                    break
            logged += asked and called_hint
            doubled += requested_calls > 1
            unasked += requested_calls > 0 and not asked
            empties += not said
            leaks += any(re.search(r"<\|?channel|Platform note|\[SYSTEM", s) for s in said)
            if first:
                firsts.append(first)
            sentences.extend(said)
            print(f"  [{name}] {line[:60]!r:64} -> {' '.join(said)[:150]}", flush=True)

    near, seen = 0, []
    for sentence in sentences:
        w = words(sentence)
        if len(w) >= 5 and any(len(w & v) / len(w | v) >= 0.8 for v in seen):
            near += 1
        seen.append(w)
    return {
        "turns": len(lines) + 1,
        "sentences": len(sentences),
        "near_repeats": f"{near} ({100 * near / max(1, len(sentences)):.0f}%)",
        "hint_requests_logged": f"{logged}/{hint_requests}",
        "counted_twice": doubled,
        "counted_unasked": unasked,
        "empty_replies": empties,
        "markup_leaks": leaks,
        "first_sentence_median_ms": round(1000 * statistics.median(firsts)) if firsts else None,
    }


async def main():
    results = {}
    for name, lines in [("realistic", REALISTIC), ("repeated", REPEATED)]:
        results[name] = await run(lines, name)
    print(json.dumps({LABEL: results}, indent=1))
    with open(HERE / "eval-results.jsonl", "a") as out:
        out.write(json.dumps({"label": LABEL, "at": time.strftime("%Y-%m-%d %H:%M"), **results}) + "\n")


asyncio.run(main())
