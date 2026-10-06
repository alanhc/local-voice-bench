"""Stage 0, LLM half: how long gemma takes to start speaking as the interviewer.

Uses CodeTrial's real interviewer instructions (tests/golden/prompts.json,
`instructions`) and tools shaped like `live_tool_declarations`, against
llama-server's OpenAI endpoint, thinking off, streamed. Measures, from the
moment the candidate's turn is complete:

- first token, and the end of the first sentence (what TTS can start on);
- cold, with a system prompt the server has not cached, and warm, the same
  conversation one turn later, which is the ordinary case mid-interview;
- a hint request, where the model has to call tools and hear back before it
  speaks, so the tool round trip is inside the wait.
"""

import json
import os
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

import httpx

BASE = os.environ.get("LLM_BASE", "http://127.0.0.1:8180")
RUNS = int(os.environ.get("RUNS", "5"))
INSTRUCTIONS = json.loads((Path(__file__).parent / "golden-prompts.json").read_text())["instructions"]

TOOLS = [
    {"type": "function", "function": {"name": "read_editor", "description": "The recorded framework phases, editor's language and numbered code, latest test run and minutes left.", "parameters": {"type": "object", "properties": {"fromLine": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "log_hint", "description": "Record a hint. Before a hint the candidate asked for, call with requested true and give the clue it returns; after any other hint, call with requested false.", "parameters": {"type": "object", "properties": {"requested": {"type": "boolean"}}, "required": ["requested"]}}},
    {"type": "function", "function": {"name": "record_framework_evidence", "description": "Record trusted REACTO or STAR evidence only after it is present in candidate speech, an editor snapshot, or a test event.", "parameters": {"type": "object", "properties": {"phase": {"type": "string"}, "source": {"type": "string"}, "kind": {"type": "string"}, "confidence": {"type": "integer"}, "summary": {"type": "string"}}, "required": ["phase", "source", "kind", "confidence", "summary"]}}},
    {"type": "function", "function": {"name": "end_interview", "description": "Close the interview because it is genuinely finished.", "parameters": {"type": "object", "properties": {}}}},
]
TOOL_ANSWERS = {
    "read_editor": "Editor language: python\n(the editor is currently empty)\n\nNo test run was recorded.\n\nTIMER: about 41 minutes remain on the candidate's countdown.",
    "log_hint": "Recorded. Total hints so far: 1. Hint rung 1, the only clue to give now: Take one transaction of amount 4 with a target of 0. What exact amount would its partner need to have? Say it as one question or nudge in your own words, fitted to their current code, and stop for their response.",
    "record_framework_evidence": "Recorded.",
    "end_interview": "Ended.",
}
HISTORY = [
    {"role": "user", "content": "[SYSTEM EVENT] The interview starts now. Greet the candidate in at most four short sentences."},
    {"role": "assistant", "content": "Hi, I'm Jim. Our payments team needs to find two transactions in a statement that add up to a disputed charge. Which language would you like to use?"},
    {"role": "user", "content": "I'd like to use Python for this one."},
    {"role": "assistant", "content": "Gotcha, Python it is. Could you restate the inputs, outputs and any constraints in your own words?"},
]
SENTENCE_END = re.compile(r"[.!?](\s|$)")


def turn(client, messages):
    """One streamed completion. Returns (first token, first sentence, end,
    text, tool calls), times in seconds from the request."""
    started = time.perf_counter()
    first = sentence = None
    text = ""
    calls = {}
    with client.stream(
        "POST",
        f"{BASE}/v1/chat/completions",
        json={
            "messages": messages,
            "tools": TOOLS,
            "temperature": 0.7,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0].get("delta") or {}
            if delta.get("content") or delta.get("tool_calls"):
                first = first or time.perf_counter() - started
            if delta.get("content"):
                text += delta["content"]
                if sentence is None and SENTENCE_END.search(text):
                    sentence = time.perf_counter() - started
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), {"id": None, "name": "", "arguments": ""})
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    end = time.perf_counter() - started
    if text and sentence is None:
        sentence = end
    return first, sentence, end, text, list(calls.values())


def speak(client, messages):
    """Run the model, answering tool calls, until it says something. Returns
    the time to the first spoken sentence and the tool names it called."""
    started = time.perf_counter()
    called = []
    messages = list(messages)
    for _ in range(4):
        first, sentence, end, text, calls = turn(client, messages)
        if text.strip():
            offset = time.perf_counter() - started - end
            return offset + first, offset + sentence, called, text
        if not calls:
            return None, None, called, ""
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": call["id"] or f"call-{i}", "type": "function", "function": {"name": call["name"], "arguments": call["arguments"] or "{}"}}
            for i, call in enumerate(calls)
        ]})
        for i, call in enumerate(calls):
            called.append(call["name"])
            messages.append({"role": "tool", "tool_call_id": call["id"] or f"call-{i}", "content": TOOL_ANSWERS.get(call["name"], "OK")})
    return None, None, called, ""


def ms(values):
    values = [value for value in values if value is not None]
    if not values:
        return "n/a"
    ordered = sorted(values)
    p90 = ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]
    return f"median {statistics.median(values) * 1000:6.0f} ms   p90 {p90 * 1000:6.0f} ms"


def main():
    client = httpx.Client()
    props = client.get(f"{BASE}/props").json()
    settings = props.get("default_generation_settings", {})
    print(f"server: {Path(props.get('model_path', '?')).name}, n_ctx per slot {settings.get('n_ctx')}, slots {props.get('total_slots')}")
    tokens = client.post(f"{BASE}/tokenize", json={"content": INSTRUCTIONS}).json()["tokens"]
    print(f"interviewer instructions: {len(INSTRUCTIONS)} chars, {len(tokens)} tokens")

    def conversation(system, user):
        return [{"role": "system", "content": system}, *HISTORY, {"role": "user", "content": user}]

    restate = "So the input is a list of transaction amounts and a target, and I return the indices of the two that add up to it."

    # Cold: a fresh nonce at the top of the system prompt defeats the cache.
    cold_first, cold_sentence = [], []
    for _ in range(RUNS):
        system = f"[session {uuid.uuid4()}]\n{INSTRUCTIONS}"
        first, sentence, _, text, _ = turn(client, conversation(system, restate))
        cold_first.append(first)
        cold_sentence.append(sentence)
    print("\ncold (instructions not cached):")
    print(f"  first token     {ms(cold_first)}")
    print(f"  first sentence  {ms(cold_sentence)}")

    # Warm: the same conversation, a new candidate turn each time. The prefix
    # up to the previous turn is already in the slot's cache.
    system = f"[session {uuid.uuid4()}]\n{INSTRUCTIONS}"
    turn(client, conversation(system, restate))
    lines = [
        "Okay, so for an ordinary example, amounts four, seven and minus four with a target of zero give positions zero and two.",
        "And an edge case would be two equal amounts, like three and three with a target of six.",
        "I think I'd start with the brute force, two nested loops over the list.",
        "That's O of n squared, which might be slow for ten thousand transactions.",
        "Could I keep a dictionary of what I've seen so far?",
    ]
    warm_first, warm_sentence, sample = [], [], ""
    for line in lines[:RUNS]:
        first, sentence, _, text, _ = turn(client, conversation(system, line))
        warm_first.append(first)
        warm_sentence.append(sentence)
        sample = sample or text
    print("\nwarm (same session, next candidate turn):")
    print(f"  first token     {ms(warm_first)}")
    print(f"  first sentence  {ms(warm_sentence)}")
    print(f"  e.g. {sample.strip()[:160]!r}")

    # A hint request: tools before speech.
    hint_first, hint_sentence, tools_seen = [], [], []
    for _ in range(RUNS):
        first, sentence, called, text = speak(client, conversation(system, "I'm stuck. Can I get a hint?"))
        hint_first.append(first)
        hint_sentence.append(sentence)
        tools_seen.append("+".join(called) or "(no tool)")
    print("\nhint request (tool round trips inside the wait):")
    print(f"  first token     {ms(hint_first)}")
    print(f"  first sentence  {ms(hint_sentence)}")
    print(f"  tools called    {', '.join(tools_seen)}")


if __name__ == "__main__":
    main()
