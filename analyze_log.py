"""Summarize one interview from live_shim.py's log: latency, repeats, barge-ins.

    python analyze_log.py shim.log [first-line]
"""

import re
import statistics
import sys
from collections import Counter


def words(sentence):
    return set(re.findall(r"[a-z0-9']+", sentence.lower()))


def main(path, start=0):
    lines = open(path).read().splitlines()[start:]
    text = "\n".join(lines)
    said = re.findall(r"said: (.*)", text)
    near, seen = 0, []
    for sentence in said:
        w = words(sentence)
        if len(w) >= 5 and any(len(w & v) / len(w | v) >= 0.8 for v in seen):
            near += 1
        seen.append(w)
    turns = re.findall(r"turn done(?: \((\w+)\))?, first audio (\d+) ms", text)
    latencies = sorted(int(ms) for _, ms in turns)
    prepared = sum(1 for how, _ in turns if how == "prepared")
    print(f"sessions            {text.count('new session')}")
    print(f"candidate heard     {len(re.findall(r'heard \(', text))}")
    print(f"interviewer turns   {len(turns)}  (prepared during the silence: {prepared})")
    if latencies:
        p90 = latencies[min(len(latencies) - 1, int(round(0.9 * (len(latencies) - 1))))]
        print(f"first audio         median {statistics.median(latencies):.0f} ms, p90 {p90} ms, max {latencies[-1]} ms")
        prepared_ms = sorted(int(ms) for how, ms in turns if how == "prepared")
        if prepared_ms:
            print(f"  prepared turns    median {statistics.median(prepared_ms):.0f} ms")
    print(f"sentences spoken    {len(said)}")
    print(f"near-repeats        {near} ({100 * near / max(1, len(said)):.0f}%)")
    held = sum(int(n) for n in re.findall(r"held back (\d+) repeated", text))
    print(f"held back repeats   {held}")
    print(f"interruptions       {text.count('interrupted:')}")
    print(f"failures            {len(re.findall(r'failed', text))}")
    for sentence, count in Counter(said).most_common(3):
        if count > 1:
            print(f"  x{count}: {sentence[:100]}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
