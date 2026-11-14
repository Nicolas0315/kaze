"""Task prompts for the drafter comparison (shared by the driver and the worker)."""
REC = '{"id": %d, "name": "item-%d", "tags": ["a", "b"], "ok": true, "note": "row %d"}'
FENCE = chr(96) * 3

TASKS = {
    "code": (
        "Write a Python function called normalize_rows that takes a list of dicts and "
        "returns a list of dicts with lowercased string values. Only output code.\n"
    ),
    "story": (
        "Write the opening of a short story about a lighthouse keeper who finds a "
        "message in a bottle that was written in his own handwriting.\n"
    ),
    "translation": (
        "Translate the following into Japanese, keeping the tone formal.\n"
        "English: The committee will review all proposals before the end of the quarter, "
        "and applicants will be notified in writing.\nJapanese: "
    ),
    "structured": (
        "Return only JSON matching this schema.\n"
        '{"type":"object","properties":{"name":{"type":"string"},'
        '"age":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}},'
        '"required":["name","age"]}\nJSON: '
    ),
    "code edit": (
        "Rewrite the whole file below with RETRY changed from 3 to 5. Output the complete "
        "file, no commentary.\n" + FENCE + "python\nimport time\n\nRETRY = 3\nTIMEOUT = 30\n\n"
        "def fetch(url):\n    for attempt in range(RETRY):\n        try:\n"
        "            return get(url, timeout=TIMEOUT)\n        except TimeoutError:\n"
        "            time.sleep(1)\n    raise RuntimeError('giving up')\n" + FENCE + "\n"
    ),
    "jsonl edit": (
        "Rewrite these JSONL rows with ok set to false. Output every row, no commentary.\n"
        + "\n".join(REC % (i, i, i) for i in range(12)) + "\n"
    ),
    "long jsonl": (
        # a loose instruction let the model answer with prose and a code fence, so the
        # ratio measured chatter rather than the drafter
        "Continue the JSONL below. Output only the next rows, one per line, no prose "
        "and no code fences.\n"
        + "\n".join(REC % (i, i, i) for i in range(40)) + "\n"
    ),
}

CONFIGS = {"plain": 0, "t1 drafter": 2, "ours (adaptive)": 16}
