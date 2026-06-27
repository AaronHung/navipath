#!/usr/bin/env python3
"""Export a Cursor agent transcript (.jsonl) to readable Markdown.

Usage:
    python tools/export_transcript.py <transcript.jsonl> [out.md]

Recovers the human-readable conversation (user queries + assistant replies),
with a compact note of each tool call and a truncated tool result, so you can
re-read past sessions even if the Cursor UI no longer lists them.
"""
import json
import sys
from pathlib import Path

TOOL_RESULT_TRUNC = 800  # chars to keep from each tool result


def _text_of(item):
    if isinstance(item, dict):
        return item.get("text") or item.get("content") or ""
    return str(item)


def render(jsonl_path: str, out_path: str | None = None) -> str:
    jsonl_path = Path(jsonl_path)
    out_path = Path(out_path) if out_path else jsonl_path.with_suffix(".md")
    lines_out = [f"# Recovered transcript: {jsonl_path.name}\n"]
    n_user = n_asst = 0

    for raw in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except Exception:
            continue
        role = d.get("role")
        if role not in ("user", "assistant"):
            continue
        content = (d.get("message") or {}).get("content")
        if content is None:
            continue
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        chunks = []
        for item in content:
            itype = item.get("type") if isinstance(item, dict) else None
            if itype == "text" or itype is None:
                t = _text_of(item).strip()
                if t:
                    chunks.append(t)
            elif itype == "tool_use":
                name = item.get("name", "tool")
                inp = item.get("input", {})
                s = json.dumps(inp, ensure_ascii=False)
                if len(s) > 400:
                    s = s[:400] + " …"
                chunks.append(f"> 🔧 **{name}** `{s}`")
            elif itype == "tool_result":
                body = item.get("content")
                if isinstance(body, list):
                    body = " ".join(_text_of(b) for b in body)
                body = str(body or "").strip()
                if body:
                    if len(body) > TOOL_RESULT_TRUNC:
                        body = body[:TOOL_RESULT_TRUNC] + " …[truncated]"
                    chunks.append(f"> 📤 result: {body}")
        if not chunks:
            continue

        if role == "user":
            n_user += 1
            lines_out.append(f"\n---\n\n## 🧑 USER\n\n" + "\n\n".join(chunks) + "\n")
        else:
            n_asst += 1
            lines_out.append(f"\n### 🤖 ASSISTANT\n\n" + "\n\n".join(chunks) + "\n")

    header = f"> user turns: {n_user} · assistant turns: {n_asst}\n"
    md = lines_out[0] + header + "".join(lines_out[1:])
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}  ({n_user} user / {n_asst} assistant turns)")
    return str(out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    render(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
