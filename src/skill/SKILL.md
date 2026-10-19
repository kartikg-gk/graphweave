---
name: graphweave
description: Answer questions about a public GitHub repository's recent activity (who worked on what, what changed, which pull requests fixed which issues, who reviewed what) with cited pull requests, issues, commits and people, and show which retrieved items the answer actually used. Also explains an existing graphweave trace file (*.json) and how to trace the user's own retrieval agent. Use when the user asks about a repo's people, changes, PRs or issues, mentions graphweave, asks "what did my agent retrieve vs use", or types /graphweave.
---

# graphweave

graphweave reads a GitHub repository's recent pull requests, issues and commits, links them into a graph, ranks what matches a question, and records the run as a trace. Each trace gets a Markdown report next to it. Read the report and answer from it. Everything runs locally; the only network calls go to GitHub's API.

## 1. Make sure it is installed

Run `graphweave-github-trace --help`. If the command is missing, install it:

```bash
pip install graphweave
```

It needs Python 3.12 or newer.

## 2. Answer a question about a repository

Use the repository as `owner/name` and the user's question in quotes. Give each question its own file:

```bash
graphweave-github-trace owner/name "the user's question" --out graphweave_out/<short-slug>.json
```

Add flags when the question needs them:

| Question is about | Add |
| --- | --- |
| who reviewed something | `--reviews` |
| which files changed | `--files` |
| older history | `--prs 50 --issues 50 --commits 50` |

The command prints a line `Report for your agent: <path>`. **Read that report file.** It holds the answer, the items the answer used, the items it did not, and every link with where it came from (`record` = GitHub's own data, `text` = read out of prose).

Then reply to the user:

- Answer in a few sentences, citing the **used** items by name with their links.
- For "who" questions, name people from the **Links** section: `AUTHORED`, `REVIEWED` and `REPORTED` links say who wrote, reviewed or opened each item.
- Say how many items were retrieved and how many were used.
- Do not add facts that are not in the report. If the report's answer says the context lacks the information, say so and suggest a sharper question or more history (`--prs 50`).

If the graphweave MCP tools are available (`trace_repo`, `explain_trace`, `list_traces`), call `trace_repo` instead of the command; it returns the same report.

## 3. Show the map (offer, do not force)

```bash
graphweave graphweave_out/<short-slug>.json
```

It opens a viewer at `http://127.0.0.1:4630` and keeps running until stopped, so start it in the background or tell the user to run it. A folder path opens every trace in it.

## 4. Explain an existing trace

For a trace file the user already has:

```bash
graphweave report path/to/trace.json
```

This writes `path/to/trace.report.md`. Read it and explain what was retrieved, what was used and why it ranked.

## 5. Trace the user's own agent

When the user wants to see what their own retrieval agent used, show them the smallest change, and do not edit their code unless asked:

```python
from graphrag.tracing import capture, save

trace = capture(question, retrieved_items, answer=answer, producer="my-agent")
save(trace, "graphweave_out/my-run.json")
```

`retrieved_items` is a list of dicts with `id`, `label` and `content`. For LangGraph, `pip install "graphweave[adapter]"` and pass `graphrag.tracing_langgraph.LangGraphTracer()` as a callback. For LlamaIndex, `pip install "graphweave[llamaindex]"`, wrap the query in `with graphrag.tracing_llamaindex.LlamaIndexTracer() as t:` and call `t.finish(question, answer)`. For DSPy, `pip install "graphweave[dspy]"`, run the program inside `with graphrag.tracing_dspy.DSPyTracer() as t:` and call `t.finish()`. If they already export OpenTelemetry spans (OpenInference), run `graphweave import-otel spans.json`. Then use step 4 on the saved file.

If the user doubts a used/unused mark, offer `graphweave verify <trace.json>`: it re-asks a model with each item removed (needs `graphweave[verify]` and an API key; one model call per item, so say so before running it).

## Problems

| You see | Do this |
| --- | --- |
| GitHub rate limit message | Set `GITHUB_TOKEN`, or wait an hour |
| Repository not found | Check `owner/name`; private repos need a token with access |
| Port already in use | Add `--port 4700` to the viewer command |
