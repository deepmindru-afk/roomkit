# What a tool conversation costs

The chat suite measures latency and contracts against a live model, whose
rounds change from one run to the next. `--suite cost` measures what a fixed
tool conversation is billed, so a change to how RoomKit assembles a request
(the tool list, the system prompt, the history) can be compared before and
after on identical traffic.

The model's answers are scripted (`cost_script.py`): every run makes the same
requests over the same rounds, in the streaming loop and the buffered one.
Each request the channel builds is also sent to the real provider with
`max_tokens=1`; its answer is discarded and its usage (input, cache read,
cache write) is what the report counts, priced from the model catalogue.

| Scenario | What it exercises |
|---|---|
| `tool_cost` | Four turns: `find_tools` reveals a tool, its result is evicted and paged back with `read_stored_result`; a skill is activated and unlocks the tool it gates; a turn answers from the tool-usage digest and the active skill; a last turn calls a tool again, the steady state |
| `force_stop_cost` | Six identical calls: the third is refused, the sixth pulls the anti-loop ripcord and the last generation is told to answer, none of its calls running |
| `long_cost` | Six policy questions answered at length, then five turns, four of them calling a tool: what changes between turns, priced against a history that outgrows it |

Each has a `_buffered` twin. Every sample marks its tools and system prompt
with a fresh nonce, so one run never reads the cache another wrote.

```bash
# Offline: the requests and what changed between them, no key, no cost.
uv run python -m benchmarks.chat --suite cost --provider mock --warmups 0 \
  --repetitions 1 --output benchmark-results/cost-offline

# Billed on Claude: about 0.40 USD for two repetitions on claude-sonnet-5.
uv run --extra anthropic python -m benchmarks.chat --suite cost \
  --provider anthropic --model claude-sonnet-5 --key-file ~/.secrets/anthropic \
  --warmups 0 --repetitions 2 --output benchmark-results/cost-before
```

Pick a model whose minimum cacheable prompt the requests exceed: 1,024 tokens
on Claude Sonnet 5, 4,096 on Claude Haiku 4.5, whose requests here would not
be cached at all. `--provider openai` with `--base-url` measures an
OpenAI-compatible service the same way, where it reports cached tokens.

`report.md` then carries two tables, and `cost.csv` the first one:

- **Cost per turn**: rounds, input, cache read, cache write, output, read rate
  (cache read over the three input counters, which are disjoint) and what was
  billed, median over the passed samples. Billed is the requests' input and
  the one output token each asked for: the input side of the conversation's
  cost, the scripted answers being free.
- **Rounds**: every provider call of each scenario's first sample, with the
  first block of its request that differs from the previous request.
  `append` means the previous request is a prefix of this one, which the cache
  can read back; `tools`, `system` or `messages` is where the cached prefix
  stops.

The offline contracts run in `tests/test_chat_cost.py`.
