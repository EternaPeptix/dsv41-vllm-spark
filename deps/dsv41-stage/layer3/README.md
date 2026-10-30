# Layer-3 parser patch: stream tool-call arguments line by line (vLLM 0.32.0, DeepSeek-V4.1)

**Off unless the API server starts with `DSV41_TOOL_STREAM_NEWLINE=1`. Installing it
changes two files and restarts nothing.**

## Why

vLLM's DeepSeek V4/V4.1 tool parser sends a tool call's argument text to the client only
when the newly generated text contains `>` (`arg_structural_chars = frozenset(">")`). A
`run_code` program therefore reaches DSH in a few large bursts, usually at its first `=>`
and at the end. Streaming speculation (the dsh-sptc plugin's layer 3) needs the program as
it is written.

## What changes (`layer3-parser.diff`)

| File | With the flag |
|---|---|
| `vllm/parser/deepseek_v4.py` | `arg_structural_chars = frozenset(">\n")`: argument text is also flushed at every newline. V4.1 inherits it. |
| `vllm/parser/engine/parser_engine.py` | Inside a string parameter that is still open (string="true", no DSML tag since the last full conversion), each flush JSON-escapes only the new text and appends it, instead of converting the whole argument again. The output is byte-identical to the full conversion, and linear instead of quadratic in the argument length. A DSML tag (or anything containing `｜DSML｜`) falls back to the full conversion. Outside an open string, newline-only flushes re-convert the whole argument only while it is under `DSV41_TOOL_STREAM_NEWLINE_MAX_CHARS` (default 8192). |

Unset, both files behave exactly as deployed (tested byte for byte). Non-streaming responses
never go through the changed code. `DSV41_TOOL_STREAM_INCREMENTAL=0` keeps newline flushes
but disables the incremental path. The variables are read once, when the API server
imports the parser; the parser runs only in the API server process (the head node).

## Tests (no server): `test/`

`zsh test/run-tests.sh` (Python 3.10+ with `regex`; on 512S2 `~/dsh-sptc/.tools/pyl3`)
runs the real parser sources (`orig/` = the deployed files, sha256 in `MANIFEST`;
`payload/` = patched), with stubs only for unrelated vLLM infrastructure, on the 95
`run_code` calls recorded in the 2026-10-26 check runs plus 12 adversarial cases (HTML with
`<`/`>` on every line, escapes and unicode, DSML look-alike text, `</` in strings,
parameters in other orders, a number parameter, sandbox fields, two calls in one message,
a 20 KB single line, a 60 KB program), fed in random 1-10 character steps, with and without
thinking, and in one piece:

- flag unset: every delta, id, content and non-streaming result equals the deployed files';
- flag set: the deltas of each call join to exactly the final arguments, equal the
  full-conversion path delta for delta, and every program newline reaches the client in
  the step that generated it (deployed: lines wait for the next `>`);
- `parse()` and `extract_tool_calls()` results are unchanged;
- cost on a 60 KB program: deployed 34 ms, flag 37 ms (without the incremental path
  1.87 s, capped 87 ms); 25 KB of HTML rows: deployed 2.1 s, flag 29 ms.

`bash test/test-install.sh` exercises install/uninstall/status on a throwaway layout.

## Install (needs a maintenance window; not done)

On each Spark (the parser runs on the API-server host, spark-one; installing on all keeps
the venvs identical):

```sh
cd ~/dsv41-stage/layer3
bash status.sh                 # deployed / patched / unknown, backups, flag of running vllm
bash install.sh --dry-run
bash install.sh                # verifies sha256, backs up to <file>.bak-layer3, copies, verifies
```

Then, at the next planned restart, add `DSV41_TOOL_STREAM_NEWLINE=1` to the API server's
environment (the launcher of the rank-0 / `vllm serve` process). `bash status.sh` shows
whether a running vllm process has it.

Undo: `bash uninstall.sh` (restores from `.bak-layer3`, verifies), or simply restart
without the variable.

`stage.sh` copies this bundle (README, MANIFEST, scripts, diff, payload) to
`~/dsv41-stage/layer3/` on the Sparks and verifies it; it touches nothing else.
