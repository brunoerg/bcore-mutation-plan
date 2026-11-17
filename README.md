# Bitcoin Core Mutation Testing Planner

Turns a Bitcoin Core pull request into a validated, evidence-based **mutation testing
plan**: whether mutation testing is worthwhile, which changed production files and
functions to mutate, and which existing unit tests, functional tests and fuzz targets
to run against those mutants.

It does **not** generate mutants, compile Bitcoin Core, run tests or compute mutation
scores. Its only output is `mutation-plan.json`.

```
git diff (merge-base..head) ─► repository analyzer ─► mutation-context.json ─► LLM (1 call) ─► validator ─► mutation-plan.json
          deterministic facts: files, functions,         provider-neutral       claude-code      deterministic
          test inventory, candidate tests + evidence                            opencode         checks against
                                                                                opencode-zen     the context
```

Deterministic tools collect the facts, the LLM makes the decisions, and a deterministic
validator checks the LLM's output against those facts. The LLM runs with no tools.

## Installation

Requires Python 3.11+ and `git`.

```bash
git clone <this repository> mutation-planner && cd mutation-planner
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'        # pydantic, tree-sitter, tree-sitter-cpp, pytest
python -m pytest -q            # 107 tests; no network, LLM or API key needed
```

The system prompt (`prompts/mutation_planner_system.md`) ships inside the package, so a regular `pip install .` also works.

## Local PR analysis

For a GitHub PR, `BASE_SHA` is the tip of the target branch and `HEAD_SHA` is the PR
tip. The planner diffs `git merge-base BASE HEAD` against `HEAD`, reads every file from
Git objects at `HEAD` (never from the working tree), and records all three SHAs.

```bash
cd ~/bitcoin && git fetch origin pull/32150/head:pr-32150 && cd -

python -m mutation_planner \
  --repo ~/bitcoin \
  --base "$(git -C ~/bitcoin rev-parse origin/master)" \
  --head "$(git -C ~/bitcoin rev-parse pr-32150)" \
  --pr-number 32150 \
  --provider claude-code --model sonnet \
  --output mutation-plan.json
```

For an already-merged PR, use the merge commit's parents: `--base M^1 --head M^2`.

Each run writes:

| file | contents |
|---|---|
| `mutation-context.json` | everything the analyzer found, which is also the replay input (`--context-output`) |
| `mutation-plan.json` | the validated plan (`--output`) |
| `mutation-run-metadata.json` | SHAs, provider, model, prompt version and SHA-256, context SHA-256, tool version, validation status and errors (`--metadata-output`) |

Exit codes: `0` ok, `1` the plan failed validation, `2` bad input or analysis error,
`3` LLM provider error, `4` the analysis was too large for the context budget.

### Dry run

```bash
python -m mutation_planner --repo ~/bitcoin --base BASE_SHA --head HEAD_SHA --dry-run
```

This analyses the repository, writes `mutation-context.json` and stops. No LLM is
called and no plan is produced.

### Context replay

`--context-file` plans from a saved context. It needs no Git checkout: validation uses
the file inventory, test inventory and snapshot metadata recorded in the context. The
same context works with every provider:

```bash
python -m mutation_planner --context-file mutation-context.json --provider claude-code  --model sonnet
python -m mutation_planner --context-file mutation-context.json --provider opencode     --model PROVIDER/MODEL
python -m mutation_planner --context-file mutation-context.json --provider opencode-zen --model MODEL_ID

# Re-validate an existing plan against its context:
python -m mutation_planner --context-file mutation-context.json --validate-plan mutation-plan.json
```

The plan schema is the same for every provider. A replayed context sends identical
input, but LLM decisions can still vary between runs.

### Useful options

| option | default | meaning |
|---|---|---|
| `--max-tests-per-file` | 10 | per-file test selection budget (a ceiling, not a target) |
| `--max-candidates-per-file` | 40 | candidate tests shown to the LLM per file; truncation is reported |
| `--max-context-chars` | 600000 | LLM context budget; see [large PRs](#large-prs) |
| `--max-call-depth` | 8 | static call-graph hops used to find indirect tests |
| `--max-retries` | 1 | LLM retries after a validation failure (the errors are sent back) |
| `--timeout` | 900 | LLM call timeout in seconds |
| `--prompt-file` | `prompts/mutation_planner_system.md` | system prompt override |
| `--pr-title`, `--pr-description` | | optional PR metadata, treated as untrusted data |

## LLM providers

### Claude Code CLI (`--provider claude-code`, default)

1. Install Claude Code: `npm install -g @anthropic-ai/claude-code` (or the native installer).
2. Log in once, interactively, with your Claude subscription: run `claude` and follow
   `/login`. No Anthropic API key is needed. (If `ANTHROPIC_API_KEY` is set, the CLI
   uses it instead; that is the supported option for CI.)
3. Run with `--provider claude-code --model sonnet`. Any model alias or full model
   name the CLI accepts works, e.g. `opus`, `haiku` or a full model ID.

The planner runs `claude --print --output-format json` in an empty temporary directory
and sends the context on stdin. Tools are locked down with the CLI's own options:

* `--tools ""` (no built-in tools), plus `--disallowedTools` for Bash, Edit, Write and the others
* `--strict-mcp-config --mcp-config '{"mcpServers":{}}'` (no MCP servers)
* `--setting-sources ""` (no user or project settings or hooks), `--disable-slash-commands`
* `--permission-mode dontAsk`, `--no-session-persistence`
* `--json-schema <plan schema>`, so the CLI returns `structured_output`

On startup it reads `claude --help`. If the installed CLI lacks the flags needed for a
tool-free run, it refuses to proceed rather than allowing broader access. It never
reads Claude Code's credential storage.

### OpenCode CLI (`--provider opencode`)

1. Install OpenCode: `curl -fsSL https://opencode.ai/install | bash`.
2. Configure a provider: `opencode auth login` (or providers in `opencode.json`).
3. List models: `opencode models`.
4. Run with `--provider opencode --model PROVIDER/MODEL`, e.g. `opencode/claude-sonnet-5-5`.

The planner runs `opencode run --pure --format json --model …` in an empty temporary
directory and sends the context on stdin. It parses the JSON event stream and takes the
plan from the final assistant message. Permissions come from an inline restricted
config (`OPENCODE_CONFIG_CONTENT`): everything is denied, including `edit`, `bash`,
`webfetch`, `websearch`, `task` and `external_directory`. The only exception is `read`,
because current OpenCode releases hang when `read` is denied; read access stays
confined to the empty temporary directory.

### OpenCode Zen API (`--provider opencode-zen`)

1. Create an API key in the OpenCode Zen console.
2. `export OPENCODE_ZEN_API_KEY=...` (`OPENCODE_API_KEY` is also accepted).
3. Pick a model ID: `curl -s https://opencode.ai/inference/v1/models | jq -r '.data[].id'`.
4. Run with `--provider opencode-zen --model claude-sonnet-5-5` (or `gpt-5.5`, `glm-5.3`, …).

Zen serves each model through its native API. The planner reads the model's API
format and base URL from the public model catalogue (each entry's `api` map), then uses
Chat Completions, Responses, Anthropic Messages or Gemini `generateContent`
accordingly. To skip the catalogue lookup, pass `--zen-api-format` and
`--zen-base-url`. HTTP 429 and 5xx responses are retried with backoff. Keys are never
logged, and error messages are redacted.

## Output interpretation

```jsonc
{
  "decision": {"mutate": true, "reason": "..."},          // eligibility
  "files": [{
    "path": "src/wallet/coinselection.cpp",               // a changed production file at head
    "priority": "high",                                   // high | medium | low
    "functions": ["SelectCoinsBnB"],                       // changed functions found by the analyzer
    "changed_lines": [{"start": 118, "end": 264}],         // head line ranges
    "tests": {
      "unit":       [{"id": "unit:coinselector_tests/bnb_search_test", "confidence": "high",
                      "evidence": ["ev-0007"], "reason": "...",
                      "command": "build/bin/test_bitcoin --run_test=coinselector_tests/bnb_search_test"}],
      "functional": [{"id": "functional:wallet_fundrawtransaction.py", "confidence": "low", ...}],
      "fuzz":       [{"id": "fuzz:bnb_finds_min_waste", "command": "FUZZ=bnb_finds_min_waste build_fuzz/bin/fuzz", ...}]
    },
    "uncertainty": "..."
  }],
  "warnings": ["..."],
  "metadata": {"provider": "...", "model": "...", "prompt_version": "1.0", "context_sha256": "...", ...}
}
```

A real plan for PR #32150 is in `examples/mutation-plan.example.json`.

* **Eligibility.** `mutate: false` means the PR changes no production logic worth
  mutating (docs, tests, comments, formatting or mechanical changes). Then `files` is
  always empty.
* **Priorities** say where to spend the mutation budget first. They are the LLM's
  judgement of how much the changed logic matters.
* **Test categories and IDs.** Each ID comes from the test inventory discovered at
  the head commit, and each test includes a ready-to-run `command`:
  * `unit:<suite>` / `unit:<suite>/<case>`: Boost.Test suites and cases, run with
    `test_bitcoin --run_test=…`. The binary comes from CMake (`test_kernel` and
    `bitcoin_ipc_test` are separate binaries).
  * `functional:<script>.py`: scripts registered in `test/functional/test_runner.py`
    (`BASE_SCRIPTS`/`EXTENDED_SCRIPTS`). Variants such as `--usecli` and the
    extended-only flag are recorded in the context.
  * `fuzz:<target>`: `FUZZ_TARGET` registrations, including wrapper macros like
    `FUZZ_TARGET_DESERIALIZE`. They need a fuzz build (`BUILD_FUZZ_BINARY`, usually
    `-DBUILD_FOR_FUZZING=ON`) and a corpus.
* **Confidence.** `high` means a direct static relationship (the test calls the
  changed function or RPC, or checks a changed string). `medium` means a broader
  static relationship (an affected type, an included header) or a short inferred path.
  `low` means a long inferred call path, a module-area link, or the model's own
  inference with no collected evidence. The validator enforces that `high` and
  `medium` need evidence, and that filename similarity alone never justifies them.
* **Evidence.** Every evidence ID points to an item in `mutation-context.json` with a
  file, line, relationship and `basis`:

  | relationship | basis | meaning |
  |---|---|---|
  | `direct_symbol_reference` | static | the test source references a changed function |
  | `class_reference` | static | the test references an affected class or struct |
  | `rpc_method_call` | static | the test calls an RPC whose handler changed |
  | `string_literal_match` | static | the test contains a string literal from a changed line (error message, option name) |
  | `header_include` | static | the test includes the changed header, or the changed `.cpp`'s header |
  | `call_path` | inferred | the test references a production caller of the changed code (bounded static call graph) |
  | `rpc_call_path` | inferred | the test calls an RPC whose handler statically reaches the changed code |
  | `module_area` | inferred | the functional-test family for the changed area, ranked by relevance |
  | `name_similarity` | heuristic | similar file names only; never sufficient on its own |
  | `coverage` | coverage | reserved for future dynamic coverage data |

* **Warnings** include files with no selected tests, tests selected without evidence,
  context truncation, deleted production files and analysis limitations.

## GitHub Actions

There are two trust-separated workflows.

* **`.github/workflows/mutation-plan.yml`: Phase A, unprivileged.** It triggers on
  `pull_request` (opened, synchronize, reopened) and `workflow_dispatch`. It holds no
  secrets. It checks out the PR as data (`persist-credentials: false`), runs the
  analyzer with `--dry-run` and uploads `mutation-context.json`. PR title and body are
  passed through environment variables, never interpolated into scripts. No script
  from the PR is executed. For `pull_request` the workflow is opt-in through
  `vars.MUTATION_PLANNER_ENABLED == 'true'`. `workflow_dispatch` can analyse any PR of
  any repository (default `bitcoin/bitcoin`), which helps with debugging.
* **`.github/workflows/mutation-plan-llm.yml`: Phase B, privileged.** It triggers on
  `workflow_run` and holds the API secret, so it never checks out PR code. It
  verifies the source workflow's path and repository, downloads the artifact from
  that specific run, and checks that the artifact holds exactly one file within a size
  bound. The file must parse under the strict context schema, which rejects unknown
  fields. For `pull_request` runs the recorded `head_sha` must also match the
  triggering run. It then plans, re-validates the written plan, and uploads
  `mutation-plan.json` and the run metadata as artifacts.

`pull_request_target` is not used.

Configuration (repository settings, Actions variables and secrets):

| name | purpose |
|---|---|
| `vars.MUTATION_PLANNER_ENABLED` | `true` to enable |
| `vars.MUTATION_PLANNER_SPEC` | pinned planner install source, e.g. `git+https://github.com/OWNER/mutation-planner@<commit-sha>`. Leave unset when the workflows live in the planner repository itself. |
| `vars.MUTATION_PLANNER_PROVIDER` | `opencode-zen` (default) or `claude-code` |
| `vars.MUTATION_PLANNER_MODEL` | model ID for the provider |
| `secrets.OPENCODE_ZEN_API_KEY` | for `opencode-zen` |
| `secrets.ANTHROPIC_API_KEY` | for `claude-code` in CI (API-key authentication) |

**CI credentials, honestly stated.** A personal Claude subscription login is meant for
interactive and local use. Do not copy session credentials into GitHub secrets. In CI,
use API-key authentication (an Anthropic API key with the Claude Code CLI, or an
OpenCode Zen key). If you don't want an API key in CI, run Phase A in CI and plan
locally: download the `mutation-context` artifact and run
`python -m mutation_planner --context-file mutation-context.json --provider claude-code`.

Phase B treats the context as untrusted. A malicious fork could change Phase A and
upload a crafted context. The schema and provenance checks limit what such a context
can do, but a plan is only as trustworthy as the context it was built from. Phase B
can also spend LLM credits on every PR; gate it with the variable above, or add a label
or author-association check. Pin actions to commit SHAs if your policy requires it.
`tests.yml` runs this project's own test suite.

## Large PRs

The changed-file inventory and PR metadata are always kept. If the LLM view exceeds
`--max-context-chars`, the planner reduces it in steps, recording each step under
`truncations`:

1. Drop non-production patch excerpts.
2. Shrink function snippets to the changed lines ±4.
3. Truncate production patches.
4. Keep the top 15 candidates per file.
5. Keep only the changed lines in snippets.
6. Drop production patches.

Candidate truncation always keeps direct candidates first and reserves slots for
indirect ones. If the view is still too large, the context is marked
`analysis_complete: false` and planning stops with exit code 4. The planner never
concludes that mutation testing is unnecessary just because a PR is large.

## How discovery works (verified against Bitcoin Core)

Discovery reads the repository's own definitions rather than guessing:

* **Unit:** `BOOST_{AUTO,FIXTURE}_TEST_SUITE`, `BOOST_*_TEST_CASE*`, with nested
  suites. Comments are stripped in a string-aware way, so `"xpub/0/*"` inside a string
  is not treated as a comment. Executables and build conditions come from CMake
  `add_executable`/`target_sources` and the `if()`s around `add_subdirectory`.
* **Functional:** `BASE_SCRIPTS`/`EXTENDED_SCRIPTS`/`NON_SCRIPTS` are read from
  `test_runner.py` with `ast`; the file is never executed. Unregistered Python files are
  reported as helpers.
* **Fuzz:** `FUZZ_TARGET(name, .init=…, .hidden=…)` plus any `#define` that wraps
  `FUZZ_TARGET`, excluding the macro bodies themselves.
* **Subtrees:** read from `test/lint/lint_ignore_dirs.py` and classified as
  `third_party`.
* **Functions:** parsed with tree-sitter-cpp. Thread-safety annotations
  (`EXCLUSIVE_LOCKS_REQUIRED(...)` etc.) are blanked before parsing so annotated
  definitions are recognised. RPC handlers are identified by
  `RPCMethod{"name"`/`RPCHelpMan{"name"`.

On a recent Bitcoin Core master this finds 158 Boost suites and 913 cases, 285
functional scripts and 238 fuzz targets. These counts match a `grep` of the
registration macros exactly. Analysis takes about 1 second per PR.

## Known limitations

* Static test selection does not prove dynamic coverage. A selected test may never
  execute the mutated line.
* Executing a changed function does not guarantee a mutant is detected; that depends
  on the test's assertions.
* Some indirect dependencies are missed. The call graph is name-based and bounded
  (depth, fan-out), so it misses virtual dispatch, callbacks, signals, function
  pointers and P2P message flows. Name-based matching can also over-approximate
  overloaded or common names. Very generic names are ignored.
* Discovery depends on repository conventions (Boost macros, `test_runner.py` lists,
  `FUZZ_TARGET`, CMake target sources). Qt tests (`test_bitcoin-qt`, QTest) and
  benchmarks are not inventoried.
* Declarations related to a changed function (e.g. the matching header prototype)
  are not extracted separately; the LLM sees the snippet and the patch.
* LLM outputs vary between runs and providers, even for a replayed context. The
  validator constrains what is accepted, not which valid choice is made.
* Fuzz targets need a fuzzing build, a corpus (e.g. `qa-assets`) and possibly
  sanitizers. Some targets are only built with optional features (`ENABLE_WALLET`,
  `ENABLE_IPC`), as recorded in `build_condition`.
* The MVP does not execute tests or compute mutation scores. It does not post PR
  comments. Dynamic coverage is not collected, although the evidence schema has a
  `coverage` basis for it.

## Project layout

```
mutation_planner/
  cli.py                 argument parsing, artifact writing, exit codes
  config.py              limits and Bitcoin Core conventions
  schemas.py             Pydantic models: context, LLM output, final plan
  git_analysis.py        safe git access (argument lists, no shell, no hooks), merge-base diff
  source_analysis.py     file classification, tree-sitter function extraction
  test_discovery.py      unit / functional / fuzz inventory, CMake parsing
  candidate_selection.py candidate tests and evidence (static, inferred, heuristic)
  analysis.py            pipeline orchestration, LLM view, context budget
  planner.py             single LLM call, validation, one retry, enrichment
  validator.py           deterministic plan validation
  llm/                   provider interface, claude_code, opencode_cli, opencode_zen, factory
prompts/mutation_planner_system.md   version-controlled system prompt
examples/mutation-plan.example.json  real plan for bitcoin/bitcoin#32150
tests/                               pytest suite (synthetic Bitcoin Core-style git repos, mocked LLMs)
```

The planner sits behind interfaces that later stages could plug into: a mutation
executor would consume `mutation-plan.json` (which already includes test commands),
and coverage data would add `coverage` evidence to the context.
