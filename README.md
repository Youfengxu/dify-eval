# difyeval

Universal Dify eval pipeline: **Dataset (samples) × Task (system under test) ×
composable Checks → Scores → Metrics → Experiment**.

Evaluate any Dify workflow — or anything that can emit a run JSON — with
declarative YAML checks; zero Python for the common case. Built for
court-citable determinism: offline byte-stable replays, an advisory-only LLM
judge panel with persisted auditable outputs, and degrade-never-raise handling
of messy data.

Requires Python ≥ 3.10. Dependencies: `PyYAML`, `requests` (and requests is
only ever touched when a *live* runner or a *live* judge is invoked).

```
pip install -e .
```

## Quickstart — the 5-command authoring path

```bash
difyeval new-case my_eval/case.yml        # 1. scaffold a commented skeleton
$EDITOR my_eval/case.yml                  # 2. point it at your workflow + checks
difyeval validate my_eval/case.yml        # 3. fail fast on config problems
difyeval run my_eval/case.yml --no-judge --save-fixture   # 4. acquire + deterministic score
difyeval replay my_eval/case.yml --no-judge               # 5. re-score offline, byte-stable
```

Add a judge key (e.g. `OPENROUTER_API_KEY`) and drop `--no-judge` to get the
advisory judge panel. Try the shipped demo right now, fully offline:

```bash
difyeval run examples/text-source-discovery/case.yml --no-judge
```

## The case.yml contract

```yaml
case_id: my_workflow_eval
mode: reference | live            # affects verdict scope wording (see Verdict)
packs: [my_pkg.checks]            # optional: importable modules registering custom checks
runner: {type: dify-workflow, base: https://dify.example.com, api_key_env: DIFY_API_KEY,
         input_map: {caption: post_text}, watch: [merge, sources]}
dataset:
  - id: s1
    inputs: {var: "..."}          # -> Dify start-node variables, mechanically
    reference: {expected: "..."}  # optional, typed by the checks that use it
    repeats: 2                    # optional, re-run the sample N times
checks:
  - {id: has_out, type: output.nonempty, gate: true}
  - {id: cites,   type: node.json_count, selector: nodes.sources.outputs.sources_json, min: 3, gate: true}
  - id: rubric
    type: judge.rubric
    items:
      - {id: a, weight: essential, item: "The report identifies X with a URL."}
  - {id: craft, type: judge.tradecraft}   # advisory, 3 default dims
expect_verdict: PASS              # optional; anchor cases pin FAIL|WARN|PASS
judge_notes: >                    # optional context shown to every judge
  ...
```

Config problems (bad YAML, unknown check type, judge check with `gate: true`,
bad selector syntax) fail fast **before any run** with exit code 2. Data
problems (missing node, undecodable JSON, absent fixture) never raise — they
degrade into failing/None Scores so every failure is visible and attributable
in the scorecard.

## Check library

| type | params | gate? | asserts |
|---|---|---|---|
| `output.nonempty` | — | yes | report text is non-empty |
| `output.contains` | `values` / `value` / `values_from: reference.<path>` | yes | ALL listed substrings present (value = fraction found) |
| `output.not_contains` | same | yes | NONE of the listed substrings present |
| `output.regex` | `pattern` | yes | regex matches the report |
| `output.json_schema` | `schema`, optional `selector` | yes | stdlib dict-shape walk: `type` (object/array/string/number/integer/boolean/null), `required`, `properties`, `items` |
| `node.exists` | `node` | yes | node id was journaled |
| `node.json_count` | `selector`, `min`/`max` | yes | JSON-decode if string → `len()` within bounds |
| `node.json_field` | `selector`, optional `field`, `equals` / `equals_from: reference.<path>` | yes | field present, or strictly equal to the expected value |
| `node.regex` | `selector`, `pattern` | yes | regex over the resolved value (non-strings JSON-dumped) |
| `judge.rubric` | `items: [{id, weight: essential\|expected\|bonus, item}]` | **never** | weighted binary rubric; every panel judge votes per item with an evidence quote; majority vote; essential failure → WARN |
| `judge.tradecraft` | optional `dims` (built-in names or `{id, criterion}`) | **never** | panel scores each dim 0–1; defaults: `uncertainty_expression`, `info_vs_judgment`, `alternatives_considered` |
| `gt.derived_rubric` | `from: reference.<path>`, `template`, `weight` | **never** | maps a structured reference list into rubric items (`"{date} — {description}"`), feeds the same rubric machinery |
| `meta.elapsed_max` | `max` (seconds) | allowed | run wall-clock under budget |
| `meta.cost_max` | `max` (number + optional `currency`, or `{CUR: amt}`) | allowed | per-currency spend under cap — **no FX conversion**; under-counting (nodes without usage) is noted |

Numeric-threshold checks within 10 % relative of their threshold are flagged
`(marginal)` in the checks table.

### Selector semantics

A selector is a dotted path into the Run dict:
`nodes.merge_s4.outputs.resolved_json`. Legal roots: `output`, `nodes`,
`usage`, `meta`, `error` (reference-side selectors — `values_from`,
`equals_from`, `from` — use `reference` / `inputs` / `metadata`). Dicts are
walked by key, lists by integer index, and when traversal hits a *string* leaf
with path remaining it is auto-JSON-decoded (Dify node outputs are routinely
JSON-in-a-string). A missing path degrades: `passed=False` for gated checks,
`None` for advisory, with evidence `selector not found: ...`.

### Custom check packs

`difyeval new-pack my_checks.py` scaffolds a module with one worked example.
A pack registers checks at import time:

```python
from difyeval.checks import register, make_score, degrade

@register("csd.accounts_recall")
def accounts_recall(spec, sample, run, ctx): ...
```

A case opts in with `packs: [my_checks]` (any importable dotted module path).
Unknown check types are fatal before any run and the error lists every
registered type.

## Runners

Every runner emits the same uniform Run contract:

```
{output: str, nodes: {node_id: {outputs, status?, elapsed?}},
 usage: {tokens?, cost?: {currency: amount}, nodes_without_usage?},
 meta: {status, elapsed, ...}, error?}
```

| type | source | node trace | notes |
|---|---|---|---|
| `file` | JSON fixture per case/sample (`path: fixtures/{sample_id}/run.json`, `{repeat}` supported; per-sample override via `metadata.run_path`) | as recorded | accepts native Run shape AND the legacy archive shape `{report, nodes, elapsed, status}` |
| `difyctl-json` | saved `difyctl run -o json` output (same path templating) | **none** — difyctl doesn't emit one; full outputs kept at `meta.outputs` | `data.outputs.<report_key>` (bare `outputs` tolerated), `report_key` default `report` |
| `dify-workflow` | live `POST /v1/workflows/run` (streaming SSE) | `node_finished` journaling; `watch:` list filters, omit to journal all | output = `workflow_finished` `data.outputs[report_key]`; `input_map` renames sample inputs to Dify variables |
| `dify-chatflow` | live `POST /v1/chat-messages` (streaming SSE) | same journaling | output = accumulated `message` answer; `query_key` (default `query`) selects the query input; usage captured from `message_end` metadata |
| `command` | your argv (`{case_file}`, `{sample_id}`, `{repeat}`, `{input.<k>}` placeholders; sample inputs as JSON on stdin) | whatever your command emits | expects Run-shaped JSON on stdout; nonzero exit / timeout / parse failure degrade to a `runner_error` Run |

Usage accounting sums per-node `outputs.usage` blocks plus top-level usage,
per currency; nodes that report no usage are counted in
`nodes_without_usage` so under-counted cost is never silent. (The workflow
runner uses the `workflow_finished` token total only when no node reported
usage — no double counting.)

## Profiles — `--profile-file`

A profile file names and describes the system under test so the eval.json is
self-describing and two Experiments can be diffed profile-vs-profile:

```yaml
profile: opus-4.6-baseline        # required — becomes the Experiment label
description: orchestrator on opus 4.6, temp 0
models: {orchestrator: anthropic/claude-opus-4.6}   # free-form, recorded verbatim
config: {temperature: 0, reasoning_budget: 8401}    # free-form, recorded verbatim
dsl: ../workflow.yml              # optional — auto --dsl (relative to this file)
runner: {base: https://dify.example.com}            # optional case.runner overrides
```

`difyeval run my_eval/case.yml --profile-file profiles/opus.yml --no-judge`

- The `profile` name becomes the Experiment label (output file names + the
  single scorecard header line, where the profile already appears).
- The whole mapping is embedded **verbatim** into `manifest.profile_config` —
  the audit record carries the profile, not a pointer to a mutable file.
- `dsl:` auto-fills `--dsl` (an explicit `--dsl` wins); `runner:` overrides
  shallow-merge over the case's runner (e.g. point the same case at another
  Dify base URL or app).
- Plain `--profile <label>` keeps working; passing both is a config error.

## Comparing experiments — `difyeval diff`

```bash
difyeval diff results/case__a__eval.json results/case__b__eval.json [--out diff.md]
```

Markdown to stdout (and `--out`): verdict A→B, a per-check table (value A/B,
Δ, pass counts, pass-flips highlighted `⚠ **FLIP**`), rubric item flips,
judge-dim median deltas, gated pass-rate delta, and cost/tokens deltas when
either side reports usage. Diffing two different `case_id`s is refused
(exit 2) unless `--force` — and a forced diff says so in its header.

## Judge↔human validation — `difyeval audit`

The propose / apply / report loop that measures whether the LLM judge panel
actually agrees with a human analyst:

```bash
difyeval audit propose --eval results/case__p__eval.json --out audits/case.yml
$EDITOR audits/case.yml     # fill every `human:` slot with agree | disagree
difyeval audit apply --sheet audits/case.yml --analyst "A. Nalyst"
difyeval audit report --sheets 'audits/*.applied.yml' [--out agreement.md]
```

- **propose** lists every rubric item (panel verdict, votes, evidence quote,
  per-judge votes) and every tradecraft dim (panel median, spread, per-judge
  scores), each with a `human: null  # agree | disagree` slot + a `note:`.
- **apply** validates (every slot must be exactly agree/disagree), stamps
  analyst + date + source sha256, and writes `<sheet>.applied.yml` —
  immutable-ish: re-applying is refused.
- **report** aggregates applied sheets into per-judge-backend agreement.
  Rubric items get % agreement **and Cohen's kappa** (binary, textbook
  formula) — the human's implied ground truth per item is the panel verdict
  when agreeing, its negation when disagreeing.

Honest-stats rules (enforced, not advisory):

- kappa is refused for **n < 10** ("insufficient n") — % agreement is still
  shown with its n;
- degenerate marginals (a rater that never varies) report
  **"n/a (no variance)"** instead of a meaningless kappa;
- tradecraft dims were validated at **panel level** (the human judged the
  panel median, not any single backend's score), so dims report
  % human-agree only — no kappa, no per-backend split;
- n is printed everywhere a percentage is.

## Ground truth that grows — provenance, accretions, `difyeval review`

For workflows whose truth grows from human review of live outputs (a discovery
pipeline, an investigative research workflow — anything where the reference
set is curated over time rather than known up front), difyeval ships an optional
convention + loop. Everything here is additive: cases without sidecars or
review are untouched, and the byte-stable `--no-judge` replay contract holds.

### Provenance tiers (`difyeval.gt`)

Any reference list may hold dict items carrying optional metadata:

```yaml
reference:
  accounts:
    items:
      - handle: "@seed_account"
        provenance: human_verified          # tier — see below
        discoverable: true                  # can the system even find this?
        validated: {by: analyst, date: 2026-07-01}
        source_run: "results/case__p__eval.json#s1:r1"
  must_reject:                              # hard-negatives the run must NOT claim
    - {value: "@false_positive", reason: "news outlet, not a buzzer"}
```

| tier | meaning | may gate? |
|---|---|---|
| `human_verified` | analyst confirmed it (authoring research, or a review `confirm` — carries `validated: {by, date}`) | yes |
| `human_supplied` | provided by a human but not independently re-verified; also the default for items with no `provenance` (a human hand-wrote them into the case) | yes |
| `pipeline_observed_unvalidated` | surfaced by a run, accreted via a review `unvalidated` decision | **never** — advisory until a human promotes it |

The trust rule mirrors judge governance: **pipeline-observed never gates
until promoted.** Two orthogonal axes for pack authors (documented in
`difyeval/gt.py`, the pack-author contract):

- `provenance` governs **trust** — may the item back a FAIL-capable
  assertion? `gt.partition_by_tier(items)` / `gt.gateable(items)` split on
  it; unknown tiers are advisory (never trust what you cannot classify).
- `discoverable: false` governs **gateability for recall checks** — the item
  is real but unreachable by the system's discovery surfaces, so recall
  packs must exclude it from gating *denominators* (failing to find the
  unfindable is not a failure). Packs implement the exclusion, e.g.
  `[it for it in gt.gateable(items) if it.get("discoverable") is not False]`.

Helpers: `iter_items(reference, "accounts.items")` (missing → `[]`),
`partition_by_tier`, `gateable`, `validate_items` (warnings: unknown tier,
`validated` without by/date), `reject_list(reference)` for `must_reject`,
and `item_identity` — the duplicate-detection key (first of
`id`/`handle`/`value`, `@`-stripped and case-insensitive; whole-dict
equality as fallback).

### The accretions sidecar

`<case minus .yml>.accretions.yml` — written only by `difyeval review
apply`, merged into sample references at load time, **append-only**; the
hand-authored case file is never rewritten (comments survive, and the
hand-authored / run-accreted boundary stays auditable):

```yaml
samples:
  _all:                        # applies to every sample — or use a sample id
    reference:
      accounts.items:          # dotted path to a list inside reference
        - {handle: "@x", provenance: human_verified, ...}
      must_reject:
        - {value: "@y", reason: "...", source_run: "..."}
```

Merge order is deterministic (base items, then `_all`, then the per-sample
block, sidecar lists in file order); duplicates by `item_identity` are
skipped with a load warning, never an error. `difyeval validate` reports
sidecar presence, per-path merge counts, and provenance warnings; the
manifest records `accretions_sha256` (the eval.json `case` echo stays the
authored base file).

### The review loop — `difyeval review`

```bash
difyeval review propose --eval results/case__p__eval.json --case my/case.yml \
    --out reviews/sheet.yml [--sample s1] [--extractor csd] [--sample-k 10 --seed 0]
$EDITOR reviews/sheet.yml     # set decision: skip | confirm | unvalidated | negative
difyeval review apply --sheet reviews/sheet.yml --case my/case.yml \
    --analyst "A. Nalyst" [--date 2026-07-23]
```

- **propose** replays the eval.json's embedded runs through a
  pack-registered extractor and lists every candidate not already in the
  merged reference (and not banned by `must_reject`), each with evidence
  and `decision: skip`. Packs register extractors at import time:

  ```python
  from difyeval.review import register_review_extractor

  @register_review_extractor("csd")
  def extract(case, sample, run, experiment):   # experiment = the eval.json mapping
      return [{"path": "accounts.items",
               "item": {"handle": "@discovered"},
               "evidence": {"urls": ["..."]}}]
  ```

  Do **not** dedupe inside the extractor — propose does it centrally.
- **apply** validates decisions and appends to the sidecar only:
  `confirm` → `provenance: human_verified` + `validated: {by, date}` +
  `source_run`; `unvalidated` → `pipeline_observed_unvalidated`;
  `negative` → a `{value, reason, source_run}` `must_reject` entry;
  `skip` → nothing. Idempotent-safe: re-applying the same sheet finds
  everything already present and no-ops. Doing nothing with a sheet — or
  never proposing one — changes nothing.
- **Tier-4 spot-validation**: `--sample-k K --seed S` takes a seeded
  deterministic subsample; apply on a sampled sheet prints a precision
  estimate with its 95 % Wilson score interval — an estimate from a
  sample, never a census.

The loop lets a `mode: live` case grow its own ground truth run over run
until the accreted, human-verified reference is substantial enough to
flip to `mode: reference`.

## Drift detection — `difyeval drift`

```bash
difyeval drift --anchors anchors.yml [--baseline-dir d] [--out drift.md] [--no-judge]
```

`anchors.yml` lists frozen (case, runs, baseline) triples — paths relative to
the anchors file; `--baseline-dir` re-roots relative baseline paths:

```yaml
anchors:
  - case: cases/demo/case.yml
    runs: [runs/demo/s1.json]          # the SAME frozen runs the baseline scored
    baseline: baselines/demo__default__eval.json
```

Each anchor's runs are re-scored NOW (drift is the one networked difyeval
command — the judge panel runs live unless `--no-judge`) and compared against
the baseline:

- **DETERMINISM-BREAK** — any deterministic (non-judge) check changed
  per (sample, repeat, check). The code/config moved under you; this is
  loudly distinct from judge drift.
- **DRIFT** — any rubric item flip, or any per-dim panel-median delta
  > 0.15.
- `--no-judge` runs the deterministic-only drift check offline; the judge
  column reports "not assessed" (as it also does when no judge key is set).

Exit 1 on any DRIFT or DETERMINISM-BREAK — CI-gateable.

## Verdict

- **FAIL** — any gated Score not passed (a degraded/unprovable gate is a
  failed gate). Gates are non-judge checks with `gate: true`.
- **WARN** — gates passed, but a rubric **essential** item failed.
- **PASS** — everything else.

A mode-aware scope statement is rendered under the verdict: in `live` mode,
PASS = process-soundness only — correctness is not established without a
reference; in `reference` mode, PASS is scoped to exactly the gated checks.

Metrics honor the honest-stats rule: mean ± stderr only for n ≥ 2, single
observations reported as-is, ranges with explicit n, plus per-sample
repeat-consistency (fraction of checks whose outcome agrees across repeats).

CLI exit codes: `0` = verdict matches `expect_verdict` (default `PASS`) /
clean drift, `1` = FAIL, mismatch, DRIFT or DETERMINISM-BREAK, `2` = config
error.

## Judge governance — advisory only, by construction

The LLM judge panel (`difyeval/judge.py`, ported from an earlier internal eval
engine) **never gates**. `gate: true` on a judge check is a config error, and
the engine forces `advisory=True` on every judge-derived Score. Rationale:
LLM judgments are probabilistic and vendor-dependent; a court-citable FAIL
must be reproducible from deterministic checks alone. Judge output still
matters — a failing *essential* rubric item drives WARN — but it can never
flip a verdict to FAIL on its own.

Panel mechanics:

- Backends activate **by env-key presence**: `MOONSHOT_API_KEY`,
  `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `GLM_API_KEY`,
  `LOCAL_LLM_API_KEY`(+`_BASE_URL`/`_MODEL`); `--judges a,b` filters.
- Per-dim **median + spread + n** across the panel;
  `independent_vendor` = ≥ 2 distinct vendors (same-vendor panels are
  flagged in the scorecard).
- Rubric items get a deterministic per-judge order shuffle
  (`random.Random(f"{case_id}:{judge_name}")`) — positional-bias mitigation
  that replays identically forever.
- One retry (5 s backoff) on transport errors / non-200 **only** — never on a
  JSON-parse failure of a model reply.
- Reports are truncated at 40 000 chars for judging; truncation is recorded
  in the manifest and warned about in the scorecard's judge block.
- Per-judge raw JSON + rationales are retained in eval.json (audit trail).

## Determinism contract

`difyeval replay <case> --no-judge` twice produces **byte-identical
scorecards** except the single `- generated: <ts>` line (enforced by a test).
All iteration over unordered data is sorted; the only wall-clock reads are
that line and eval.json's `manifest.scored_at`; the only randomness is the
seeded rubric shuffle. `--no-judge` paths never import networking.

## eval.json (schema `eval.json/1`)

```jsonc
{
  "schema": "eval.json/1",
  "case": { /* the authored case.yml, echoed verbatim */ },
  "case_path": "...", "profile": "default",
  "verdict": "PASS", "warn_reason": null, "scope": "...",
  "results": [
    {"sample_id": "s1", "repeat": 1,
     "run": {"output": "...", "nodes": {...}, "usage": {...}, "meta": {...}, "error": null},
     "scores": [
       {"check_id": "has_out", "value": true, "passed": true, "evidence": "...",
        "marginal": false, "advisory": false,
        "detail": { /* rubric items+votes, judge per_judge raw outputs & rationales */ }}
     ]}
  ],
  "metrics": {"pass_rate": {...}, "per_check": {...}, "repeat_consistency": {...}},
  "manifest": {
    "difyeval_version": "0.1.0",
    "git_sha": "... or null",
    "case_sha256": "...", "dsl_sha256": "... or null (--dsl <path>)",
    "accretions_sha256": "... or null (the case's accretions sidecar, when present)",
    "judge_backends": [{"name": "...", "vendor": "...", "model": "..."}],  // keys never stored
    "truncation": [ /* judge report-truncation events */ ],
    "scored_at": "...", "profile": "...",
    "profile_config": { /* the --profile-file mapping, verbatim; null without one */ },
    "mode": "reference"
  }
}
```

## Repo layout

```
difyeval/           the package (core, checks/, judge, runners/, verdict, report,
                    engine, gt, review, audit, diff, drift, profile, cli)
examples/text-source-discovery/   offline demo bundle (case.yml + fixtures)
tests/              unittest suite — `python -m unittest discover`
```
