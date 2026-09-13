"""``difyeval diff`` — compare two Experiments (eval.json files).

Typical use: the same case scored under two profiles. The comparison is a
pure function of the two files — no wall clock, no network. Case-id mismatch
is refused (exit 2) unless --force: diffing different cases is almost always
an operator error, and a forced diff says so in the output header.
"""
from __future__ import annotations

from statistics import median

from .core import ConfigError
from .evalio import (check_order, dim_medians, eval_case_id, eval_profile,
                     load_eval, iter_scores, rubric_entries, sum_usage)

_FLIP = "⚠ **FLIP**"
_NA = "–"


def _fmt(v, digits=4):
    if v is None:
        return _NA
    if isinstance(v, float):
        return f"{round(v, digits):g}"
    return str(v)


def _delta(a, b, digits=4):
    if a is None or b is None:
        return _NA
    d = round(b - a, digits)
    return f"{'+' if d > 0 else ''}{d:g}"


def _summarize_checks(ev: dict) -> dict:
    """Per-check aggregate across all runs: n, ok, numeric-value mean, status."""
    agg: dict[str, dict] = {}
    for _, s in iter_scores(ev):
        cid = s.get("check_id")
        if not isinstance(cid, str):
            continue
        e = agg.setdefault(cid, {"n": 0, "ok": 0, "vals": [], "all_none": True})
        e["n"] += 1
        if s.get("passed") is True:
            e["ok"] += 1
        if s.get("passed") is not None:
            e["all_none"] = False
        v = s.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            e["vals"].append(float(v))
    for e in agg.values():
        e["mean"] = round(sum(e["vals"]) / len(e["vals"]), 4) if e["vals"] else None
        e["status"] = ("n/a" if e["all_none"]
                       else "pass" if e["ok"] == e["n"] else "fail")
    return agg


def _pass_cell(e) -> str:
    if e is None:
        return _NA
    if e["status"] == "n/a":
        return "n/a"
    return f"{e['ok']}/{e['n']}"


def _rubric_item_map(ev: dict) -> dict:
    """(check, item) -> {'pass': all-runs-pass, 'text': item text}. An item
    'passes' in an eval only when it passed in every run it appeared in."""
    out: dict = {}
    for e in rubric_entries(ev):
        key = (str(e["check"]), str(e["item"]))
        cur = out.setdefault(key, {"pass": True, "text": e["text"] or ""})
        if e["panel"] != "pass":
            cur["pass"] = False
    return out


def render_diff(ea: dict, eb: dict, path_a: str, path_b: str,
                forced: bool = False) -> str:
    ca, cb = eval_case_id(ea), eval_case_id(eb)
    title = f"`{ca}`" if ca == cb else f"`{ca}` vs `{cb}` (FORCED — different cases)"
    L = [f"# difyeval diff — {title}", ""]
    L += ["|  | A | B |", "|---|---|---|",
          f"| file | `{path_a}` | `{path_b}` |",
          f"| profile | {eval_profile(ea)} | {eval_profile(eb)} |",
          f"| verdict | {ea.get('verdict', '?')} | {eb.get('verdict', '?')} |",
          f"| scored_at | {(ea.get('manifest') or {}).get('scored_at', _NA)} "
          f"| {(eb.get('manifest') or {}).get('scored_at', _NA)} |", ""]

    va, vb = ea.get("verdict", "?"), eb.get("verdict", "?")
    L.append(f"## Verdict: {va} → {vb}" + ("" if va == vb else "  ⚠ changed"))
    L.append("")

    # ---- per-check table ----
    sa, sb = _summarize_checks(ea), _summarize_checks(eb)
    order = check_order(ea)
    for cid in check_order(eb):
        if cid not in order:
            order.append(cid)
    L += ["## Checks", "", "| check | value A | value B | Δ | pass A | pass B | flip |",
          "|---|---|---|---|---|---|---|"]
    for cid in order:
        a, b = sa.get(cid), sb.get(cid)
        ma = a["mean"] if a else None
        mb = b["mean"] if b else None
        sta = a["status"] if a else None
        stb = b["status"] if b else None
        if a is None or b is None:
            flip = f"only in {'A' if a else 'B'}"
        elif sta != stb and "n/a" not in (sta, stb):
            flip = _FLIP
        elif sta != stb:
            flip = f"{sta} → {stb}"
        else:
            flip = ""
        L.append(f"| {cid} | {_fmt(ma)} | {_fmt(mb)} | {_delta(ma, mb)} | "
                 f"{_pass_cell(a)} | {_pass_cell(b)} | {flip} |")
    L.append("")

    # ---- rubric item flips ----
    ra, rb = _rubric_item_map(ea), _rubric_item_map(eb)
    L += ["## Rubric item flips", ""]
    if not ra and not rb:
        L.append("- no rubric items in either eval")
    else:
        shared = [k for k in ra if k in rb]
        flips = [k for k in shared if ra[k]["pass"] != rb[k]["pass"]]
        if flips:
            L += ["| check | item | A | B | item text |", "|---|---|---|---|---|"]
            for check, item in flips:
                k = (check, item)
                L.append(f"| {check} | {item} | "
                         f"{'pass' if ra[k]['pass'] else 'fail'} | "
                         f"{'pass' if rb[k]['pass'] else 'fail'} | "
                         f"{(ra[k]['text'] or rb[k]['text'])[:80]} |")
        else:
            L.append(f"- no flips ({len(shared)} shared item(s) agree)")
        only_a = sorted(k for k in ra if k not in rb)
        only_b = sorted(k for k in rb if k not in ra)
        for label, only in (("A", only_a), ("B", only_b)):
            if only:
                L.append(f"- items only in {label}: "
                         + ", ".join(f"{c}/{i}" for c, i in only))
    L.append("")

    # ---- judge dims median delta ----
    da, db = dim_medians(ea), dim_medians(eb)
    L += ["## Judge dims (median of per-run panel medians)", ""]
    all_dims = sorted(set(da) | set(db))
    if not all_dims:
        L.append("- no judge dims in either eval")
    else:
        L += ["| dim | A | B | Δ |", "|---|---|---|---|"]
        for d in all_dims:
            va_ = round(median(da[d]), 4) if da.get(d) else None
            vb_ = round(median(db[d]), 4) if db.get(d) else None
            L.append(f"| {d} | {_fmt(va_)} | {_fmt(vb_)} | {_delta(va_, vb_)} |")
    L.append("")

    # ---- metrics ----
    pa = (ea.get("metrics") or {}).get("pass_rate") or {}
    pb = (eb.get("metrics") or {}).get("pass_rate") or {}
    L += ["## Metrics", "",
          f"- gated pass-rate: {_fmt(pa.get('rate'))} → {_fmt(pb.get('rate'))}"
          f" (Δ {_delta(pa.get('rate'), pb.get('rate'))})"]

    # ---- cost / tokens (only when either side reports any) ----
    ua, ub = sum_usage(ea), sum_usage(eb)
    if ua["tokens"] or ub["tokens"] or ua["cost"] or ub["cost"]:
        L += ["", "## Cost / tokens", "", "| what | A | B | Δ |", "|---|---|---|---|",
              f"| tokens | {ua['tokens']} | {ub['tokens']} | "
              f"{_delta(float(ua['tokens']), float(ub['tokens']), 0)} |"]
        for cur in sorted(set(ua["cost"]) | set(ub["cost"])):
            a_, b_ = ua["cost"].get(cur), ub["cost"].get(cur)
            L.append(f"| cost {cur} | {_fmt(a_, 6)} | {_fmt(b_, 6)} | {_delta(a_, b_, 6)} |")
    return "\n".join(L) + "\n"


def diff_command(path_a: str, path_b: str, out: str | None = None,
                 force: bool = False) -> int:
    ea, eb = load_eval(path_a), load_eval(path_b)
    ca, cb = eval_case_id(ea), eval_case_id(eb)
    if ca != cb and not force:
        raise ConfigError(f"case_id mismatch: {ca!r} vs {cb!r} — these are different "
                          f"cases; pass --force to diff them anyway")
    md = render_diff(ea, eb, path_a, path_b, forced=(ca != cb))
    print(md, end="")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"diff written: {out}")
    return 0
