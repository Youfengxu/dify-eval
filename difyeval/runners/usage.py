"""Usage aggregation: sum per-node usage blocks + top-level usage.

Dify LLM/agent nodes emit an ``outputs.usage`` block ({total_tokens,
total_price, currency, ...}); chatflow message_end carries a top-level usage
for the answer generation. This helper sums both into the Run.usage contract:
``{tokens?, cost?: {currency: amount}, nodes_without_usage?}`` — per-currency,
NO FX conversion, and an explicit count of nodes that reported no usage so
cost under-counting is visible, never silent.
"""
from __future__ import annotations


def _num(v):
    """Dify emits prices as strings ('0.00123'); coerce defensively."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def normalize_usage_block(u: dict) -> dict:
    """One Dify usage block -> {tokens?, cost?: {cur: amt}}."""
    out: dict = {}
    tt = _num(u.get("total_tokens"))
    if tt is not None:
        out["tokens"] = int(tt)
    tp = _num(u.get("total_price"))
    if tp is not None:
        cur = str(u.get("currency") or "USD")
        out["cost"] = {cur: tp}
    return out


def aggregate_usage(nodes: dict | None, top: dict | None = None) -> dict:
    """Sum nodes.*.outputs.usage blocks + an optional top-level usage block
    (already in Run.usage shape: {tokens?, cost?})."""
    tokens = 0
    cost: dict[str, float] = {}
    nodes_without = 0
    for nid in sorted(nodes or {}):
        node = nodes[nid] or {}
        u = (node.get("outputs") or {}).get("usage")
        if isinstance(u, dict):
            norm = normalize_usage_block(u)
        else:
            norm = {}
        if not norm:
            nodes_without += 1
            continue
        tokens += norm.get("tokens", 0)
        for cur, amt in (norm.get("cost") or {}).items():
            cost[cur] = cost.get(cur, 0.0) + amt
    if top:
        t = top.get("tokens")
        if isinstance(t, (int, float)) and not isinstance(t, bool):
            tokens += int(t)
        for cur in sorted(top.get("cost") or {}):
            amt = _num((top.get("cost") or {})[cur])
            if amt is not None:
                cost[cur] = cost.get(cur, 0.0) + amt
    out: dict = {}
    if tokens:
        out["tokens"] = tokens
    if cost:
        out["cost"] = {cur: round(cost[cur], 10) for cur in sorted(cost)}
    if nodes_without:
        out["nodes_without_usage"] = nodes_without
    return out
