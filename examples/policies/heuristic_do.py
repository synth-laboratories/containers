"""Hunt the tree then collect. Used by examples/craftax_code_policy_eval.py."""

def choose_actions(*, observation_text, session, valid_actions, engine, seed, ply, readout):
    grid = str((readout or {}).get("ascii") or observation_text or "")
    rows = [row for row in grid.splitlines() if row]
    px = py = tx = ty = None
    for y, row in enumerate(rows):
        if "P" in row:
            px, py = row.index("P"), y
        if "T" in row:
            tx, ty = row.index("T"), y
    if tx is None:
        return {"actions": ["do" if "do" in (valid_actions or []) else "noop"], "policy_reason": "collect"}
    if px < tx:
        return {"actions": ["east"], "policy_reason": "hunt"}
    if px > tx:
        return {"actions": ["west"], "policy_reason": "hunt"}
    if py < ty:
        return {"actions": ["south"], "policy_reason": "hunt"}
    if py > ty:
        return {"actions": ["north"], "policy_reason": "hunt"}
    return {"actions": ["do"], "policy_reason": "collect"}
