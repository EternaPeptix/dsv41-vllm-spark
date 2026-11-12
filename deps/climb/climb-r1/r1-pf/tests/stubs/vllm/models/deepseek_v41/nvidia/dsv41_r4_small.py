"""Test stub of the CERT head hook point."""
MAINPROJ_SPLIT = True
CALLS = []


def certified_logits(st, x, full=False, debug=None):
    CALLS.append(("certified_logits", st, x))
    return "logits"


def target_logits(model, hidden_states, skip_gather=False):
    return certified_logits(None, hidden_states)  # global lookup at call time, as in the live module
