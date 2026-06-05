import re
from typing import List, Tuple

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def _normalize_uci(raw: str) -> str:
    s = raw.strip().lower().replace(" ", "")
    s = s.replace("×", "x")
    if "x" in s and len(s) <= 5:
        s = s.replace("x", "")
    return s


def chess_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Parse model outputs for UCI moves inside <action>...</action>.
    Returns (uci_strings, validity_flags). Empty string denotes an invalid / missing move.
    """
    valids = [0] * len(actions)
    out: List[str] = []

    for i, original in enumerate(actions):
        text = original.lower()
        start_tag = "<action>"
        end_tag = "</action>"
        sidx = text.find(start_tag)
        eidx = text.find(end_tag)

        think_s = original.find("<think>")
        think_e = original.find("</think>")
        if think_s == -1 or think_e == -1:
            out.append("")
            continue

        if sidx == -1 or eidx == -1 or eidx <= sidx:
            out.append("")
            continue

        inner = text[sidx + len(start_tag) : eidx].strip()
        inner = _normalize_uci(inner)
        if _UCI_RE.match(inner):
            out.append(inner)
            valids[i] = 1
        else:
            out.append("")

    return out, valids
