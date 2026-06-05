#!/usr/bin/env python3
"""Turn the lichess "Coordinate training" drill into LLM-solvable tasks (Python port).

Mirrors gen_coordinates.js exactly, including the seeded LCG so the generated
tasks are byte-identical. The trainer has two real modes
(ui/coordinateTrainer):
  - findSquare : a coordinate name is shown, you CLICK that square on the board
  - nameSquare : a square is highlighted on the board, you TYPE its name
plus a color choice (white/black) that flips the board orientation. A third mode
`squareColor` (light/dark) is added for pure coordinate reasoning.

Output:
  coordinates.jsonl       - one task/line {id, system, user, meta:{answer}}
  coordinates_sample.md   - a few rendered examples

Grading is exact-match:
    from gen_coordinates import grade
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = ["a", "b", "c", "d", "e", "f", "g", "h"]


def file_idx(f):
    return ord(f) - 97


def sq_name(fi, rank):
    return FILES[fi] + str(rank)


def square_color(sq):
    return "light" if (file_idx(sq[0]) + int(sq[1])) % 2 == 0 else "dark"


def grid_pos(sq, orientation):
    fi, rank = file_idx(sq[0]), int(sq[1])
    if orientation == "white":
        return {"col": fi + 1, "row": 8 - rank + 1}
    return {"col": 8 - fi, "row": rank}


def render_board(orientation, mark):
    ranks = [8, 7, 6, 5, 4, 3, 2, 1] if orientation == "white" else [1, 2, 3, 4, 5, 6, 7, 8]
    files = FILES if orientation == "white" else list(reversed(FILES))
    out = "   " + " ".join(files) + "\n"
    for rank in ranks:
        out += str(rank) + " |"
        out += " ".join("X" if (mark and f + str(rank) == mark) else "." for f in files)
        out += "\n"
    return out.rstrip()


SYSTEM = (
    'You are training on chess board coordinates (the Lichess "Coordinate training" drill).\n'
    "Conventions:\n"
    "- Squares are named by file (a-h, left-to-right from White's side) then rank (1-8, bottom-to-top from White's side), e.g. e4.\n"
    "- The board may be shown from White's OR Black's perspective. From Black's perspective the board is rotated 180 degrees: rank 1 is at the top and file h is on the left.\n"
    "- The board grid uses 'X' to mark a square and '.' for the others; row/column labels are printed around it.\n"
    "Output ONLY a final JSON object on the last line:\n"
    '  {"answer":"<value>"}\n'
    'where <value> is exactly what the task asks for (a square name like "e4", a "col,row" pair, or "light"/"dark").'
)


def name_square_task(sq, orientation):
    user = "\n".join([
        "Mode: name the square.",
        "The board is shown from %s's perspective." % ("White" if orientation == "white" else "Black"),
        "One square is marked with X. What is its algebraic name (e.g. e4)?",
        "",
        render_board(orientation, sq),
        "",
        'Answer with the square name, e.g. {"answer":"e4"}.',
    ])
    return {"user": user, "answer": sq}


def find_square_task(sq, orientation):
    user = "\n".join([
        "Mode: find the square.",
        "The board is shown from %s's perspective (blank board below for reference)." % ("White" if orientation == "white" else "Black"),
        'Where is square %s? Give its position as "col,row", 1-indexed, where col counts from the LEFT and row counts from the TOP of the board as drawn.' % sq,
        "",
        render_board(orientation, None),
        "",
        'Answer like {"answer":"3,5"}.',
    ])
    gp = grid_pos(sq, orientation)
    return {"user": user, "answer": "%d,%d" % (gp["col"], gp["row"])}


def square_color_task(sq):
    user = "\n".join([
        "Mode: square color.",
        "Is the square %s a light square or a dark square?" % sq,
        'Answer {"answer":"light"} or {"answer":"dark"}.',
    ])
    return {"user": user, "answer": square_color(sq)}


# ---- deterministic generation (seeded LCG, matches JS) ----
def make_rng(seed):
    state = {"s": seed & 0xFFFFFFFF}

    def rng():
        state["s"] = (state["s"] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state["s"] / 4294967296

    return rng


def rand_square(rng):
    fi = int(rng() * 8)
    rank = 1 + int(rng() * 8)
    return sq_name(fi, rank)


def pick(rng, arr):
    return arr[int(rng() * len(arr))]


def generate(n=60, seed=42):
    rng = make_rng(seed)
    modes = ["findSquare", "nameSquare", "squareColor"]
    tasks = []
    for i in range(n):
        mode = modes[i % len(modes)]
        orientation = pick(rng, ["white", "black"])
        sq = rand_square(rng)
        if mode == "findSquare":
            t = find_square_task(sq, orientation)
        elif mode == "nameSquare":
            t = name_square_task(sq, orientation)
        else:
            t = square_color_task(sq)
        tasks.append({
            "id": "coord-%s-%d" % (mode, i + 1),
            "kind": "coordinate",
            "mode": mode,
            "orientation": None if mode == "squareColor" else orientation,
            "square": sq,
            "system": SYSTEM,
            "user": t["user"],
            "meta": {"answer": t["answer"]},
        })
    return tasks


def _norm(x):
    return "".join(str("" if x is None else x).strip().lower().split())


def grade(task, model_answer):
    want = _norm(task["meta"]["answer"])
    got = _norm(model_answer)
    success = got == want
    return {"success": success, "reward": 1 if success else 0,
            "expected": task["meta"]["answer"], "got": model_answer}


if __name__ == "__main__":
    tasks = generate(60, 42)
    with open(os.path.join(HERE, "coordinates.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(t, ensure_ascii=False, separators=(",", ":")) for t in tasks) + "\n")

    parts = []
    for m in ["findSquare", "nameSquare", "squareColor"]:
        t = next(x for x in tasks if x["mode"] == m)
        parts.append("## %s\n\n**USER**\n\n```\n%s\n```\n\n**ANSWER (ground truth):** `%s`\n"
                     % (t["id"], t["user"], t["meta"]["answer"]))
    with open(os.path.join(HERE, "coordinates_sample.md"), "w") as f:
        f.write("# Coordinate-trainer task samples\n\n" + "\n---\n\n".join(parts))

    passed = sum(1 for t in tasks if grade(t, t["meta"]["answer"])["success"])
    print("coordinate tasks: %d (modes: findSquare/nameSquare/squareColor, both orientations)" % len(tasks))
    print("wrote coordinates.jsonl, coordinates_sample.md")
    print("self-test (ground truth grades as success): %d/%d" % (passed, len(tasks)))

    blk = next(x for x in generate(3, 7) if x["mode"] == "nameSquare")
    print("\nspot-check nameSquare %s marked=%s:" % (blk["orientation"], blk["square"]))
    print("\n".join(blk["user"].split("\n")[4:]))
