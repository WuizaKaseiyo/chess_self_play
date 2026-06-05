"""Environment smoke test for the `chess` venv on a GPU node.

Validates, in order:
  1. interpreter + GPU visibility (torch.cuda)
  2. the heavy training stack imports (vllm / ray / transformers / flash_attn)
  3. the chesslesson env stack imports (gym / chess) and our integration runs
     (ChessLessonWorker: projection -> score -> binary reward)
  4. a tiny CUDA tensor op to confirm the CUDA runtime actually works
  5. (optional) the ray-backed vectorised env: build -> reset -> step

Exit code is non-zero if any mandatory check fails, so condor marks the job
failed and the .err/.out tell you exactly what broke.
"""
import os
import sys
import traceback

FAIL = []


def check(name, fn, mandatory=True):
    try:
        val = fn()
        print(f"[ OK ] {name}: {val}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()
        if mandatory:
            FAIL.append(name)


print("=" * 60)
print("python :", sys.executable)
print("version:", sys.version.replace("\n", " "))
print("=" * 60)

# 1. torch + GPU
import torch  # noqa: E402

check("torch.__version__", lambda: torch.__version__)
check("torch.version.cuda", lambda: torch.version.cuda)
check("torch.cuda.is_available", lambda: torch.cuda.is_available())
check("torch.cuda.device_count", lambda: torch.cuda.device_count())
check("gpu names", lambda: [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# 2. heavy stack imports
check("import vllm", lambda: __import__("vllm").__version__)
check("import ray", lambda: __import__("ray").__version__)
check("import transformers", lambda: __import__("transformers").__version__)
check("import flash_attn", lambda: __import__("flash_attn").__version__)

# 3. chesslesson env stack + our integration
check("import gym", lambda: __import__("gym").__version__)
check("import chess (python-chess)", lambda: __import__("chess").__version__)


def chesslesson_integration():
    sys.path.insert(0, "$HOME/verl-agent")
    from chess_game.chesslesson_envs import ChessLessonWorker, chesslesson_projection

    w = ChessLessonWorker({"include_chess": True, "include_coord": True})
    n = w.task_count()

    payload, valid = chesslesson_projection(
        ["<think>x</think><action>a1e1</action>", "no tags",
         "<think>y</think><action>light</action>"]
    )
    assert valid == [1, 0, 1], f"projection valid mismatch: {valid}"

    idx = next(i for i, t in enumerate(w.tasks) if t["id"] == "check1-1")
    w.reset(idx)
    _, rew, done, info = w.step("<action>a1e1</action>")
    assert rew == 1.0 and info["won"] and done, f"check1-1 scoring wrong: {rew} {info}"
    assert "graded" not in info, "graded should be gone (binary reward only)"
    return f"{n} tasks, projection OK, check1-1->reward=1.0, binary reward OK"


check("chesslesson integration", chesslesson_integration)

# 4. CUDA tensor op
def cuda_matmul():
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible")
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    return f"512x512 matmul on GPU ok (sum={c:.1f})"


check("cuda matmul", cuda_matmul)

# 5. ray-backed vectorised env (optional: validates the full env path)
def ray_env():
    sys.path.insert(0, "$HOME/verl-agent")
    from chess_game.chesslesson_envs import build_chesslesson_envs

    envs = build_chesslesson_envs(
        seed=0, env_num=1, group_n=2,
        resources_per_worker={"num_cpus": 0.1},
        is_train=True,
        env_kwargs={"include_chess": True, "include_coord": True},
    )
    obs, infos = envs.reset()
    next_obs, rews, dones, infos = envs.step(["<action>a1e1</action>"] * len(obs))
    envs.close()
    return f"reset->{len(obs)} obs, step->rewards={list(rews)}, dones={list(dones)}"


check("ray vectorised env", ray_env, mandatory=False)

print("=" * 60)
if FAIL:
    print("RESULT: FAILED ->", ", ".join(FAIL))
    sys.exit(1)
print("RESULT: ALL MANDATORY CHECKS PASSED")
