"""The run loop must pace itself on every path through it.

A repair path that skipped the pacing sleep spun at full speed, hammering the
MCP server with several HTTP round trips per iteration. From outside it looked
like the script had hung.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "drive.py"


def collect_loop() -> ast.While:
    tree = ast.parse(SOURCE.read_text())
    collect = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "collect")
    return next(node for node in ast.walk(collect) if isinstance(node, ast.While))


def test_the_run_loop_never_skips_its_pacing():
    # `continue` jumps past the sleep at the bottom of the loop. Any exit from
    # the body has to be a break (leaving the loop) or fall through to it.
    loop = collect_loop()
    assert not [node for node in ast.walk(loop) if isinstance(node, ast.Continue)]


def test_the_loop_still_ends_with_a_sleep():
    loop = collect_loop()
    assert any(isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "sleep"
               for node in ast.walk(loop.body[-1]))
