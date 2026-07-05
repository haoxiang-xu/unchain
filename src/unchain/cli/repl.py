"""Minimal interactive REPL — reference implementation for the interject primitives.

Run with: unchain-repl --provider ollama --model llama3
While a run is in flight, type:
    /btw <question>   immediate side answer (main run unaffected)
    /fyi <text>       inject into the current run at the next iteration
    /steer <text>     queue for a new run right after this one finishes
"""

from __future__ import annotations

import argparse
import threading
from typing import Any, Callable

from ..agent import Agent, InteractionModule
from ..interaction import FyiChannel, ProgressDigest, SteerBuffer, build_btw_prompt
from ..kernel.lifecycle_events import last_assistant_text

_PREFIXES = {"/btw": "btw", "/fyi": "fyi", "/steer": "steer"}


def route_input(line: str) -> tuple[str, str]:
    stripped = line.strip()
    if not stripped:
        return ("empty", "")
    for prefix, channel in _PREFIXES.items():
        if stripped == prefix or stripped.startswith(prefix + " "):
            body = stripped[len(prefix):].strip()
            return (channel, body) if body else ("empty", "")
    return ("unknown", stripped)


def _fallback_renderer(event: dict[str, Any]) -> None:
    """Rich-free stand-in for ``TerminalRenderer``.

    The REPL's *only* job is to eventually print the model's answer, so this
    still has to happen even without rich installed. Prints the plain-text
    answer on ``final_message`` events; everything else is silently ignored
    (no live token/tool rendering without rich).
    """
    if not isinstance(event, dict) or event.get("type") != "final_message":
        return
    content = event.get("content") or ""
    print(f"\n{content}\n")


def _make_renderer() -> Callable[[dict[str, Any]], None]:
    """Best-effort TerminalRenderer — the 'render' extra (rich) is optional.

    Falls back to ``_fallback_renderer`` if rich isn't installed, so the
    final answer is still printed (and ProgressDigest, which is attached
    separately, still works).
    """
    try:
        from ..render import TerminalRenderer
    except ImportError:
        print("[unchain-repl] 'rich' not installed — the plain-text answer will still be "
              "printed, but live rendering is disabled "
              "(pip install unchain[render] to enable). Progress digest still works.")
        return _fallback_renderer
    return TerminalRenderer()


def _make_agent(args: argparse.Namespace, fyi_channel: FyiChannel) -> Agent:
    return Agent(
        name="repl_agent",
        provider=args.provider,
        model=args.model,
        instructions="You are a helpful assistant in a terminal session.",
        modules=(InteractionModule(fyi_channel=fyi_channel),),
    )


def _side_answer(args: argparse.Namespace, task: str, digest: ProgressDigest, question: str) -> None:
    try:
        side = Agent(
            name="repl_side_agent",
            provider=args.provider,
            model=args.side_model or args.model,
            instructions="",
        )
        result = side.run(build_btw_prompt(task, digest.summary(), question), max_iterations=1)
        print(f"\n[btw] {last_assistant_text(result.messages)}\n")
    except Exception as exc:  # noqa: BLE001 — a /btw failure must be visible, not silent
        print(f"\n[btw failed] {exc}\n")


def _run_worker(
    agent: Agent, task: str, callback: Callable[[dict[str, Any]], None], done: threading.Event
) -> None:
    """Run ``agent`` in a worker thread, always signalling ``done``.

    If ``agent.run`` raises, the exception is caught and printed so the
    input loop (which waits on ``done``) never hangs silently.
    """
    try:
        agent.run(task, callback=callback)
    except Exception as exc:  # noqa: BLE001 — surface run failures instead of hanging the loop
        print(f"\n[run failed] {exc}\n")
    finally:
        done.set()


def main() -> None:
    parser = argparse.ArgumentParser(prog="unchain-repl")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--side-model", dest="side_model", default=None)
    args = parser.parse_args()

    print("unchain interject REPL — /btw /fyi /steer during a run, plain text to start one, /quit to exit")
    while True:
        try:
            task = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            continue
        if task == "/quit":
            break

        fyi_channel = FyiChannel()
        steer_buffer = SteerBuffer()
        digest = ProgressDigest()
        renderer = _make_renderer()

        def callback(event: dict) -> None:
            digest(event)
            renderer(event)

        current_task = task
        while True:  # steer 链:一个 task 可能连续触发多轮 run
            agent = _make_agent(args, fyi_channel)
            done = threading.Event()
            worker = threading.Thread(
                target=_run_worker,
                args=(agent, current_task, callback, done),
                daemon=True,
            )
            worker.start()

            while not done.is_set():
                try:
                    line = input()
                except (EOFError, KeyboardInterrupt):
                    break
                channel, body = route_input(line)
                if channel == "btw":
                    threading.Thread(
                        target=_side_answer, args=(args, current_task, digest, body), daemon=True
                    ).start()
                elif channel == "fyi":
                    fyi_channel.post(body)
                    print(f"[fyi queued] will be injected at the next iteration ({fyi_channel.pending_count()} pending)")
                elif channel == "steer":
                    steer_buffer.post(body)
                    print(f"[steer queued] will start a new run after this one ({steer_buffer.pending_count()} pending)")
                elif channel == "unknown":
                    print("[?] run in progress — prefix with /btw, /fyi or /steer")
            worker.join()

            merged = steer_buffer.drain_merged()
            if merged is None:
                break
            print("\n[steer] starting follow-up run\n")
            current_task = merged


if __name__ == "__main__":
    main()
