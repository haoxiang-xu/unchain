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
from ..kernel.types import KernelRunResult

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
    agent: Agent,
    task: str | list[dict[str, Any]],
    callback: Callable[[dict[str, Any]], None],
    done: threading.Event,
    result_holder: list[Any] | None = None,
) -> None:
    """Run ``agent`` in a worker thread, always signalling ``done``.

    If ``agent.run`` raises, the exception is caught and printed so the
    input loop (which waits on ``done``) never hangs silently.

    If ``result_holder`` is given, the run's result (typically a
    ``KernelRunResult``) is appended to it on success and left untouched on
    failure. This is how the steer chain in ``main()`` recovers the prior
    run's transcript to carry conversation history into the follow-up run
    (see ``build_followup_messages``) — without it, the discarded return
    value would make every steer-triggered run start from scratch.
    """
    try:
        result = agent.run(task, callback=callback)
    except Exception as exc:  # noqa: BLE001 — surface run failures instead of hanging the loop
        print(f"\n[run failed] {exc}\n")
    else:
        if result_holder is not None:
            result_holder.append(result)
    finally:
        done.set()


def build_followup_messages(
    prior_messages: list[dict[str, Any]], merged: str
) -> list[dict[str, Any]]:
    """Build the next steer-chained run's input as full conversation history.

    A steer chain is *one conversation* split across several ``agent.run()``
    calls, not a series of unrelated tasks — so the follow-up run must see
    everything said (and answered) so far, plus the newly merged steer text
    as the next user turn. Passing ``merged`` alone would silently drop the
    prior transcript, which is incoherent with ``fyi``/``ProgressDigest``
    state carrying across the same chain (see the comment in ``main()``).
    """
    return [*prior_messages, {"role": "user", "content": merged}]


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

        # fyi_channel / steer_buffer / digest are intentionally scoped to the
        # whole task (the steer chain below), not to a single run: they are
        # constructed once per user task and reused across every
        # steer-triggered follow-up run in the inner loop. A leftover fyi
        # posted too late to be drained inside run N is still valid context
        # for run N+1 — with message history now carried across the chain
        # (see build_followup_messages), run N+1 is the *same conversation*
        # as run N, so replaying it at the next run's first iteration is
        # coherent, not a leak.
        fyi_channel = FyiChannel()
        steer_buffer = SteerBuffer()
        digest = ProgressDigest()
        renderer = _make_renderer()

        def callback(event: dict) -> None:
            digest(event)
            renderer(event)

        current_task: str | list[dict[str, Any]] = task
        while True:  # steer 链:一个 task 可能连续触发多轮 run
            agent = _make_agent(args, fyi_channel)
            done = threading.Event()
            result_holder: list[KernelRunResult] = []
            worker = threading.Thread(
                target=_run_worker,
                args=(agent, current_task, callback, done),
                kwargs={"result_holder": result_holder},
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
                    # Pass the original plain-text task, not current_task: once
                    # the steer chain carries history, current_task becomes a
                    # list[dict] (see build_followup_messages) and _side_answer
                    # needs a short string to describe "what the main agent is
                    # working on", not a stringified message list.
                    threading.Thread(
                        target=_side_answer, args=(args, task, digest, body), daemon=True
                    ).start()
                elif channel == "fyi":
                    fyi_channel.post(body)
                    print(f"[fyi queued] will be injected at the next iteration ({fyi_channel.pending_count()} pending)")
                elif channel == "steer":
                    steer_buffer.post(body)
                    print(f"[steer queued] will start a new run after this one ({steer_buffer.pending_count()} pending)")
                elif channel == "unknown":
                    print("[?] run in progress — prefix with /btw, /fyi or /steer")

            try:
                worker.join()
            except KeyboardInterrupt:
                print("\n[interrupted] waiting for the current run to stop...\n")
                # worker is a daemon thread wrapping a blocking provider call —
                # it cannot be killed from here. Give it one short grace
                # window to finish on its own, then give up and abandon the
                # steer chain rather than risk hanging forever (or eating a
                # second Ctrl+C as a crash) on a wedged provider call.
                try:
                    worker.join(timeout=5.0)
                except KeyboardInterrupt:
                    pass
                if worker.is_alive():
                    print("[interrupted] run still in progress, abandoning the steer chain\n")
                    break

            merged = steer_buffer.drain_merged()
            if merged is None:
                break
            print("\n[steer] starting follow-up run\n")
            if result_holder:
                current_task = build_followup_messages(result_holder[0].messages, merged)
            else:
                # The run failed (see [run failed] above) or otherwise left
                # no captured transcript — fall back to the merged steer
                # text alone rather than crash on a missing history.
                current_task = merged


if __name__ == "__main__":
    main()
