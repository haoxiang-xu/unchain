"""Closed v2 execution state. V1 descriptors remain read-only compatibility data."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from unchain.journal.models import ArtifactRef, AttemptRef

from .errors import SkeletonValidationError
from .io import _validate_json
from .preparation import PreparedWorkflowPlan
from .agent_preparation import checked_execution_plan, execution_definition_digest


STATE_SCHEMA = "unchain.skeleton_execution_state.v2"
MAX_CONTENT_BYTES = 8 * 1024 * 1024
TERMINAL_STATUSES = frozenset({"completed", "cancelled", "failed", "outcome_unknown"})
_STATUSES = TERMINAL_STATUSES | {"ready", "running", "waiting", "result_received"}
_REF_KEYS = {"schema", "ref", "media_type", "byte_length", "sha256", "preview"}
_RESOURCE_KEYS = {"schema", "kind", "id", "revision", "fragment"}
_CODE_BINDING_KEYS = {"schema", "source_ref", "source_sha256", "profile", "artifacts"}


def fail(code: str, path: tuple, message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def canonical_bytes(value: Any) -> bytes:
    _validate_json(value, ("content",))
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8", "surrogatepass")
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise SkeletonValidationError("invalid_json", ("content",), "content cannot be encoded") from exc
    if len(raw) > MAX_CONTENT_BYTES:
        fail("limit_bytes", ("content",), "content exceeds eight MiB")
    return raw


def decode_json(content: bytes) -> Any:
    if type(content) is not bytes or len(content) > MAX_CONTENT_BYTES:
        fail("limit_bytes", ("content",), "content must be bounded bytes")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                fail("duplicate_name", ("content",), "duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(content.decode("utf-8", "surrogatepass"), object_pairs_hook=pairs)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SkeletonValidationError("invalid_json", ("content",), "invalid JSON content") from exc
    if canonical_bytes(value) != content:
        fail("invalid_json", ("content",), "content is not canonical JSON")
    return value


def _closed(value, keys, path):
    if type(value) is not dict or set(value) != keys:
        fail("invalid_shape", path, "closed object fields do not match")
    return value


def _text(value, path):
    if type(value) is not str or not value:
        fail("invalid_type", path, "expected nonempty text")


def _sha(value, path):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail("invalid_value", path, "expected SHA-256 digest")


def artifact_ref(value: Any, path: tuple = ("artifact",), *, json_only=False) -> ArtifactRef:
    _closed(value, _REF_KEYS, path)
    _closed(value["ref"], _RESOURCE_KEYS, path + ("ref",))
    try:
        ref = ArtifactRef.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise SkeletonValidationError("invalid_value", path, "invalid artifact descriptor") from exc
    if canonical_bytes(ref.to_dict()) != canonical_bytes(value):
        fail("invalid_shape", path, "artifact descriptor must be canonical")
    if ref.ref.kind != "artifact" or ref.ref.revision != 1 or ref.ref.fragment or ref.preview:
        fail("invalid_value", path, "artifact must be an unfragmented immutable reference without preview")
    if ref.byte_length > MAX_CONTENT_BYTES:
        fail("limit_bytes", path, "artifact exceeds eight MiB")
    if json_only and ref.media_type != "application/json":
        fail("invalid_value", path + ("media_type",), "receipt must reference JSON")
    return ref


def node_identity(state: dict, node_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(canonical_bytes([state["graph_id"], state["attempt"], node_id])).hexdigest()
    return "node." + digest, "node-attempt." + digest


def execution_artifacts(state: dict) -> tuple[ArtifactRef, ...]:
    values = [state["inputs_ref"]]
    values.extend(state["attachments"][key] for key in sorted(state["attachments"]))
    values.extend(
        state["code_bindings"][key]["source_ref"]
        for key in sorted(state.get("code_bindings", {}))
    )
    for row in state["completed"]:
        values.extend(row[key] for key in ("input_ref", "result_ref", "output_ref"))
    if state["active"] is not None:
        active = state["active"]
        values.append(active["input_ref"])
        if active["result_ref"] is not None:
            values.append(active["result_ref"])
        if active["wait"] is not None:
            values.append(active["wait"]["request_ref"])
            if active["wait"]["receipt_ref"] is not None:
                values.append(active["wait"]["receipt_ref"])
        if active.get("resume", {}).get("request_ref") is not None:
            values.append(active["resume"]["request_ref"])
    refs = {}
    for value in values:
        ref = artifact_ref(value)
        key = (ref.ref.resource_id, ref.ref.revision)
        if key in refs and refs[key] != ref:
            fail("identity_mismatch", ("artifact",), "conflicting artifact descriptors")
        refs[key] = ref
    return tuple(refs.values())


def _code_bindings(value: Any, attachments: dict[str, Any], path: tuple) -> None:
    if type(value) is not dict:
        fail("invalid_type", path, "Code bindings must be an object")
    for node_id, binding in value.items():
        _text(node_id, path + (node_id,))
        _closed(binding, _CODE_BINDING_KEYS, path + (node_id,))
        if binding["schema"] != "unchain.skeleton.code-binding.v1":
            fail("unknown_variant", path + (node_id, "schema"), "unsupported Code binding")
        artifact_ref(binding["source_ref"], path + (node_id, "source_ref"))
        _sha(binding["source_sha256"], path + (node_id, "source_sha256"))
        profile = _closed(binding["profile"], {"id", "kind", "command", "sha256"}, path + (node_id, "profile"))
        _text(profile["id"], path + (node_id, "profile", "id"))
        if type(profile["kind"]) is not str:
            fail("invalid_type", path + (node_id, "profile", "kind"), "runtime profile kind must be text")
        if profile["kind"] not in {"venv", "bundled"}:
            fail("unknown_variant", path + (node_id, "profile", "kind"), "unsupported runtime profile")
        if type(profile["command"]) is not list or not profile["command"] or any(type(item) is not str or not item for item in profile["command"]):
            fail("invalid_value", path + (node_id, "profile", "command"), "runtime command is invalid")
        _sha(profile["sha256"], path + (node_id, "profile", "sha256"))
        if type(binding["artifacts"]) is not list:
            fail("invalid_type", path + (node_id, "artifacts"), "Code artifacts must be an array")
        names = set()
        for index, item in enumerate(binding["artifacts"]):
            item_path = path + (node_id, "artifacts", index)
            _closed(item, {"name", "attachment", "ref"}, item_path)
            _text(item["name"], item_path + ("name",)); _text(item["attachment"], item_path + ("attachment",))
            if item["name"] in names:
                fail("duplicate_name", item_path + ("name",), "duplicate Code artifact name")
            names.add(item["name"])
            if item["attachment"] not in attachments or canonical_bytes(item["ref"]) != canonical_bytes(attachments[item["attachment"]]):
                fail("identity_mismatch", item_path + ("ref",), "Code artifact is not the granted attachment")
            artifact_ref(item["ref"], item_path + ("ref",))


def _code_bindings_match_plan(value: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> None:
    expected: dict[str, tuple[str, list[dict[str, str]]]] = {}
    for node_id, node in nodes.items():
        config = node.get("config")
        if type(config) is not dict or "code" not in config:
            continue
        code = config["code"]
        path = ("state", "code_bindings", node_id)
        if node.get("type") != "code":
            fail("identity_mismatch", path, "Code binding config targets a non-Code node")
        if type(code) is not dict:
            fail("invalid_shape", path, "fixed Code config is invalid")
        profile_id = code.get("environment_profile_id")
        artifacts = code.get("artifacts", [])
        if type(profile_id) is not str or not profile_id or type(artifacts) is not list:
            fail("invalid_shape", path, "fixed Code config identity is invalid")
        mappings = []
        for index, item in enumerate(artifacts):
            if type(item) is not dict or set(item) != {"name", "attachment"}:
                fail("invalid_shape", path + ("artifacts", index), "fixed Code artifact mapping is invalid")
            if type(item["name"]) is not str or type(item["attachment"]) is not str:
                fail("invalid_shape", path + ("artifacts", index), "fixed Code artifact mapping is invalid")
            mappings.append({"name": item["name"], "attachment": item["attachment"]})
        expected[node_id] = (profile_id, mappings)
    if set(value) != set(expected):
        fail("identity_mismatch", ("state", "code_bindings"), "Code bindings differ from fixed Code nodes")
    for node_id, binding in value.items():
        profile_id, mappings = expected[node_id]
        actual = [{"name": item["name"], "attachment": item["attachment"]} for item in binding["artifacts"]]
        if binding["profile"]["id"] != profile_id or actual != mappings:
            fail("identity_mismatch", ("state", "code_bindings", node_id), "Code binding differs from fixed node config")


def _node_row(row, state, path, *, active):
    keys = {"node_id", "node_invocation_id", "node_attempt_id", "input_ref"}
    keys |= {"owner_token", "result_ref", "wait"} if active else {"result_ref", "output_ref", "edge_id"}
    if active and type(row) is dict and set(row) == keys | {"resume"}:
        resume_keys = set(row["resume"]) if type(row["resume"]) is dict else set()
        if resume_keys == {"kind"}:
            resume = _closed(row["resume"], {"kind"}, path + ("resume",))
        else:
            resume = _closed(row["resume"], {"kind", "wait_id", "request_ref"}, path + ("resume",))
            _text(resume["wait_id"], path + ("resume", "wait_id"))
            artifact_ref(resume["request_ref"], path + ("resume", "request_ref"), json_only=True)
        if resume["kind"] != "child":
            fail("unknown_variant", path + ("resume", "kind"), "unsupported resumable continuation")
    else:
        _closed(row, keys, path)
    _text(row["node_id"], path + ("node_id",))
    expected = node_identity(state, row["node_id"])
    if (row["node_invocation_id"], row["node_attempt_id"]) != expected:
        fail("identity_mismatch", path, "node invocation is not bound to this graph attempt")
    artifact_ref(row["input_ref"], path + ("input_ref",), json_only=True)
    if not active or row["result_ref"] is not None:
        artifact_ref(row["result_ref"], path + ("result_ref",), json_only=True)
    if not active:
        artifact_ref(row["output_ref"], path + ("output_ref",), json_only=True)
        if row["edge_id"] is not None:
            _text(row["edge_id"], path + ("edge_id",))
    else:
        _text(row["owner_token"], path + ("owner_token",))
        if row["wait"] is not None:
            wait = _closed(row["wait"], {"kind", "id", "request_ref", "receipt_ref"}, path + ("wait",))
            if wait["kind"] not in ("interaction", "child"):
                fail("unknown_variant", path + ("wait", "kind"), "unsupported wait kind")
            _text(wait["id"], path + ("wait", "id"))
            artifact_ref(wait["request_ref"], path + ("wait", "request_ref"), json_only=True)
            if wait["receipt_ref"] is not None:
                artifact_ref(wait["receipt_ref"], path + ("wait", "receipt_ref"), json_only=True)


def validate_execution_state(state: Any, plan: PreparedWorkflowPlan | None = None) -> dict:
    _validate_json(state, ("state",))
    keys = {"schema", "graph_id", "attempt", "definition_digest", "call_stack", "inputs_ref",
            "attachments", "completed", "active", "status", "paused", "error", "root_committed"}
    if type(state) is not dict or not keys <= set(state) or not set(state) <= keys | {"code_bindings"}:
        fail("invalid_shape", ("state",), "closed object fields do not match")
    if state["schema"] != STATE_SCHEMA:
        fail("unknown_variant", ("state", "schema"), "unsupported execution state")
    _text(state["graph_id"], ("state", "graph_id"))
    try:
        attempt = AttemptRef.from_dict(state["attempt"])
    except (TypeError, ValueError) as exc:
        raise SkeletonValidationError("invalid_value", ("state", "attempt"), "invalid attempt") from exc
    if canonical_bytes(attempt.to_dict()) != canonical_bytes(state["attempt"]):
        fail("invalid_shape", ("state", "attempt"), "attempt must be canonical")
    _sha(state["definition_digest"], ("state", "definition_digest"))
    frames = state["call_stack"]
    if type(frames) is not list or not frames or len(frames) > 32:
        fail("invalid_value", ("state", "call_stack"), "call stack must contain 1 to 32 frames")
    graph_ids, attempts, callers = set(), set(), set()
    for index, frame in enumerate(frames):
        final = index == len(frames) - 1
        keys = {"execution_id", "graph_invocation_id", "attempt_id", "definition_id", "definition_digest"}
        if not final:
            keys.add("caller_node_invocation_id")
        _closed(frame, keys, ("state", "call_stack", index))
        for key in keys:
            _text(frame[key], ("state", "call_stack", index, key))
        _sha(frame["definition_digest"], ("state", "call_stack", index, "definition_digest"))
        if frame["execution_id"] != attempt.generation.execution_id:
            fail("identity_mismatch", ("state", "call_stack", index), "foreign execution frame")
        if frame["graph_invocation_id"] in graph_ids or frame["attempt_id"] in attempts:
            fail("duplicate_name", ("state", "call_stack", index), "duplicate graph or attempt")
        graph_ids.add(frame["graph_invocation_id"]); attempts.add(frame["attempt_id"])
        if not final:
            if frame["caller_node_invocation_id"] in callers:
                fail("duplicate_name", ("state", "call_stack", index), "duplicate caller")
            callers.add(frame["caller_node_invocation_id"])
    final = frames[-1]
    if (final["graph_invocation_id"], final["attempt_id"], final["definition_digest"]) != (
        state["graph_id"], attempt.attempt_id, state["definition_digest"]
    ):
        fail("identity_mismatch", ("state", "call_stack"), "current frame does not match execution")
    artifact_ref(state["inputs_ref"], ("state", "inputs_ref"), json_only=True)
    if type(state["attachments"]) is not dict:
        fail("invalid_type", ("state", "attachments"), "attachments must be an object")
    for key, ref in state["attachments"].items():
        _text(key, ("state", "attachments"))
        artifact_ref(ref, ("state", "attachments", key))
    if "code_bindings" in state:
        _code_bindings(state["code_bindings"], state["attachments"], ("state", "code_bindings"))
    if type(state["status"]) is not str or state["status"] not in _STATUSES:
        fail("unknown_variant", ("state", "status"), "unknown execution status")
    if type(state["paused"]) is not bool or type(state["root_committed"]) is not bool:
        fail("invalid_type", ("state",), "control flags must be boolean")
    if type(state["completed"]) is not list:
        fail("invalid_type", ("state", "completed"), "completed must be an array")
    seen = set()
    for index, row in enumerate(state["completed"]):
        _node_row(row, state, ("state", "completed", index), active=False)
        if row["node_id"] in seen:
            fail("duplicate_name", ("state", "completed", index), "node already completed")
        seen.add(row["node_id"])
    active = state["active"]
    if active is not None:
        _node_row(active, state, ("state", "active"), active=True)
        if active["node_id"] in seen:
            fail("invalid_value", ("state", "active"), "active node already completed")
    status = state["status"]
    if status == "completed" and not state["completed"]:
        fail("missing_value", ("state", "completed"), "completed graph requires an End receipt")
    if status in {"ready", "completed"} and active is not None:
        fail("invalid_value", ("state", "active"), "ready/completed cannot have an active node")
    unknown_child_wait = False
    if status in {"running", "waiting", "result_received", "outcome_unknown"}:
        if active is None:
            fail("missing_value", ("state", "active"), "active record required")
        if (active["result_ref"] is not None) != (status == "result_received"):
            fail("invalid_value", ("state", "active", "result_ref"), "result does not match phase")
        unknown_child_wait = (status == "outcome_unknown" and active["wait"] is not None
                              and active["wait"]["kind"] == "child"
                              and type(state["error"]) is dict
                              and state["error"].get("code") == "child_outcome_unknown")
        if (active["wait"] is not None) != (status == "waiting") and not unknown_child_wait:
            fail("invalid_value", ("state", "active", "wait"), "wait does not match phase")
        if "resume" in active and (
            status != "running" or active["wait"] is not None or active["result_ref"] is not None
        ):
            fail("invalid_value", ("state", "active", "resume"), "resume marker requires a running continuation")
    if status in TERMINAL_STATUSES and state["paused"]:
        fail("invalid_value", ("state", "paused"), "terminal state cannot be paused")
    if state["root_committed"] and (status != "completed" or len(frames) != 1):
        fail("invalid_value", ("state", "root_committed"), "only root End may commit")
    if state["error"] is not None:
        error = _closed(state["error"], {"code", "path"}, ("state", "error"))
        _text(error["code"], ("state", "error", "code"))
        if type(error["path"]) is not list or any(type(v) not in (str, int) for v in error["path"]):
            fail("invalid_type", ("state", "error", "path"), "invalid diagnostic path")
        if status not in {"failed", "outcome_unknown"}:
            fail("invalid_value", ("state", "error"), "unexpected failure diagnostic")
    elif status == "failed":
        fail("missing_value", ("state", "error"), "failed state needs a diagnostic")
    if plan is not None:
        checked = checked_execution_plan(plan)
        if execution_definition_digest(checked) != state["definition_digest"] or final["definition_id"] != checked.document_id:
            fail("identity_mismatch", ("state", "definition_digest"), "definition differs from fixed plan")
        nodes = {n["id"]: n for n in checked.snapshot()["nodes"]}
        if unknown_child_wait and nodes[active["node_id"]]["type"] != "workflow_call":
            fail("invalid_value", ("state", "active", "wait"),
                 "only a Workflow Call may retain an unknown child wait")
        bindings = state.get("code_bindings", {})
        if "code_bindings" in state:
            _code_bindings_match_plan(bindings, nodes)
        edges = {e.id: e for e in checked.flow_edges}
        current = next(n["id"] for n in nodes.values() if n["type"] == "start")
        for index, row in enumerate(state["completed"]):
            if row["node_id"] != current:
                fail("path_mismatch", ("state", "completed", index), "receipt is not on selected path")
            if nodes[current]["type"] == "end":
                if row["edge_id"] is not None or index != len(state["completed"]) - 1:
                    fail("path_mismatch", ("state", "completed", index), "End cannot continue")
                current = None
            else:
                edge = edges.get(row["edge_id"])
                if edge is None or edge.source_node_id != current:
                    fail("path_mismatch", ("state", "completed", index, "edge_id"), "foreign selected edge")
                current = edge.target_node_id
        if (status == "completed") != (current is None):
            fail("path_mismatch", ("state", "status"), "End receipt and terminal status disagree")
        if active is not None:
            if active["node_id"] != current or nodes[current]["type"] not in {"code", "tool_call", "skill_read", "workflow_call", "agent"}:
                fail("path_mismatch", ("state", "active"), "active handler is not the next executable node")
    execution_artifacts(state)
    return deepcopy(state)


def _same(left, right):
    return canonical_bytes(left) == canonical_bytes(right)


def validate_execution_transition(previous: dict | None, current: dict, plan: PreparedWorkflowPlan | None) -> None:
    current = validate_execution_state(current, plan)
    if previous is None:
        if current["status"] != "ready" or current["completed"] or current["active"] or current["paused"] or current["root_committed"]:
            fail("invalid_transition", ("state",), "new execution must start ready")
        return
    previous = validate_execution_state(previous, plan)
    for key in ("schema", "graph_id", "attempt", "definition_digest", "call_stack", "inputs_ref", "attachments", "code_bindings"):
        if not _same(previous.get(key), current.get(key)):
            fail("identity_mismatch", ("state", key), "fixed execution input changed")
    before, after = previous["status"], current["status"]
    if before in TERMINAL_STATUSES:
        acknowledged = deepcopy(previous)
        acknowledged["root_committed"] = True
        if before == "completed" and not previous["root_committed"] and _same(acknowledged, current):
            return
        fail("invalid_transition", ("state",), "terminal execution cannot advance")
    if current["root_committed"]:
        fail("invalid_transition", ("state", "root_committed"), "End must precede root acknowledgement")
    old_rows, rows = previous["completed"], current["completed"]
    if len(rows) < len(old_rows) or not _same(rows[:len(old_rows)], old_rows) or len(rows) > len(old_rows) + 1:
        fail("invalid_transition", ("state", "completed"), "completion must extend the path once")
    added = len(rows) > len(old_rows)
    if added:
        if before not in {"ready", "result_received"} or after not in {"ready", "completed"} or previous["paused"]:
            fail("invalid_transition", ("state",), "cannot commit a node from this phase")
        row = rows[-1]
        if before == "result_received":
            for key in ("node_id", "node_invocation_id", "node_attempt_id", "input_ref", "result_ref"):
                if not _same(previous["active"][key], row[key]):
                    fail("identity_mismatch", ("state", "completed", key), "completion replaced active receipt")
        elif plan is not None:
            node = next(n for n in plan.snapshot()["nodes"] if n["id"] == row["node_id"])
            if node["type"] not in {"start", "end", "if_else", "switch_case"}:
                fail("invalid_transition", ("state", "completed"), "handler requires an execution intent and raw receipt")
        return
    if after in {"cancelled", "failed"}:
        if not _same(current["active"], previous["active"]):
            fail("invalid_transition", ("state", "active"), "failure/cancellation must preserve pending work")
        if (after == "failed" and before == "running"
                and current["error"]["code"] not in {
                    "child_failed", "child_cancelled", "hook_invalid_decision",
                }):
            fail("invalid_transition", ("state",), "running side effect has unknown outcome")
        return
    if before == "waiting" and after == "outcome_unknown":
        if (previous["active"]["wait"]["kind"] != "child"
                or current["error"]["code"] != "child_outcome_unknown"
                or not _same(current["active"], previous["active"])):
            fail("invalid_transition", ("state", "active"),
                 "unknown child outcome must preserve its exact pending wait")
        return
    if before == after:
        if before == "waiting" and previous["paused"] == current["paused"]:
            prior = deepcopy(previous["active"])
            next_active = deepcopy(current["active"])
            if (
                prior["wait"] is not None
                and next_active["wait"] is not None
                and prior["wait"]["receipt_ref"] is None
                and next_active["wait"]["receipt_ref"] is not None
            ):
                prior["wait"]["receipt_ref"] = next_active["wait"]["receipt_ref"]
                if _same(prior, next_active):
                    return
            if (
                prior["wait"] is not None
                and next_active["wait"] is not None
                and prior["wait"]["kind"] == "child"
                and next_active["wait"]["kind"] == "child"
                and prior["wait"]["receipt_ref"] is None
                and next_active["wait"]["receipt_ref"] is None
            ):
                prior["wait"] = next_active["wait"]
                if _same(prior, next_active):
                    return
        control = deepcopy(previous); control["paused"] = current["paused"]
        if not _same(control, current) or control["paused"] == previous["paused"]:
            fail("invalid_transition", ("state",), "only pause flag may change without progress")
        return
    allowed = {("ready", "running"), ("running", "result_received"), ("running", "waiting"),
               ("running", "outcome_unknown"), ("waiting", "result_received"), ("waiting", "running")}
    if (before, after) not in allowed:
        fail("invalid_transition", ("state", "status"), "unsupported phase transition")
    if previous["paused"] != current["paused"] and after != "outcome_unknown":
        fail("invalid_transition", ("state", "paused"), "phase transition cannot change pause")
    if before == "ready":
        if previous["paused"]:
            fail("invalid_transition", ("state",), "paused execution cannot dispatch")
        return
    resuming = (before, after) == ("waiting", "running")
    if resuming:
        # A journal-only reader can prove the fixed identities and transition,
        # but cannot infer a node's type without its prepared definition.
        # Writes always require the actual plan in WorkflowExecutionStore.save.
        node = (next(n for n in plan.snapshot()["nodes"] if n["id"] == previous["active"]["node_id"])
                if plan is not None else None)
        wait_kind = previous["active"]["wait"]["kind"]
        resumable = node is None or (
            (node["type"] == "agent" and wait_kind in {"interaction", "child"})
            or (node["type"] == "tool_call" and wait_kind == "interaction")
        )
        if (not resumable
                or previous["paused"] or previous["active"]["owner_token"] == current["active"]["owner_token"]):
            fail("invalid_transition", ("state", "active"), "only an unpaused resumable wait may acquire a fresh owner")
        expected_resume = ({"kind": "child"}
                           if wait_kind == "child" and (node is None or node["type"] == "agent") else None)
        actual_resume = current["active"].get("resume")
        if actual_resume is not None and (
            expected_resume is None or actual_resume.get("kind") != expected_resume["kind"]
        ):
            fail("invalid_transition", ("state", "active", "resume"), "resume marker does not match the resumed wait")
        if expected_resume is not None and actual_resume is not None:
            previous_wait = previous["active"]["wait"]
            if (
                type(actual_resume) is not dict
                or set(actual_resume) != {"kind", "wait_id", "request_ref"}
                or actual_resume["wait_id"] != previous_wait["id"]
                or not _same(actual_resume["request_ref"], previous_wait["request_ref"])
            ):
                fail("identity_mismatch", ("state", "active", "resume"),
                     "child resume marker is not bound to the replaced wait")
    elif current["active"] is not None and "resume" in current["active"]:
        fail("invalid_transition", ("state", "active", "resume"), "resume marker is only valid while resuming a child wait")
    keys = ("node_id", "node_invocation_id", "node_attempt_id", "input_ref")
    if not resuming:
        keys += ("owner_token",)
    for key in keys:
        if not _same(previous["active"][key], current["active"][key]):
            fail("identity_mismatch", ("state", "active", key), "active execution identity changed")
