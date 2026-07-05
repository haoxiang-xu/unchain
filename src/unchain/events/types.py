from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal


RuntimeEventType = Literal[
    "session.started",
    "run.started",
    "run.completed",
    "run.failed",
    "turn.started",
    "turn.completed",
    "step.started",
    "step.delta",
    "step.completed",
    "interaction.requested",
    "interaction.resolved",
    "interaction.fyi_injected",
    "artifact.created",
    "artifact.updated",
]

Visibility = Literal["user", "debug", "internal"]
SurfaceSlot = Literal["trace_inline", "iteration_summary", "run_summary", "side_panel", "debug"]
SurfaceScope = Literal["turn", "run", "session"]
SurfaceDefaultState = Literal["expanded", "collapsed", "hidden"]

RUNTIME_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "session.started",
        "run.started",
        "run.completed",
        "run.failed",
        "turn.started",
        "turn.completed",
        "step.started",
        "step.delta",
        "step.completed",
        "interaction.requested",
        "interaction.resolved",
        "interaction.fyi_injected",
        "artifact.created",
        "artifact.updated",
    }
)

VISIBILITIES: frozenset[str] = frozenset({"user", "debug", "internal"})
SURFACE_SLOTS: frozenset[str] = frozenset(
    {"trace_inline", "iteration_summary", "run_summary", "side_panel", "debug"}
)
SURFACE_SCOPES: frozenset[str] = frozenset({"turn", "run", "session"})
SURFACE_DEFAULT_STATES: frozenset[str] = frozenset({"expanded", "collapsed", "hidden"})


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(inner)
            for key, inner in value.items()
            if isinstance(key, (str, int, float, bool))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


@dataclass(frozen=True)
class RuntimeEventLinks:
    parent_run_id: str | None = None
    parent_event_id: str | None = None
    caused_by_event_id: str | None = None
    step_id: str | None = None
    tool_call_id: str | None = None
    interaction_id: str | None = None
    artifact_id: str | None = None
    workspace_change_set_id: str | None = None
    plan_id: str | None = None
    channel_id: str | None = None
    team_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "parent_run_id": self.parent_run_id,
            "parent_event_id": self.parent_event_id,
            "caused_by_event_id": self.caused_by_event_id,
            "step_id": self.step_id,
            "tool_call_id": self.tool_call_id,
            "interaction_id": self.interaction_id,
            "artifact_id": self.artifact_id,
            "workspace_change_set_id": self.workspace_change_set_id,
            "plan_id": self.plan_id,
            "channel_id": self.channel_id,
            "team_id": self.team_id,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeEventLinks":
        if not isinstance(raw, dict):
            return cls()
        values = {key: _str_or_none(raw.get(key)) for key in cls().to_dict()}
        return cls(**values)


@dataclass(frozen=True)
class RuntimeEventSurface:
    slot: SurfaceSlot = "trace_inline"
    scope: SurfaceScope = "turn"
    group: str = ""
    default_state: SurfaceDefaultState = "collapsed"
    priority: int = 100

    def __post_init__(self) -> None:
        if self.slot not in SURFACE_SLOTS:
            raise ValueError(f"surface.slot must be one of {sorted(SURFACE_SLOTS)}")
        if self.scope not in SURFACE_SCOPES:
            raise ValueError(f"surface.scope must be one of {sorted(SURFACE_SCOPES)}")
        if self.default_state not in SURFACE_DEFAULT_STATES:
            raise ValueError(
                f"surface.default_state must be one of {sorted(SURFACE_DEFAULT_STATES)}"
            )
        object.__setattr__(self, "group", str(self.group or ""))
        object.__setattr__(self, "priority", int(self.priority))

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "scope": self.scope,
            "group": self.group,
            "default_state": self.default_state,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "RuntimeEventSurface":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            slot=raw.get("slot", "trace_inline"),
            scope=raw.get("scope", "turn"),
            group=str(raw.get("group") or ""),
            default_state=raw.get("default_state", "collapsed"),
            priority=int(raw.get("priority") or 100),
        )


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    type: str
    timestamp: str
    session_id: str
    run_id: str
    agent_id: str
    seq: int
    turn_id: str | None = None
    links: RuntimeEventLinks = field(default_factory=RuntimeEventLinks)
    surface: RuntimeEventSurface = field(default_factory=RuntimeEventSurface)
    visibility: Visibility = "user"
    payload: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    schema_version: str = "v4"

    def __post_init__(self) -> None:
        if self.schema_version != "v4":
            raise ValueError("schema_version must be 'v4'")
        if self.type not in RUNTIME_EVENT_TYPES:
            raise ValueError(f"unknown event type: {self.type}")
        if self.visibility not in VISIBILITIES:
            raise ValueError(f"visibility must be one of {sorted(VISIBILITIES)}")
        object.__setattr__(self, "seq", int(self.seq))
        if self.payload is None:
            object.__setattr__(self, "payload", {})
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "turn_id": self.turn_id,
            "seq": self.seq,
            "links": self.links.to_dict(),
            "surface": self.surface.to_dict(),
            "visibility": self.visibility,
            "payload": _json_safe(copy.deepcopy(self.payload or {})),
            "metadata": _json_safe(copy.deepcopy(self.metadata or {})),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, strict: bool = True) -> "RuntimeEvent":
        if not isinstance(raw, dict):
            raise TypeError("runtime event must be a dict")
        if raw.get("schema_version") != "v4":
            raise ValueError("schema_version must be 'v4'")
        event_type = raw.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("type is required")
        if strict and event_type not in RUNTIME_EVENT_TYPES:
            raise ValueError(f"unknown event type: {event_type}")
        visibility = raw.get("visibility", "user")
        if visibility not in VISIBILITIES:
            raise ValueError(f"visibility must be one of {sorted(VISIBILITIES)}")

        def string_field(key: str, *, required: bool = True) -> str:
            value = raw.get(key)
            if isinstance(value, str):
                return value
            if required:
                raise ValueError(f"{key} is required")
            return ""

        turn_id = raw.get("turn_id")
        return cls(
            schema_version="v4",
            event_id=string_field("event_id"),
            type=event_type,
            timestamp=string_field("timestamp"),
            session_id=string_field("session_id"),
            run_id=string_field("run_id"),
            agent_id=string_field("agent_id"),
            turn_id=turn_id if isinstance(turn_id, str) and turn_id else None,
            seq=int(raw.get("seq") or 0),
            links=RuntimeEventLinks.from_dict(raw.get("links")),
            surface=RuntimeEventSurface.from_dict(raw.get("surface")),
            visibility=visibility,  # type: ignore[arg-type]
            payload=copy.deepcopy(raw.get("payload")) if isinstance(raw.get("payload"), dict) else {},
            metadata=copy.deepcopy(raw.get("metadata")) if isinstance(raw.get("metadata"), dict) else {},
        )
