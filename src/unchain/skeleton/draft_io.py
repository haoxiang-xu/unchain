"""Editing-time IO diagnostics, independent of complete execution configuration."""
from __future__ import annotations

from typing import Any, Callable

from .branch_admission import analyze_branch_execution
from .diagnostics import SkeletonDiagnostic
from .errors import SkeletonValidationError
from .static_io import StaticIOSurfaces

_POOLS = {"toolkit_pool", "subagent_pool", "hooks_pool"}
_BRANCHES = {"if_else", "switch_case"}


def _rows(value: Any, path: tuple[str | int, ...], validate: Callable,
          diagnostics: list[SkeletonDiagnostic]) -> list[tuple[int, dict[str, Any]]]:
    if type(value) is not list:
        try:
            validate(value, path)
        except SkeletonValidationError as error:
            diagnostics.append(SkeletonDiagnostic(error.code, error.path, str(error)))
        return []
    names: set[str] = set()
    valid = []
    for index, row in enumerate(value):
        # Reuse admission's row grammar, but collect independent row failures.
        try:
            validate([row], path)
        except SkeletonValidationError as error:
            actual_path = path + (index,) + error.path[len(path) + 1:]
            diagnostics.append(SkeletonDiagnostic(error.code, actual_path, str(error)))
        else:
            valid.append((index, row))
        name = row.get("name") if type(row) is dict else None
        if type(name) is str and name:
            if name in names:
                section = "binding" if path[-1] == "inputs" else "output"
                diagnostics.append(SkeletonDiagnostic("duplicate_name", path + (index, "name"),
                                                      f"{section} name is duplicated"))
            names.add(name)
    return valid


def diagnose_execution_io(skeleton: dict[str, Any]) -> tuple[SkeletonDiagnostic, ...]:
    """Inspect declared IO without preparing Kernels, providers or child workflows."""
    # execution imports preparation, whose structural gate imports validation.
    # Defer this provider-free import until the modules have finished loading.
    from .execution import _validate_bindings, _validate_outputs

    diagnostics: list[SkeletonDiagnostic] = []
    surfaces = StaticIOSurfaces(skeleton)

    def references(expression: dict[str, Any], path: tuple[str | int, ...], node_id: str) -> None:
        pending = [(expression, path)]
        while pending:
            value, value_path = pending.pop()
            if value["kind"] == "first_present":
                pending.extend((child, value_path + ("values", index))
                               for index, child in reversed(list(enumerate(value["values"]))))
            elif value["kind"] == "ref":
                problem = surfaces.reference_problem(value, node_id)
                if problem is not None:
                    suffix, message = problem
                    diagnostics.append(SkeletonDiagnostic("missing_value", value_path + suffix, message))

    for index, node in enumerate(skeleton["nodes"]):
        config = node["config"]
        if "execution" not in config:
            continue  # Preserve structural-only drafts with no declared IO yet.
        path = ("document", "nodes", index, "config", "execution")
        execution = config["execution"]
        if node["type"] in _POOLS:
            diagnostics.append(SkeletonDiagnostic("unknown_field", path, "pools do not have execution IO"))
            continue
        if type(execution) is not dict:
            diagnostics.append(SkeletonDiagnostic("invalid_type", path, "execution config must be an object"))
            continue
        sections = ("outputs",) if node["type"] == "start" else ("inputs",) if node["type"] in _BRANCHES else ("inputs", "outputs")
        for section in ("inputs", "outputs"):
            if section not in sections:
                if section in execution:
                    diagnostics.append(SkeletonDiagnostic("unknown_field", path + (section,),
                                                          "node does not support this IO section"))
                continue
            if section not in execution:
                diagnostics.append(SkeletonDiagnostic("missing_field", path + (section,), "required IO section is missing"))
                continue
            validator = _validate_bindings if section == "inputs" else _validate_outputs
            rows = _rows(execution[section], path + (section,), validator, diagnostics)
            for row_index, row in rows:
                row_path = path + (section, row_index)
                if section == "inputs":
                    references(row["value"], row_path + ("value",), node["id"])
                elif node["type"] == "end" and row["source"]["kind"] != "input":
                    diagnostics.append(SkeletonDiagnostic("invalid_value", row_path + ("source", "kind"),
                                                          "end outputs may only use inputs"))
                elif row["source"]["kind"] == "input":
                    problem = surfaces.output_problem(row["source"], node["id"])
                    if problem is not None:
                        diagnostics.append(SkeletonDiagnostic("missing_value", row_path + ("source", "path", 0),
                                                              problem))
        if node["type"] in _BRANCHES:
            diagnostics.extend(analyze_branch_execution(node, surfaces, path).diagnostics)
    return tuple(diagnostics)
