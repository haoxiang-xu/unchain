"""Provider-free branch diagnostics and conservative missing-input admission."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from .conditions import _evaluate_condition, _validate_condition, _validate_operand
from .diagnostics import SkeletonDiagnostic
from .errors import SkeletonValidationError

if TYPE_CHECKING:
    from .static_io import StaticIOSurfaces

_MISSING = "known_missing"
_OTHER_ERROR = "other_error"


@dataclass(frozen=True)
class BranchAdmission:
    diagnostics: tuple[SkeletonDiagnostic, ...]
    rejection: SkeletonDiagnostic | None = None


def _missing_operand(operand: dict[str, Any], names: set[str] | None,
                     path: tuple[str | int, ...]) -> SkeletonDiagnostic | None:
    if operand["kind"] != "input" or not operand["path"]:
        return None
    root = operand["path"][0]
    # Resolved inputs are always an object, including an undeclared draft surface.
    if type(root) is int or names is not None and root not in names:
        return SkeletonDiagnostic("missing_value", path + ("path",),
                                  "condition input is unavailable")
    return None


def _condition_outcomes(condition: dict[str, Any], names: set[str] | None,
                        path: tuple[str | int, ...], diagnostics: list[SkeletonDiagnostic]
                        ) -> set[bool | str]:
    kind = condition["kind"]
    if kind in {"all", "any"}:
        continuing = kind == "all"
        outcomes: set[bool | str] = {continuing}
        for index, child in enumerate(condition["conditions"]):
            if continuing not in outcomes:
                break
            child_outcomes = _condition_outcomes(child, names, path + ("conditions", index), diagnostics)
            outcomes = (outcomes - {continuing}) | child_outcomes
        return outcomes

    operands = [(side, condition[side]) for side in ("left", "right") if side in condition]
    missing = [problem for side, operand in operands
               if (problem := _missing_operand(operand, names, path + (side,))) is not None]
    if missing:
        # Both operands are read before comparison; missing precedes null/type checks.
        diagnostics.extend(missing)
        return {_MISSING}
    if all(operand["kind"] == "literal" for _, operand in operands):
        try:
            return {_evaluate_condition(condition, {}, path)}
        except SkeletonValidationError:
            # Existing literal null/type errors stay runtime errors, never new gates.
            return {_OTHER_ERROR}
    if "right" in condition and any(operand["kind"] == "literal" and operand["value"] is None
                                    for _, operand in operands):
        # Missing or null may win here, but neither can continue a condition group.
        return {_OTHER_ERROR}
    operator = condition["operator"]
    if operator in {"gt", "gte", "lt", "lte"} and any(
            operand["kind"] == "literal" and type(operand["value"]) not in {int, float}
            for _, operand in operands):
        return {_OTHER_ERROR}  # Invalid literal types cannot continue a group.
    if operator == "contains" and condition["left"]["kind"] == "literal" and type(
            condition["left"]["value"]) not in {str, list}:
        return {_OTHER_ERROR}
    if condition["operator"] in {"is_null", "is_not_null"}:
        parts = condition["left"]["path"]
        if not parts:
            return {condition["operator"] == "is_not_null"}  # The whole input object.
        if len(parts) == 1 and names is not None and parts[0] in names:
            return {True, False}  # A resolved root binding is present, including null.
    # Do not infer payload types, deeper JSON leaves or correlations between reads.
    return {True, False, _OTHER_ERROR}


def analyze_branch_execution(node: dict[str, Any], surfaces: StaticIOSurfaces,
                             path: tuple[str | int, ...]) -> BranchAdmission:
    """Diagnose reachable missing roots; reject only unavoidable known missing."""
    execution = node["config"]["execution"]
    field = "condition" if node["type"] == "if_else" else "value"
    if field not in execution:
        return BranchAdmission((SkeletonDiagnostic("missing_field", path + (field,),
                                                  "required branch expression is missing"),))
    value = execution[field]
    value_path = path + (field,)
    try:
        # Validate every AST child, even behind a provably decisive short circuit.
        if field == "condition":
            _validate_condition(value, value_path)
        else:
            _validate_operand(value, value_path)
    except SkeletonValidationError as error:
        return BranchAdmission((SkeletonDiagnostic(error.code, error.path, str(error)),))

    names = surfaces.names(node["id"], "inputs")
    if field == "value":
        problem = _missing_operand(value, names, value_path)
        return BranchAdmission((problem,), problem) if problem is not None else BranchAdmission(())
    diagnostics: list[SkeletonDiagnostic] = []
    outcomes = _condition_outcomes(value, names, value_path, diagnostics)
    rejection = diagnostics[0] if outcomes == {_MISSING} else None
    return BranchAdmission(tuple(diagnostics), rejection)
