"""An authored Skill identity must not become a name-priority lookup (#352)."""
from dataclasses import asdict

import pytest

from unchain.skeleton import SkeletonSkillRuntime, SkeletonValidationError, prepare_workflow
from unchain.skills.models import SkillIdentity
from unchain.skills.registry import SkillRegistry, SkillsConfig
from unchain.tools.models import SkillDescriptor
from tests.test_skeleton_tool_runtime import _runner, _skill_document


def _write(root, body):
    path = root / "notes" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: notes\ndescription: Selected notes\n---\n{body}\n", encoding="utf-8")
    return path.resolve()


def _registry(tmp_path):
    home, project = tmp_path / "home", tmp_path / "project"
    user = _write(home / ".unchain/skills", "USER CONTENT")
    local = _write(project / ".unchain/skills", "PROJECT CONTENT")
    registry = SkillRegistry(SkillsConfig(home=home, project_root=project))
    identity = SkillIdentity("user-unchain", str(user), "notes")
    return registry, identity, user, local


def test_saved_identity_reads_user_even_when_project_has_the_same_name(tmp_path):
    registry, identity, user, local = _registry(tmp_path)
    winner = registry.get("notes")
    assert winner.summary.path == local
    document = _skill_document(asdict(identity))
    runtime = SkeletonSkillRuntime(prepare_workflow(document), registry)
    loaded = runtime._read(asdict(identity))
    assert loaded.summary.identity == identity
    assert loaded.summary.path == user
    assert loaded.body == "USER CONTENT"
    assert loaded.summary.description == "Selected notes"
    # Exact reads do not change ordinary name/alias priority.
    assert registry.get("notes") == winner
    assert registry.resolve("notes") == winner.summary
    run = _runner(tmp_path, document, {"skill_read": runtime})
    run.create({})
    assert run.run().outputs == {"value": {"content": loaded.body, "name": "notes"}}


def test_exact_identity_never_falls_back_to_a_same_name_or_arbitrary_path(tmp_path):
    registry, identity, user, local = _registry(tmp_path)
    document = _skill_document(asdict(identity))
    runtime = SkeletonSkillRuntime(prepare_workflow(document), registry)
    # A valid file outside the configured roots is not a registry capability.
    foreign = _write(tmp_path / "foreign", "FOREIGN CONTENT")
    for wrong in (
        {**asdict(identity), "source_id": str(foreign)},
        {**asdict(identity), "source_id": str(local)},
        {**asdict(identity), "source": "project-unchain"},
        {**asdict(identity), "name": "Notes"},
    ):
        with pytest.raises(SkeletonValidationError) as caught:
            runtime._read(wrong)
        assert caught.value.code == "identity_mismatch"
    user.unlink()
    with pytest.raises(SkeletonValidationError) as caught:
        runtime._read(asdict(identity))
    assert caught.value.code == "identity_mismatch"
    assert registry.get("notes").summary.path == local


def test_exact_read_preserves_a_run_snapshot_but_new_run_reads_new_bytes(tmp_path):
    registry, identity, user, _ = _registry(tmp_path)
    document = _skill_document(asdict(identity))
    runtime = SkeletonSkillRuntime(prepare_workflow(document), registry)
    first = _runner(tmp_path, document, {"skill_read": runtime})
    first.create({})
    assert first.run().outputs["value"]["content"] == "USER CONTENT"
    user.write_text(user.read_text().replace("USER CONTENT", "UPDATED CONTENT"))
    cold = _runner(tmp_path, document, {"skill_read": runtime})
    assert cold.run().outputs["value"]["content"] == "USER CONTENT"
    fresh = _runner(tmp_path, document, {"skill_read": runtime}, graph="next")
    fresh.create({})
    assert fresh.run().outputs["value"]["content"] == "UPDATED CONTENT"


def test_exact_pack_read_keeps_descriptor_tools_and_metadata(tmp_path):
    descriptor = SkillDescriptor(name="notes", description="Pack notes", body="Use {tools}",
                                 source="pack", source_id="pack-1", tools=("read",), metadata={"version": "v1"})
    registry = SkillRegistry(SkillsConfig(include_project_dirs=False, include_user_dirs=False,
                                         extra_skills=(descriptor,)))
    identity = {"source": "pack", "source_id": "pack-1", "name": "notes"}
    runtime = SkeletonSkillRuntime(prepare_workflow(_skill_document(identity)), registry)
    loaded = runtime._read(identity)
    assert loaded.body == "Use `read`"
    assert loaded.tools == ("read",) and loaded.metadata == {"version": "v1"}
    loaded.metadata["version"] = "changed"
    assert runtime._read(identity).metadata == {"version": "v1"}


def test_ambiguous_pack_identity_is_not_chosen_by_registration_order(tmp_path):
    descriptors = tuple(SkillDescriptor(name="notes", description="Pack notes", body=body,
                                       source="pack", source_id="pack-1") for body in ("A", "B"))
    identity = {"source": "pack", "source_id": "pack-1", "name": "notes"}
    for entries in (descriptors, descriptors[::-1]):
        registry = SkillRegistry(SkillsConfig(include_project_dirs=False, include_user_dirs=False,
                                             extra_skills=entries))
        runtime = SkeletonSkillRuntime(prepare_workflow(_skill_document(identity)), registry)
        with pytest.raises(SkeletonValidationError) as caught:
            runtime._read(identity)
        assert caught.value.code == "identity_mismatch"
