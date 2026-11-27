from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import FrozenInstanceError
import pytest
from unchain.skeleton import SkeletonValidationError, prepare_workflow

FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_preparation"
def fixture(): return json.loads((FIXTURES / "workflow.json").read_text())

def test_prepares_document_order_edges_and_deterministic_topology():
    source = fixture()
    nested = {"layers": [{"value": "original"}]}
    source["nodes"][2]["config"]["nested"] = nested
    source["nodes"][3]["config"]["nested"] = nested
    plan = prepare_workflow(source)
    assert plan.node_ids == ("start", "if", "left", "right", "end")
    assert plan.topological_node_ids == plan.node_ids
    assert tuple(edge.id for edge in plan.flow_edges) == ("one", "two", "three", "four", "five")
    source["nodes"][2]["config"]["nested"]["layers"][0]["value"] = "source-mutated"
    first, second = plan.snapshot(), plan.snapshot()
    first["nodes"][2]["config"]["nested"]["layers"][0]["value"] = "snapshot-mutated"
    assert first["nodes"][3]["config"]["nested"]["layers"][0]["value"] == "original"
    assert second["nodes"][2]["config"]["nested"]["layers"][0]["value"] == "original"
    assert second["nodes"][3]["config"]["nested"]["layers"][0]["value"] == "original"
    assert plan.snapshot()["nodes"][2]["config"]["nested"]["layers"][0]["value"] == "original"
    with pytest.raises(FrozenInstanceError): plan.document_id = "changed"
    with pytest.raises(FrozenInstanceError): plan.flow_edges[0].id = "changed"

def test_wrong_kind_and_existing_preflight_errors_pass_through():
    agent = {"schema_version":1,"kind":"agent","id":"agent","name":"Agent","nodes":[{"id":"s","type":"start","config":{}},{"id":"a","type":"agent","config":{}},{"id":"e","type":"end","config":{}}],"edges":[{"id":"one","kind":"flow","source_node_id":"s","source_port_id":"out","target_node_id":"a","target_port_id":"in"},{"id":"two","kind":"flow","source_node_id":"a","source_port_id":"out","target_node_id":"e","target_port_id":"in"}]}
    with pytest.raises(SkeletonValidationError) as error: prepare_workflow(agent)
    assert (error.value.code, error.value.path) == ("wrong_kind", ("document", "kind"))
    broken = fixture(); broken["edges"].pop()
    with pytest.raises(SkeletonValidationError) as error: prepare_workflow(broken)
    assert (error.value.code, error.value.path) == ("flow_output_required", ("document", "nodes", 3))
    malformed = fixture()
    malformed["schema_version"] = True
    with pytest.raises(SkeletonValidationError) as error: prepare_workflow(malformed)
    assert (error.value.code, error.value.path) == ("invalid_value", ("document", "schema_version"))

def test_linear_and_branch_edge_permutation_have_stable_topological_ties():
    linear = {"schema_version":1,"kind":"workflow","id":"linear","name":"Linear","nodes":[{"id":"s","type":"start","config":{}},{"id":"c","type":"code","config":{}},{"id":"e","type":"end","config":{}}],"edges":[{"id":"a","kind":"flow","source_node_id":"s","source_port_id":"out","target_node_id":"c","target_port_id":"in"},{"id":"b","kind":"flow","source_node_id":"c","source_port_id":"out","target_node_id":"e","target_port_id":"in"}]}
    assert prepare_workflow(linear).topological_node_ids == ("s", "c", "e")
    source = fixture(); reordered = fixture(); reordered["edges"] = list(reversed(reordered["edges"]))
    assert prepare_workflow(source).topological_node_ids == prepare_workflow(reordered).topological_node_ids == ("start", "if", "left", "right", "end")

def test_fresh_process_does_not_import_provider_or_agent_packages():
    code = '''import importlib, json, os, sys
blocked = ("unchain.agent", "unchain.agent.model_io", "unchain.providers", "openai", "anthropic", "google", "ollama", "httpx")
import_hits = 0
class Guard:
 def find_spec(self, name, path=None, target=None):
  global import_hits
  if name == "skeleton_import_guard_probe":
   import_hits += 1
   raise ImportError("guard fired")
  if any(name == root or name.startswith(root + ".") for root in blocked): raise ImportError("blocked import: " + name)
sys.meta_path.insert(0, Guard())
try: importlib.import_module("skeleton_import_guard_probe")
except ImportError: pass
else: raise AssertionError("import guard negative control failed")
if import_hits != 1: raise AssertionError("meta-path guard did not execute")
keys = {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_BASE_URL", "HYPERSPACE_API_KEY", "HYPERSPACE_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"}
hits = {"getitem": set(), "get": set(), "contains": set(), "getenv": set()}
orig = os.environ
class Env(dict):
 def __getitem__(self, key):
  if key in keys:
   hits["getitem"].add(key)
   raise AssertionError("credential read: " + key)
  return orig[key]
 def get(self, key, default=None):
  if key in keys:
   hits["get"].add(key)
   raise AssertionError("credential read: " + key)
  return orig.get(key, default)
 def __contains__(self, key):
  if key in keys:
   hits["contains"].add(key)
   raise AssertionError("credential membership: " + key)
  return key in orig
os.environ = Env(orig)
def getenv(key, default=None):
 if key in keys:
  hits["getenv"].add(key)
  raise AssertionError("credential getenv: " + key)
 return os.environ.get(key, default)
os.getenv = getenv
for key in keys:
 for read in (lambda key=key: os.environ.get(key), lambda key=key: os.environ[key], lambda key=key: key in os.environ, lambda key=key: os.getenv(key)):
  try: read()
  except AssertionError: pass
  else: raise AssertionError("environment guard negative control failed")
if any(seen != keys for seen in hits.values()): raise AssertionError("environment guard did not execute")
from unchain.skeleton import prepare_workflow
prepare_workflow(json.load(open(sys.argv[1])))
'''
    result = subprocess.run([sys.executable, "-c", code, str(FIXTURES / "workflow.json")], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
