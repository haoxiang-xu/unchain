import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace
import pytest
from unchain.skeleton import WorkflowExecutionError, WorkflowNodeCompleted, prepare_workflow, run_workflow

def doc(): return json.loads((Path(__file__).parent / "fixtures" / "skeleton_execution" / "if_workflow.json").read_text())
def switch_doc(): return json.loads((Path(__file__).parent / "fixtures" / "skeleton_execution" / "switch_workflow.json").read_text())
def rejoin_doc(): return json.loads((Path(__file__).parent / "fixtures" / "skeleton_execution" / "if_rejoin_workflow.json").read_text())

def test_runs_one_if_path_and_detaches_handler_result():
    calls=[]; shared={"answer":{"items":["yes"]}}
    def code(context, inputs): calls.append(context.node_id); return WorkflowNodeCompleted(shared)
    result=run_workflow(prepare_workflow(doc()), {"flag":True}, {"code":code})
    assert (result.outputs, result.completed_node_ids, result.selected_edge_ids)==({"answer":{"items":["yes"]}},("start","if","left","end"),("a","b","d"))
    shared["answer"]["items"].append("changed")
    assert result.outputs["answer"]["items"] == ["yes"] and calls == ["left"]

def test_false_path_and_runner_failures_are_closed():
    with pytest.raises(WorkflowExecutionError) as error: run_workflow(prepare_workflow(doc()), {"flag":False}, {})
    assert error.value.code == "missing_value"
    with pytest.raises(WorkflowExecutionError) as error: run_workflow(prepare_workflow(doc()), [], {"code":lambda *_: WorkflowNodeCompleted({})})
    assert (error.value.code,error.value.path)==("invalid_type",("inputs",))
    with pytest.raises(WorkflowExecutionError) as error: run_workflow(prepare_workflow(doc()), {"flag":True}, {"code":lambda *_: WorkflowNodeCompleted({},"bad")})
    assert error.value.code == "invalid_value"

def test_preflights_unselected_execution_config_before_handler():
    source=doc(); source["nodes"][3]["config"]["execution"]={"inputs":[]}
    called=[]
    with pytest.raises(WorkflowExecutionError): run_workflow(prepare_workflow(source), {"flag":True}, {"code":lambda *_: called.append(1)})
    assert called == []

def test_preflights_hostile_cp1_fields_and_end_result_before_handlers():
    for field, value, path in (("name", [], ("name",)), ("missing", {}, ("missing",))):
        source=doc(); source["nodes"][3]["config"]["execution"]["inputs"]=[{"name":"x","value":{"kind":"literal","value":1},"missing":"null"}]
        source["nodes"][3]["config"]["execution"]["inputs"][0][field]=value
        called=[]
        with pytest.raises(WorkflowExecutionError) as error: run_workflow(prepare_workflow(source), {"flag":True}, {"code":lambda *_: called.append(1)})
        assert error.value.path[-1:] == path and called == []
    source=doc(); source["nodes"][4]["config"]["execution"]["outputs"][0]["source"]["kind"]="result"
    with pytest.raises(WorkflowExecutionError) as error: run_workflow(prepare_workflow(source), {"flag":True}, {"code":lambda *_: WorkflowNodeCompleted({"answer":1})})
    assert error.value.code == "invalid_value" and error.value.path[-1] == "kind"

def test_switch_outlets_are_exact_and_preflight_before_handlers():
    source=doc(); node=source["nodes"][1]; node["type"]="switch_case"
    node["config"]["execution"]={"inputs":[],"value":{"kind":"literal","value":1},"cases":[{"id":"one","value":1}]}
    source["edges"][1]["source_port_id"]="one"; source["edges"][2]["source_port_id"]="default"
    result=run_workflow(prepare_workflow(source), {"flag":False}, {"code":lambda *_: WorkflowNodeCompleted({"answer":"one"})})
    assert result.completed_node_ids == ("start","if","left","end")
    source["edges"].append({"id":"extra","kind":"flow","source_node_id":"if","source_port_id":"extra","target_node_id":"right","target_port_id":"in"}); called=[]
    with pytest.raises(WorkflowExecutionError): run_workflow(prepare_workflow(source), {"flag":False}, {"code":lambda *_: called.append(1)})
    assert called == []

def test_switch_fixture_selects_declared_case_and_default():
    source=switch_doc(); calls=[]
    def code(context, _): calls.append(context.node_id); return WorkflowNodeCompleted({"value":context.node_id})
    assert run_workflow(prepare_workflow(source), {}, {"code":code}).outputs == {"value":"case"}
    source["nodes"][1]["config"]["execution"]["value"]={"kind":"literal","value":"other"}
    assert run_workflow(prepare_workflow(source), {}, {"code":code}).outputs == {"value":"fallback"}
    assert calls == ["case","fallback"]

def test_false_rejoin_and_first_present_null_and_missing_default():
    calls=[]
    def code(context, _): calls.append(context.node_id); return WorkflowNodeCompleted({"answer":None if context.node_id=="left" else "right"})
    assert run_workflow(prepare_workflow(doc()), {"flag":False}, {"code":code}).outputs == {"answer":"right"}
    assert run_workflow(prepare_workflow(doc()), {"flag":True}, {"code":code}).outputs == {"answer":None}
    assert calls == ["right","left"]
    source=doc(); source["nodes"][4]["config"]["execution"]["inputs"]=[{"name":"answer","value":{"kind":"first_present","values":[{"kind":"ref","node_id":"missing","path":[]},{"kind":"ref","node_id":"also-missing","path":[]}]},"missing":"default","default":"fallback"}]
    assert run_workflow(prepare_workflow(source), {"flag":True}, {"code":code}).outputs == {"answer":"fallback"}

def test_explicit_code_rejoin_executes_once_per_selected_branch():
    source=rejoin_doc()
    calls=[]
    def code(context,_): calls.append(context.node_id); return WorkflowNodeCompleted({"answer":context.node_id})
    true=run_workflow(prepare_workflow(source),{"flag":True},{"code":code})
    false=run_workflow(prepare_workflow(source),{"flag":False},{"code":code})
    assert true.completed_node_ids == ("start","if","left","join","end")
    assert false.completed_node_ids == ("start","if","right","join","end")
    assert calls == ["left","join","right","join"]

def test_preflight_path_and_checked_plan_block_handlers():
    source=doc(); source["nodes"][3]["config"]["execution"]["outputs"][0]["source"]["path"]=[True]
    called=[]
    with pytest.raises(WorkflowExecutionError) as error: run_workflow(prepare_workflow(source),{"flag":True},{"code":lambda *_:called.append(1)})
    assert error.value.code == "invalid_path" and called == []
    plan=prepare_workflow(doc()); forged=replace(plan,node_ids=())
    with pytest.raises(WorkflowExecutionError) as error: run_workflow(forged,{"flag":True},{"code":lambda *_:called.append(1)})
    assert (error.value.code,error.value.path)==("invalid_plan",("plan",)) and called == []

def test_handler_exception_noncompletion_nonjson_and_port_fail_closed():
    plan=prepare_workflow(doc())
    for handler, code in ((lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),"handler_error"),(lambda *_: {"answer":"no"},"invalid_type"),(lambda *_: WorkflowNodeCompleted({"answer":object()}),"invalid_json"),(lambda *_: WorkflowNodeCompleted({"answer":"no"},"side"),"invalid_value")):
        with pytest.raises(WorkflowExecutionError) as error: run_workflow(plan,{"flag":True},{"code":handler})
        assert error.value.code == code

def test_runtime_binding_errors_identify_the_failing_node_config():
    failures=[]
    for node_id in ("left","join"):
        source=rejoin_doc()
        node_index=next(index for index,node in enumerate(source["nodes"]) if node["id"]==node_id)
        source["nodes"][node_index]["config"]["execution"]["inputs"]=[{"name":"x","value":{"kind":"ref","node_id":"start","path":["absent"]},"missing":"error"}]
        with pytest.raises(WorkflowExecutionError) as error:
            run_workflow(prepare_workflow(source),{"flag":True},{"code":lambda *_:WorkflowNodeCompleted({"answer":"ok"})})
        expected=("document","nodes",node_index,"config","execution","inputs",0,"value")
        assert (error.value.code,error.value.path)==("missing_value",expected)
        current={"document":source}
        for part in error.value.path: current=current[part]
        assert current["node_id"]=="start"
        failures.append(error.value.path)
    assert failures[0] != failures[1]

def test_switch_errors_use_resolvable_numeric_document_paths():
    source=switch_doc(); source["nodes"][1]["config"]["execution"]["cases"][0]["value"]=[]
    with pytest.raises(WorkflowExecutionError) as error:
        run_workflow(prepare_workflow(source),{}, {"code":lambda *_:WorkflowNodeCompleted({"value":"ok"})})
    assert (error.value.code,error.value.path)==("invalid_type",("document","nodes",1,"config","execution","cases",0,"value"))
    current={"document":source}
    for part in error.value.path: current=current[part]
    assert current == []

    source=switch_doc(); source["nodes"][1]["config"]["execution"]["cases"].append({"id":"new","value":1})
    with pytest.raises(WorkflowExecutionError) as error:
        run_workflow(prepare_workflow(source),{}, {"code":lambda *_:WorkflowNodeCompleted({"value":"ok"})})
    assert (error.value.code,error.value.path)==("invalid_value",("document","nodes",1,"config","execution","cases"))
    current={"document":source}
    for part in error.value.path: current=current[part]
    assert current[-1] == {"id":"new","value":1}

def test_handler_failures_use_numeric_runtime_node_paths():
    plan=prepare_workflow(doc())
    cases=(
        ({},("execution","nodes",3,"handler")),
        ({"code":lambda *_:(_ for _ in ()).throw(RuntimeError("boom"))},("execution","nodes",2,"handler")),
        ({"code":lambda *_:{"answer":"no"}},("execution","nodes",2,"result")),
        ({"code":lambda *_:WorkflowNodeCompleted({"answer":object()})},("execution","nodes",2,"result","answer")),
        ({"code":lambda *_:WorkflowNodeCompleted({"answer":"no"},"side")},("execution","nodes",2,"selected_port")),
    )
    for handlers,expected in cases:
        with pytest.raises(WorkflowExecutionError) as error:
            run_workflow(plan,{"flag":False if not handlers else True},handlers)
        assert error.value.path == expected

def test_runtime_output_condition_and_switch_value_errors_map_to_config():
    cases=[]
    source=doc(); source["nodes"][2]["config"]["execution"]["outputs"][0]["source"]["path"]=["absent"]
    cases.append((source,{"flag":True},{"code":lambda *_:WorkflowNodeCompleted({"answer":"ok"})},("document","nodes",2,"config","execution","outputs",0,"source","path")))

    source=doc(); source["nodes"][1]["config"]["execution"]["condition"]["left"]["path"]=["absent"]
    cases.append((source,{"flag":True},{"code":lambda *_:WorkflowNodeCompleted({"answer":"ok"})},("document","nodes",1,"config","execution","condition","left","path")))

    source=switch_doc(); source["nodes"][1]["config"]["execution"]["value"]={"kind":"input","path":["absent"]}
    cases.append((source,{}, {"code":lambda *_:WorkflowNodeCompleted({"value":"ok"})},("document","nodes",1,"config","execution","value","path")))

    for source,inputs,handlers,expected in cases:
        with pytest.raises(WorkflowExecutionError) as error:
            run_workflow(prepare_workflow(source),inputs,handlers)
        assert error.value.path == expected
        current={"document":source}
        for part in error.value.path: current=current[part]

def test_derived_data_limit_errors_identify_the_current_runtime_node():
    large=[0]*55000

    source=rejoin_doc()
    source["nodes"][2]["config"]["execution"]["outputs"]=[{"name":name,"source":{"kind":"result","path":["answer"]}} for name in ("a","b")]
    calls=[]
    with pytest.raises(WorkflowExecutionError) as error:
        run_workflow(prepare_workflow(source),{"flag":True},{"code":lambda context,_:calls.append(context.node_id) or WorkflowNodeCompleted({"answer":large})})
    assert error.value.code == "limit_nodes"
    assert error.value.path[:5] == ("execution","nodes",2,"outputs","a")
    assert calls == ["left"]

    source=rejoin_doc()
    source["nodes"][4]["config"]["execution"]["inputs"]=[{"name":name,"value":{"kind":"ref","node_id":"left","path":["answer"]},"missing":"error"} for name in ("a","b")]
    calls=[]
    with pytest.raises(WorkflowExecutionError) as error:
        run_workflow(prepare_workflow(source),{"flag":True},{"code":lambda context,_:calls.append(context.node_id) or WorkflowNodeCompleted({"answer":large})})
    assert error.value.code == "limit_nodes"
    assert error.value.path[:5] == ("execution","nodes",4,"resolved_inputs","a")
    assert calls == ["left"]

    source=rejoin_doc(); calls=[]
    with pytest.raises(WorkflowExecutionError) as error:
        run_workflow(prepare_workflow(source),{"flag":True},{"code":lambda context,_:calls.append(context.node_id) or WorkflowNodeCompleted({"answer":large})})
    assert error.value.code == "limit_nodes"
    assert error.value.path[:6] == ("execution","nodes",5,"upstream_outputs","left","answer")
    assert calls == ["left","join"]

def test_root_input_limit_error_stays_at_the_public_input_argument():
    with pytest.raises(WorkflowExecutionError) as error:
        run_workflow(prepare_workflow(doc()),{"flag":True,"large":[0]*100001},{"code":lambda *_:WorkflowNodeCompleted({"answer":"unused"})})
    assert error.value.code == "limit_nodes"
    assert error.value.path[0] == "inputs"

def test_fresh_process_active_import_and_environment_guards():
    code='''import json, os, sys
blocked=("unchain.agent","unchain.kernel","unchain.providers","unchain.tools","unchain.journal","unchain.persistence","openai","anthropic","google","ollama")
blocked_hits=[]; probe_hits=[]; env_hits=[]
class Guard:
 def find_spec(self,name,path=None,target=None):
  if name=="skeleton_execution_probe": probe_hits.append(name); raise ImportError("probe")
  if any(name==root or name.startswith(root+".") for root in blocked): blocked_hits.append(name); raise ImportError("blocked "+name)
sys.meta_path.insert(0,Guard())
try: __import__("skeleton_execution_probe")
except ImportError: pass
assert probe_hits==["skeleton_execution_probe"]
keys={"OPENAI_API_KEY","ANTHROPIC_API_KEY","GEMINI_API_KEY","GOOGLE_API_KEY","OLLAMA_BASE_URL","HYPERSPACE_API_KEY","HYPERSPACE_BASE_URL"}
def normalize(k): return k.decode("utf-8") if type(k) is bytes else k
class Env(dict):
 def __init__(self,base): super().__init__(base); self.base=base
 def blocked(self,k,where):
  if normalize(k) in keys: env_hits.append((where,normalize(k))); raise AssertionError(where)
 def __getitem__(self,k):
  self.blocked(k,"[]")
  return self.base[k]
 def get(self,k,d=None):
  self.blocked(k,"get")
  return self.base.get(k,d)
 def __contains__(self,k):
  self.blocked(k,"contains")
  return k in self.base
old=os.environ; oldb=os.environb; os.environ=Env(old); os.environb=Env(oldb)
def getenv(k,d=None):
 if normalize(k) in keys: env_hits.append(("getenv",normalize(k))); raise AssertionError("getenv")
 return os.environ.get(k,d)
def getenvb(k,d=None):
 if normalize(k) in keys: env_hits.append(("getenvb",normalize(k))); raise AssertionError("getenvb")
 return os.environb.get(k,d)
os.getenv=getenv; os.getenvb=getenvb
for k in ("OPENAI_API_KEY","ANTHROPIC_API_KEY"):
 for read in (lambda k=k:os.environ[k],lambda k=k:os.environ.get(k),lambda k=k:k in os.environ,lambda k=k:os.getenv(k),lambda k=k:os.environb[k.encode()],lambda k=k:os.environb.get(k.encode()),lambda k=k:k.encode() in os.environb,lambda k=k:os.getenvb(k.encode())):
  try: read()
  except AssertionError: pass
  else: raise AssertionError("guard missed")
if len(env_hits) != 16: raise AssertionError("positive environment probes did not fire")
env_hits.clear()
from unchain.skeleton import WorkflowNodeCompleted,prepare_workflow,run_workflow
doc=json.load(open(sys.argv[1])); run_workflow(prepare_workflow(doc),{"flag":True},{"code":lambda *_:WorkflowNodeCompleted({"answer":"ok"})})
assert blocked_hits==[] and env_hits==[]
'''
    result=subprocess.run([sys.executable,"-c",code,str(Path(__file__).parent / "fixtures" / "skeleton_execution" / "if_workflow.json")],capture_output=True,text=True,check=False)
    assert result.returncode == 0, result.stderr
