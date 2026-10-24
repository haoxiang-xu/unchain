"""Offline harness coverage only. Not real-model quality evidence."""
import collections,json
from pathlib import Path
import live_eval as e

class FakeTransport(e.LiveTransport):
    def __init__(self,path):
        super().__init__(path)
        self.steps=collections.Counter()
    def infer(self,system,prompt,kind):
        number=len(self.calls)+1
        self.calls.append({'kind':kind,'path':str(self.run_dir/'offline'),'status':'fake'})
        if kind=='summary':
            payload=json.loads(prompt)
            summary={'schema':'unchain.context_summary.v1',**{k:[] for k in ['goal','constraints','decisions','progress','open_items','next_steps']}}
            summary['goal']=[{'text':'Continue the staged fixture task.','source_ids':[payload['sources'][0]['source_id']]}]
            return json.dumps(summary),{},number
        payload=json.loads(prompt)
        names={t['name'] for t in payload['tools']}
        key=(self.label,tuple(sorted(names)))
        step=self.steps[key]; self.steps[key]+=1
        calls=[]
        incident='incident_handoff' in self.label
        if 'verify_hypothesis' in names and step==0:
            calls=[{'name':'verify_hypothesis','arguments':{'evidence_id':'EV-bridge-44','quoted_error':'CACHE_EPOCH_MISMATCH E42','observed_epoch':12}}]
        elif 'run_replay' in names and step==0:
            calls=[{'name':'run_replay','arguments':{'ttl_ms':2750,'verification_receipt':'VR-482-K6'}}]
        elif 'write_module' in names:
            corrected='run_checks' in names
            if step==0:
                code='def reconcile(entries):\n    last = {}\n    for i, e in enumerate(entries):\n        if e["status"] == "settled":\n            last[e["id"]] = (i, e["cents"])\n    ids = '+('sorted(last, key=lambda k: last[k][0])' if corrected else 'sorted(last)')+'\n    return {"total_cents": sum(x[1] for x in last.values()), "settled_ids": ids}\n'
                calls=[{'name':'write_module','arguments':{'code':code}}]
            elif corrected and step==1:
                calls=[{'name':'run_checks','arguments':{}}]
        elif step==0:
            work=self.run_dir/self.label/'workspace'
            if incident:
                ev={'evidence_id':'EV-bridge-44','request_id':'R-740','quoted_error':'CACHE_EPOCH_MISMATCH E42','observed_epoch':12,'proof_nonce':'PROOF-N7Q4M2-V8D1','root_cause':'stale retry cache'}
                value={'incident_id':'INC-482','root_cause':'stale retry cache','ttl_ms':2750,'evidence_id':'EV-bridge-44','evidence_sha256':e.sha(ev),'quoted_error':'CACHE_EPOCH_MISMATCH E42','proof_nonce':'PROOF-N7Q4M2-V8D1','evidence_tool_call_id':'setup-incident_handoff-1','verification_receipt':'VR-482-K6','replay_receipt':'RR-2750-V3','audit_token':'AUDIT-17-J6K9P4-R2W8','blocked_item':'P-release','publication_status':'denied'}
                calls=[{'name':'write_incident_report','arguments':{'report_json':json.dumps(value)}}]
            else:
                value={'spec_id':'SPEC-D7-Q9','ordering':'last_settled_occurrence','module_sha256':e.sha((work/'reconcile.py').read_bytes()),'audit_token':'AUDIT-17-J6K9P4-R2W8','blocked_item':'P-publish','publication_status':'denied'}
                calls=[{'name':'write_delivery_manifest','arguments':{'manifest_json':json.dumps(value)}}]
        return json.dumps({'message':'Offline control.','tool_calls':calls}),{},number

path=e.ROOT/'offline-preflight'
path.mkdir()
t=FakeTransport(path)
rows=[]
for case in ('incident_handoff','coding_handoff'):
    row=e.run_cell(case,'summary',t,path)
    rows.append(row)
    assert row['status']=='passed',row
    comp=[json.loads((path/f'{case}-summary'/f'compression-{i}.json').read_text()) for i in (1,2,3)]
    assert len({c['archive_id'] for c in comp})==3
    assert all(c['archive_integrity'] and c['cold_replay'] and c['model_calls']==1 for c in comp)
e.write_json(path/'preflight-report.json',{'offline_only':True,'rows':rows,'distinct_summary_calls':sum(c['kind']=='summary' for c in t.calls)})
print('OFFLINE PREFLIGHT PASSED: 2 cases x 3 distinct compression requests, real local tool execution and continuation.')
