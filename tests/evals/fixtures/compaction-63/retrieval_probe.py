"""One separate exact-evidence recovery probe, not a retry of a failed cell."""
from pathlib import Path
import json,copy
import live_eval as e
from unchain.context import CompressionResult,DirectoryCompressionStore

ROOT=e.ROOT
plan={'purpose':'Exercise real-model archive recovery of exact old evidence absent from the summary text.',
      'input':'Latest accepted coding view from the fixed candidate; retain the canonical-source worker configuration.',
      'request':'Recover original intake module and complete original failing-test stderr, actual original read/check call IDs, and original hash. Save a recovery receipt; do not edit or rerun original task.',
      'oracle':'Exact equality to original seed tool results, actual archived call IDs, and at least one real context_archive_read call.',
      'max_iterations':8,'repetitions':1,'semantic_retries':0,
      'conclusion_boundary':'This checks scoped archive recovery only; cannot repair or qualify a failed main cell.'}
e.write_json(ROOT/'retrieval-probe-plan.json',plan)

def main():
    out=ROOT/'retrieval-probe'
    out.mkdir()
    cell=ROOT/'attempt-1/coding_handoff-summary'
    stages=sorted(cell.glob('compression-*.json'))
    assert stages
    stage=int(stages[-1].stem.rsplit('-',1)[1])
    record=json.loads(stages[-1].read_text())
    view=json.loads((cell/f'view-{stage}.json').read_text())
    store=DirectoryCompressionStore(cell/'archives')
    result=CompressionResult(messages=tuple(view),mode='summary',compacted=True,
                             estimated_tokens=record['estimated_input_tokens'],archive_id=record['archive_id'])
    underlying=result.read_tool(store)
    seed=json.loads((cell/'seed-history.json').read_text())
    originals={b['tool_use_id']:json.loads(b['content']) for b in seed[3]['content']}
    expected={'original_code':originals['setup-coding_handoff-0']['content'],
              'original_test_stderr':originals['setup-coding_handoff-3']['stderr'],
              'original_module_sha256':originals['setup-coding_handoff-3']['module_sha256'],
              'read_call_id':'setup-coding_handoff-0','check_call_id':'setup-coding_handoff-3'}
    # Ensure the full exact requested old text is absent from visible text (JSON-escaped or raw).
    assert expected['original_test_stderr'] not in e.canonical(view)
    assert e.canonical(expected['original_test_stderr'])[1:-1] not in e.canonical(view)
    e.write_json(out/'oracle.json',expected)
    transport=e.LiveTransport(out); transport.label='exact_evidence_recovery'
    audit=[]
    def context_archive_read(source_id: str='',offset: int=0,limit: int=2048,call_id: str='',tool_name: str='',occurrence: int=0) -> dict:
        """Read original source/catalog or exact call_id/tool_name + zero-based occurrence; Unicode pages, limit <=4096. Follow next_offset."""
        match=next((json.loads(args) for (label,name,args),queue in transport.pending.items()
                    if label==transport.label and name=='context_archive_read' and queue
                    and json.loads(args).get('source_id','')==source_id
                    and json.loads(args).get('offset',0)==offset
                    and json.loads(args).get('limit',2048)==limit
                    and json.loads(args).get('call_id','')==call_id
                    and json.loads(args).get('tool_name','')==tool_name
                    and json.loads(args).get('occurrence',0)==occurrence),None)
        if match is None:raise RuntimeError('unscheduled retrieval')
        read_call_id=transport.consume('context_archive_read',match)
        response=underlying(source_id,offset,limit,call_id=call_id,tool_name=tool_name,occurrence=occurrence)
        audit.append({'name':'context_archive_read','arguments':match,'call_id':read_call_id,
                      'response':response})
        e.write_json(out/'tool-audit.json',audit)
        return response
    def save_recovered_evidence(payload_json: str) -> dict:
        """Save recovered intake evidence: original_code, original_test_stderr, original_module_sha256, read_call_id, check_call_id. Exact original strings are required."""
        call_id=transport.consume('save_recovered_evidence',{'payload_json':payload_json})
        value=json.loads(payload_json)
        failures=[key for key,item in expected.items() if value.get(key)!=item]
        if not any(a['name']=='context_archive_read' for a in audit):failures.append('no_actual_archive_read')
        audit.append({'name':'save_recovered_evidence','call_id':call_id,'failures':failures})
        e.write_json(out/'tool-audit.json',audit)
        if not failures:e.write_json(out/'recovered-evidence.json',value)
        return {'saved':not failures,'mismatched_fields':failures}
    prompt='''Separate archival audit. Recover the EXACT ORIGINAL intake reconcile.py source and the COMPLETE ORIGINAL failing run_checks stderr (including newlines), before any code edit, plus the original module_sha256 and actual original read/check tool call IDs. These originals are in the archive, and the old full stderr is not in the current summary. Use context_archive_read to inspect the original JSON and tool-call/result links; do not infer or recreate the stderr by rerunning tests. No selector reads the catalog; call_id or tool_name plus zero-based chronological occurrence reads an exact exchange; source_id with offset/limit reads original JSON text. For a large source, use total_chars and offsets to inspect the beginning and tail as needed. Save the five fields with save_recovered_evidence. Do not modify code, run checks, publish, or repeat the original delivery task.'''
    inputs=copy.deepcopy(view)+[{'role':'user','content':prompt}]
    e.write_json(out/'input.json',inputs)
    status={'status':'running','source_stage':stage,'archive_id':record['archive_id'],'plan':plan}
    try:
        run=e.worker(transport,[context_archive_read,save_recovered_evidence]).run(inputs,
            max_iterations=8,max_context_window_tokens=200000,
            tool_runtime_config={'tool_result_budget':{'max_result_chars':1000000,'max_batch_chars':2000000,'min_chars_to_budget':1000000}})
        e.write_json(out/'output.json',{'status':run.status,'messages':run.messages})
        saved=out/'recovered-evidence.json'
        passed=saved.exists() and json.loads(saved.read_text())==expected and run.status in ('complete','completed')
        status.update(status='passed' if passed else 'failed',runtime_status=run.status,
                      archive_reads=sum(a['name']=='context_archive_read' for a in audit))
    except Exception as error:
        status.update(status='failed',error=str(error))
    status.update(model_calls=len(transport.calls),list_price_cost_usd=transport.total_cost)
    e.write_json(out/'report.json',status)
    print(json.dumps(status,ensure_ascii=False))
    return 0 if status['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
