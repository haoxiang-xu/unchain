"""Independent byte/chain checks; does not import the compression implementation."""
from pathlib import Path
import hashlib,json,collections
ROOT=Path(__file__).resolve().parent

def digest(value):
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()

def chain(messages,ids):
    rows=[]; pending=collections.defaultdict(collections.deque)
    for message,source in zip(messages,ids):
        calls=[];results=[]
        for block in message.get('content',[]) if isinstance(message.get('content'),list) else []:
            if block.get('type')=='tool_use':calls.append(block['id'])
            if block.get('type')=='tool_result':results.append(block['tool_use_id'])
        if message.get('type')=='mcp_approval_request':calls.append(message['id'])
        if message.get('type')=='mcp_approval_response':results.append(message['approval_request_id'])
        for call in sorted(set(calls)):
            item={'call_id':call,'call_source_ids':[source],'result_source_ids':[]}
            rows.append(item);pending[call].append(item)
        for result in sorted(set(results)):
            if pending[result]:pending[result].popleft()['result_source_ids'].append(source)
            else:rows.append({'call_id':result,'call_source_ids':[],'result_source_ids':[source]})
    return rows

def review(attempt):
    rows=[]
    for cell in sorted((ROOT/attempt).glob('*summary*')):
        if not list((cell/'archives').glob('*.json')): continue
        canonical_file=cell/'canonical-history.json'
        original=json.loads((canonical_file if canonical_file.exists() else cell/'seed-history.json').read_text())
        original=[m for m in original if m.get('role') not in ('system','developer')]
        for p in sorted((cell/'archives').glob('*.json')):
            a=json.loads(p.read_text());body={k:v for k,v in a.items() if k!='content_sha256'}
            citations=[ref for section in ('goal','constraints','decisions','progress','open_items','next_steps') for item in a['summary'][section] for ref in item['source_ids']]
            expected_chain=chain(a['messages'],a['source_ids'])
            checks={'exact_canonical_prefix':a['messages']==original[:len(a['messages'])],
                    'source_digest':digest(a['messages'])==a['source_sha256'],
                    'content_digest':digest(body)==a['content_sha256'],
                    'independent_tool_chain':expected_chain==a['tool_chain'],
                    'no_orphan_exchanges':all(x['call_source_ids'] and x['result_source_ids'] for x in expected_chain),
                    'source_ids_exist':all(ref in a['source_ids'] for ref in citations)}
            rows.append({'cell':cell.name,'archive':p.name,'messages':len(a['messages']),
                         'exchanges':len(expected_chain),'checks':checks,'all_pass':all(checks.values())})
    return {'attempt':attempt,'scope':'Exact source preservation and structural references only; not a semantic grounding verdict.',
            'archives':rows,'all_pass':all(row['all_pass'] for row in rows)}

if __name__=='__main__':
    reports=[review(p.name) for p in sorted(ROOT.glob('attempt-*')) if p.is_dir()]
    (ROOT/'independent-archive-review.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
    for r in reports:print(r['attempt'],len(r['archives']),r['all_pass'])
