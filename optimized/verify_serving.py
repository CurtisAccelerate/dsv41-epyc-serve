"""Require serving admission and committed optimized worker receipts."""
import argparse, json, pathlib, urllib.request
def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:8095')
    a=p.parse_args()
    states=list(pathlib.Path('/w9/state').glob('dsv41-*/status.json'))
    if not states:raise RuntimeError('No serving receipt; model has not qualified')
    path=max(states,key=lambda p:p.stat().st_mtime);state=json.loads(path.read_text())
    if state.get('phase')!='serving' or not state.get('public_admission') or state.get('poisoned'):
        raise RuntimeError('Optimized serving is closed: '+json.dumps(state))
    production=json.loads((path.parent/'optimized-serving.json').read_text())
    expected='next-attention-sampler-native-mixed'
    rows=production.get('committed',[])
    if (not production.get('passed') or len(rows)!=4
        or sorted(r.get('rank',-1) for r in rows)!=list(range(4))
        or any(r.get('bank')!=expected or not r.get('production_enabled') for r in rows)):
        raise RuntimeError('Four committed optimized graph bank receipts required')
    with urllib.request.urlopen(a.url+'/v1/models',timeout=10) as r:models=json.load(r)
    request=urllib.request.Request(a.url+'/v1/chat/completions',
        data=json.dumps(dict(model=models['data'][0]['id'],messages=[dict(role='user',content='Reply with the number 2.')],
                             max_tokens=16,temperature=0)).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=120) as r:answer=json.load(r)
    if not answer.get('choices') or answer.get('usage',{}).get('completion_tokens',0)<=0:
        raise RuntimeError('No generated response')
    print('DSV41_SERVING_VERIFIED '+json.dumps(dict(passed=True,phase=state['phase'],
        public_admission=True,bank=expected,committed_ranks=4,response=answer,receipt=str(path))),flush=True)
if __name__=='__main__':main()
