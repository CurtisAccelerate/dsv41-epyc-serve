"""Validate assembled release metadata, launch command and controller rejection diagnostics."""
import ast,asyncio,copy,hashlib,importlib.util,json,pathlib,tempfile,types,os,re
from unittest.mock import patch

def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def census_branch_test():
    from hybrid import p2p4_journal
    source=pathlib.Path('/usr/local/lib/python3.12/dist-packages/hybrid/initial_state_plan.py').read_text()
    collect=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='collect')
    start=next(i for i,n in enumerate(collect.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='excluded' for t in n.targets))
    code=compile(ast.Module(body=collect.body[start:-1],type_ignores=[]),'<actual-census-transport-branch>','exec')
    dummy=object()
    seq=types.SimpleNamespace(device='cuda:0',data_ptr=lambda:100,numel=lambda:4,element_size=lambda:8)
    reg=dict(owner=types.SimpleNamespace(seq=seq,dev_base=200,seg_bytes=64))
    for p2p,host,accept in ((False,False,False),(False,True,True),(True,False,True),(True,True,False)):
        scope=dict(__package__='hybrid',CP=types.SimpleNamespace(REG=[reg] if host else []),runner=object())
        with patch.dict(os.environ,DSV41_TRANSPORT='p2p4' if p2p else 'legacy'),patch.object(p2p4_journal,'PeerGuard',lambda runner:dummy):
            try:exec(code,scope)
            except ValueError:assert not accept
            else:
                assert accept
                assert (scope['protocol_guard'] is dummy)==p2p
                assert bool(scope['excluded'])==host
    from hybrid.initial_state_plan import InitialJournal
    journal=InitialJournal.__new__(InitialJournal)
    def reject():raise RuntimeError('test-invalid-transport')
    journal.protocol_guard=types.SimpleNamespace(check=reject)
    for method in (journal.restore,journal.verify):
        try:method()
        except RuntimeError as e:assert str(e)=='test-invalid-transport'
        else:raise AssertionError('Journal continued with invalid transport')

async def controller_test():
    spec=importlib.util.spec_from_file_location('tested_control','/w9/resident-api/w9_resident_api/control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for mismatch in (False,True):
        with tempfile.TemporaryDirectory() as folder:
            c=module.Control.__new__(module.Control);c.root=pathlib.Path(folder);c.collective_profile='stock'
            c.poisoned=False
            async def qualify():pass
            async def quarantine(reason,poison):c.poisoned=poison
            async def rpc(name,args):
                method,token,profile=args
                return [dict(ok=True,rank=rank,result=dict(token=token,profile=profile,
                    phase=dict(prepare='prepared',apply='applied',commit='committed')[method],
                    counts={'61440':2 if mismatch and rank==0 else 1},
                    semantic=[dict(world_size=4,payloads=[[61440,'torch.bfloat16',122880,'torch.float32',1]])])) for rank in range(4)]
            c.qualify=qualify;c.quarantine=quarantine;c.rpc=rpc
            try:await c.collective('six')
            except RuntimeError as e:
                assert mismatch and 'counts differ' in str(e) and c.poisoned
            else:assert not mismatch and c.collective_profile=='six'
            records=list(c.root.glob('collective-*.json'));assert len(records)==1
            saved=json.loads(records[0].read_text());assert len(saved['steps'][0]['rows'])==4
            if mismatch:assert saved['steps'][0]['rows'][0]['result']['counts']!=saved['steps'][0]['rows'][1]['result']['counts']

def main():
    count=0
    for path in ('/opt/dsv41/manifest.json','/w26-preflight/manifest.json',
        '/usr/local/lib/python3.12/dist-packages/hybrid/attention-imports.json'):
        data=json.loads(pathlib.Path(path).read_text())
        for row in data.get('files',[])+data.get('data_files',[]):
            target=row.get('target',row.get('path'))
            if target and row.get('sha256'):
                assert digest(target)==row['sha256'],target
                count+=1
    command=json.loads(pathlib.Path('/opt/dsv41/serve-command.json').read_text())
    assert command[0]=='python3' and command[1].endswith('.py')
    assert '--disable-custom-all-reduce' in command
    from hybrid.transport_policy import validate_topology
    paired=[[j in (i,i^1) for j in range(4)] for i in range(4)]
    assert validate_topology(paired,'hostpair') and validate_topology([[True]*4 for _ in range(4)],'hostpair')
    trace=ast.parse(pathlib.Path('/usr/local/lib/python3.12/dist-packages/hybrid/dsv41_r1_trace.py').read_text())
    nodes=[n for n in trace.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NAME_CLASSES' for t in n.targets) or isinstance(n,ast.FunctionDef) and n.name=='classify_kernel']
    namespace=dict(re=re);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<trace-classifier>','exec'),namespace)
    assert namespace['classify_kernel']('pair_write_bf16')=='collectives'
    asyncio.run(controller_test())
    census_branch_test()
    print('P2P4_STARTUP_ASSEMBLY_PASSED '+json.dumps(dict(checked_file_references=count,
        python_preflight=True,count_mismatch_still_rejected=True,rejected_rank_maps_persisted=True)))

if __name__=='__main__':main()
