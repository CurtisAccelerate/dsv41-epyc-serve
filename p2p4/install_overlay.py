"""Install the EPYC four-way transport into the pinned complete serving image."""
import ast,hashlib,json,pathlib,shutil,sys

BASE_IMAGE='sha256:e3bc13c775dee4013f5cfe28688482e0ec8ccb9b045ccb012c9f6e53b56c2726'
root=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else pathlib.Path('/')
here=pathlib.Path(__file__).resolve().parent
site=root/'usr/local/lib/python3.12/dist-packages'
changed={}
mode=__import__('os').environ.get('DSV41_TRANSPORT','p2p4')
if mode not in ('p2p4','hostpair'):raise ValueError('Unknown image transport')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
base_manifest=root/'opt/dsv41/manifest.json'
if sha(base_manifest)!='2bb8ed7693624245583ad6960d37e787edc7bbeb59be89f5eb48334fcf310de4':raise RuntimeError('Install only on the pinned complete base image')
for item in json.loads(base_manifest.read_text())['files']:
    if sha(root/item['path'].lstrip('/'))!=item['sha256']:raise RuntimeError('Base image source has drifted: '+item['path'])
def patch(path,old,new):
    text=path.read_text()
    if old not in text:raise RuntimeError('Pinned source patch does not match '+str(path))
    changed.setdefault(path,sha(path));path.write_text(text.replace(old,new))

module=site/'hybrid/p2p4_transport.py';shutil.copy2(here/'p2p4_transport.py',module)
for name in ('p2p4_provenance.py','p2p4_journal.py','transport_policy.py','collective_error.py','optimized_receipts.py'):
    shutil.copy2(here/name,site/'hybrid'/name)
cuda=site/'vllm/platforms/cuda.py'
patch(cuda,'if envs.VLLM_HOST_ALLREDUCE:',
      'if envs.VLLM_HOST_ALLREDUCE or __import__("os").environ.get("DSV41_TRANSPORT")=="p2p4":')
comm=site/'vllm/sm80/distributed/communicator.py'
patch(comm,'envs.VLLM_HOST_ALLREDUCE\n        or envs.VLLM_CUSTOM_ALLREDUCE_FLAGS',
      '(envs.VLLM_HOST_ALLREDUCE or os.environ.get("DSV41_TRANSPORT")=="p2p4")\n        or envs.VLLM_CUSTOM_ALLREDUCE_FLAGS')
patch(comm,'        if self.world_size > 1:\n            self._log_all_reduce_backend_selection()',
'''        self._p2p4 = os.environ.get('DSV41_TRANSPORT')=='p2p4' and unique_name.split(':')[0]=='tp'
        if self._p2p4:
            from hybrid import p2p4_transport as peer
            if self.world_size!=4 or self.hostshm_comm is not None:
                raise RuntimeError('Four-way P2P requires TP4 and no host communicator')
            peer.register(self.pynccl_comm)
        if self.world_size > 1:
            self._log_all_reduce_backend_selection()''')
patch(comm,'        enabled_ar_backends: list[str] = []',
'''        enabled_ar_backends: list[str] = []
        if getattr(self,'_p2p4',False):enabled_ar_backends.append('FOURWAY_P2P')''')
patch(comm,'    def all_reduce(self, input_):\n',
'''    def all_reduce(self, input_):
        if getattr(self,'_p2p4',False):
            from hybrid.p2p4_transport import reduce
            return reduce(input_)
''')
cp=root/'w9/integration/w9_exl_ab/collective_profiles.py'
patch(cp,'def invoke(worker,method,token,profile):\n global TX,PROFILE\n',
'''def invoke(worker,method,token,profile):
 global TX,PROFILE
 if os.environ.get('DSV41_TRANSPORT')=='p2p4':
  from hybrid.p2p4_transport import profile_invoke
  return profile_invoke(worker,method,token,profile,sys.modules[__name__])
''')
patch(cp,"def status():return dict(profile=PROFILE,pending=TX,registered=len(REG),patchable_nodes=0 if BANK is None else len(BANK))",
'''def status():
 result=dict(profile=PROFILE,pending=TX,registered=len(REG),patchable_nodes=0 if BANK is None else len(BANK))
 if os.environ.get('DSV41_TRANSPORT')=='p2p4':
  from hybrid.p2p4_transport import status as peer_status
  result.update(peer_status())
 return result''')
boot=site/'hybrid/boot_native_contract.py'
patch(boot,"        if c.PROFILE!='six' or c.TX is not None or not c.REG:raise Rejected('Expected committed six profile')",
'''        from . import p2p4_transport as peer
        if peer.enabled():
            peer.require()
            if c.REG:raise Rejected('Host registry is forbidden in P2P4 mode')
        if c.PROFILE!='six' or c.TX is not None or (not c.REG and not peer.enabled()):raise Rejected('Expected committed collective profile')''')
patch(boot,'    def collective_signature(self,graphs,*,baseline_raw):\n',
'''    def collective_signature(self,graphs,*,baseline_raw):
        from . import p2p4_transport as peer
        if peer.enabled():return peer.graph_signature(graphs)
''')
patch(boot,'    def check_communication_health(self):\n',
'''    def check_communication_health(self):
        from . import p2p4_transport as peer
        if peer.enabled():peer.health();return
''')
patch(boot,'        self.roots=(self.backend,tuple(CP.REG),self.entry,self.baseline_graphs)',
'''        self.roots=(self.backend,tuple(CP.REG),self.entry,self.baseline_graphs)
        from . import p2p4_transport as peer
        if peer.enabled():self.roots+=(peer.require(),)''')
r2=site/'hybrid/r2_contract.py'
patch(r2,'import hashlib\n','import hashlib\nimport os\n')
patch(r2,'    if value == PAIR:\n        return PAIR',"    if value == 'p2p4':return value\n    if value == PAIR:\n        return PAIR")
patch(r2,'self.collective if self.collective == PAIR else list(self.collective)',
      'self.collective if isinstance(self.collective,str) else list(self.collective)')
patch(r2,'default_collective(PAD_TO)',"('p2p4' if env.get('DSV41_TRANSPORT')=='p2p4' else default_collective(PAD_TO))")
patch(r2,'            coll = colls.get(served_b, default_collective(served_b))',
'''            coll = colls.get(served_b, 'p2p4' if env.get('DSV41_TRANSPORT')=='p2p4' else default_collective(served_b))
            if env.get('DSV41_TRANSPORT')=='p2p4' and coll!='p2p4':
                raise ValueError('P2P4 mode cannot select a host or pair collective')''')
bank=site/'hybrid/r2_bank.py'
patch(bank,'        if self.attention_state is not None:\n            st = self.attention_state',
'''        from . import p2p4_transport as peer
        if peer.enabled():out['collective']=peer.graph_receipt(self.entry['graph'],ar_bytes(self.plan.served_b))
        if self.attention_state is not None:
            st = self.attention_state''')
patch(bank,'self.plan.collective if self.plan.collective == PAIR else list(self.plan.collective)',
      'self.plan.collective if isinstance(self.plan.collective,str) else list(self.plan.collective)')
startup=site/'hybrid/r2_startup.py'
patch(startup,'                    if owner is None:\n                        raise ValueError("host all-reduce registry missing")',
'''                    from . import p2p4_transport as peer
                    if peer.enabled():
                        rec['p2p4_parity']=peer.collective_parity(worker,rows_of(b))
                        return True
                    if owner is None:
                        raise ValueError("host all-reduce registry missing")''')
# The old bitwise geometry check is inapplicable to a changed reduction order.
patch(startup,'profile=plan.record()["collective"], bitwise=ok)',
      'profile=plan.record()["collective"], passed=ok, bitwise=(None if os.environ.get("DSV41_TRANSPORT")=="p2p4" else ok))')
verify=root/'opt/dsv41/verify_serving.py'
patch(verify,"    production=json.loads((path.parent/'optimized-serving.json').read_text())",
'''    import os
    selected=os.environ.get('DSV41_TRANSPORT')
    for rank in state.get('ranks',[]):
        transport=rank.get('collective',{})
        if selected=='p2p4' and (transport.get('transport')!='p2p4' or transport.get('world_size')!=4 or transport.get('payload_host_bytes')!=0):
            raise RuntimeError('Four-way device-memory serving transport is not attested')
        if selected=='hostpair' and (transport.get('registered')!=1 or len(rank.get('pair',[]))!=1 or not rank['pair'][0].get('ready')):
            raise RuntimeError('Host/pair serving transport is not attested')
    if len(state.get('ranks',[]))!=4:raise RuntimeError('Four transport receipts required')
    production=json.loads((path.parent/'optimized-serving.json').read_text())''')
command=root/'opt/dsv41/serve-command.json';cmd=json.loads(command.read_text())
if cmd[0].endswith('.py'):cmd.insert(0,'python3')
if '--disable-custom-all-reduce' not in cmd:cmd.append('--disable-custom-all-reduce')
command.write_text(json.dumps(cmd,indent=2))
defaults=root/'opt/dsv41/defaults.json';env=json.loads(defaults.read_text())
env.update(DSV41_TRANSPORT=mode,VLLM_HOST_ALLREDUCE='1' if mode=='hostpair' else '0',W15_PAIR_TRANSPORT='1' if mode=='hostpair' else '0',
           DSV41_P2P_PAYLOAD_DTYPE='native',NCCL_SHM_DISABLE='0' if mode=='hostpair' else '1',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P',
           NCCL_DEBUG_FILE='/tmp/dsv41-nccl4.%p.log')
defaults.write_text(json.dumps(env,indent=2))
entry=root/'opt/dsv41/entrypoint.py'
patch(entry,"    args=sys.argv[1:];cmd=args.pop(0) if args else 'serve'",
'''    from hybrid.transport_policy import validate_runtime
    validate_runtime(os.environ)
    args=sys.argv[1:];cmd=args.pop(0) if args else 'serve' ''')
pair=root/'w9/integration/w9_exl_ab/pair_transport.py'
patch(pair,"   for rank,x in enumerate(identities):\n    if rank!=owner.rank:assert self.driver.can_access(self.identity['ordinal'],x['ordinal'])==(rank==(owner.rank^1))",
'''   from hybrid.transport_policy import validate_peer_row
   row=[rank==owner.rank or bool(self.driver.can_access(self.identity['ordinal'],x['ordinal'])) for rank,x in enumerate(identities)]
   validate_peer_row(owner.rank,row)''')
probe=root/'opt/dsv41/p2p_check.py'
patch(probe,"    if not all(all(row) for row in peers):raise RuntimeError('Driver does not permit P2P on every GPU pair')",
'''    from hybrid.transport_policy import validate_topology
    transport=os.environ.get('DSV41_TRANSPORT','p2p4')
    validate_topology(peers,transport)''')
patch(probe,"    if not lines or not all('via P2P' in log for log in rank_logs) or fallbacks:",
      "    if transport=='p2p4' and (not lines or not all('via P2P' in log for log in rank_logs) or fallbacks):")
patch(probe,'        allreduce_correct=True,nccl_p2p_connections=len(lines),log_directory=directory))',
      '        transport=transport,allreduce_correct=True,nccl_p2p_connections=len(lines),fallback_lines=len(fallbacks),log_directory=directory))')
trace=site/'hybrid/dsv41_r1_trace.py'
patch(trace,'r"cross_device|custom_ar", re.I))','r"cross_device|custom_ar|pair_write_bf16", re.I))')
builder=root/'w9/engram-loader/numa_builder.py'
patch(builder,'def numa_evidence(mm,nbytes,node):\n',
'''def numa_evidence(mm,nbytes,node):
    if len(mm)!=nbytes or nbytes<=0:raise ValueError('Exact payload mapping extent required')
    # numa_maps counts populated PTEs, not all tmpfs page-cache pages. Read one
    # byte per page before observing placement; never write or migrate data.
    prefault_pages=len(mm[::mmap.PAGESIZE])
''')
patch(builder,"'resident_pages':expected,'kernel_page_bytes':pagesize}",
      "'resident_pages':expected,'kernel_page_bytes':pagesize,'read_only_prefault_pages':prefault_pages}")
# Persist every returned rank row before any comparison can reject it.
control=root/'w9/resident-api/w9_resident_api/control.py'
patch(control,"record['steps'].append(dict(method=method,rows=rows))",
      "record['steps'].append(dict(method=method,rows=rows));self.save('collective-'+token+'.json',record)")
patch(control,"  if self.poisoned:raise RuntimeError('Measured serve refused: poisoned')",
'''  if self.poisoned:raise RuntimeError('Measured serve refused: poisoned')
  from hybrid.optimized_receipts import require_optimized
  require_optimized(json.loads((self.root/'attention-initial.json').read_text()))''')
plan=site/'hybrid/initial_state_plan.py'
patch(plan,"    if not excluded:raise ValueError('Missing explicit host collective protocol exclusion')",
'''    protocol_guard=None
    from . import p2p4_transport as peer
    if peer.enabled():
        from .p2p4_journal import PeerGuard
        protocol_guard=PeerGuard(runner)
        if excluded:raise ValueError('Unexpected host protocol state in P2P4 census')
    elif not excluded:raise ValueError('Missing explicit host collective protocol exclusion')''')
patch(plan,'qualified=False,forbidden_device_ranges=excluded,',
      'qualified=False,forbidden_device_ranges=excluded,protocol_guard=protocol_guard,')
patch(plan,"        self.torch=torch\n        self.gpu=ByteJournal",
'''        self.torch=torch
        self.protocol_guard=plan.get('protocol_guard')
        if self.protocol_guard is not None:self.protocol_guard.check()
        self.gpu=ByteJournal''')
patch(plan,'    def restore(self):\n        self.torch.cuda.synchronize()',
'''    def restore(self):
        if self.protocol_guard is not None:self.protocol_guard.check()
        self.torch.cuda.synchronize()''')
patch(plan,'    def verify(self):\n        bindings_ok=',
'''    def verify(self):
        if self.protocol_guard is not None:self.protocol_guard.check()
        bindings_ok=''')
# Repair the documented stale source pin BEFORE propagating edited identities.
attention=site/'hybrid/attention-imports.json';data=json.loads(attention.read_text())
changed.setdefault(attention,sha(attention))
for item in data['files']:
    if item['module']=='w9_resident_api.control':
        item['sha256']=changed[control]
attention.write_text(json.dumps(data,indent=2))
# Update content identities of edited source, including dependencies embedded in
# the component source manifests. Parameter ABI and compiled expert pins are unchanged.
texts=[]
for folder in (site/'hybrid',root/'w9',root/'w26-preflight'):
    texts += [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
transitions={old:sha(p) for p,old in changed.items() if sha(p)!=old}
import re
for wave in range(30):
    mapping=dict(transitions)
    for key in mapping:
        for _ in range(50):
            v=mapping[key]
            if v not in transitions:break
            mapping[key]=transitions[v]
    again=False
    for p in texts:
        text=p.read_text();new=re.sub(r'\b[0-9a-f]{64}\b',lambda m:mapping.get(m.group(),m.group()),text)
        if new!=text:
            old=sha(p);p.write_text(new);transitions[old]=sha(p);again=True
    if not again:break
else:raise RuntimeError('Source identity references did not converge')
preflight=root/'w26-preflight/manifest.json';d=json.loads(preflight.read_text())
for item in d['files']+d['data_files']:item['sha256']=sha(root/item['target'].lstrip('/'))
d['required_environment']['W15_PAIR_TRANSPORT']=env['W15_PAIR_TRANSPORT']
d['required_environment']['VLLM_HOST_ALLREDUCE']=env['VLLM_HOST_ALLREDUCE']
d['required_environment']['DSV41_TRANSPORT']=mode
preflight.write_text(json.dumps(d,indent=2))
manifest=root/'opt/dsv41/manifest.json';d=json.loads(manifest.read_text())
paths={row['path'] for row in d['files']};paths.add('/usr/local/lib/python3.12/dist-packages/hybrid/p2p4_transport.py')
for name in ('p2p4_provenance.py','p2p4_journal.py','transport_policy.py','collective_error.py','optimized_receipts.py'):
    paths.add('/usr/local/lib/python3.12/dist-packages/hybrid/'+name)
d['files']=[dict(path=p,sha256=sha(root/p.lstrip('/'))) for p in sorted(paths)]
manifest.write_text(json.dumps(d,indent=2))
for p in changed:
    if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
ast.parse(module.read_text())
print('FOUR_WAY_P2P_OVERLAY_INSTALLED')
