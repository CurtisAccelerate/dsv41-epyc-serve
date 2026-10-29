"""Optional host-AR profiles, changed only by the drained all-rank controller.

Original CUDA source/argument ABI retained. Candidate changes BF16 hidden5120
six/48-row launches only. DMA nodes for48 rows are disabled by exact source,
destination and byte-range matches; all other collectives stay unchanged.
"""
import ctypes as C,os,sys
from .node_versions import KernelParams,graph_objects
from . import runtime as R
from .versions import Rejected
from .collective_args import Arguments,read_values
REG=[];PROFILE='stock';TX=None;BANK=None;SIGNATURE=None
PROFILES=('stock','six','six48','pair_hybrid')
LIB=C.CDLL('libcuda.so.1')
def call(name,*args):
 f=getattr(LIB,name);f.restype=C.c_int;code=f(*args)
 if code:raise RuntimeError(f'{name}: CUDA {code}')
# ctypes requires fields assigned once, after the complete declaration.
fields=[]
for prefix,suffix in [('src','0'),('dst','1')]:
 fields.extend([(prefix+k,C.c_size_t) for k in ('XInBytes','Y','Z','LOD')])
 fields.extend([(prefix+'MemoryType',C.c_int),(prefix+'Host',C.c_void_p),(prefix+'Device',C.c_uint64),(prefix+'Array',C.c_void_p),('reserved'+suffix,C.c_void_p),(prefix+'Pitch',C.c_size_t),(prefix+'Height',C.c_size_t)])
fields.extend([(k,C.c_size_t) for k in ('WidthInBytes','Height','Depth')])
class CopyParams(C.Structure):_fields_=fields
def graph_nodes(graph):
 raw=C.c_void_p(graph.raw_cuda_graph());n=C.c_size_t();call('cuGraphGetNodes',raw,None,C.byref(n));buf=(C.c_void_p*n.value)();call('cuGraphGetNodes',raw,buf,C.byref(n))
 for v in buf:
  node=C.c_void_p(v);kind=C.c_int();call('cuGraphNodeGetType',node,C.byref(kind));yield node,kind.value
def register(owner):
 """Capture unused microscopic graphs once, solely to retain function handles."""
 from . import pair_transport as PT
 if PT.INTERNAL:return
 if os.environ.get('W14_COLLECTIVE_PROFILES')!='1' or owner.world_size!=4:return
 import torch
 if torch.cuda.is_current_stream_capturing():raise RuntimeError('Host AR registry must precede graph capture')
 x=torch.zeros(6*5120,device=owner.device,dtype=torch.bfloat16);y=torch.empty_like(x);torch.cuda.synchronize()
 stream=torch.cuda.Stream(device=owner.device);funcs={};graphs=[]
 for algo in (0,1,2):
  g=torch.cuda.CUDAGraph(keep_graph=True)
  with torch.cuda.graph(g,stream=stream):owner.mod.all_reduce(x,y,owner.dev_base,owner.host_ptr,owner.seq,owner.rank,4,owner.slot_bytes,1,1024,owner.spin_cycles,algo)
  g.instantiate();found=[]
  for node,kind in graph_nodes(g):
   if kind==0:
    p=KernelParams();call('cuGraphKernelNodeGetParams',node,C.byref(p));found.append(p.func)
  if len(found)!=1:raise RuntimeError('Host AR registry kernel count differs')
  funcs[algo]=found[0];graphs.append(g)
 entry=dict(owner=owner,funcs=funcs,graphs=graphs,buffers=(x,y),stream=stream);REG.append(entry)
 try:PT.initialize(owner)
 except BaseException:
  REG.remove(entry)
  raise
def selected(inp,blocks,threads,algo):
 r2=sys.modules.get('hybrid.r2_bank')
 if r2 is not None:
  chosen=r2.collective_hook(inp,blocks,threads,algo)
  if chosen is not None:return chosen
 if os.environ.get('W14_COLLECTIVE_PROFILES')=='1' and PROFILE!='stock' and str(inp.dtype)=='torch.bfloat16':
  size=inp.numel()*inp.element_size()
  if size==61440 or PROFILE=='six48' and size==491520:return 1,512,1
 return blocks,threads,algo
def read_arg(p,i,t):return t.from_address(p.kernelParams[i]).value
def predecessors(node):
 n=C.c_size_t();call('cuGraphNodeGetDependencies',node,None,C.byref(n));buf=(C.c_void_p*n.value)();call('cuGraphNodeGetDependencies',node,buf,C.byref(n))
 return [C.c_void_p(x) for x in buf]
def dependents(node):
 n=C.c_size_t();call('cuGraphNodeGetDependentNodes',node,None,C.byref(n));buf=(C.c_void_p*n.value)();call('cuGraphNodeGetDependentNodes',node,buf,C.byref(n))
 return [x for x in buf]
def inventory():
 global BANK,SIGNATURE
 if not REG:raise Rejected('No qualified host collective registry')
 mapping={}
 for reg in REG:
  for algo,func in reg['funcs'].items():mapping.setdefault(func,[]).append((reg,algo))
 bank=[];seen=set();signature=[];counts={};owned_copies=set()
 for (wrapper_id,key),wrapper in R.S.all_graph_refs.items():
  for graph in graph_objects(wrapper,key):
   gid=graph.raw_cuda_graph();eid=graph.raw_cuda_graph_exec();signature.append((wrapper_id,repr(key),gid,eid))
   if gid in seen:continue
   seen.add(gid);kernels=[];copies=[]
   for node,kind in graph_nodes(graph):
    if kind==0:
     p=KernelParams();call('cuGraphKernelNodeGetParams',node,C.byref(p))
     if p.func in mapping:
      if not p.kernelParams or p.extra:raise Rejected('Collective kernel argument representation differs')
      choices=[(reg,algo) for reg,algo in mapping[p.func] if (reg['owner'].dev_base,reg['owner'].seq.data_ptr())==(read_arg(p,2,C.c_uint64),read_arg(p,3,C.c_uint64))]
      if len(choices)!=1:raise Rejected('Ambiguous collective group identity')
      kernels.append((node,p,*choices[0]))
    elif kind==1:
     p=CopyParams();call('cuGraphMemcpyNodeGetParams',node,C.byref(p));copies.append((node,p))
    elif kind==4:raise Rejected('Child CUDA graph is outside collective profile qualification')
   for node,p,reg,algo in kernels:
    size=read_arg(p,5,C.c_int64)*16
    o=reg['owner'];src=read_arg(p,0,C.c_uint64)
    if size<=0 or size>o.host_cap or size%16:raise Rejected('Unexpected captured host collective size')
    if (read_arg(p,2,C.c_uint64),read_arg(p,3,C.c_uint64),read_arg(p,4,C.c_int),read_arg(p,6,C.c_int64))!=(o.dev_base,o.seq.data_ptr(),o.rank,o.slot_bytes):raise Rejected('Collective argument identity differs')
    if (p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ)!=(1,1,1,1024,1,1):raise Rejected('Collective baseline dimensions differ')
    expected_algo=2 if size>=o.dma_min_bytes else 1 if size>=o.two_shot_min_bytes else 0
    if algo!=expected_algo:raise Rejected('Collective baseline algorithm differs')
    matched=[]
    if algo==2:
     dst=o.host_ptr+256*1024+(2*4+2+o.rank)*o.slot_bytes;lo=size*o.rank//4;hi=size*(o.rank+1)//4
     expected=set((src+a,dst+a,b-a) for a,b in [(0,lo),(hi,size)] if b>a)
     found=set();current=node;copy_map={cn.value:cp for cn,cp in copies}
     for _ in range(len(expected)):
      deps=predecessors(current)
      if len(deps)!=1 or deps[0].value not in copy_map:raise Rejected('DMA copy nodes are not an exclusive immediate chain')
      cn=deps[0];cp=copy_map[cn.value]
      if dependents(cn)!=[current.value]:raise Rejected('DMA copy has another dependent consumer')
      identity=(cp.srcDevice+cp.srcXInBytes,(cp.dstHost or 0)+cp.dstXInBytes,cp.WidthInBytes)
      normalized=not any((cp.srcY,cp.srcZ,cp.srcLOD,cp.dstY,cp.dstZ,cp.dstLOD))
      if not(normalized and cp.srcMemoryType==2 and cp.dstMemoryType==1 and cp.Height==cp.Depth==1 and identity in expected and identity not in found):raise Rejected('DMA immediate copy range differs')
      if (gid,cn.value) in owned_copies:raise Rejected('DMA node belongs to more than one collective')
      owned_copies.add((gid,cn.value));matched.append(cn);found.add(identity);current=cn
     if found!=expected:raise Rejected('DMA copy-node coverage differs')
    from . import pair_transport as PT
    pair=PT.state(o)
    original_args=Arguments(read_values(p));pair_args=None
    if pair is not None and size==491520:pair_args=Arguments(pair.kernel_values(original_args.values()))
    bank.append(dict(graph=graph,exe=C.c_void_p(eid),node=node,original=p,original_args=original_args,
                     candidate=reg['funcs'][1],pair_candidate=None if pair is None else pair.small_fn.value,
                     pair_args=pair_args,pair_state=pair,copies=matched,bytes=size));counts[size]=counts.get(size,0)+1
 signature=sorted(signature)
 if SIGNATURE is not None and SIGNATURE!=signature:raise Rejected('Collective graph handles changed')
 if not any(counts.values()):raise Rejected('No eligible captured collective nodes')
 if BANK is None:BANK=bank;SIGNATURE=signature
 return counts
def apply(profile,fault=False):
 changes=0
 for item in BANK:
  use=(profile in ('six','six48','pair_hybrid') and item['bytes']==61440) or (profile=='six48' and item['bytes']==491520)
  pair=profile=='pair_hybrid' and item['bytes']==491520
  p=KernelParams.from_buffer_copy(bytes(item['original']))
  item['original_args'].bind(p)
  if use:p.func=item['candidate'];p.gridDimX=1;p.blockDimX=512
  if pair:
   if item['pair_candidate'] is None:raise Rejected('Pair candidate was not initialized before model capture')
   if item['pair_args'] is None:raise Rejected('Dedicated pair argument block missing')
   p.func=item['pair_candidate'];p.gridDimX=8;p.blockDimX=256
   item['pair_args'].bind(p)
  call('cuGraphExecKernelNodeSetParams',item['exe'],item['node'],C.byref(p))
  for node in item['copies']:call('cuGraphNodeSetEnabled',item['exe'],node,C.c_uint(0 if use or pair else 1))
  if pair:changes+=1
  marker=__import__('pathlib').Path('/work/COLLECTIVE_FAIL_AFTER')
  if fault and pair and os.environ.get('W14_TINY_PROOF')=='1' and int(R.S.backend.device.index)==1 and marker.exists() and changes>=int(marker.read_text()):raise Rejected('Intentional tiny rank1 partial collective apply rejection')
def invoke(worker,method,token,profile):
 global TX,PROFILE
 try:
  if profile not in PROFILES:raise Rejected('Unknown collective profile')
  from . import pair_transport as PT
  PT.check_errors(force=True)
  if not R.S.ready or R.S.poisoned or R.S.pending or R.S.version_pending or R.S.capturing or R.S.quarantine_reason:raise Rejected('Worker not idle for collective switch')
  if method=='prepare':
   if TX is not None:raise Rejected('Collective transaction already pending')
   counts=inventory()
   if profile=='pair_hybrid':
    if counts.get(61440,0)<=0 or counts.get(491520,0)<=0:raise Rejected('Hybrid requires six-row and48-row captured nodes')
    if any(i['pair_candidate'] is None for i in BANK if i['bytes']==491520):raise Rejected('Incomplete pair48 graph coverage')
   R.S.backend.synchronize();TX=dict(token=token,profile=profile,previous=PROFILE,phase='prepared',counts=counts)
  elif method=='apply':
   if TX is None or TX['token']!=token or TX['profile']!=profile or TX['phase']!='prepared':raise Rejected('Collective apply phase mismatch')
   apply(profile,fault=True);TX['phase']='applied'
  elif method=='commit':
   if TX is None or TX['token']!=token or TX['profile']!=profile or TX['phase']!='applied':raise Rejected('Collective commit phase mismatch')
   PROFILE=profile;result=dict(TX,phase='committed');TX=None;return dict(ok=True,rank=int(worker.rank),result=result)
  elif method=='abort':
   if TX is not None:
    if TX['token']!=token:raise Rejected('Collective abort token mismatch')
    apply(TX['previous']);TX=None
   return dict(ok=True,rank=int(worker.rank),result=dict(token=token,phase='aborted',profile=PROFILE))
  else:raise Rejected('Unknown collective method')
  return dict(ok=True,rank=int(worker.rank),result=dict(TX))
 except Rejected as e:return dict(ok=False,rank=int(worker.rank),rejected=True,error=str(e),poisoned=R.S.poisoned)
 except Exception as e:
  R.S.poisoned=True;R.S.ready=False;return dict(ok=False,rank=int(worker.rank),rejected=False,error=repr(e),poisoned=True)
def status():return dict(profile=PROFILE,pending=TX,registered=len(REG),patchable_nodes=0 if BANK is None else len(BANK))
