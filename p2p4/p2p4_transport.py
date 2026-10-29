"""TP4 device-memory transport with captured-graph provenance.

The legacy 'six' control label is retained only for graph-bank compatibility.
It never selects a host-memory kernel in this mode.
"""
import collections, ctypes as C, os, pathlib

TRANSPORT=None
CAPTURES={}

def enabled():return os.environ.get('DSV41_TRANSPORT')=='p2p4'

def _call(name,*args):
    if not hasattr(_call,'driver'):_call.driver=C.CDLL('libcuda.so.1')
    f=getattr(_call.driver,name);f.restype=C.c_int;code=f(*args)
    if code:raise RuntimeError(name+': CUDA '+str(code))

def _captured_graph(stream):
    status=C.c_int();identity=C.c_uint64();graph=C.c_void_p()
    _call('cuStreamGetCaptureInfo_v2',C.c_void_p(stream),C.byref(status),C.byref(identity),
          C.byref(graph),None,None)
    if status.value!=1 or not graph.value:raise RuntimeError('Active capture graph identity unavailable')
    return graph.value

def _p2p_log():
    raw=os.environ.get('NCCL_DEBUG_FILE','')
    if not raw:raise RuntimeError('P2P4 requires a per-process NCCL_DEBUG_FILE')
    import socket
    path=pathlib.Path(raw.replace('%p',str(os.getpid())).replace('%h',socket.gethostname()))
    text=path.read_text(errors='replace')
    if 'via P2P' not in text:raise RuntimeError('Serving communicator has no direct P2P evidence: '+str(path))
    if any('via SHM' in line or 'via NET' in line for line in text.splitlines()):
        raise RuntimeError('Serving communicator used SHM/NET data fallback: '+str(path))
    return dict(passed=True,direct_p2p=True,shm_or_net_data_fallback=False)

class Transport:
    def __init__(self,comm,expected_world=4):
        if not comm.available or comm.disabled or comm.world_size!=expected_world:
            raise RuntimeError('Required PyNCCL peer communicator unavailable')
        self.comm=comm;self.world=comm.world_size;self.rank=comm.rank;self.calls=0
        self.expected_world=expected_world;self._identity=self.binding()
    def binding(self):
        return (id(self.comm),int(self.comm.comm.value),str(self.comm.device),self.comm.world_size)
    def require(self):
        if self.binding()!=self._identity or self.comm.disabled or not self.comm.available:
            raise RuntimeError('P2P communicator changed or became unavailable')
        if any(os.environ.get(k)!=v for k,v in
               {'NCCL_P2P_DISABLE':'0','NCCL_P2P_LEVEL':'SYS','VLLM_HOST_ALLREDUCE':'0','W15_PAIR_TRANSPORT':'0'}.items()):
            raise RuntimeError('P2P4 transport requires host/pair paths disabled and all-pairs P2P enabled')
    def reduce(self,inp):
        import torch
        self.require()
        if not inp.is_cuda or inp.device!=self.comm.device:raise RuntimeError('Wrong collective device')
        # Keep accumulation in FP32 for FP16/BF16 activations, as the previous
        # host kernel did. NCCL's reduction order still needs quality qualification.
        wire=inp.float() if inp.dtype in (torch.float16,torch.bfloat16) else inp
        out=self.comm.all_reduce(wire)
        if out is None:raise RuntimeError('Required P2P all-reduce declined the tensor')
        self.calls+=1
        if torch.cuda.is_current_stream_capturing():
            graph=_captured_graph(torch.cuda.current_stream().cuda_stream)
            CAPTURES.setdefault(graph,[]).append((inp.numel()*inp.element_size(),str(inp.dtype),
                                                 wire.numel()*wire.element_size(),str(wire.dtype)))
        return out.to(inp.dtype) if wire.dtype!=inp.dtype else out
    def status(self):
        self.require()
        return dict(transport='p2p4',backend='pynccl-device-memory',world_size=self.world,
                    payload_host_bytes=0,host_collective_enabled=False,pair_collective_enabled=False,
                    fp32_accumulation_for_half=True,captured_graphs=len(CAPTURES),enqueue_calls=self.calls)

def register(comm):
    global TRANSPORT
    if TRANSPORT is not None:raise RuntimeError('Only one TP4 peer communicator is supported')
    TRANSPORT=Transport(comm)
    return TRANSPORT

def require():
    if not enabled() or TRANSPORT is None:raise RuntimeError('P2P4 serving transport was not initialized')
    TRANSPORT.require()
    if TRANSPORT.world!=4:raise RuntimeError('Serving transport must span all four TP ranks')
    return TRANSPORT

def reduce(inp):return require().reduce(inp)

def graph_signature(graphs):
    from .private_nodes import Driver,nodes
    result=collections.Counter();driver=Driver()
    for graph in graphs:
        records=CAPTURES.get(graph.raw_cuda_graph())
        if not records:raise RuntimeError('Graph has no recorded TP4 device-memory all-reduces')
        names=[]
        for _,params in nodes(driver,graph):
            name=C.c_char_p();driver.call('cuFuncGetName',C.byref(name),C.c_void_p(params.func))
            value=(name.value or b'').decode(errors='replace')
            if 'nccl' in value.lower():names.append(value)
        if not names:raise RuntimeError('Graph has no executable NCCL kernel nodes')
        result.update(('p2p4',*row) for row in records)
        result[('nccl-kernels',0,len(names),tuple(sorted(names)))]+=1
    return result

def graph_receipt(graph,required_bytes):
    require();signature=graph_signature((graph,))
    calls=sum(n for k,n in signature.items() if k[0]=='p2p4')
    planned=sum(n for k,n in signature.items() if k[0]=='p2p4' and k[1]==required_bytes)
    if not planned:raise RuntimeError('No required residual all-reduce captured in P2P graph')
    return dict(ok=True,transport='p2p4',world_size=4,payload_host_bytes=0,calls=calls,
                planned=planned,unavailable=0,other_sizes={},bytes=required_bytes,profile='p2p4',signature=repr(signature))

def status():
    return require().status() if enabled() else {}

def health():
    if enabled():require()

def inventory():
    from w9_exl_ab import runtime as R
    from w9_exl_ab.node_versions import graph_objects
    require();counts=collections.Counter();seen=set()
    for (_,key),wrapper in R.S.all_graph_refs.items():
        for graph in graph_objects(wrapper,key):
            if graph.raw_cuda_graph() in seen:continue
            seen.add(graph.raw_cuda_graph())
            if graph.raw_cuda_graph() not in CAPTURES:continue
            for row,n in graph_signature((graph,)).items():
                if row[0]=='p2p4':counts[row[1]]+=n
    if not counts.get(61440):raise RuntimeError('No six-row serving graph has P2P all-reduce coverage')
    return dict(counts)

def profile_invoke(worker,method,token,profile,CP):
    require()
    if profile not in ('stock','six'):raise ValueError('Host/pair profile unavailable in four-way P2P mode')
    if method=='prepare':
        if CP.TX is not None:raise RuntimeError('Collective profile transaction already pending')
        counts=inventory();_p2p_log()
        CP.TX=dict(token=token,profile=profile,previous=CP.PROFILE,phase='prepared',counts=counts)
        return dict(ok=True,rank=int(worker.rank),result=dict(CP.TX,transport='p2p4',executable_updates=0))
    if method=='abort':
        CP.TX=None;return dict(ok=True,rank=int(worker.rank),result=dict(token=token,phase='aborted',profile=CP.PROFILE))
    if CP.TX is None or CP.TX['token']!=token or CP.TX['profile']!=profile:raise RuntimeError('Wrong collective transaction')
    if method=='apply' and CP.TX['phase']=='prepared':
        CP.TX['phase']='applied';return dict(ok=True,rank=int(worker.rank),result=dict(CP.TX,transport='p2p4',executable_updates=0))
    if method=='commit' and CP.TX['phase']=='applied':
        CP.PROFILE=profile;result=dict(CP.TX,phase='committed',transport='p2p4',executable_updates=0);CP.TX=None
        return dict(ok=True,rank=int(worker.rank),result=result)
    raise RuntimeError('Wrong collective transaction phase')

def collective_parity(worker,rows):
    import torch,torch.distributed as dist
    from vllm.distributed import get_tp_group
    t=require();group=get_tp_group().cpu_group
    g=torch.Generator(device=t.comm.device).manual_seed(1017+t.rank)
    tested=[]
    for scale in (1.,1024.,1e-5):
        inp=(torch.randn((rows,5120),device=t.comm.device,generator=g)*scale).to(torch.bfloat16)
        local=inp.float().cpu();peers=[None]*t.world
        dist.all_gather_object(peers,local,group=group)
        reference=torch.stack(peers).double().sum(0).to(torch.bfloat16)
        out=t.reduce(inp);torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(),reference,rtol=1/64,atol=1e-7)
        tested.append(dict(scale=scale,max_absolute_error=float((out.cpu().float()-reference.float()).abs().max())))
    return dict(passed=True,independent_cpu_fp64_reference=True,transport='p2p4',world_size=4,
                tested=tested,scope='Collective operator check; model quality and performance require separate measurement')
