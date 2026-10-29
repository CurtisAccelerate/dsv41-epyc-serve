"""Narrow48-row W17 schedule with private host/IPC arena and sequence counters.

Six rows keep their host incumbent; all large routes stay original NCCL.
Initialization precedes model capture. Switching only edits retained graph nodes.
"""
import hashlib,os,time
from pathlib import Path
import torch,torch.distributed as dist
from .pair_driver import Driver
from .pair_policy import CAP,approved_large,pair_route,shape_allowed,validate_startup_votes
from .pair_arena import DedicatedArena,NBYTES,BLOCKS,THREADS,TILE
INTERNAL=False;STATES={};NEXT_CHECK=0.;FAULT_RETAINED=[]
CUBIN_SHA='04ddd502f8cd651eb16d336804f0b1a2fc2a93061683f6dccb2ff009b63d1d65'
def enabled():return os.environ.get('W15_PAIR_TRANSPORT')=='1'
class PairState:
 def __init__(self,owner):
  self.owner=owner;self.ready=False;self.large=None;self.large_sizes=();self.calls={};self.memory=[]
  def phase(fn):
   if not owner._step(fn):raise RuntimeError('W19 pair startup disagreement: '+str(owner._local_error))
  self.phase=phase
  def validate():
   assert owner.world_size==4 and owner.block_cap==1 and owner.host_cap==512*1024
   assert not approved_large(os.environ.get('W15_PAIR_LARGE_BYTES_JSON','[]')),'Large routes must remain empty'
   self.binary=Path(__file__).with_name('pair_write.sm80.cubin')
   assert hashlib.sha256(self.binary.read_bytes()).hexdigest()==CUBIN_SHA,'Unqualified W19 cubin'
  phase(validate);phase(lambda:setattr(self,'driver',Driver()))
  phase(lambda:setattr(self,'identity',self.driver.identity()))
  identities=[None]*4;dist.all_gather_object(identities,self.identity,group=owner.group);self.topology=identities
  def topology():
   assert len({x['uuid'] for x in identities})==4 and len({x['ordinal'] for x in identities})==4
   for rank,x in enumerate(identities):
    if rank!=owner.rank:assert self.driver.can_access(self.identity['ordinal'],x['ordinal'])==(rank==(owner.rank^1))
  phase(topology)
  self.arena=DedicatedArena(owner,phase)
  def alloc():
   before,_=torch.cuda.mem_get_info(owner.device);needed=4*self.arena.slot_bytes
   assert before-needed>=1024**3,'W19 inbox would cross1GiB free-memory floor'
   self.small_inbox=self.driver.alloc(needed);after,_=torch.cuda.mem_get_info(owner.device)
   self.memory.append(dict(before_free=before,after_free=after,inbox_bytes=needed))
  phase(alloc);local={};phase(lambda:local.update(handle=self.driver.handle(self.small_inbox)))
  handles=[None]*4;dist.all_gather_object(handles,local['handle'],group=owner.group)
  phase(lambda:setattr(self,'small_peer',self.driver.open_peer(handles[owner.rank^1])))
  phase(lambda:setattr(self,'small_fn',self.driver.load(self.binary,self.small_inbox,self.small_peer)))
  phase(lambda:torch.cuda.synchronize(owner.device))
  def floor():assert torch.cuda.mem_get_info(owner.device)[0]>=1024**3,'W19 postinit VRAM floor'
  phase(floor);phase(self.check);self.ready=True
 def eligible(self,x):
  from .collective_profiles import PROFILE
  return (self.ready and x.dtype==torch.bfloat16 and x.is_cuda and x.device==self.owner.device
          and shape_allowed(tuple(x.shape),x.numel()) and x.numel()*2==NBYTES
          and pair_route(PROFILE,NBYTES,()))
 def kernel_values(self,original):
  assert len(original)==8 and original[4]==self.owner.rank and original[5]*16==NBYTES
  values=list(original);values[2]=self.arena.dev_base;values[3]=self.arena.seq.data_ptr();values[6]=self.arena.slot_bytes
  return values
 def reduce(self,x):
  assert self.eligible(x);self.check();x=self.owner._aligned(x.contiguous());out=torch.empty_like(x);a=self.arena
  self.driver.launch(self.small_fn,x.data_ptr(),out.data_ptr(),a.dev_base,a.seq.data_ptr(),a.rank,
                     NBYTES,a.slot_bytes,a.spin_cycles,torch.cuda.current_stream(x.device).cuda_stream,BLOCKS,THREADS)
  self.calls[str(NBYTES)]=self.calls.get(str(NBYTES),0)+1;return out
 def check(self):
  for arena in (self.owner,self.arena):
   flags=arena.error_flags()
   if any(flags):raise RuntimeError('Sticky W19 collective error: '+str(flags))
 def status(self):
  return dict(ready=self.ready,small_slot=self.arena.slot_bytes,owner_small_slot=self.owner.slot_bytes,
              large_slot=0,small_error_flags=self.arena.error_flags(),owner_error_flags=self.owner.error_flags(),
              large_error_flags=[],eager_calls=dict(self.calls),rank=self.owner.rank,peer_rank=self.owner.rank^1,
              device_inbox_bytes=4*self.arena.slot_bytes,own_small_inbox=self.small_inbox,own_large_inbox=None,
              device_identity=self.identity,topology=self.topology,large_approved_bytes=[],cubin_sha256=CUBIN_SHA,
              initialization_memory=self.memory,dedicated_arena=self.arena.identity(),policy='W19-fixed48-pipe1024',
              completed_generations=self.arena.completed_generations(),
              numa_query_attempts=dict(before_pin=self.arena.before_pin['query_attempts'],after_pin=self.arena.after_pin['query_attempts']),
              numa_initial_thread_policy=self.arena.initial_thread_policy,
              numa_first_writes=self.arena.touch_records,numa_stage_log=str(self.arena.trace_path),
              completed_generations_scope='phase1 max parity,8blocks4ranks; counts only while drained paused,no wrap')
def initialize(owner):
 if not enabled() or INTERNAL:return
 def one():assert not STATES,'Only one native TP4 group is qualified'
 if not owner._step(one):raise RuntimeError('W19 group identity disagreement')
 STATES[id(owner)]=PairState(owner)
def state(owner):return STATES.get(id(owner))
def active():
 from .collective_profiles import PROFILE
 return PROFILE=='pair_hybrid'
def eligible(owner,x):
 s=state(owner);return s is not None and s.eligible(x)
def check_errors(force=False):
 global NEXT_CHECK
 now=time.monotonic()
 if force or now>=NEXT_CHECK:
  for s in STATES.values():s.check()
  NEXT_CHECK=now+.05
def status():return [s.status() for s in STATES.values()]
def require_startup():
 """All-rank fail-closed guard immediately after TP communicator construction."""
 from vllm.distributed.parallel_state import get_tp_group
 group=get_tp_group().cpu_group;requested=enabled();ready=False;error=None
 try:
  if requested:
   assert len(STATES)==1,'Missing or multiple pair states'
   s=next(iter(STATES.values()));assert s.ready and not s.owner.disabled
   assert s.arena.before_pin['exact_matches'] and s.arena.after_pin['exact_matches']
   s.check();ready=True
 except Exception as exc:error=repr(exc)
 local=dict(requested=requested,ready=ready,error=error);votes=[None]*dist.get_world_size(group=group)
 dist.all_gather_object(votes,local,group=group);validate_startup_votes(votes)
def fault_proof():
 """Tiny-only RPC: two disposable independent arenas; active graph arena untouched."""
 assert os.environ.get('W14_TINY_PROOF')=='1' and len(STATES)==1
 s=next(iter(STATES.values()));s.check();before=s.arena.identity();receipts=[]
 for phase_id in (0,1):
  a=DedicatedArena(s.owner,s.phase);local={}
  s.phase(lambda:local.update(driver=Driver()));d=local['driver'];FAULT_RETAINED.append((a,d))
  def alloc():
   free,_=torch.cuda.mem_get_info(s.owner.device);assert free-8*NBYTES>=1024**3
   local['own']=d.alloc(4*NBYTES);local['handle']=d.handle(local['own'])
  s.phase(alloc);handles=[None]*4;dist.all_gather_object(handles,local['handle'],group=s.owner.group)
  s.phase(lambda:local.update(peer=d.open_peer(handles[s.owner.rank^1])))
  s.phase(lambda:local.update(fn=d.load(s.binary,local['own'],local['peer'])))
  def independent():
   assert d.modules[-1].value!=s.driver.modules[-1].value and local['fn'].value!=s.small_fn.value,'Fault module aliases active module'
  s.phase(independent)
  s.phase(lambda:d.global32(d.modules[-1],'fail_rank',2))
  s.phase(lambda:d.global32(d.modules[-1],'fail_phase',phase_id))
  x=torch.full((NBYTES//2,),s.owner.rank+1.,dtype=torch.bfloat16,device=s.owner.device);y=torch.full_like(x,-123)
  stream=torch.cuda.Stream(device=s.owner.device);torch.cuda.synchronize();started=time.monotonic()
  d.launch(local['fn'],x.data_ptr(),y.data_ptr(),a.dev_base,a.seq.data_ptr(),s.owner.rank,NBYTES,NBYTES,50_000_000,stream.cuda_stream,BLOCKS,THREADS)
  stream.synchronize();dist.barrier(group=s.owner.group)
  elapsed=time.monotonic()-started;flags=a.error_flags()
  def verify():
   assert elapsed<10 and all(f&0x80000000 for f in flags),(phase_id,elapsed,flags)
   if phase_id==0:assert bool(torch.all(y==-123))
  s.phase(verify);receipts.append(dict(phase=phase_id,seconds=elapsed,error_flags=flags,discard_outputs=True,
                                    output_untouched=bool(torch.all(y==-123))))
 s.check();assert s.arena.identity()==before
 return dict(pass_all=True,faults=receipts,active_arena_unchanged=True,retained_disposable_arenas=2)
