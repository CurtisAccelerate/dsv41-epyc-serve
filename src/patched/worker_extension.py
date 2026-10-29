"""Named vLLM worker extension; no callable serialization or worker constructor changes."""
class WorkerExtension:
 def w47_production_status(self):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'status')
 def w47_production_prepare(self,request):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'prepare',request)
 def w47_production_commit(self,token):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'commit',token)
 def w47_production_abort(self,reason):
  from hybrid.production_control import worker_invoke
  return worker_invoke(self,'abort',reason)
 def w47_production_measured_prepare(self,token):
  return measured_serve(self,'prepare',token)
 def w47_production_measured_commit(self,token):
  return measured_serve(self,'commit',token)
 def w9_status(self):
  import torch
  from w9_exl_ab import runtime as R
  from w9_exl_ab.collective_profiles import status as collective_status
  from w9_exl_ab.pair_transport import check_errors, status as pair_status
  try:check_errors(force=True)
  except Exception:
   R.S.poisoned=True;R.S.ready=False
  return dict(pair=pair_status(),collective=collective_status(),pid=__import__("os").getpid(),pending=R.S.pending,capturing=R.S.capturing,quarantine_reason=R.S.quarantine_reason,active_versions=dict(R.S.active_versions),active_hashes=dict(R.S.active_hashes),version_pending=R.S.version_pending,graph_sets={a:sorted(repr(v) for v in R.S.graphs[a]) for a in R.ARMS},rank=int(self.rank),ready=R.S.ready,arm=R.S.arm,epoch=R.S.epoch,poisoned=R.S.poisoned,capture_generation=R.S.capture_generation,graphs={a:len(v) for a,v in R.S.graphs.items()},launches=R.S.launches,graph_budget=R.S.graph_budget,graph_used=R.S.graph_used,graph_receipts=R.S.graph_receipts,device=str(R.S.backend.device),cc=R.S.backend.cc,hashes=R.HASHES,free_total=torch.cuda.mem_get_info(),allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())

 def w9_prepare(self,arm,epoch):
  from .runtime import rpc_prepare
  return rpc_prepare(self,arm,epoch)

 def w9_commit(self,arm,epoch):
  from .runtime import rpc_commit
  return rpc_commit(self,arm,epoch)

 def w9_version_stage(self,token,arm,version,manifest_name,manifest_sha):
  from .versions import invoke
  return invoke(self,'stage',token,arm,version,manifest_name,manifest_sha)
 def w9_version_capture(self,token):
  from .versions import invoke
  return invoke(self,'capture',token)
 def w9_version_commit(self,token):
  from .versions import invoke
  return invoke(self,'commit',token)
 def w9_version_abort(self,token):
  from .versions import invoke
  return invoke(self,'abort',token)
 def w9_version_status(self):
  from .versions import invoke
  return invoke(self,'status')

 def w14_collective_profile(self,method,token,profile):
  from .collective_profiles import invoke
  return invoke(self,method,token,profile)

 def w26_attention_bank_status(self):
  from hybrid.attention_worker import invoke
  return invoke(self,'status')

 def w26_attention_bank_prepare(self,request):
  from hybrid.attention_worker import invoke
  return invoke(self,'prepare',request)

 def w26_attention_bank_commit(self,token):
  from hybrid.attention_worker import invoke
  return invoke(self,'commit',token)

 def w26_attention_bank_abort(self,reason):
  from hybrid.attention_worker import invoke
  return invoke(self,'abort',reason)

 def w26_attention_benchmark_arm(self,session,identity_sha256):
  from hybrid.attention_worker import invoke
  return invoke(self,'arm',session,identity_sha256)

 def w26_attention_initial_prepare(self,owner_id):
  from hybrid.attention_startup import prepare_worker
  return prepare_worker(self,owner_id)

def measured_serve(worker,method,token):
 import os
 from hybrid.cumulative_contract import MIXED
 from hybrid.cumulative_worker import WorkerCumulative
 a=getattr(worker,'_w26_attention_adapter',None)
 if type(a) is not WorkerCumulative:raise RuntimeError('Measured serve requires initialized cumulative worker')
 c=a.controller;identity=a.current_identity()
 if not isinstance(token,str) or not token:raise ValueError('Invalid measured token')
 if method=='prepare':
  if os.environ.get('DSV41_R2','0') in ('','0'):raise ValueError('Measured serve requires DSV41_R2')
  if c.production_active or c.production_pending is not None or c.pending is not None or c.session_token is not None or c.selected!='baseline' or a.session_armed or a._production_binding is not None:raise ValueError('Measured prepare requires idle unarmed baseline')
  if not c.availability()[MIXED]:raise ValueError('Mixed arm is not ready')
  digest=identity.arm_digest(MIXED)
  c.production_pending=(token,digest,'measured-serving')
  return dict(rank=int(worker.rank),ok=True,prepared=True,token=token,bank=MIXED,identity_sha256=digest)
 if method=='commit':
  pending=c.production_pending
  if pending is None or pending[0]!=token or pending[2]!='measured-serving' or not c.availability()[MIXED] or identity.arm_digest(MIXED)!=pending[1]:raise ValueError('Measured commit is not prepared')
  c.selected=MIXED;c.session_token=None;c.production_approval='measured-serving';c.production_active=True;c.production_pending=None;a.session_armed=False
  a._production_binding=(id(a._family),id(c.entries['next_mixed']),id(c.mixed_native),c.production_approval)
  a.current_identity();a.light_bindings()
  return dict(rank=int(worker.rank),ok=True,committed=True,token=token,bank=MIXED,identity_sha256=pending[1],production_enabled=True,session_armed=False)
 raise ValueError('Unknown measured serve method')
