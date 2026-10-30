"""Exercise actual communicator selection, profile transaction and R2 parity on TP4.

Uses small synthetic graphs, not a model or a serving-admission substitute.
"""
import argparse,datetime,importlib,json,os,pathlib,socket,types

def worker(rank,port,folder):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    import torch,torch.distributed as dist
    import vllm.distributed.parallel_state as ps
    from vllm.platforms import current_platform
    from w9_exl_ab import collective_profiles as CP,runtime as R,pair_transport as PT
    from hybrid import r2_bank
    from hybrid.r2_contract import Contract
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo',rank=rank,world_size=4,timeout=datetime.timedelta(seconds=240))
    device_group=dist.new_group(backend='nccl')
    ps.set_custom_all_reduce(False)
    qualname=current_platform.get_device_communicator_cls();module,name=qualname.rsplit('.',1)
    comm=getattr(importlib.import_module(module),name)(dist.group.WORLD,device=torch.device('cuda',rank),device_group=device_group,unique_name='tp:0')
    owner=comm.hostshm_comm
    assert owner is not None and not owner.disabled and comm.ca_comm is None
    assert comm.fi_pcie_ipc_ar_comm is None and len(CP.REG)==1 and PT.state(owner).ready
    def reference(x):
        peers=[None]*4;dist.all_gather_object(peers,x.float().cpu())
        return torch.stack(peers).double().sum(0).to(x.dtype)
    roots=[];graphs=[]
    with torch.inference_mode():
        CP.PROFILE='stock'
        for rows in (6,48):
            x=torch.randn(rows,5120,device=rank,dtype=torch.bfloat16)
            stream=torch.cuda.Stream();graph=torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph,stream=stream):out=comm.all_reduce(x)
            graph.instantiate();graphs.append(graph);roots.append((x,out,graph))
        wrapper=types.SimpleNamespace(entries={'fixture':types.SimpleNamespace(capture=types.SimpleNamespace(kernel_patchable_graphs=graphs,num_graphs=2))})
        # Only the test's tiny retained graph set is registered, in this fresh process.
        R.S.all_graph_refs={(id(wrapper),'fixture'):wrapper}
        R.S.ready=True;R.S.poisoned=False;R.S.pending=None;R.S.version_pending=None;R.S.capturing=False;R.S.quarantine_reason=None
        R.S.backend=types.SimpleNamespace(synchronize=lambda:torch.cuda.synchronize(),device=torch.device('cuda',rank))
        transaction=[]
        for method in ('prepare','apply','commit'):
            row=CP.invoke(types.SimpleNamespace(rank=rank),method,'factory-test','six')
            votes=[None]*4;dist.all_gather_object(votes,row)
            if not all(v['ok'] for v in votes):raise RuntimeError('Actual profile transaction rejected: '+repr(votes))
            transaction.append(dict(method=method,phase=row['result']['phase'],counts=row['result']['counts']))
        for x,out,graph in roots:
            expected=reference(x);out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(out.cpu(),expected,rtol=1/64,atol=1e-7)
        contract=Contract.from_env(os.environ);parity=[]
        for plan in contract.captured():
            g=torch.Generator(device=rank).manual_seed(1017+rank);rows=plan.served_by[0]
            base=torch.randn((rows,5120),device=rank,generator=g)
            sign=-1. if rank%2 else 1.
            inputs=(base,base*1024.*sign+torch.randn(base.shape,device=rank,generator=g),
                    base*torch.logspace(-6,6,5120,device=rank))
            ctx=r2_bank.CaptureContext(plan);equal=True
            for x in inputs:
                x=x.to(torch.bfloat16);ref=comm.all_reduce(x)
                r2_bank.ACTIVE=ctx
                try:got=comm.all_reduce(x)
                finally:r2_bank.ACTIVE=None
                equal=equal and bool(torch.equal(ref,got))
            torch.cuda.synchronize();PT.check_errors(force=True)
            ok=equal and ctx.collective['planned']==3 and ctx.collective['unavailable']==0
            votes=[None]*4;dist.all_gather_object(votes,ok)
            if not all(votes):raise RuntimeError('Actual R2 bitwise parity failed at B'+str(plan.b))
            parity.append(dict(batch=plan.b,rows=rows,bitwise=True,planned=ctx.collective['planned']))
        # The actual communicator must service a large tensor through its upstream fallback.
        x=torch.randn(2048,5120,device=rank,dtype=torch.bfloat16)
        assert not owner.should_host_ar(x)
        values=[None]*4;dist.all_gather_object(values,x.cpu())
        out=comm.all_reduce(x);torch.cuda.synchronize()
        from hybrid.collective_error import check_reduction
        numerical=check_reduction(values,out)
        result=dict(passed=True,rank=rank,communicator=qualname,host_selected=True,pair_ready=True,
                    profile_transaction=transaction,r2_bitwise_parity=parity,large_fallback_correct=True,large_numerical_contract=numerical,
                    scope='Actual dispatch/profile/R2 operator integration with synthetic graphs; no model startup')
        pathlib.Path(folder,f'rank-{rank}.json').write_text(json.dumps(result,indent=2))
        print('FACTORY_R2_PASSED '+json.dumps(dict(rank=rank,batches=len(parity))),flush=True)
    dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='/tmp/dsv41-factory-qualification');a=p.parse_args()
    import torch,torch.multiprocessing as mp
    if torch.cuda.device_count()!=4:raise RuntimeError('Exactly four GPUs required')
    folder=pathlib.Path(a.output);folder.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,str(folder)),nprocs=4,join=True)
    result=dict(passed=True,world_size=4,ranks=[json.loads((folder/f'rank-{r}.json').read_text()) for r in range(4)],full_model_qualified=False)
    (folder/'result.json').write_text(json.dumps(result,indent=2));print('FACTORY_R2_QUALIFIED',flush=True)
if __name__=='__main__':main()
