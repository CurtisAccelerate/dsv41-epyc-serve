"""Four-rank HOSTSHM/pair operator qualification, independent of model admission."""
import argparse,datetime,json,os,pathlib,socket,ctypes as C

def worker(rank,port,folder):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    import torch,torch.distributed as dist
    from vllm.sm80.distributed.host_shm_all_reduce import HostShmAllreduce
    from w9_exl_ab import collective_profiles as CP,pair_transport as PT
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo',rank=rank,world_size=4,timeout=datetime.timedelta(seconds=240))
    owner=HostShmAllreduce(dist.group.WORLD,rank)
    if owner.disabled or len(CP.REG)!=1 or len(PT.STATES)!=1:raise RuntimeError('Actual host/pair initialization failed')
    state=PT.state(owner);checks=[];roots=[]
    def reference(x):
        values=[None]*4;dist.all_gather_object(values,x.float().cpu())
        return torch.stack(values).double().sum(0).to(x.dtype)
    def verify(actual,expected):
        torch.testing.assert_close(actual.cpu(),expected,rtol=1/64,atol=1e-7)
    with torch.inference_mode():
        for profile,rows_list in (('stock',(1,6,12,18,24,30,36,48)),('six',(6,)),('pair_hybrid',(6,48))):
            CP.PROFILE=profile
            for rows in rows_list:
                x=torch.randn(rows,5120,device=f'cuda:{rank}',dtype=torch.bfloat16)
                assert owner.should_host_ar(x)
                expected=reference(x);out=owner.host_all_reduce(x);torch.cuda.synchronize();verify(out,expected)
                stream=torch.cuda.Stream();graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.stream(stream):owner.host_all_reduce(x)
                torch.cuda.synchronize();dist.barrier()
                with torch.cuda.graph(graph,stream=stream):out=owner.host_all_reduce(x)
                graph.instantiate();roots.append((graph,x,out))
                from hybrid.private_nodes import Driver,nodes
                driver=Driver();names=[]
                for _,params in nodes(driver,graph):
                    name=C.c_char_p();driver.call('cuFuncGetName',C.byref(name),C.c_void_p(params.func))
                    names.append((name.value or b'').decode(errors='replace'))
                if profile=='pair_hybrid' and rows==48 and not any('pair_write_bf16' in n for n in names):
                    raise RuntimeError('Selected pair graph lacks the actual pair kernel')
                for change in range(3):
                    x.copy_(torch.randn_like(x)*(change+1)+rank)
                    expected=reference(x);out.fill_(float('nan'));graph.replay()
                    torch.cuda.synchronize();verify(out,expected)
                state.check()
                checks.append(dict(profile=profile,rows=rows,eager=True,graph_replays=3,changed_input=True,poisoned_output=True,kernel_names=names))
        large=torch.empty((2048,5120),device=f'cuda:{rank}',dtype=torch.bfloat16)
        if owner.should_host_ar(large):raise RuntimeError('Large prefill incorrectly routed to fixed-cap host kernel')
        torch.cuda.synchronize();dist.barrier();status=state.status()
        if not all(value>0 for row in status['completed_generations'] for value in row):
            raise RuntimeError('Pair graph executions have no completed device generations')
        if any(status['small_error_flags']) or any(status['owner_error_flags']):raise RuntimeError('Sticky protocol error')
        # Public receipt excludes machine identities, addresses and local paths.
        result=dict(passed=True,rank=rank,world_size=4,checks=checks,
                    cubin_sha256=PT.CUBIN_SHA,completed_generations=status['completed_generations'],
                    error_flags=status['small_error_flags'],host_error_flags=status['owner_error_flags'],
                    large_prefill_declined=True,pair_scope='48x5120 BF16 only',
                    scope='Four-rank operator/graph qualification; full-model and fully-connected EPYC qualification remain separate')
        pathlib.Path(folder,f'rank-{rank}.json').write_text(json.dumps(result,indent=2))
        print('HOSTPAIR_RANK_PASSED '+json.dumps(dict(rank=rank,cases=len(checks))),flush=True)
    dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='/tmp/dsv41-hostpair-qualification')
    a=p.parse_args()
    import torch,torch.multiprocessing as mp
    from hybrid.transport_policy import validate_topology
    if torch.cuda.device_count()!=4:raise RuntimeError('Expose exactly four GPUs')
    matrix=[[i==j or bool(torch.cuda.can_device_access_peer(i,j)) for j in range(4)] for i in range(4)]
    validate_topology(matrix,'hostpair')
    folder=pathlib.Path(a.output);folder.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(port,str(folder)),nprocs=4,join=True)
    result=dict(passed=True,world_size=4,peer_matrix=matrix,independent_cpu_fp64_reference=True,
                ranks=[json.loads((folder/f'rank-{r}.json').read_text()) for r in range(4)],full_model_qualified=False)
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    print('HOSTPAIR_QUALIFIED '+json.dumps(dict(passed=True,result=str(folder/'result.json'))),flush=True)
if __name__=='__main__':main()
