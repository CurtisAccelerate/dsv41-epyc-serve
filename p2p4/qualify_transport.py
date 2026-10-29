"""Independent operator, changed-input and poisoned-output CUDA graph checks."""
import argparse,datetime,json,os,pathlib,socket,tempfile

def worker(rank,world,port,folder):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),
                     NCCL_DEBUG_FILE=f'{folder}/nccl-{os.getpid()}.log')
    import torch,torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from hybrid import p2p4_transport as peer
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo',rank=rank,world_size=world,timeout=datetime.timedelta(seconds=180))
    comm=PyNcclCommunicator(dist.group.WORLD,device=rank)
    transport=peer.Transport(comm,expected_world=world)
    checks=[];roots=[]
    def reference(x):
        values=[None]*world;dist.all_gather_object(values,x.float().cpu(),group=dist.group.WORLD)
        return torch.stack(values).double().sum(0).to(x.dtype)
    def verify(actual,expected):
        torch.testing.assert_close(actual.cpu(),expected,rtol=2*torch.finfo(actual.dtype).eps,atol=1e-6)
    with torch.inference_mode():
        for dtype in (torch.bfloat16,torch.float16,torch.float32):
            for rows in (1,6,12,18,24,30,36,48,128,2048):
                x=torch.randn(rows,5120,device=f'cuda:{rank}',dtype=dtype)
                expected=reference(x);actual=transport.reduce(x);torch.cuda.synchronize();verify(actual,expected)
                checks.append(dict(rows=rows,dtype=str(dtype),eager=True))
            for rows in (6,48):
                x=torch.randn(rows,5120,device=f'cuda:{rank}',dtype=dtype)
                stream=torch.cuda.Stream();graph=torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.stream(stream):
                    for _ in range(3):transport.reduce(x)
                torch.cuda.synchronize();dist.barrier()
                with torch.cuda.graph(graph,stream=stream):out=transport.reduce(x)
                graph.instantiate();roots.append((graph,x,out,stream))
                signature=peer.graph_signature((graph,))
                if not any(k[0]=='p2p4' and k[1]==x.numel()*x.element_size() for k in signature):
                    raise RuntimeError('Actual graph lacks recorded device-memory all-reduce')
                for change in range(3):
                    x.copy_(torch.randn_like(x)*(change+1)+rank)
                    expected=reference(x);out.fill_(float('nan'));graph.replay()
                    torch.cuda.synchronize();verify(out,expected)
                checks.append(dict(rows=rows,dtype=str(dtype),graph_replays=3,
                    changed_inputs=True,poisoned_output=True,nccl_kernel_nodes=True))
        receipt=dict(passed=True,rank=rank,world_size=world,transport=transport.status(),
                     checks=checks,actual_peer_log=peer._p2p_log(),
                     scope='Two-rank protocol check only' if world==2 else 'Four-rank transport operator and CUDA graph qualification')
        pathlib.Path(folder,f'rank-{rank}.json').write_text(json.dumps(receipt,indent=2))
        print('P2P_TRANSPORT_RANK_PASSED '+json.dumps(dict(rank=rank,world_size=world,cases=len(checks))),flush=True)
    dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world-size',type=int,choices=(2,4),default=4)
    p.add_argument('--allow-pair-test',action='store_true')
    p.add_argument('--output',default='/tmp/dsv41-p2p4-qualification')
    a=p.parse_args()
    if a.world_size==2 and not a.allow_pair_test:raise RuntimeError('Two-rank test needs explicit --allow-pair-test; it does not qualify TP4')
    import torch,torch.multiprocessing as mp
    if torch.cuda.device_count()!=a.world_size:raise RuntimeError('Expose exactly the requested GPU count')
    peers=[[i==j or bool(torch.cuda.can_device_access_peer(i,j)) for j in range(a.world_size)] for i in range(a.world_size)]
    if not all(all(row) for row in peers):raise RuntimeError('All-pairs peer access unavailable: '+repr(peers))
    folder=pathlib.Path(a.output);folder.mkdir(parents=True,exist_ok=False)
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    mp.spawn(worker,args=(a.world_size,port,str(folder)),nprocs=a.world_size,join=True)
    rows=[json.loads((folder/f'rank-{rank}.json').read_text()) for rank in range(a.world_size)]
    result=dict(passed=True,world_size=a.world_size,full_four_way_qualified=a.world_size==4,
                independent_cpu_fp64_reference=True,changed_input_and_poisoned_output_checks=True,
                captured_nccl_kernel_evidence=True,ranks=rows)
    (folder/'result.json').write_text(json.dumps(result,indent=2))
    print('P2P_TRANSPORT_QUALIFIED '+json.dumps(dict(passed=True,world_size=a.world_size,
          full_four_way_qualified=a.world_size==4,result=str(folder/'result.json'))),flush=True)
if __name__=='__main__':main()
