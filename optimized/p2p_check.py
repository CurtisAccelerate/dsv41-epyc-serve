"""Validate every visible peer and NCCL all-reduce before model allocation."""
import datetime, json, os, pathlib, socket, tempfile

def worker(rank, world, port, directory):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),
        NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P',
        NCCL_DEBUG_FILE=f'{directory}/nccl-{rank}.log')
    import torch, torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',rank=rank,world_size=world,timeout=datetime.timedelta(seconds=120))
    x=torch.full((1024*1024,),float(rank+1),device=f'cuda:{rank}')
    dist.all_reduce(x);torch.cuda.synchronize()
    if not bool(torch.all(x==world*(world+1)//2).item()):raise RuntimeError('NCCL reduction incorrect')
    dist.destroy_process_group()

def main():
    import torch, torch.multiprocessing as mp
    if os.environ.get('NCCL_P2P_DISABLE')!='0' or os.environ.get('NCCL_P2P_LEVEL')!='SYS':
        raise RuntimeError('P2P requires NCCL_P2P_DISABLE=0 and NCCL_P2P_LEVEL=SYS')
    n=torch.cuda.device_count()
    if n!=4:raise RuntimeError(f'This TP4 build requires exactly four visible GPUs, got {n}')
    peers=[[i==j or bool(torch.cuda.can_device_access_peer(i,j)) for j in range(n)] for i in range(n)]
    print('DSV41_P2P_CAPABILITY '+json.dumps(dict(peer_matrix=peers)),flush=True)
    if not all(all(row) for row in peers):raise RuntimeError('Driver does not permit P2P on every GPU pair')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    directory=tempfile.mkdtemp(prefix='dsv41-p2p-')
    mp.spawn(worker,args=(n,port,directory),nprocs=n,join=True)
    rank_logs=[(pathlib.Path(directory)/f'nccl-{rank}.log').read_text(errors='replace') for rank in range(n)]
    logs='\n'.join(rank_logs)
    lines=[line for line in logs.splitlines() if 'via P2P' in line]
    fallbacks=[line for line in logs.splitlines() if 'via SHM' in line or 'via NET' in line]
    if not lines or not all('via P2P' in log for log in rank_logs) or fallbacks:
        raise RuntimeError('NCCL did not demonstrate P2P data transport on every rank without SHM/NET fallback: '+directory)
    print('DSV41_P2P_VERIFIED '+json.dumps(dict(passed=True,world_size=n,
        allreduce_correct=True,nccl_p2p_connections=len(lines),log_directory=directory)),flush=True)
if __name__=='__main__':main()
