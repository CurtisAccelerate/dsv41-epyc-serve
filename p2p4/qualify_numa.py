"""CPU-only proof that read-only prefault preserves exact NUMA/data validation."""
import argparse,hashlib,json,mmap,pathlib,sys,tempfile
sys.path.insert(0,'/w9/engram-loader')
from numa_builder import numa_evidence

def main():
    p=argparse.ArgumentParser();p.add_argument('--node',type=int,default=0);a=p.parse_args()
    size=8*1024**2
    with tempfile.TemporaryFile(dir='/dev/shm') as f:
        f.truncate(size);mm=mmap.mmap(f.fileno(),size);mm[:]=b'X'*size
        original=hashlib.sha256(mm).hexdigest();numa_evidence(mm,size,a.node)
        mm.madvise(mmap.MADV_DONTNEED)
        evidence=numa_evidence(mm,size,a.node)
        if evidence['read_only_prefault_pages']!=size//4096:raise RuntimeError('Not every page was observed')
        if hashlib.sha256(mm).hexdigest()!=original:raise RuntimeError('Prefault changed payload bytes')
        try:numa_evidence(mm,size,a.node+1)
        except RuntimeError:pass
        else:raise RuntimeError('Foreign NUMA placement accepted')
        print('DSV41_NUMA_OBSERVATION_VERIFIED '+json.dumps(dict(passed=True,pages=size//4096,
            exact_node=a.node,unpopulated_mapping_handled=True,wrong_node_rejected=True,payload_unchanged=True,gpu_used=False)))
if __name__=='__main__':main()
