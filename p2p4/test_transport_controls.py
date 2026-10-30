import ast,copy,itertools,os,pathlib,sys,types,unittest
from unittest.mock import patch
from transport_policy import validate_peer_row,validate_topology,validate_runtime

class TopologyTests(unittest.TestCase):
    def test_full_matrix_both_modes(self):
        for mode in ('hostpair','p2p4'):self.assertTrue(validate_topology([[True]*4 for _ in range(4)],mode))
    def test_pair_matrix_host_only(self):
        matrix=[[j in (i,i^1) for j in range(4)] for i in range(4)]
        self.assertTrue(validate_topology(matrix,'hostpair'))
        with self.assertRaises(ValueError):validate_topology(matrix,'p2p4')
    def test_all_optional_edge_combinations(self):
        edges=[(i,j) for i in range(4) for j in range(4) if j not in (i,i^1)]
        for bits in itertools.product((False,True),repeat=len(edges)):
            matrix=[[j in (i,i^1) for j in range(4)] for i in range(4)]
            for (i,j),value in zip(edges,bits):matrix[i][j]=value
            self.assertTrue(validate_topology(matrix,'hostpair'))
    def test_every_missing_partner_rejected(self):
        for rank in range(4):
            matrix=[[True]*4 for _ in range(4)];matrix[rank][rank^1]=False
            for mode in ('p2p4','hostpair'):
                with self.assertRaises(ValueError):validate_topology(matrix,mode)
    def test_invalid(self):
        for matrix in ([],[[True]*4]*3,[[1]*4]*4,[[True]*3]*4):
            with self.assertRaises(ValueError):validate_topology(matrix,'hostpair')
        with self.assertRaises(ValueError):validate_topology([[True]*4]*4,'unknown')
    def test_runtime_modes_and_incompatible_flags(self):
        for mode in ('hostpair','p2p4'):
            env=dict(DSV41_TRANSPORT=mode,VLLM_HOST_ALLREDUCE='1' if mode=='hostpair' else '0',
                W15_PAIR_TRANSPORT='1' if mode=='hostpair' else '0',NCCL_SHM_DISABLE='0' if mode=='hostpair' else '1',
                NCCL_P2P_DISABLE='0',NCCL_P2P_LEVEL='SYS')
            self.assertEqual(validate_runtime(env),mode)
            for key in ('VLLM_HOST_ALLREDUCE','W15_PAIR_TRANSPORT','NCCL_SHM_DISABLE','NCCL_P2P_DISABLE','NCCL_P2P_LEVEL'):
                with self.assertRaises(ValueError):validate_runtime(dict(env,**{key:'wrong'}))

package=types.ModuleType('payload_fixture');package.__path__=[str(pathlib.Path(__file__).parent)]
sys.modules['payload_fixture']=package
import importlib
peer=importlib.import_module('payload_fixture.p2p4_transport')

class Tensor:
    is_cuda=True;device='cuda:0';dtype='bf16';casts=0
    def float(self):self.casts+=1;out=Tensor();out.dtype='fp32';return out
    def to(self,dtype):out=Tensor();out.dtype=dtype;return out

class PayloadTests(unittest.TestCase):
    def run_policy(self,policy):
        fake_torch=types.SimpleNamespace(float16='fp16',bfloat16='bf16',cuda=types.SimpleNamespace(is_current_stream_capturing=lambda:False))
        t=peer.Transport.__new__(peer.Transport);t.payload_dtype=policy;t.calls=0;t.require=lambda:None
        wires=[]
        def reduce(wire):wires.append(wire);return wire
        t.comm=types.SimpleNamespace(device='cuda:0',all_reduce=reduce)
        x=Tensor()
        with patch.dict(sys.modules,torch=fake_torch):out=t.reduce(x)
        self.assertEqual(out.dtype,'bf16')
        return x,wires[0]
    def test_native_no_widening(self):
        x,wire=self.run_policy('native');self.assertIs(wire,x);self.assertEqual(x.casts,0)
    def test_fp32_explicit(self):
        x,wire=self.run_policy('fp32');self.assertEqual(wire.dtype,'fp32');self.assertEqual(x.casts,1)
    def test_mode_validation(self):
        for mode in ('native','fp32'):
            with patch.dict(os.environ,DSV41_P2P_PAYLOAD_DTYPE=mode):self.assertEqual(peer.payload_mode(),mode)
        with patch.dict(os.environ,DSV41_P2P_PAYLOAD_DTYPE='bf32'):
            with self.assertRaises(ValueError):peer.payload_mode()

if __name__=='__main__':unittest.main(verbosity=2)
