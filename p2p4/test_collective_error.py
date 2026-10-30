import unittest
try:import torch
except ImportError:torch=None
from collective_error import check_reduction

@unittest.skipIf(torch is None,'Run in serving image: torch required')
class ErrorTests(unittest.TestCase):
    def test_exact(self):
        p=[torch.tensor([1.,2.],dtype=torch.bfloat16)]*4
        self.assertTrue(check_reduction(p,torch.tensor([4.,8.],dtype=torch.bfloat16))['passed'])
    def test_cancellation_rounding(self):
        p=[torch.tensor([x],dtype=torch.bfloat16) for x in (1.,0.00390625,-1.,0.)]
        out=p[0]+p[1]+p[2]+p[3]
        self.assertTrue(check_reduction(p,out)['passed'])
    def test_wrong_result_rejected(self):
        p=[torch.ones(4,dtype=torch.bfloat16)]*4
        with self.assertRaises(RuntimeError):check_reduction(p,torch.zeros(4,dtype=torch.bfloat16))
    def test_nonfinite_rejected(self):
        p=[torch.ones(1,dtype=torch.bfloat16)]*4
        with self.assertRaises(RuntimeError):check_reduction(p,torch.tensor([float('nan')],dtype=torch.bfloat16))
    def test_fp32_wire_output_cast(self):
        p=[torch.tensor([0.1],dtype=torch.bfloat16)]*4
        out=sum(v.float() for v in p).to(torch.bfloat16)
        self.assertTrue(check_reduction(p,out,torch.float32)['passed'])
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):check_reduction([torch.ones(2,3)]*4,torch.ones(3))
    def test_nonfinite_input_rejected(self):
        with self.assertRaises(RuntimeError):check_reduction([torch.tensor([float('inf')])]*4,torch.tensor([4.]))
if __name__=='__main__':unittest.main(verbosity=2)
