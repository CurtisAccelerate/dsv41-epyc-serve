import copy,unittest
from optimized_receipts import require_optimized,MIXED

def fixture():
    row=dict(rank=0,ok=True,ready=True,available_arms={MIXED:True},
        sampler_preparation=dict(ready=True,whole_call_exact=True,all_rank_ready=True),
        r2_preparation=dict(ready=True,contract_sha256='a'*64,contract=dict(pad7=True,
            plans=[dict(b=b,served_by=[48,8] if b in (7,8) else [6*b,b]) for b in range(2,9)]),
            graphs={str(b):dict(published=True,components=['attention','experts','collective'],excluded={}) for b in (2,3,4,5,6,8)},
            samplers={str(b):dict(sampler=dict(ready=True,whole_call_bitwise=True),sampled=dict(ready=True,chain_bitwise_vs_eager=True)) for b in range(2,9)}))
    rows=[copy.deepcopy(row) for _ in range(4)]
    for rank,row in enumerate(rows):row['rank']=rank
    return dict(rows=rows)

class OptimizedTests(unittest.TestCase):
    def test_complete(self):self.assertTrue(require_optimized(fixture())['passed'])
    def test_rejections(self):
        mutations=[lambda d:d['rows'].pop(),lambda d:d['rows'][0].update(ready=False),
            lambda d:d['rows'][0]['available_arms'].update({MIXED:False}),
            lambda d:d['rows'][0]['sampler_preparation'].update(whole_call_exact=False),
            lambda d:d['rows'][0]['r2_preparation'].update(ready=False),
            lambda d:d['rows'][0]['r2_preparation']['graphs']['8'].update(excluded={'collective':'rejected'}),
            lambda d:d['rows'][0]['r2_preparation']['graphs']['2'].update(components=['attention']),
            lambda d:d['rows'][0]['r2_preparation']['samplers']['7']['sampler'].update(ready=False),
            lambda d:d['rows'][0]['r2_preparation']['contract']['plans'][5].update(served_by=[42,7]),
            lambda d:d['rows'][0]['r2_preparation'].update(contract_sha256='b'*64)]
        for mutate in mutations:
            d=fixture();mutate(d)
            with self.assertRaises(RuntimeError):require_optimized(d)
if __name__=='__main__':unittest.main(verbosity=2)
