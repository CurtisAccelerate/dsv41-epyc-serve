"""Lifecycle regression tests; no CUDA hardware required."""
import gc,unittest,weakref
from p2p4_provenance import Registry, install_hooks,REGISTRY

class Graph:
    def __init__(self,raw=7):self.raw=raw
    def raw_cuda_graph(self):return self.raw
class Transport:
    def __init__(self):self.handle=11;self.valid=True
    def binding(self):return (self.handle,4)
    def require(self):
        if not self.valid:raise RuntimeError('Transport unavailable')

class ProvenanceTests(unittest.TestCase):
    def setUp(self):self.r=Registry();self.t=Transport();self.g=Graph()
    def record(self,g=None,identity=(1,7),rows=1):
        g=g or self.g;self.r.begin(g,identity)
        for _ in range(rows):self.r.append(identity,self.t,(61440,'bf16',122880,'fp32'))
        self.r.seal(g);return self.r.get(g)
    def test_exact_single(self):self.assertEqual(len(self.record().rows),1)
    def test_raw_reuse_new_owner(self):
        self.record();other=Graph();self.record(other,(2,7))
        self.assertEqual(len(self.r.get(other).rows),1)
    def test_recapture_same_owner(self):
        old=self.record();new=self.record(identity=(2,7))
        self.assertGreater(new.generation,old.generation);self.assertEqual(len(new.rows),1)
    def test_garbage_collection(self):
        g=Graph();self.record(g);ref=weakref.ref(g);del g;gc.collect()
        self.assertIsNone(ref());self.assertEqual(len(self.r),0)
    def test_reset(self):
        self.record();self.r.invalidate(self.g)
        with self.assertRaises(RuntimeError):self.r.get(self.g)
    def test_missing(self):
        with self.assertRaises(RuntimeError):self.r.get(self.g)
    def test_unsealed(self):
        self.r.begin(self.g,(1,7))
        with self.assertRaises(RuntimeError):self.r.get(self.g)
    def test_unknown_append(self):
        with self.assertRaises(RuntimeError):self.r.append((1,7),self.t,(1,))
    def test_append_after_end(self):
        self.record()
        with self.assertRaises(RuntimeError):self.r.append((1,7),self.t,(1,))
    def test_handle_changed(self):
        self.record();self.g.raw=8
        with self.assertRaises(RuntimeError):self.r.get(self.g)
    def test_transport_changed(self):
        self.record();self.t.handle=12
        with self.assertRaises(RuntimeError):self.r.get(self.g)
    def test_unavailable(self):
        self.record();self.t.valid=False
        with self.assertRaises(RuntimeError):self.r.get(self.g)
    def test_real_count_difference_preserved(self):
        a=self.record();b=self.record(Graph(),(2,7),rows=2)
        self.assertNotEqual(a.rows,b.rows)
    def test_hooks(self):
        current=[(3,7)]
        class HookGraph(Graph):
            def capture_begin(self):pass
            def capture_end(self):pass
            def reset(self):pass
        install_hooks(HookGraph,lambda:current[0])
        g=HookGraph();g.capture_begin();REGISTRY.append(current[0],self.t,(1,));g.capture_end()
        self.assertEqual(REGISTRY.get(g).rows,[(1,)])
        g.reset()
        with self.assertRaises(RuntimeError):REGISTRY.get(g)
        current[0]=(4,7);g.capture_begin();g.capture_end()
        self.assertEqual(REGISTRY.get(g).rows,[])
    def test_failed_end_invalidates(self):
        class Bad(Graph):
            def capture_begin(self):pass
            def capture_end(self):raise RuntimeError('CUDA failed')
            def reset(self):pass
        install_hooks(Bad,lambda:(5,7));g=Bad();g.capture_begin()
        with self.assertRaises(RuntimeError):g.capture_end()
        with self.assertRaises(RuntimeError):REGISTRY.get(g)

if __name__=='__main__':unittest.main(verbosity=2)
