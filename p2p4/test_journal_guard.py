"""Ownership validation regressions; exercise the actual guard without GPU access."""
import importlib,os,pathlib,sys,types,unittest
from unittest.mock import patch

package=types.ModuleType('guard_fixture');package.__path__=[str(pathlib.Path(__file__).parent)]
sys.modules['guard_fixture']=package
peer=importlib.import_module('guard_fixture.p2p4_transport')
PeerGuard=importlib.import_module('guard_fixture.p2p4_journal').PeerGuard
N=types.SimpleNamespace

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.comm=N(device='cuda:0');self.t=N(comm=self.comm,world=4,handle=17)
        self.t.binding=lambda:(id(self.comm),self.t.handle,str(self.comm.device),self.t.world)
        self.t.require=lambda:None
        self.dc=N(pynccl_comm=self.comm,_p2p4=True,hostshm_comm=None)
        self.group=N(world_size=4,device_communicator=self.dc);self.cp=N(REG=[])
        distributed=N(get_tp_group=lambda:self.group)
        self.patches=[patch.dict(sys.modules,{'vllm.distributed':distributed,'w9_exl_ab':N(collective_profiles=self.cp)}),
            patch.dict(os.environ,DSV41_TRANSPORT='p2p4',NCCL_SHM_DISABLE='1'),patch.object(peer,'TRANSPORT',self.t)]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.guard=PeerGuard(N(device='cuda:0'))
    def rejected(self,fn):
        fn()
        with self.assertRaises(RuntimeError):self.guard.check()
    def test_valid(self):self.assertTrue(self.guard.check())
    def test_missing_transport(self):self.rejected(lambda:setattr(peer,'TRANSPORT',None))
    def test_wrong_world(self):self.rejected(lambda:setattr(self.t,'world',2))
    def test_group_replaced(self):self.rejected(lambda:setattr(self,'group',N(world_size=4,device_communicator=self.dc)))
    def test_device_comm_replaced(self):self.rejected(lambda:setattr(self.group,'device_communicator',N(pynccl_comm=self.comm)))
    def test_nccl_replaced(self):self.rejected(lambda:setattr(self.dc,'pynccl_comm',object()))
    def test_nccl_handle_changed(self):self.rejected(lambda:setattr(self.t,'handle',18))
    def test_host_registry_present(self):self.rejected(lambda:self.cp.REG.append(object()))
    def test_host_comm_present(self):self.rejected(lambda:setattr(self.dc,'hostshm_comm',object()))
    def test_group_world_changed(self):self.rejected(lambda:setattr(self.group,'world_size',2))
    def test_mode_changed(self):self.rejected(lambda:os.environ.update(DSV41_TRANSPORT='other'))
    def test_shm_enabled(self):self.rejected(lambda:os.environ.update(NCCL_SHM_DISABLE='0'))
    def test_p2p_unselected(self):self.rejected(lambda:setattr(self.dc,'_p2p4',False))
    def test_wrong_device(self):self.rejected(lambda:setattr(self.comm,'device','cuda:1'))

if __name__=='__main__':unittest.main(verbosity=2)
