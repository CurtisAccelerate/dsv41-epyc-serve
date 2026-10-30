"""Retain and validate TP ownership; NCCL protocol bytes are never journaled."""
import os

class PeerGuard:
    def __init__(self,runner):
        from . import p2p4_transport as peer
        from vllm.distributed import get_tp_group
        from w9_exl_ab import collective_profiles as CP
        self.peer,self.get_group,self.cp=peer,get_tp_group,CP
        self.group=get_tp_group();self.device_comm=self.group.device_communicator
        self.transport=peer.require();self.comm=self.transport.comm
        self.binding=self.transport.binding();self.device=str(runner.device)
        self.check()
    def check(self):
        t=self.peer.require();g=self.get_group();dc=g.device_communicator
        if (t is not self.transport or g is not self.group or dc is not self.device_comm
            or dc.pynccl_comm is not self.comm or t.binding()!=self.binding
            or g.world_size!=4 or str(self.comm.device)!=self.device
            or not getattr(dc,'_p2p4',False) or dc.hostshm_comm is not None or self.cp.REG
            or os.environ.get('NCCL_SHM_DISABLE')!='1'):
            raise RuntimeError('P2P journal transport ownership changed or host state exists')
        return True
