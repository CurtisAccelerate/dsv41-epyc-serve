"""Check the transport plan and communicator selection without loading a model."""
import os
from hybrid.r2_contract import Contract

def main():
    env=dict(DSV41_R2='1',DSV41_TRANSPORT='p2p4',VLLM_EXL3_TC_MOE_V2='1')
    plan=Contract.from_env(env)
    assert len(plan.plans)==7
    assert all(p.collective=='p2p4' and p.record()['collective']=='p2p4' for p in plan.plans)
    assert plan.plan(7).served_by==plan.plan(8).descriptor
    for override in ('2=1x512a1','8=pair'):
        try:Contract.from_env(dict(env,DSV41_R2_COLLECTIVE=override))
        except ValueError:pass
        else:raise AssertionError('Host/pair override was accepted')
    legacy=Contract.from_env(dict(DSV41_R2='1'))
    assert all(p.collective!='p2p4' for p in legacy.plans)
    from vllm.platforms import current_platform
    if current_platform.is_cuda():
        from vllm.sm80.distributed.communicator import needs_fork_communicator
        assert os.environ['VLLM_HOST_ALLREDUCE']==('1' if os.environ['DSV41_TRANSPORT']=='hostpair' else '0')
        assert needs_fork_communicator()
        assert current_platform.get_device_communicator_cls()=='vllm.sm80.distributed.communicator.Sm80CudaCommunicator'
    print('P2P4_CONTRACT_CHECK_PASSED')

if __name__=='__main__':main()
