"""Validate links a transport actually uses; extra peer capability is harmless."""
def validate_runtime(env):
    mode=env.get('DSV41_TRANSPORT')
    if mode not in ('hostpair','p2p4'):raise ValueError('Unknown transport mode')
    expected=dict(VLLM_HOST_ALLREDUCE='1' if mode=='hostpair' else '0',
        W15_PAIR_TRANSPORT='1' if mode=='hostpair' else '0',
        NCCL_SHM_DISABLE='0' if mode=='hostpair' else '1',NCCL_P2P_DISABLE='0',NCCL_P2P_LEVEL='SYS')
    if any(env.get(k)!=v for k,v in expected.items()):raise ValueError('Mixed or incompatible transport flags')
    if mode=='hostpair' and env.get('VLLM_CUSTOM_ALLREDUCE_PCIE_P2P','0')!='0':
        raise ValueError('Host transport may not stand aside for another custom backend')
    if mode=='p2p4' and env.get('DSV41_P2P_PAYLOAD_DTYPE','native') not in ('native','fp32'):
        raise ValueError('Invalid P2P payload policy')
    return mode

def validate_peer_row(rank,row):
    if rank not in range(4) or len(row)!=4 or any(type(x) is not bool for x in row):
        raise ValueError('Expected a four-rank boolean peer row')
    if not row[rank] or not row[rank^1]:
        raise ValueError('Required pair peer link unavailable')

def validate_topology(peers,transport):
    if len(peers)!=4:raise ValueError('Exactly four peer rows required')
    for rank,row in enumerate(peers):validate_peer_row(rank,row)
    if transport=='p2p4':
        if not all(all(row) for row in peers):raise ValueError('Full P2P requires all directed peer links')
    elif transport!='hostpair':raise ValueError('Unknown transport mode')
    return True
