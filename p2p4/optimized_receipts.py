"""Strict read-only audit of the complete optimized startup receipts."""
MIXED='next-attention-sampler-native-mixed'

def require_optimized(initial):
    rows=initial.get('rows',[])
    if len(rows)!=4 or sorted(r.get('rank',-1) for r in rows)!=list(range(4)):
        raise RuntimeError('Four distinct initial worker receipts required')
    contracts=[]
    for row in rows:
        rank=row['rank']
        def require(ok,reason):
            if not ok:raise RuntimeError(f'Rank {rank}: '+reason)
        require(row.get('ok') is True and row.get('ready') is True,'initial worker not ready')
        require(row.get('available_arms',{}).get(MIXED) is True,'optimized mixed arm unavailable')
        sampler=row.get('sampler_preparation') or {}
        require(all(sampler.get(k) is True for k in ('ready','whole_call_exact','all_rank_ready')),'B1 sampler not qualified')
        r2=row.get('r2_preparation') or {}
        require(r2.get('ready') is True,'R2 not ready')
        graphs=r2.get('graphs',{})
        require(set(graphs)=={'2','3','4','5','6','8'},'R2 capture ladder incomplete')
        for b,graph in graphs.items():
            require(graph.get('published') is True and set(graph.get('components',[]))=={'attention','experts','collective'}
                    and not graph.get('excluded'),'R2 B'+b+' has an unpublished or excluded component')
        samplers=r2.get('samplers',{})
        require(set(samplers)=={str(b) for b in range(2,9)},'R2 sampler ladder incomplete')
        for b,states in samplers.items():
            require(states.get('sampler',{}).get('ready') is True and states.get('sampler',{}).get('whole_call_bitwise') is True,'R2 B'+b+' greedy sampler not exact/ready')
            require(states.get('sampled',{}).get('ready') is True and states.get('sampled',{}).get('chain_bitwise_vs_eager') is True,'R2 B'+b+' sampled sampler not exact/ready')
        contract=r2.get('contract',{})
        plans={p['b']:p for p in contract.get('plans',[])}
        require(set(plans)==set(range(2,9)) and contract.get('pad7') is True,'R2 batch plan incomplete')
        require(plans[7].get('served_by')==plans[8].get('served_by'),'B7 does not use qualified B8 graph')
        contracts.append(r2.get('contract_sha256'))
    if any(not isinstance(c,str) or len(c)!=64 for c in contracts) or len(set(contracts))!=1:
        raise RuntimeError('R2 contract identity differs across ranks')
    return dict(passed=True,ranks=4,mixed_arm=MIXED,all_r2_components=True,
                captured_batches=[2,3,4,5,6,8],padded_batch=7,samplers=list(range(1,9)),
                scope='Verified optimized-bank preparation; runtime request hits still require traces')
