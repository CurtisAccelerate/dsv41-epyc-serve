#!/usr/bin/env python3
"""Source and transport checks precede model loading."""
import hashlib, json, os, pathlib, subprocess, sys

ROOT=pathlib.Path('/opt/dsv41')
def check():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    failures=[]
    for item in manifest['files']:
        p=pathlib.Path(item['path'])
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            failures.append(item['path'])
    if failures:raise RuntimeError('Installed runtime differs: '+repr(failures))
    subprocess.run(['python3','/w26-preflight/preflight_launch.py','--check-only'],check=True)
    import torch, numpy, safetensors, vllm, vllm_exl3, exllamav3
    import runpy
    resolver=runpy.run_path('/w26-preflight/preflight_launch.py')['resolve']
    for name in ('w9_exl_ab.runtime','w9_exl_ab.versions','w9_exl_ab.params',
                 'w9_exl_ab.node_versions','w9_exl_ab.collective_args',
                 'w9_resident_api.launch','hybrid.r2_startup',
                 'vllm.sm80.distributed.communicator'):
        resolver(name)
    gpu_imports=bool(torch.cuda.is_available())
    if gpu_imports:
        from glm53_exl3_plugin.exl3 import Exl3Config
        import vllm.sm80.distributed.communicator
    print('DSV41_RUNTIME_CHECK '+json.dumps(dict(passed=True,files=len(manifest['files']),
          torch=torch.__version__,vllm=vllm.__version__,gpu_imports=gpu_imports,
          scope='Installed byte identities, dependency imports, GPU module origins; CUDA imports when a GPU is exposed')),flush=True)

def prepare_engram():
    from backing import hash_file
    root=pathlib.Path(os.environ['W9_ENGRAM_BACKING'])
    manifest=root/'manifest.json'
    if not manifest.is_file():
        if os.environ.get('DSV41_PREPARE_ENGRAM','1')!='1':
            raise RuntimeError('Engram generation missing; run prepare-engram or enable DSV41_PREPARE_ENGRAM')
        nodes=os.environ.get('DSV41_RANK_NODES','0 0 0 0').split()
        if len(nodes)!=4:raise RuntimeError('DSV41_RANK_NODES must contain four NUMA nodes')
        plan=pathlib.Path('/tmp/dsv41-engram-plan.json')
        subprocess.run(['python3','/w9/engram-loader/make_plan_cli.py',
            '--index',os.environ['STAR_DSV41_SELECTED_INDEX'],
            '--config',os.environ['W9_ENGRAM_CONFIG'],
            '--native-source','/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py',
            '--output',str(plan),'--rank-nodes',*nodes],check=True)
        subprocess.run(['python3','/w9/engram-loader/numa_builder.py','--plan',str(plan),
                        '--root',str(root),'--prepare'],check=True)
    actual=hash_file(manifest)
    expected=os.environ.get('W9_ENGRAM_MANIFEST_SHA256')
    if expected and expected!=actual:raise RuntimeError('Engram manifest differs from explicit pin')
    os.environ['W9_ENGRAM_MANIFEST_SHA256']=actual
    print('DSV41_ENGRAM_READY '+json.dumps(dict(manifest_sha256=actual)),flush=True)

def serve(extra):
    check()
    if os.environ.get('DSV41_REQUIRE_P2P','1')=='1':
        subprocess.run(['python3',str(ROOT/'p2p_check.py')],check=True)
    prepare_engram()
    # A fresh queue per process prevents stale benchmark state or credentials.
    import tempfile
    state=tempfile.mkdtemp(prefix='dsv41-',dir='/w9/state')
    os.environ['W9_RESIDENT_API_CONTROL']=state
    os.environ['W9_EXL_AB_DIAGNOSTICS_DIR']=state
    command=json.loads((ROOT/'serve-command.json').read_text())
    if os.environ.get('DSV41_NUMA_BIND','1')=='1':
        nodes=os.environ.get('DSV41_RANK_NODES','0 0 0 0').split()
        position=command.index('--pipeline-parallel-size')
        command[position:position]=['--numa-bind','--numa-bind-nodes',*nodes]
    command+=extra
    print('DSV41_SERVE_START '+json.dumps(dict(command=command,p2p=os.environ['NCCL_P2P_DISABLE'],
        p2p_level=os.environ['NCCL_P2P_LEVEL'],runner=os.environ['VLLM_USE_V2_MODEL_RUNNER'],
        required_bank='next-attention-sampler-native-mixed',state=state)),flush=True)
    os.execvp(command[0],command)

def main():
    for key,value in json.loads((ROOT/'defaults.json').read_text()).items():
        os.environ.setdefault(key,value)
    args=sys.argv[1:];cmd=args.pop(0) if args else 'serve'
    if cmd=='check':check()
    elif cmd=='p2p-check':os.execvp('python3',['python3',str(ROOT/'p2p_check.py')])
    elif cmd=='prepare-engram':prepare_engram()
    elif cmd=='serve':serve(args)
    elif cmd=='verify-serving':os.execvp('python3',['python3',str(ROOT/'verify_serving.py'),*args])
    else:raise SystemExit('usage: serve [vLLM arguments] | check | p2p-check | prepare-engram | verify-serving')
if __name__=='__main__':main()
