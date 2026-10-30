"""Verify actual serving and all optimized bank preparations in the running container."""
import json,pathlib,subprocess,sys
from hybrid.optimized_receipts import require_optimized

def main():
    files=list(pathlib.Path('/w9/state').glob('dsv41-*/status.json'))
    if not files:raise RuntimeError('No serving owner status')
    path=max(files,key=lambda p:p.stat().st_mtime)
    state=json.loads(path.read_text())
    if state.get('phase')!='serving' or state.get('public_admission') is not True or state.get('poisoned'):
        raise RuntimeError('Serving is not admitted')
    result=require_optimized(json.loads(path.with_name('attention-initial.json').read_text()))
    subprocess.run([sys.executable,'/opt/dsv41/entrypoint.py','verify-serving',*sys.argv[1:]],check=True)
    print('DSV41_OPTIMIZED_PATH_VERIFIED '+json.dumps(result),flush=True)
if __name__=='__main__':main()
