import json, pathlib, sys
states=list(pathlib.Path('/w9/state').glob('dsv41-*/status.json'))
if not states:sys.exit(1)
state=json.loads(max(states,key=lambda p:p.stat().st_mtime).read_text())
sys.exit(0 if state.get('phase')=='serving' and state.get('public_admission') and not state.get('poisoned') else 1)
