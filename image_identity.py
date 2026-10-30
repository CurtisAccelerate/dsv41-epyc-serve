"""Verify base identity across classic and containerd Docker image stores."""
import json,pathlib,subprocess

def verify_base(tag,expected_id):
    actual=json.loads(subprocess.check_output(['docker','image','inspect',tag],text=True))[0]
    if actual['Id']==expected_id:return 'config-digest'
    # Docker 29's containerd store may expose the OCI manifest digest as Id.
    # Require identical content-addressed rootfs AND every runtime-config field.
    expected=json.loads(pathlib.Path(__file__).with_name('p2p4').joinpath('base-identity.json').read_text())
    for key in ('RootFS','Config','Architecture','Os'):
        if actual.get(key)!=expected[key]:raise RuntimeError('Base image '+key+' differs from pinned original')
    return 'exact-rootfs-and-runtime-config'
