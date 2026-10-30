#!/usr/bin/env python3
"""Install the complete serving base and build its pinned EPYC TP4 transport."""
import argparse, hashlib, pathlib, subprocess, sys
from image_identity import verify_base

BASE='dsv41-optimized:20261029-p2p'
BASE_ID='sha256:e3bc13c775dee4013f5cfe28688482e0ec8ccb9b045ccb012c9f6e53b56c2726'
MANIFESTS={"hostpair":"12f432e447ea4dd9dfe706ab6b5d4befd1c6ad261624b0663019f61683bf5f98","p2p4":"9494693734f79db67bbfecc8ff6312372c18fe3facaade67eea513263082fc6c"}

def inspect(image,format):
    p=subprocess.run(['docker','image','inspect',image,'--format',format],text=True,capture_output=True)
    return p.stdout.strip() if p.returncode==0 else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='dsv41-download',help='Base image download/cache directory')
    p.add_argument('--transport',choices=('both','hostpair','p2p4'),default='both')
    a=p.parse_args();here=pathlib.Path(__file__).resolve().parent
    actual=inspect(BASE,'{{.Id}}')
    if actual is None:
        subprocess.run([sys.executable,str(here/'install_image.py'),'--directory',a.directory],check=True)
        actual=inspect(BASE,'{{.Id}}')
    print('Verified base identity: '+verify_base(BASE,BASE_ID))
    for mode in ('hostpair','p2p4') if a.transport=='both' else (a.transport,):
        tag='dsv41-optimized:20261030-transport-v4-'+mode
        subprocess.run(['docker','build','--network','none','--target',mode,'-t',tag,str(here/'p2p4')],check=True)
        label='hostpair' if mode=='hostpair' else 'nccl-p2p4'
        if inspect(tag,'{{index .Config.Labels "io.dsv41.transport"}}')!=label:
            raise RuntimeError('Image transport label differs')
        raw=subprocess.check_output(['docker','run','--rm','--network','none','--entrypoint','cat',tag,
                                     '/opt/dsv41/manifest.json'])
        if hashlib.sha256(raw).hexdigest()!=MANIFESTS[mode]:raise RuntimeError('Installed source manifest differs')
        subprocess.run(['docker','run','--rm','--network','none',tag,'check'],check=True)
        subprocess.run(['docker','run','--rm','--network','none','--entrypoint','python3',tag,
            '/opt/dsv41/p2p4-source/check_startup.py'],check=True)
        print('Complete serving image installed: '+tag)
    subprocess.run([sys.executable,str(here/'check_network.py'),'--image',tag],check=True)
    print('Next run the four-GPU qualification in p2p4/README.md, then bash p2p4/run.sh.')

if __name__=='__main__':main()
