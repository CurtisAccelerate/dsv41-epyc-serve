#!/usr/bin/env python3
"""Install the complete serving base and build its pinned EPYC TP4 transport."""
import argparse, hashlib, pathlib, subprocess, sys

BASE='dsv41-optimized:20261029-p2p'
BASE_ID='sha256:e3bc13c775dee4013f5cfe28688482e0ec8ccb9b045ccb012c9f6e53b56c2726'
TAG='dsv41-optimized:20261029-epyc-p2p4'
MANIFEST='59d2c888070b4e71a31bf00a214b44038995ab1c377c65fa79aab67d48613ca2'

def inspect(image,format):
    p=subprocess.run(['docker','image','inspect',image,'--format',format],text=True,capture_output=True)
    return p.stdout.strip() if p.returncode==0 else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default='dsv41-download',help='Base image download/cache directory')
    a=p.parse_args();here=pathlib.Path(__file__).resolve().parent
    actual=inspect(BASE,'{{.Id}}')
    if actual is None:
        subprocess.run([sys.executable,str(here/'install_image.py'),'--directory',a.directory],check=True)
        actual=inspect(BASE,'{{.Id}}')
    if actual!=BASE_ID:raise RuntimeError('Base image identity differs; restore the pinned base with install_image.py')
    subprocess.run(['docker','build','--network','none','-t',TAG,str(here/'p2p4')],check=True)
    if inspect(TAG,'{{index .Config.Labels "io.dsv41.transport"}}')!='nccl-p2p4':
        raise RuntimeError('Four-way image transport label missing')
    raw=subprocess.check_output(['docker','run','--rm','--network','none','--entrypoint','cat',TAG,
                                 '/opt/dsv41/manifest.json'])
    if hashlib.sha256(raw).hexdigest()!=MANIFEST:raise RuntimeError('Installed source manifest differs')
    subprocess.run(['docker','run','--rm','--network','none',TAG,'check'],check=True)
    print('Complete EPYC P2P4 serving image installed: '+TAG)
    print('Next run the four-GPU qualification in p2p4/README.md, then bash p2p4/run.sh.')

if __name__=='__main__':main()
