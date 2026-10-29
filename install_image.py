#!/usr/bin/env python3
"""Download, verify and load the complete optimized image from GitHub Releases."""
import argparse, hashlib, json, pathlib, subprocess, sys, time, urllib.request

RELEASE='https://github.com/CurtisAccelerate/dsv41-epyc-serve/releases/download/optimized-20261029-p2p'
EXPECTED_IMAGE='sha256:e3bc13c775dee4013f5cfe28688482e0ec8ccb9b045ccb012c9f6e53b56c2726'
EXPECTED_ARCHIVE='30d2ded393da7802df029a5f6d525e77189cd30578d980ad87e8c50d1f928913'
TAG='dsv41-optimized:20261029-p2p'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def download(name,destination,expected=None,size=None):
    if destination.exists() and expected and destination.stat().st_size==size and digest(destination)==expected:
        print('Verified cached '+name,flush=True);return
    partial=destination.with_suffix(destination.suffix+'.partial')
    for attempt in range(3):
        try:
            request=urllib.request.Request(RELEASE+'/'+name,headers={'User-Agent':'dsv41-image-installer'})
            with urllib.request.urlopen(request,timeout=120) as response,partial.open('wb') as output:
                h=hashlib.sha256();total=0
                for b in iter(lambda:response.read(8*1024**2),b''):
                    output.write(b);h.update(b);total+=len(b)
            if expected and (h.hexdigest()!=expected or total!=size):raise RuntimeError('Checksum/size mismatch: '+name)
            partial.replace(destination);print('Downloaded and verified '+name,flush=True);return
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=pathlib.Path,default=pathlib.Path('dsv41-download'))
    p.add_argument('--download-only',action='store_true')
    a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
    manifest_path=a.directory/'image-parts.json';download('image-parts.json',manifest_path)
    manifest=json.loads(manifest_path.read_text())
    if manifest['image_id']!=EXPECTED_IMAGE or manifest['archive_sha256']!=EXPECTED_ARCHIVE:
        raise RuntimeError('Release manifest does not identify the pinned optimized image')
    paths=[];combined=hashlib.sha256()
    for item in manifest['parts']:
        name=item['name']
        if pathlib.PurePosixPath(name).name!=name or '/' in name or '\\' in name:raise RuntimeError('Unsafe asset name')
        path=a.directory/name;download(name,path,item['sha256'],item['size_bytes']);paths.append(path)
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):combined.update(b)
    if combined.hexdigest()!=EXPECTED_ARCHIVE:raise RuntimeError('Combined Docker archive checksum differs')
    kit=manifest['kit']
    if kit['name']!='dsv41-optimized-kit.zip':raise RuntimeError('Unexpected source kit name')
    download(kit['name'],a.directory/kit['name'],kit['sha256'],kit['size_bytes'])
    if a.download_only:
        print('All image parts and source kit verified. Rerun without --download-only to load.');return
    print('Loading verified image; allow at least 50 GB of free Docker storage.',flush=True)
    process=subprocess.Popen(['docker','load'],stdin=subprocess.PIPE)
    try:
        for path in paths:
            with path.open('rb') as f:
                for b in iter(lambda:f.read(8*1024**2),b''):process.stdin.write(b)
        process.stdin.close()
    except BaseException:
        process.kill();process.wait();raise
    if process.wait():raise RuntimeError('docker load failed')
    actual=subprocess.check_output(['docker','image','inspect',TAG,'--format','{{.Id}}'],text=True).strip()
    if actual!=EXPECTED_IMAGE:raise RuntimeError('Loaded image identity differs: '+actual)
    subprocess.run(['docker','run','--rm','--network','none',TAG,'check'],check=True)
    print('Verified complete optimized image: '+TAG)
    print('Unzip dsv41-optimized-kit.zip, set EXL3_DIR, FP8_DIR and ENGRAM_DIR, then bash run.sh.')

if __name__=='__main__':main()
