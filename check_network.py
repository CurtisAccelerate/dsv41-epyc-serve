"""CPU-only host-network HTTP smoke test before reserving GPUs."""
import argparse,json,socket,subprocess,time,urllib.request,uuid

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',default='dsv41-optimized:20261030-transport-v4-hostpair')
    a=p.parse_args();name='dsv41-http-proof-'+uuid.uuid4().hex
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    process=None
    try:
        process=subprocess.Popen(['docker','run','--name',name,'--network','host','--runtime','runc',
            '-e','NVIDIA_VISIBLE_DEVICES=void','-e','CUDA_VISIBLE_DEVICES=',
            '--entrypoint','python3',a.image,'-m','http.server',str(port),'--bind','127.0.0.1'],
            stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
        for _ in range(40):
            if process.poll() is not None:raise RuntimeError('CPU HTTP container exited: '+process.stderr.read())
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/',timeout=1) as response:
                    if response.status!=200:raise RuntimeError('Unexpected HTTP status')
                print('DSV41_NETWORK_VERIFIED '+json.dumps(dict(passed=True,network='host',http_status=200,gpu_used=False)))
                return
            except OSError:time.sleep(.25)
        raise RuntimeError('Host-network HTTP endpoint unreachable; do not start model benchmark')
    finally:
        subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        if process is not None:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.terminate();process.wait()
if __name__=='__main__':main()
