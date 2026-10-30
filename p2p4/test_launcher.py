import json,os,pathlib,subprocess,tempfile,unittest

@unittest.skipUnless(os.name=='posix','Linux launcher test runs inside serving image')
class LauncherTests(unittest.TestCase):
    def launch(self,mode,network):
        with tempfile.TemporaryDirectory() as folder:
            root=pathlib.Path(folder);docker=root/'docker'
            docker.write_text('''#!/usr/bin/env bash
if [ "$1" = image ]; then
  if [[ "$*" == *startup-contract* ]]; then echo transport-v4
  elif [ "$DSV41_TRANSPORT" = hostpair ]; then echo hostpair
  else echo nccl-p2p4; fi
else python3 -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "$@"; fi
''');docker.chmod(0o755)
            env=dict(os.environ,PATH=folder+':'+os.environ['PATH'],EXL3_DIR=folder,FP8_DIR=folder,ENGRAM_DIR=folder,
                     DSV41_TRANSPORT=mode,DSV41_GPUS='device=0,1,2,3',DSV41_PORT='18395')
            if network is not None:env['DSV41_NETWORK']=network
            else:env.pop('DSV41_NETWORK',None)
            return subprocess.run(['bash',str(pathlib.Path(__file__).with_name('run.sh'))],env=env,text=True,capture_output=True)
    def test_host_default_and_modes(self):
        for mode in ('hostpair','p2p4'):
            p=self.launch(mode,None);self.assertEqual(p.returncode,0,p.stderr);args=json.loads(p.stdout)
            self.assertIn('--network',args);self.assertNotIn('-p',args)
            self.assertEqual(args[args.index('--network')+1],'host')
            self.assertEqual(args[args.index('--host')+1],'127.0.0.1')
            self.assertEqual(args[args.index('--port')+1],'18395')
    def test_bridge_explicit(self):
        p=self.launch('hostpair','bridge');self.assertEqual(p.returncode,0,p.stderr);args=json.loads(p.stdout)
        self.assertEqual(args[args.index('-p')+1],'127.0.0.1:18395:8095')
        self.assertEqual(args[args.index('--host')+1],'0.0.0.0')
        self.assertEqual(args[args.index('--port')+1],'8095')
    def test_invalid_network(self):self.assertNotEqual(self.launch('hostpair','invalid').returncode,0)
if __name__=='__main__':unittest.main(verbosity=2)
