import copy,json,pathlib,sys,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from image_identity import verify_base

class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.expected=json.loads(pathlib.Path(__file__).with_name('base-identity.json').read_text())
    def verify(self,data):
        with patch('image_identity.subprocess.check_output',return_value=json.dumps([data])):
            return verify_base('fixture','sha256:pinned-config')
    def test_classic(self):self.assertEqual(self.verify(dict(Id='sha256:pinned-config')),'config-digest')
    def test_containerd(self):
        self.assertEqual(self.verify(dict(self.expected,Id='sha256:oci-manifest')),'exact-rootfs-and-runtime-config')
    def test_wrong_layer(self):
        changed=copy.deepcopy(self.expected);changed['RootFS']['Layers'][0]='sha256:wrong';changed['Id']='sha256:oci'
        with self.assertRaises(RuntimeError):self.verify(changed)
    def test_changed_environment(self):
        changed=copy.deepcopy(self.expected);changed['Config']['Env'].append('VLLM_HOST_ALLREDUCE=1');changed['Id']='sha256:oci'
        with self.assertRaises(RuntimeError):self.verify(changed)
    def test_changed_entrypoint(self):
        changed=copy.deepcopy(self.expected);changed['Config']['Entrypoint']=['wrong'];changed['Id']='sha256:oci'
        with self.assertRaises(RuntimeError):self.verify(changed)

if __name__=='__main__':unittest.main(verbosity=2)
