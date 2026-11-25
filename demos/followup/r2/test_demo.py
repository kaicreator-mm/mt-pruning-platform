import copy,json,struct,tempfile,unittest
from pathlib import Path
from preflight import scan,header
from reuse import compose

def shard(p,tensors,data):
    h=json.dumps(tensors).encode(); p.write_bytes(struct.pack('<Q',len(h))+h+data)
class Preflight(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.p=Path(self.tmp.name)
        (self.p/'config.json').write_text('{}'); self.f=self.p/'model-00001.safetensors'
        (self.p/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':8},'weight_map':{'x':self.f.name}}))
        self.t={'x':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}
    def test_missing(self): self.assertEqual(scan(self.p)['missing_shards'],[self.f.name])
    def test_partial(self):
        shard(self.f,self.t,b'0000'); self.assertEqual(scan(self.p)['incomplete_shards'],[self.f.name])
    def test_complete(self):
        shard(self.f,self.t,b'00000000'); r=scan(self.p); self.assertEqual(r['inventory_bytes'],8); self.assertFalse(r['payload_hash_verified']); self.assertTrue(r['status'].startswith('HEADER_INVENTORY_COMPLETE'))
    def test_shape(self):
        self.t['x']['shape']=[3]; shard(self.f,self.t,b'00000000'); self.assertTrue(scan(self.p)['errors'])
    def test_index(self):
        shard(self.f,{'y':self.t['x']},b'00000000'); self.assertTrue(scan(self.p)['errors'])
    def test_trailing(self):
        shard(self.f,self.t,b'000000000'); self.assertTrue(scan(self.p)['errors'])
    def test_path(self):
        (self.p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'x':'../bad.safetensors'}})); self.assertRaises(ValueError,scan,self.p)
class Reuse(unittest.TestCase):
    def setUp(self):
        import itertools
        self.key={k:'a'*(40 if k=='model_revision' else 64) for k in ('model_revision','tokenizer_sha256','prompt_sha256','router_sha256','collector_sha256','protocol_sha256')}
        self.base={d:{'identity':copy.deepcopy(self.key),'direction':d,'role':'original_teacher_calibration','corpus_sha256':'b'*64,'layers':{'1':{'prefill':[5,2,1,0],'decode':[4,3,1,0]}}} for a,b in itertools.permutations(['zh','en','my','th'],2) for d in [a+'-'+b]}
    def rs(self,third): return [v for k,v in self.base.items() if set(k.split('-'))<={'zh','en',third}]
    def test_reuse(self):
        a=compose(self.rs('my'),['zh','en','my'],2,1); b=compose(self.rs('th'),['zh','en','th'],2,1); self.assertEqual(a['source_records']['zh-en'],b['source_records']['zh-en']); self.assertEqual(a['layers']['1']['keep_ids'],[0,1]); self.assertFalse(a['quality_verified'])
    def test_missing(self): self.assertRaises(ValueError,compose,self.rs('my')[:-1],['zh','en','my'],2,1)
    def test_identity(self):
        r=self.rs('my'); r[0]['identity']['router_sha256']='c'*64; self.assertRaises(ValueError,compose,r,['zh','en','my'],2,1)
    def test_candidate(self):
        r=self.rs('my'); r[0]['role']='pruned_candidate'; self.assertRaises(ValueError,compose,r,['zh','en','my'],2,1)
    def test_invalid(self):
        r=self.rs('my'); r[0]['layers']['1']['decode'][0]=-1; self.assertRaises(ValueError,compose,r,['zh','en','my'],2,1)
if __name__=='__main__': unittest.main()
