import tempfile, unittest
from pathlib import Path
from cache_plan import Cache, FIELDS, key, disk_plan

class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.p=Path(self.tmp.name); self.a=self.p/'raw.json'; self.a.write_text('evidence'); self.c=Cache(self.p/'cache.db'); self.addCleanup(self.c.close)
        self.i=dict(dict.fromkeys(FIELDS,'a'*64),split='validation'); self.r={'status':'COMPLETE','split':'validation','score':50}
    def test_hit_reopen(self):
        self.c.put(self.i,self.r,self.a); self.c.put(self.i,self.r,self.a)
        other=Cache(self.p/'cache.db'); self.addCleanup(other.close); self.assertEqual(other.get(self.i,self.a),self.r)
    def test_identity_change(self):
        self.c.put(self.i,self.r,self.a)
        for f in FIELDS: self.assertIsNone(self.c.get(dict(self.i,**{f:'b'*64}),self.a))
    def test_split(self):
        self.c.put(self.i,self.r,self.a); self.assertIsNone(self.c.get(dict(self.i,split='test'),self.a))
        with self.assertRaises(ValueError): self.c.put(self.i,dict(self.r,split='test'),self.a)
    def test_missing_field(self):
        with self.assertRaises(ValueError): key({'teacher':'a'*64})
    def test_partial(self):
        with self.assertRaises(ValueError): self.c.put(self.i,dict(self.r,status='BLOCKED'),self.a)
    def test_changed_artifact(self):
        self.c.put(self.i,self.r,self.a); self.a.write_text('changed')
        with self.assertRaises(ValueError): self.c.get(self.i,self.a)
    def test_missing_artifact(self):
        self.c.put(self.i,self.r,self.a); self.a.unlink()
        with self.assertRaises(ValueError): self.c.get(self.i,self.a)
    def test_conflict(self):
        self.c.put(self.i,self.r,self.a)
        with self.assertRaises(ValueError): self.c.put(self.i,dict(self.r,score=49),self.a)
        self.assertEqual(self.c.get(self.i,self.a),self.r)
    def test_corrupt_receipt(self):
        self.c.put(self.i,self.r,self.a)
        with self.c.db: self.c.db.execute("UPDATE done SET payload='{}'")
        with self.assertRaises(ValueError): self.c.get(self.i,self.a)

class DiskTests(unittest.TestCase):
    def test_budget(self):
        r=disk_plan(1000,200,[5,5],[80,80],8,{'a':[60,60],'b':[40,40]},850,50)
        self.assertEqual(r['all_candidate_payloads'],1400); self.assertEqual(r['serial_extra_peak_estimate'],850); self.assertTrue(r['serial_fits_estimate'])
    def test_over_budget(self):
        self.assertFalse(disk_plan(1000,200,[5,5],[80,80],8,{'a':[60,60]},849,50)['serial_fits_estimate'])
    def test_invalid(self):
        with self.assertRaises(ValueError): disk_plan(1000,200,[5,5],[80,80],8,{'a':[100,100]},1000)
    def test_layer_limit(self):
        with self.assertRaises(ValueError): disk_plan(1000,200,[5,5],[80,80],8,{'a':[90,10]},1000)
    def test_topk(self):
        with self.assertRaises(ValueError): disk_plan(1000,200,[5,5],[80,80],8,{'a':[7,10]},1000)
    def test_shape(self):
        with self.assertRaises(ValueError): disk_plan(1000,200,[5,5],[80,80],8,{'a':[60]},1000)
if __name__=='__main__': unittest.main()
