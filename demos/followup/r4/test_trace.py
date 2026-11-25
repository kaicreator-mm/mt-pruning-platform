import tempfile, unittest
from pathlib import Path
from trace import compare, stability, Journal
class TestTrace(unittest.TestCase):
    def test_mapping(self):
        self.assertEqual(compare([{'key':'a','ids':[2,4]}],[{'key':'a','ids':[0,1]}],[2,4])['mean_topk_overlap'],1)
    def test_change(self):
        self.assertEqual(compare([{'key':'a','ids':[2,3]}],[{'key':'a','ids':[0,1]}],[2,4])['mean_topk_overlap'],.5)
    def test_alignment(self):
        with self.assertRaises(ValueError): compare([{'key':'a','ids':[2]}],[{'key':'b','ids':[0]}],[2])
    def test_mapping_invalid(self):
        with self.assertRaises(ValueError): compare([{'key':'a','ids':[2]}],[{'key':'a','ids':[0]}],[2,2])
    def test_stable(self):
        x=stability([[8,4,1]]*10,[2,4],2,[1,2,3]); self.assertEqual(x[0]['mean_jaccard'],1); self.assertAlmostEqual(x[1]['mean_heldout_mass'],12/13)
    def test_stability_invalid(self):
        with self.assertRaises(ValueError): stability([[1,2]]*3,[3],1,[1,2])
    def test_bad_mass(self):
        with self.assertRaises(ValueError): stability([[1,2],[0,0],[1,2]],[1],1,[1,2])
    def test_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.db'; j=Journal(p,{'teacher':'a'}); j.put('b',{'x':2}); j.close(); j=Journal(p,{'teacher':'a'}); j.put('b',{'x':2}); j.put('a',{'x':1}); self.assertEqual(list(j.records()),[('a',{'x':1}),('b',{'x':2})]); j.close()
    def test_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(Path(d)/'db',{'teacher':'a'}); j.put('a',1)
            with self.assertRaises(ValueError): j.put('a',2)
            self.assertEqual(list(j.records()),[('a',1)]); j.close()
    def test_identity(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'db'; Journal(p,{'teacher':'a'}).close()
            with self.assertRaises(ValueError): Journal(p,{'teacher':'b'})
    def test_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            j=Journal(Path(d)/'db',{'teacher':'a'}); j.put('a',1)
            with j.db: j.db.execute("UPDATE samples SET payload='2'")
            with self.assertRaises(ValueError): list(j.records())
            j.close()
if __name__=='__main__': unittest.main()
