import unittest
from bridge import DIRS, PHASES
from prepare_candidates import candidates

class Plan(unittest.TestCase):
    def setUp(self):
        self.a={'status':'AGGREGATE_COVERAGE_CHECKED_NOT_PROVENANCE','expert_count':4,'top_k':2,'hashes':{'corpus':'abc'},'profiles':{'1':{d:{p:[4.,3.,2.,1.] for p in PHASES} for d in DIRS},'2':{d:{p:[1.,2.,3.,4.] for p in PHASES} for d in DIRS}}}
    def test_export(self):
        r=candidates(self.a,[2,4]); self.assertEqual(r['candidates']['2']['kept'],{'1':[0,1],'2':[2,3]}); self.assertEqual(r['candidates']['4']['kept']['1'],list(range(4))); self.assertFalse(r['quality_verified'])
    def test_worst(self): self.assertEqual(candidates(self.a,[2],'worst_first')['candidates']['2']['kept']['2'],[2,3])
    def test_random(self): self.assertEqual(candidates(self.a,[2],'random'),candidates(self.a,[2],'random'))
    def test_invalid(self):
        for k in ([1],[5],[2,2],[]):
            with self.assertRaises(ValueError): candidates(self.a,k)
    def test_missing_group(self):
        del self.a['profiles']['1']['zh-my']
        with self.assertRaises(ValueError): candidates(self.a,[2])
if __name__=='__main__': unittest.main()
