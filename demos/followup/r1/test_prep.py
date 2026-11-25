import unittest
from prep import *

class DemoTests(unittest.TestCase):
    def rows(self,prefix,n):
        return [{'id':str(i),**{l:f'{prefix}-{l}-{i}' for l in LANGS}} for i in range(n)]
    def test_split(self):
        a,b=self.rows('a',30),self.rows('b',45); x=split_data(a,b); self.assertEqual(x,split_data(a,b)); self.assertEqual([len(v) for v in x.values()],[144,120,120]); sets=[{r['group_id'] for r in v} for v in x.values()]; self.assertFalse(any(a&b for a,b in itertools.combinations(sets,2)))
    def test_cross_split_duplicate(self):
        with self.assertRaises(ValueError): split_data(self.rows('a',30),self.rows('a',45))
    def test_alignment(self):
        with self.assertRaises(ValueError): align({l:[{'id':l,'split':'dev','text':l}] for l in LANGS},'dev')
    def profile(self): return {'layer.1':{d:{p:[1+i if d=='zh-my' else 128-i for i in range(128)] for p in ('prefill','decode')} for d in DIRS}}
    def test_plan(self):
        for m in ('balanced','worst_first','random'):
            a=plan(self.profile(),64,m); self.assertEqual(a,plan(self.profile(),64,m)); self.assertEqual(len(set(a['layers']['layer.1']['keep_ids'])),64)
        self.assertAlmostEqual(min(plan(self.profile(),128)['layers']['layer.1']['retained_mass'].values()),1.)
    def test_invalid_profile(self):
        a=self.profile(); del a['layer.1']['zh-my']; self.assertRaises(ValueError,plan,a,64)
    def test_budget(self): self.assertRaises(ValueError,plan,self.profile(),7)
    def test_number(self): self.assertEqual(numbers('၁၂ ၃'),numbers('12 3'))
    def test_gate(self):
        rows=[{'id':d,'direction':d,'reference':'ref 12','source':'12'} for d in DIRS]; b={d:{'text':'ref 12','truncated':False} for d in DIRS}; c={**b,'zh-my':{'text':'','truncated':False}}
        # Injected fake scoring only tests gate logic, not sacrebleu.
        score=lambda h,r:100 if h==r else 0
        self.assertEqual(evaluate(rows,b,b,score)['status'],'PILOT_GATE_PASS'); self.assertEqual(evaluate(rows,b,c,score)['status'],'PILOT_GATE_FAIL'); self.assertRaises(ValueError,evaluate,rows,b,{},score)
if __name__=='__main__': unittest.main()
