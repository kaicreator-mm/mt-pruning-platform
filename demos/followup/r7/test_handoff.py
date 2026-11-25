import tempfile, unittest, json
from pathlib import Path
from handoff import DIRS, SPLITS, sha, corpus, evaluate, digits


class HandoffTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup); self.p=Path(t.name)
        self.m={'corpus':'FLORES-200','directions':list(DIRS),'sizes_per_direction':dict.fromkeys(SPLITS,1),'split_sizes':{'dev':100,'devtest':100},'indices':{l:{'calibration':[1],'validation':[2],'test':[3]} for l in ('zh','en','my')},'records':18}
        self.context=[]
        for s in SPLITS:
            for d in DIRS:
                src,tgt=d.split('-'); i=self.m['indices'][src][s][0]
                self.context.append(dict(id=f'{s}:{d}:{i}',split=s,direction=d,src_lang=src,tgt_lang=tgt,flores_split='dev' if s=='calibration' else 'devtest',line=i,source=f'{src}-{s}',reference=f'{tgt}-{s}',prompt_ids=[1,2]))
        self.freeze(); self.base=self.runfile('b',128); self.cand=self.runfile('c',64)
    def write(self,name,rows):
        p=self.p/name; p.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8'); return p
    def freeze(self):
        self.cp=self.write('corpus',self.context); self.m['corpus_sha256']=sha(self.cp); self.mp=self.p/'manifest'; self.mp.write_text(json.dumps(self.m)); self.msha=sha(self.mp)
    def runfile(self,name,keep,split='validation',bad=False):
        h=dict(type='run',corpus_sha256=self.m['corpus_sha256'],num_experts=keep,masked_keep=None,splits=[split],max_new_tokens=10,decoding='greedy')
        rs=[dict(type='sentence',id=c['id'],split=c['split'],direction=c['direction'],line=c['line'],prompt_tokens=2,new_tokens=1,tokens=[4],finish='eos',text='wrong' if bad else c['reference']) for c in self.context if c['split']==split]
        return self.write(name,[h]+rs)
    def calc(self,cs=None,metric=None):
        return evaluate(self.mp,self.msha,self.cp,self.base,cs or {64:self.cand},metric or (lambda h,r:100.0*sum(a==b for a,b in zip(h,r))/len(r)))
    def alter(self,path,fn):
        rs=[json.loads(x) for x in path.read_text().splitlines()]; fn(rs); self.write(path.name,rs)
    def test_choose_smallest(self):
        a=self.runfile('a',96); self.assertEqual(self.calc({96:a,64:self.cand})['chosen_keep'],64)
    def test_fallback(self):
        a=self.runfile('a',64,bad=True); self.assertTrue(self.calc({64:a})['fallback_to_teacher'])
    def test_reorder(self):
        self.alter(self.cand,lambda r:r.__setitem__(slice(1,None),list(reversed(r[1:])))); self.assertEqual(self.calc()['chosen_keep'],64)
    def test_missing(self):
        self.alter(self.cand,lambda r:r.pop()); self.assertRaises(ValueError,self.calc)
    def test_duplicate(self):
        self.alter(self.cand,lambda r:r.append(r[-1])); self.assertRaises(ValueError,self.calc)
    def test_wrong_direction(self):
        self.alter(self.cand,lambda r:r[1].update(direction='en-my')); self.assertRaises(ValueError,self.calc)
    def test_wrong_model(self):
        self.alter(self.cand,lambda r:r[0].update(num_experts=96)); self.assertRaises(ValueError,self.calc)
    def test_masked(self):
        self.alter(self.cand,lambda r:r[0].update(masked_keep=64)); self.assertRaises(ValueError,self.calc)
    def test_test_header(self):
        self.alter(self.cand,lambda r:r[0]['splits'].append('test')); self.assertRaises(ValueError,self.calc)
    def test_test_row(self):
        other=self.runfile('test',64,'test'); row=json.loads(other.read_text().splitlines()[1]); self.alter(self.cand,lambda r:r.append(row)); self.assertRaises(ValueError,self.calc)
    def test_decoding(self):
        self.alter(self.cand,lambda r:r[0].update(max_new_tokens=11)); self.assertRaises(ValueError,self.calc)
    def test_empty(self):
        self.alter(self.cand,lambda r:r[1].update(text='')); self.assertTrue(self.calc(metric=lambda h,r:99.0)['fallback_to_teacher'])
    def test_bad_score(self):
        self.assertRaises(ValueError,self.calc,metric=lambda h,r:float('nan'))
    def test_manifest_hash(self):
        self.mp.write_text('{}'); self.assertRaises(ValueError,self.calc)
    def test_corpus_hash(self):
        self.cp.write_text(''); self.assertRaises(ValueError,self.calc)
    def test_parallel_leak(self):
        self.m['indices']['zh']['test']=[2]; self.freeze(); self.assertRaises(ValueError,self.calc)
    def test_text_leak(self):
        self.context[-1]['source']='my-validation'; self.freeze(); self.assertRaises(ValueError,self.calc)
    def test_parallel_alignment(self):
        self.context[1]['reference']='different'; self.freeze(); self.assertRaises(ValueError,self.calc)
    def test_digit_fold(self):
        self.assertEqual(digits('၁၂ ３'),digits('3 12'))
    def test_boundary(self):
        for value,chosen in ((89.,64),(88.99,None),(92.,64)):
            a=self.runfile('a',64,bad=True)
            self.assertEqual(self.calc({64:a},lambda h,r:90.0 if h==r else value)['chosen_keep'],chosen)
    def test_no_test_authorization(self):
        self.assertFalse(self.calc()['test_execution_authorized'])
    def test_prompt_count(self):
        self.alter(self.cand,lambda r:r[1].update(prompt_tokens=1)); self.assertRaises(ValueError,self.calc)


if __name__=='__main__': unittest.main()
