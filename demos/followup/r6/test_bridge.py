import copy, json, tempfile, unittest
from pathlib import Path
import numpy as np
from bridge import DIRS, PHASES, adapt, quality_gate, sha

class Adapter(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup); self.p=Path(t.name)
        self.c=[{'id':d,'split':'calibration','direction':d,'line':i,'prompt_ids':[1,2,3]} for i,d in enumerate(DIRS)]
        self.corpus=self.p/'corpus.jsonl'; self.write(self.corpus,self.c)
        self.h={'type':'run','corpus_sha256':sha(self.corpus),'num_experts':4,'masked_keep':None,'splits':['calibration']}
        self.r=[{'type':'sentence','id':r['id'],'split':'calibration','direction':r['direction'],'line':r['line'],'prompt_tokens':3,'new_tokens':2,'tokens':[4,5]} for r in self.c]
        self.record=self.p/'records.jsonl'; self.stats=self.p/'stats'; self.summary={'directions':list(DIRS),'layers':[1,2],'sentences':dict.fromkeys(DIRS,1),'tokens':{d:{'prefill':3,'decode':2} for d in DIRS}}
        self.arrays={'sentences':np.ones(6,dtype=np.int64)}
        for phase,n in [('prefill',3),('decode',2)]:
            a=np.tile([n,n,0,0],(6,2,1)); self.arrays.update({f'count_{phase}':a,f'mass_{phase}':a.astype(float)*.5,f'tokens_{phase}':np.full(6,n)})
    def write(self,path,records): path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    def run_adapter(self):
        self.write(self.record,[self.h,*self.r]); (self.p/'stats.json').write_text(json.dumps(self.summary)); np.savez(self.p/'stats.npz',**self.arrays)
        return adapt(self.corpus,self.record,self.stats,corpus_sha256=sha(self.corpus),layer_ids=[1,2],n_experts=4,top_k=2,expected_n=1)
    def test_good(self):
        r=self.run_adapter(); self.assertEqual(r['profiles']['1']['zh-my']['prefill'],[1.5,1.5,0.,0.]); self.assertEqual(r['status'],'AGGREGATE_COVERAGE_CHECKED_NOT_PROVENANCE')
    def test_adapter_to_selection(self):
        from prepare_candidates import candidates
        r=candidates(self.run_adapter(),[2,4]); self.assertEqual(r['candidates']['2']['kept'],{'1':[0,1],'2':[0,1]}); self.assertFalse(r['quality_verified'])
    def test_partial_records(self):
        self.r.pop()
        with self.assertRaisesRegex(ValueError,'missing/extra'): self.run_adapter()
    def test_duplicate(self):
        self.r.append(self.r[0])
        with self.assertRaisesRegex(ValueError,'duplicate'): self.run_adapter()
    def test_resume_stats_missing(self):
        self.arrays['sentences'][0]=0
        with self.assertRaisesRegex(ValueError,'sentence mismatch'): self.run_adapter()
    def test_resume_token_missing(self):
        self.arrays['tokens_decode'][0]=0
        with self.assertRaisesRegex(ValueError,'token mismatch'): self.run_adapter()
    def test_count(self):
        self.arrays['count_decode'][0,0,0]=1
        with self.assertRaisesRegex(ValueError,'count coverage'): self.run_adapter()
    def test_mass(self):
        self.arrays['mass_prefill'][0,0,2]=1
        with self.assertRaisesRegex(ValueError,'routing mass'): self.run_adapter()
    def test_axis_order(self):
        self.summary['directions'].reverse()
        with self.assertRaisesRegex(ValueError,'axis'): self.run_adapter()
    def test_masked(self):
        self.h['masked_keep']=2
        with self.assertRaisesRegex(ValueError,'unmasked'): self.run_adapter()
    def test_wrong_prompt_length(self):
        self.r[0]['prompt_tokens']=4
        with self.assertRaisesRegex(ValueError,'prompt token'): self.run_adapter()

class Gate(unittest.TestCase):
    def table(self,delta):
        return [{'split':'validation','direction':d,'n':20,'baseline_chrf_pp':50.,'baseline_flags':{},'candidates':{'64':{'chrf_pp':50.+delta,'delta':delta,'flags':{},'gate_pass':False}}} for d in DIRS]
    def gate(self,table): return quality_gate(table,'validation','64',20)['status']
    def test_boundary(self): self.assertEqual(self.gate(self.table(-1.)),'REPORT_GATE_PASS')
    def test_improvement(self): self.assertEqual(self.gate(self.table(2.)),'REPORT_GATE_PASS')
    def test_drop(self): self.assertEqual(self.gate(self.table(-1.01)),'REPORT_GATE_FAIL')
    def test_incomplete(self):
        with self.assertRaises(ValueError): self.gate(self.table(0.)[:-1])
    def test_duplicate(self):
        r=self.table(0.); r.append(r[0])
        with self.assertRaises(ValueError): self.gate(r)
    def test_candidate_missing(self):
        r=self.table(0.); r[0]['candidates']={}
        with self.assertRaises(ValueError): self.gate(r)
    def test_error(self):
        r=self.table(0.); r[0]['candidates']['64']['flags']['truncated']=1
        self.assertEqual(self.gate(r),'REPORT_GATE_FAIL')
    def test_delta_inconsistent(self):
        r=self.table(0.); r[0]['candidates']['64']['delta']=1.
        with self.assertRaises(ValueError): self.gate(r)
    def test_sparse_flags(self):
        r=self.table(0.); r[0]['candidates']['64']['flags']['wrong_script']=1
        self.assertEqual(self.gate(r),'REPORT_GATE_PASS')
    def test_unknown_candidate_flag(self):
        r=self.table(0.); r[0]['candidates']['64']['flags']['typo']=20
        with self.assertRaisesRegex(ValueError,'diagnostic flags'): self.gate(r)
    def test_unknown_baseline_flag(self):
        r=self.table(0.); r[0]['baseline_flags']['typo']=0
        with self.assertRaisesRegex(ValueError,'diagnostic flags'): self.gate(r)
    def test_missing_flags(self):
        for side in ('baseline','candidate'):
            r=self.table(0.)
            if side=='baseline': del r[0]['baseline_flags']
            else: del r[0]['candidates']['64']['flags']
            with self.assertRaisesRegex(ValueError,'diagnostic flags'): self.gate(r)
if __name__=='__main__': unittest.main()
