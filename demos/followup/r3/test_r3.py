import copy, hashlib, json, tempfile, unittest
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file
from stream_prune import execute, digest, dump, header
from search import profile, select, sensitivity, choose, GROUPS, DIRS

class Streaming(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.base=Path(self.tmp.name); self.root=self.base/'teacher'; self.root.mkdir()
        self.tensors={'l0.expert':torch.arange(16*64*64,dtype=torch.float32).reshape(16,64,64),'l0.router':torch.arange(16*64,dtype=torch.bfloat16).reshape(16,64),'l0.bias':torch.arange(16,dtype=torch.float32),'shared':torch.arange(64*64,dtype=torch.bfloat16).reshape(64,64),'attention':torch.arange(64,dtype=torch.int64)}
        a={k:v for k,v in self.tensors.items() if k in ('l0.expert','shared')}; b={k:v for k,v in self.tensors.items() if k not in a}; self.wm={k:'a.safetensors' if k in a else 'b.safetensors' for k in self.tensors}; save_file(a,str(self.root/'a.safetensors'),metadata={'format':'pt'}); save_file(b,str(self.root/'b.safetensors'))
        dump(self.root/'config.json',{'num_experts':16,'top_k':4,'hidden_size':64}); dump(self.root/'model.safetensors.index.json',{'metadata':{'total_size':sum(x.numel()*x.element_size() for x in self.tensors.values())},'weight_map':self.wm}); self.plan={'original_count':16,'keep_count':8,'top_k':4,'count_field':'num_experts','top_k_field':'top_k','layers':{'0':[0,2,4,6,8,10,12,14]},'required_roles':['expert','router','bias'],'tensors':{k:({'role':k.split('.')[-1],'layer':'0','axis':0} if k.startswith('l0.') else {'role':'copy'}) for k in self.tensors},'source_sha256':{p.name:digest(p) for p in self.root.iterdir()}}
    def run_plan(self,p=None,name='candidate'): return execute(self.root,self.base/name,p or self.plan)
    def test_actual_safetensors_reload(self):
        r=self.run_plan(); out={**load_file(str(self.base/'candidate/a.safetensors')),**load_file(str(self.base/'candidate/b.safetensors'))}; ids=self.plan['layers']['0']; self.assertEqual(set(out),set(self.tensors)); self.assertTrue(all(torch.equal(out[k],v[ids] if k.startswith('l0.') else v) for k,v in self.tensors.items())); self.assertEqual(header(self.base/'candidate/a.safetensors')[1],{'format':'pt'}); self.assertLess(r['output_bytes'],r['source_bytes']); print('tensor_payload_bytes',r['source_bytes'],r['output_bytes'])
    def test_reproducible(self):
        a=self.run_plan(); b=self.run_plan(name='second'); self.assertEqual(a,b); self.assertEqual(digest(self.base/'candidate/receipt.json'),digest(self.base/'second/receipt.json'))
    def test_noop(self):
        p=copy.deepcopy(self.plan); p['keep_count']=16; p['layers']['0']=list(range(16)); r=self.run_plan(p); self.assertEqual(r['source_bytes'],r['output_bytes']); self.assertTrue(torch.equal(load_file(str(self.base/'candidate/a.safetensors'))['l0.expert'],self.tensors['l0.expert']))
    def test_missing_role(self):
        p=copy.deepcopy(self.plan); p['tensors']['l0.bias']={'role':'copy'}; self.assertRaises(ValueError,self.run_plan,p); self.assertFalse((self.base/'candidate').exists())
    def test_unknown_inventory(self):
        p=copy.deepcopy(self.plan); del p['tensors']['shared']; self.assertRaises(ValueError,self.run_plan,p)
    def test_bad_axis(self):
        p=copy.deepcopy(self.plan); p['tensors']['l0.router']['axis']=1; self.assertRaises(ValueError,self.run_plan,p)
    def test_bad_ids(self):
        p=copy.deepcopy(self.plan); p['layers']['0']=[0]*8; self.assertRaises(ValueError,self.run_plan,p)
    def test_modified_teacher(self):
        with (self.root/'a.safetensors').open('ab') as f:f.write(b'x')
        self.assertRaises(ValueError,self.run_plan)
    def test_wrong_shape_rolls_back(self):
        p=copy.deepcopy(self.plan); p['tensors']['shared']={'role':'expert','layer':'0','axis':0}; self.assertRaises(ValueError,self.run_plan,p); self.assertFalse((self.base/'candidate').exists()); self.assertFalse(list(self.base.glob('.prune-*')))
    def test_existing_output(self):
        self.run_plan(); self.assertRaises(ValueError,self.run_plan)

class Search(unittest.TestCase):
    def setUp(self):
        self.events=[{'layer':'0','direction':d,'phase':p,'ids':[0,1],'weights':[.8,.2],'output_norms':[1.,9.]} for d,p in sorted(GROUPS)]; self.events += [{'layer':'0','direction':d,'phase':p,'ids':[0,2],'weights':[.8,.2],'output_norms':[1.,2.]} for d,p in sorted(GROUPS)]
    def test_metrics_distinct(self):
        s=profile(self.events,4,2); self.assertEqual(select(s,4,2,2,'frequency')['0']['ids'],[0,1]); self.assertEqual(select(s,4,2,2,'mass')['0']['ids'],[0,1]); events=copy.deepcopy(self.events); [e['output_norms'].__setitem__(1,12.) for e in events if e['ids'][1]==2]; self.assertEqual(select(profile(events,4,2),4,2,2,'saliency')['0']['ids'],[1,2])
    def test_coactivation(self):
        s=profile(self.events,4,2); r=select(s,4,2,2,'coactivation'); self.assertEqual(r['0']['ids'],[0,1]); self.assertEqual(r['0']['pair_coverage'],.5)
    def test_random_repeat(self):
        s=profile(self.events,4,2); self.assertEqual(select(s,4,2,2,'random'),select(s,4,2,2,'random'))
    def test_missing_group(self): self.assertRaises(ValueError,profile,[e for e in self.events if e['direction']!='zh-my'],4,2)
    def test_nan(self):
        self.events[0]['output_norms'][0]=float('nan'); self.assertRaises(ValueError,profile,self.events,4,2)
    def test_sensitivity(self):
        r=sensitivity({'0':{4:5,8:0},'1':{4:1,8:0}},12,4); self.assertEqual(r['counts'],{'0':8,'1':4}); self.assertEqual(r['estimated_additive_loss'],1)
    def test_infeasible(self): self.assertRaises(ValueError,sensitivity,{'0':{4:0}},3,4)
    def report(self,name,size,score=80,errors=0,split='validation'):
        return {'id':name,'bytes':size,'split':split,'validation_identity':'a'*64,'directions':{d:{'n':20,'baseline':80,'candidate':score,'baseline_errors':{'empty':0,'truncated':0,'number_mismatch':0},'candidate_errors':{'empty':errors,'truncated':0,'number_mismatch':0}} for d in DIRS}}
    def test_choose_smallest_pass(self): self.assertEqual(choose([self.report('small',10,78),self.report('mid',15,79),self.report('big',20)])['candidate'],'mid')
    def test_one_direction_failure(self):
        r=self.report('small',10); r['directions']['zh-my']['candidate']=78; self.assertEqual(choose([r])['status'],'FALLBACK_TO_TEACHER')
    def test_error_regression(self): self.assertIsNone(choose([self.report('small',10,80,1)])['candidate'])
    def test_test_rejected(self): self.assertRaises(ValueError,choose,[self.report('small',10,split='test')])
    def test_mixed_identity(self):
        a=self.report('a',10); b=self.report('b',20); b['validation_identity']='b'*64; self.assertRaises(ValueError,choose,[a,b])
    def test_mixed_baseline(self):
        a=self.report('a',10); b=self.report('b',20); b['directions']['zh-my']['baseline']=79; self.assertRaises(ValueError,choose,[a,b])
    def test_missing_direction(self):
        r=self.report('small',10); del r['directions']['zh-my']; self.assertRaises(ValueError,choose,[r])
if __name__=='__main__': unittest.main()
