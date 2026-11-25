import unittest,json,copy
from pathlib import Path
from stress import expand,diagnostics
class Stress(unittest.TestCase):
    def setUp(self): self.data=json.loads(Path(__file__).with_name('stress_r3.json').read_text(encoding='utf-8'))
    def test_72_no_reference(self):
        r=expand(self.data); self.assertEqual(len(r),72); self.assertEqual(len({x['id'] for x in r}),72); self.assertTrue(all('reference' not in x for x in r))
    def test_literal_loss(self):
        r=expand(self.data)[0]; self.assertEqual(diagnostics(r,r['source'])['literal_count_mismatches'],[]); self.assertEqual(diagnostics(r,'')['literal_count_mismatches'],r['required_literals'])
    def test_duplicate_rejected(self):
        self.data['cases'].append(copy.deepcopy(self.data['cases'][0])); self.assertRaises(ValueError,expand,self.data)
    def test_wrong_contract(self):
        self.data['cases'][0]['required'].append('missing'); self.assertRaises(ValueError,expand,self.data)
