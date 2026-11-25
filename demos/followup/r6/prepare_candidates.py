"""Alternative selection plan, D01b-compatible JSON. No runner or weights mutation."""
import argparse, importlib.util, json
from pathlib import Path
from bridge import adapt, need

def prep_module():
    spec = importlib.util.spec_from_file_location('r1_prep', Path(__file__).resolve().parents[1]/'r1'/'prep.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def candidates(aggregate, keeps, method='balanced', seed=42):
    need(aggregate['status']=='AGGREGATE_COVERAGE_CHECKED_NOT_PROVENANCE', 'unknown aggregate status')
    n,k=aggregate['expert_count'],aggregate['top_k']
    need(keeps and len(set(keeps))==len(keeps) and all(type(x) is int and k<=x<=n for x in keeps), 'invalid keeps')
    prep=prep_module(); result={}
    for keep in keeps:
        p=prep.plan(aggregate['profiles'],keep,method=method,seed=seed,top_k=k,expert_count=n)
        result[str(keep)]={'kept':{l:r['keep_ids'] for l,r in p['layers'].items()}, 'retained_mass_by_direction_phase_layer':{l:r['retained_mass'] for l,r in p['layers'].items()}}
    return {'status':'EXPERIMENTAL_PLAN_ONLY_NOT_QUALITY', 'num_experts':n, 'top_k':k, 'method':method, 'seed':seed, 'input_hashes':aggregate['hashes'], 'candidates':result, 'quality_verified':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True); p.add_argument('--records',required=True); p.add_argument('--stats',required=True)
    p.add_argument('--corpus-sha256',required=True); p.add_argument('--layer-ids',required=True,help='comma-separated actual sparse layer IDs')
    p.add_argument('--experts',type=int,required=True); p.add_argument('--top-k',type=int,required=True); p.add_argument('--per-direction',type=int,required=True)
    p.add_argument('--keeps',required=True); p.add_argument('--method',choices=['balanced','random','worst_first'],default='balanced'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    aggregate=adapt(a.corpus,a.records,a.stats,corpus_sha256=a.corpus_sha256,layer_ids=[int(x) for x in a.layer_ids.split(',')],n_experts=a.experts,top_k=a.top_k,expected_n=a.per_direction)
    result=candidates(aggregate,[int(x) for x in a.keeps.split(',')],a.method,a.seed)
    with Path(a.out).open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
if __name__=='__main__': main()
