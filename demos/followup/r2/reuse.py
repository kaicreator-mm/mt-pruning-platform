"""Direction profile cache composition. Teacher-only, never infer quality from coverage."""
import argparse, hashlib, itertools, json, math, re
from pathlib import Path

def need(ok,msg):
    if not ok: raise ValueError(msg)
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def compose(records,languages,keep,top_k=8):
    need(len(languages)==3 and len(set(languages))==3 and {'zh','en'}<=set(languages),'require zh/en plus one distinct language')
    dirs={a+'-'+b for a,b in itertools.permutations(languages,2)}
    need(records and type(keep) is int and type(top_k) is int and 1<=top_k<=keep,'bad budget')
    # Cache identity includes teacher revision, tokenizer/prompt/router and collection protocol, not language alone.
    fields=('model_revision','tokenizer_sha256','prompt_sha256','router_sha256','collector_sha256','protocol_sha256')
    key=records[0]['identity']; need(set(key)==set(fields),'bad identity fields')
    need(all(re.fullmatch(r'[0-9a-f]{40}' if k=='model_revision' else r'[0-9a-f]{64}',key[k]) for k in fields),'identity must be pinned')
    bydir={}; layers=None; count=None
    for r in records:
        need(r['identity']==key and r['role']=='original_teacher_calibration','incompatible or candidate profile')
        d=r['direction']; need(d in dirs and d not in bydir,'unexpected/duplicate direction')
        need(bool(re.fullmatch(r'[0-9a-f]{64}',r['corpus_sha256'])),'missing corpus hash')
        p=r['layers']; need(p and isinstance(p,dict),'empty layers')
        if layers is None: layers=set(p)
        need(set(p)==layers,'layer mismatch')
        for layer,phases in p.items():
            need(set(phases)=={'prefill','decode'},'missing phase')
            for vec in phases.values():
                if count is None: count=len(vec)
                need(len(vec)==count and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in vec) and 0<sum(vec)<math.inf,'invalid vector')
        bydir[d]=r
    need(set(bydir)==dirs and keep<=count,'missing directions or budget too large')
    output={}
    for layer in sorted(layers):
        vectors={f'{d}/{phase}': [v/sum(a) for v in a] for d in sorted(dirs) for phase,a in bydir[d]['layers'][layer].items()}
        ids=sorted(sorted(range(count),key=lambda i:(-sum(v[i] for v in vectors.values()),i))[:keep])
        output[layer]={'keep_ids':ids,'retained_mass':{k:sum(v[i] for i in ids) for k,v in vectors.items()}}
    return {'status':'REUSED_TEACHER_PROFILE_PLAN_ONLY','identity':key,'languages':languages,'keep':keep,'top_k':top_k,'expert_count':count,'source_records':{d:digest(bydir[d]) for d in sorted(dirs)},'layers':output,'quality_verified':False}

def main():
    p=argparse.ArgumentParser(); p.add_argument('records'); p.add_argument('output'); p.add_argument('--third',required=True); p.add_argument('--keep',type=int,required=True); a=p.parse_args()
    r=compose(json.loads(Path(a.records).read_text(encoding='utf-8')),['zh','en',a.third],a.keep)
    with Path(a.output).open('x',encoding='utf-8') as f: json.dump(r,f,indent=2,allow_nan=False)
if __name__=='__main__': main()
