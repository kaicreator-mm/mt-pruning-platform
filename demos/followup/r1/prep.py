"""CPU-only follow-up demo helpers. No model inference or production acceptance."""
import argparse, hashlib, itertools, json, math, random, re, unicodedata
from collections import Counter
from pathlib import Path

LANGS=('zh','en','my')
DIRS=tuple(f'{a}-{b}' for a,b in itertools.permutations(LANGS,2))
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s): return ' '.join(unicodedata.normalize('NFC',s).split())
def require(ok,msg):
    if not ok: raise ValueError(msg)
def order(seed,x): return hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest()
def align(rows,split):
    maps={}
    for lang in LANGS:
        m={}
        for r in rows[lang]:
            require(r['split']==split,'wrong source split')
            i=str(r['id']); t=r['text']
            require(i not in m and isinstance(t,str) and norm(t),'duplicate or empty row')
            m[i]=t
        maps[lang]=m
    ids=set(maps['zh'])
    require(ids and all(set(m)==ids for m in maps.values()),'unaligned IDs')
    return [{'id':i,**{l:maps[l][i] for l in LANGS}} for i in sorted(ids)]
def split_data(dev,devtest,counts=(24,20,20),seed=42):
    # Sample a sentence before expanding directions. Disallow exact duplicate text across splits.
    seen={l:set() for l in LANGS}
    def unique(rows):
        out=[]
        for r in sorted(rows,key=lambda r:order(seed,r['id'])):
            ns={l:norm(r[l]) for l in LANGS}
            if any(ns[l] in seen[l] for l in LANGS): continue
            for l in LANGS: seen[l].add(ns[l])
            out.append(r)
        return out
    a=unique(dev); b=unique(devtest)
    n,v,t=counts; require(min(counts)>0 and len(a)>=n and len(b)>=v+t,'insufficient unique aligned rows')
    result={}
    for name,origin,rs in [('calibration','dev',a[:n]),('validation','devtest',b[:v]),('test','devtest',b[v:v+t])]:
        result[name]=[{'id':f'{origin}:{r["id"]}:{d}','group_id':f'{origin}:{r["id"]}','direction':d,'source':r[d.split('-')[0]],'reference':r[d.split('-')[1]]} for r in rs for d in DIRS]
    return result

def plan(profiles,keep,method='balanced',seed=42,top_k=8,expert_count=128):
    # Input: {layer: {direction: {prefill: [routing_mass], decode: [...]}}}.
    require(isinstance(expert_count,int) and top_k<=keep<=expert_count and profiles,'invalid expert budget/profile')
    require(method in ('balanced','random','worst_first'),'unknown method')
    result={}
    for layer,groups in sorted(profiles.items()):
        require(set(groups)==set(DIRS),'missing or unexpected direction')
        vec=[]
        for d in DIRS:
            require(set(groups[d])=={'prefill','decode'},'missing phase')
            for phase in ('prefill','decode'):
                a=groups[d][phase]; require(len(a)==expert_count and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in a) and 0<sum(a)<math.inf,'invalid routing mass')
                vec.append([x/sum(a) for x in a])
        if method=='random': ids=random.Random(order(seed,layer)).sample(range(expert_count),keep)
        elif method=='balanced': ids=sorted(range(expert_count),key=lambda i:(-sum(v[i] for v in vec),i))[:keep]
        else:
            ids=[]; coverage=[0.]*len(vec)
            for _ in range(keep):
                # Greedy worst-group protection heuristic; not a global optimizer.
                i=max((i for i in range(expert_count) if i not in ids),key=lambda i:(min(c+v[i] for c,v in zip(coverage,vec)),sum(v[i] for v in vec),-i))
                ids.append(i); coverage=[c+v[i] for c,v in zip(coverage,vec)]
        ids=sorted(ids); result[layer]={'keep_ids':ids,'retained_mass':{f'{d}/{p}':sum(vec[j][i] for i in ids) for j,(d,p) in enumerate(itertools.product(DIRS,('prefill','decode')))}}
    return {'method':method,'keep':keep,'top_k':top_k,'layers':result,'status':'PLAN_ONLY_NOT_QUALITY'}

def numbers(s):
    # Diagnostic only: catches decimal digits across scripts; ignores punctuation/semantic formats.
    s=''.join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in s)
    return Counter(re.findall(r'[0-9]+',s))
def evaluate(rows,baseline,candidate,score=None):
    if score is None:
        from sacrebleu.metrics import CHRF
        metric=CHRF(word_order=2); score=lambda h,r:metric.corpus_score(h,[r]).score
    ids=[r['id'] for r in rows]; require(len(ids)==len(set(ids)) and ids,'duplicate/empty reference')
    for model in (baseline,candidate):
        require(set(model)==set(ids),'missing/extra predictions')
        for x in model.values(): require(isinstance(x.get('text'),str) and type(x.get('truncated')) is bool,'invalid prediction status')
    report={}
    for d in DIRS:
        group=[r for r in rows if r['direction']==d]; require(group,'missing direction')
        ref=[r['reference'] for r in group]; b=[baseline[r['id']] for r in group]; c=[candidate[r['id']] for r in group]
        def metrics(xs):
            h=[x['text'] for x in xs]; s=score(h,ref); require(math.isfinite(s) and 0<=s<=100,'invalid score')
            return {'chrfpp':s,'empty':sum(not x.strip() for x in h),'truncated':sum(x['truncated'] for x in xs),'number_mismatch':sum(numbers(y)!=numbers(r['source']) for y,r in zip(h,group))}
        mb,mc=metrics(b),metrics(c); delta=mc['chrfpp']-mb['chrfpp']; report[d]={'n':len(group),'baseline':mb,'candidate':mc,'delta':delta,'pilot_gate':delta>=-1 and all(mc[k]<=mb[k] for k in ('empty','truncated','number_mismatch'))}
    require(sum(v['n'] for v in report.values())==len(rows),'unknown direction')
    return {'directions':report,'status':'PILOT_GATE_PASS' if all(x['pilot_gate'] for x in report.values()) else 'PILOT_GATE_FAIL','production_acceptance':False}

def fetch(revision,out):
    from datasets import load_dataset
    require(bool(re.fullmatch(r'[0-9a-f]{40}',revision)),'pin a full dataset revision')
    codes={'zh':'cmn_Hans','en':'eng_Latn','my':'mya_Mymr'}; repo='openlanguagedata/flores_plus'
    aligned={}
    for split in ('dev','devtest'):
        aligned[split]=align({l:load_dataset(repo,c,split=split,revision=revision,token=True,trust_remote_code=False) for l,c in codes.items()},split)
    result=split_data(aligned['dev'],aligned['devtest']); out=Path(out); out.mkdir(parents=True,exist_ok=False)
    for name,rows in result.items(): write(out/(name+'.json'),rows)
    write(out/'manifest.json',{'repo':repo,'revision':revision,'language_codes':codes,'seed':42,'counts':{k:len(v) for k,v in result.items()},'sha256':{k:digest(out/(k+'.json')) for k in result},'redistribute_publicly':False})

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='cmd',required=True)
    f=sub.add_parser('fetch'); f.add_argument('revision'); f.add_argument('out')
    f=sub.add_parser('plan'); f.add_argument('profiles'); f.add_argument('out'); f.add_argument('--keep',type=int,required=True); f.add_argument('--method',choices=['balanced','random','worst_first'],default='balanced')
    f=sub.add_parser('evaluate'); f.add_argument('rows'); f.add_argument('baseline'); f.add_argument('candidate'); f.add_argument('out')
    a=p.parse_args()
    if a.cmd=='fetch': fetch(a.revision,a.out)
    elif a.cmd=='plan': write(a.out,plan(read(a.profiles),a.keep,a.method))
    else: write(a.out,evaluate(read(a.rows),read(a.baseline),read(a.candidate)))
if __name__=='__main__': main()
