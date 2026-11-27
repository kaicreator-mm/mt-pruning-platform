"""R23B fit-only score-mask generation, no quality or deployment admission.
All methods use the same 96/expert layer budget and exact fit profiles.
No prior/fresh holdout profiles are read. Rank by group-normalized scores.
"""
import argparse,hashlib,json,math,pathlib
P=pathlib.Path
DIRECTIONS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
METHODS=("frequency","router_weight","activation_norm","reap","global_weighted","protective_max")
FIELDS=("count","weight","norm","weighted_norm","energy")
def need(x,msg):
    if not x:raise ValueError(msg)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def blob(raw):return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
def check(p,n,k,scale):
    t=p["tokens"];need(type(t) is int and t>0,"positive token count")
    for f in FIELDS:
        need(len(p[f])==n and all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in p[f]),"profile field "+f)
    need(all(type(x) is int and x<=t for x in p["count"]) and sum(p["count"])==t*k,"count conservation")
    need(math.isclose(sum(p["weight"]),t*scale,rel_tol=1e-5,abs_tol=1e-7),"weight conservation")
    for i,c in enumerate(p["count"]):
        if c==0:need(all(p[f][i]==0 for f in FIELDS),"unobserved nonzero")
        else:
            need(p["weight"][i]<=c*scale*(1+1e-5),"weight bound")
            need(p["weighted_norm"][i]**2<=c*p["energy"][i]*(1+1e-9)+1e-12,"Cauchy bound")
def merge(a,b):
    if a is None:return b
    return {"tokens":a["tokens"]+b["tokens"],**{f:[x+y for x,y in zip(a[f],b[f])] for f in FIELDS}}
def normalize(v):
    total=math.fsum(v)
    return [x/total if total else 0. for x in v]
def score(p,n,method):
    c=p["count"];t=p["tokens"]
    if method=="frequency":return [x/t for x in c]
    if method=="router_weight":return [x/t for x in p["weight"]]
    if method=="activation_norm":return [p["norm"][i]/c[i] if c[i] else 0. for i in range(n)]
    if method=="reap":return [p["weighted_norm"][i]/c[i] if c[i] else 0. for i in range(n)]
    if method in ("global_weighted","protective_max"):return [x/t for x in p["weighted_norm"]]
    raise ValueError("unknown method")
def make(profiles,original,layers,n=128,k=8,keep=96,scale=2.826):
    need(0<k<=keep<n,"budget")
    need(len(profiles)==12 and len({x["id"] for x in profiles})==12 and all(x["subset"]=="fit" and x["id"].startswith("calibration:") for x in profiles),"fit scope only")
    groups={};counts={d:0 for d in DIRECTIONS}
    for x in profiles:
        d=x["direction"];need(d in counts,"direction");counts[d]+=1
        need(set(x["profile"])=={f"{l}/{phase}" for l in layers for phase in ("prefill","decode")},"layer/phase inventory")
        for key,p in x["profile"].items():
            check(p,n,k,scale);g=(d,key);groups[g]=merge(groups.get(g),p)
    need(set(counts.values())=={2},"balanced fit directions")
    need(set(original)==set(map(str,layers)) and all(v==sorted(set(v)) and len(v)==keep and all(type(i) is int and 0<=i<n for i in v) for v in original.values()),"original mask")
    masks={"original96":original};support={}
    for l in layers:
        support[str(l)]={d:{p:sum(c>0 for c in groups[d,f"{l}/{p}"]["count"]) for p in ("prefill","decode")} for d in DIRECTIONS}
    for method in METHODS:
        out={}
        for l in layers:
            cells=[(d,normalize(score(groups[d,f"{l}/{phase}"],n,method))) for d in DIRECTIONS for phase in ("prefill","decode")]
            if method=="protective_max":
                values=[max((2 if d in DIRECTIONS[:2] else 1)*s[i] for d,s in cells) for i in range(n)]
            else:values=[math.fsum((2 if d in DIRECTIONS[:2] else 1)*s[i] for d,s in cells) for i in range(n)]
            prior=set(original[str(l)])
            # Unobserved is not zero value evidence; tie prefers original coverage.
            out[str(l)]=sorted(sorted(range(n),key=lambda i:(-values[i],i not in prior,i))[:keep])
        masks[method]=out
    return dict(status="FIT_ONLY_MASK_INVENTORY_NOT_ADMITTED",methods={m:{"kept":v,"sha256":canonical(v)} for m,v in masks.items()},support=support,fit_ids=[x["id"] for x in profiles],scope="equal normalized direction/phase groups; primary direction weight2; protective_max uses max weighted group; no holdout fitting")
def selftest():
    profile={"tokens":2,"count":[2,1,1,0],"weight":[1.,.5,.5,0.],"norm":[4.,6.,2.,0.],"weighted_norm":[2.,3.,1.,0.],"energy":[2.,9.,1.,0.]}
    rows=[dict(id=f"calibration:{d}:{i}",subset="fit",direction=d,profile={"1/prefill":profile,"1/decode":profile}) for d in DIRECTIONS for i in range(2)]
    got=make(rows,{"1":[0,3]},[1],n=4,k=2,keep=2,scale=1.)
    need(got["methods"]["frequency"]["kept"]=={"1":[0,1]} and got["methods"]["reap"]["kept"]=={"1":[0,1]},"ranking oracle")
    need(got["methods"]["reap"]["sha256"]==canonical({"1":[0,1]}),"digest")
    def rejects(fn):
        try:fn()
        except ValueError:return
        raise AssertionError("bad input accepted")
    rejects(lambda:make([dict(rows[0],subset="holdout")]+rows[1:],{"1":[0,3]},[1],n=4,k=2,keep=2,scale=1.))
    rejects(lambda:make(rows,{"1":[0,0]},[1],n=4,k=2,keep=2,scale=1.))
    rejects(lambda:check(dict(profile,count=[2,1,0,0]),4,2,1.))
    rejects(lambda:check(dict(profile,weight=[1,.5,.4,.1]),4,2,1.))
    rejects(lambda:check(dict(profile,energy=[0.,9.,1.,0.]),4,2,1.))
    print(json.dumps(dict(status="PASS_SYNTHETIC_FIT_ONLY_MASK_SELECTION",methods=len(got["methods"]),negatives=5,real_selection="NOT_RUN")))
if __name__=="__main__":selftest()
