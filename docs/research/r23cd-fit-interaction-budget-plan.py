"""R23C/D predeclared fit-only interaction families and original-centered budget curves.
No mask chosen from holdout results. Scores describe observed contributions,
not causal importance. Use same fixed original96 baseline at K=96.
"""
import argparse,hashlib,json,math,pathlib,os
P=pathlib.Path
MAP_SHA="a9fb951b36e6bab3231dfc7f5ed6caf4648f2cb8bbe1a4d52809a2354c032001"
ORIGINAL_BLOB="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
ORIGINAL_MASK="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
INTERACTION_BLOB="0bc99fcec770371e076c625a9ec9a540aaf65ba3"
DIRECTIONS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def layer_plan(groups,base,interaction,n=128,k=8,budgets=(64,80,96,112,128)):
    need(base==sorted(set(base)) and all(type(i) is int and 0<=i<n for i in base),"base")
    need(tuple(sorted(set(budgets)))==budgets and k<=budgets[0] and budgets[-1]==n and len(base) in budgets,"budgets")
    scores=[0.]*n;support=[0]*n
    for phase in ("prefill","decode"):
        a=groups["all/"+phase];need(len(a["support"])==n,"support")
        for i,s in enumerate(a["support"]):
            need(type(s) is int and 0<=s<=a["samples"],"support count")
            support[i]=max(support[i],s)
        for d in DIRECTIONS:
            a=groups[d+"/"+phase];v=a["weighted_norm"];need(len(v)==n and all(math.isfinite(z) and z>=0 for z in v),"finite contribution")
            t=math.fsum(v);need(t>0,"group contribution")
            for i,x in enumerate(v):scores[i]+=(2 if d in DIRECTIONS[:2] else 1)*x/t
    eligible=[s if support[i]>=2 else None for i,s in enumerate(scores)]
    pair=interaction.shortlist(base,eligible,n=n)
    ms=pair["masks"] if pair else {name:base.copy() for name in ("base","swap_a","swap_b","swap_ab")}
    need(ms["base"]==base,"pair baseline")
    prior=set(base);inside=sorted(base,key=lambda i:(-scores[i],-support[i],i));outside=sorted((i for i in range(n) if i not in prior),key=lambda i:(-scores[i],-support[i],i))
    order=inside+outside
    curves={f"k{b}":sorted(order[:b]) for b in budgets}
    need(curves[f"k{len(base)}"]==base and curves[f"k{n}"]==list(range(n)),"curve endpoints")
    need(all(set(curves[f"k{a}"])<=set(curves[f"k{b}"]) for a,b in zip(budgets,budgets[1:])),"nested")
    return dict(pair_supported=pair is not None,drop=pair["drop"] if pair else [],add=pair["add"] if pair else [],importance=scores,sample_support_lower_bound=support,masks={"original96":ms["base"],**{k:v for k,v in ms.items() if k!="base"},**{k:v for k,v in curves.items() if k!=f"k{len(base)}"}})
def build(data,base,interaction):
    need(data["summary"]["status"]=="DESCRIPTIVE_FIT_MAP_NOT_SELECTION" and data["summary"]["samples"]==12 and data["summary"]["expert_nodes"]==6016,"map identity")
    layers=[str(i) for i in range(1,48)];need(set(data["layers"])==set(base)==set(layers),"layers")
    plans={l:layer_plan(data["layers"][l],base[l],interaction) for l in layers}
    names=tuple(plans["1"]["masks"]);need(all(tuple(p["masks"])==names for p in plans.values()),"mask inventory")
    masks={n:{l:plans[l]["masks"][n] for l in layers} for n in names}
    need(all(len(v)==(96 if n in ("original96","swap_a","swap_b","swap_ab") else int(n[1:])) for n,m in masks.items() for v in m.values()),"budgets")
    return dict(status="FIT_ONLY_INTERACTION_AND_BUDGET_PLAN_NOT_ADMITTED",map_sha256=MAP_SHA,original_mask_sha256=ORIGINAL_MASK,method="direction/phase normalized weighted-norm sum; Chinese-Myanmar directions weight2; swap requires lower-bound support >=2 distinct fit samples in at least one phase; one pair per layer",limits="Original-centered nested budget curves, not optimized arbitrary K masks; no claim of functional causality or physical heterogeneous support",supported_layers=sum(x["pair_supported"] for x in plans.values()),fit_ids=list(data["samples"]),layers=plans,masks={n:dict(kept=m,sha256=sha(canon(m))) for n,m in masks.items()})
def selftest(interaction):
    g={}
    for p in ("prefill","decode"):
        g["all/"+p]={"support":[2]*8,"samples":2}
        for d in DIRECTIONS:g[d+"/"+p]={"weighted_norm":[1.,2.,3.,4.,5.,6.,7.,8.]}
    q=layer_plan(g,[0,1,2,3],interaction,n=8,k=2,budgets=(2,4,6,8))
    need(q["drop"]==[0,1] and q["add"]==[7,6] and q["masks"]["k2"]==[2,3] and q["masks"]["k6"]==[0,1,2,3,6,7],"rank/nested oracle")
    for p in ("prefill","decode"):g["all/"+p]["support"]=[2,2,2,2,0,0,0,0]
    q=layer_plan(g,[0,1,2,3],interaction,n=8,k=2,budgets=(2,4,6,8));need(not q["pair_supported"] and q["masks"]["swap_ab"]==[0,1,2,3],"unsupported no-op")
    print(json.dumps(dict(status="PASS_SYNTHETIC_INTERACTION_BUDGET_PLAN",checks=2)))
def main():
    p=argparse.ArgumentParser();p.add_argument("--interaction",required=True,type=P);p.add_argument("--map",type=P);p.add_argument("--original",type=P);p.add_argument("--out",type=P);p.add_argument("--selftest",action="store_true");q=p.parse_args()
    import importlib.util
    need(blob(q.interaction.read_bytes())==INTERACTION_BLOB,"interaction pin")
    sp=importlib.util.spec_from_file_location("interaction",q.interaction);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    if q.selftest:selftest(m);return
    need(q.map and q.original and q.out,"explicit input paths")
    raw=q.map.read_bytes();orig=q.original.read_bytes();need(sha(raw)==MAP_SHA and blob(orig)==ORIGINAL_BLOB,"input pins")
    base=json.loads(orig)["candidates"]["96"]["kept"];need(sha(canon(base))==ORIGINAL_MASK,"original mask")
    out=build(json.loads(raw),base,m);b=canon(out)+b"\n"
    with q.out.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=out["status"],supported_layers=out["supported_layers"],fit_samples=len(out["fit_ids"]),mask_hashes={n:x["sha256"] for n,x in out["masks"].items()},plan_sha256=sha(b),bytes=len(b))),flush=True)
if __name__=="__main__":main()
