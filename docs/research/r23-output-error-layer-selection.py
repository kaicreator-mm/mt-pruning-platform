"""R23 output-sensitive selection from bounded, fixed same-budget masks.

Fit layer choices using only calibration-fit local reconstruction errors.
Holdout can reject, never choose or refit. No translation quality claim.
"""
import argparse,hashlib,json,math
from pathlib import Path
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"))
PRIMARY=range(4)
SECONDARY=range(4,12)
BASE="original96"
NAMES=(BASE,"guarded96","balanced","robust","myanmar2")
def need(v,m):
    if not v:raise ValueError(m)
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def blob(path):
    b=path.read_bytes();return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def masks_from(original,guarded,p04):
    for p,h in [(original,"d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"),(guarded,"150616a10fbf503f815fb2f7fc281a81dcc28fea"),(p04,"daeca1c6e4e55b86d3981c6246d63e89cf7907e2")]:need(blob(p)==h,"mask blob")
    a=json.loads(original.read_bytes())["candidates"]["96"]["kept"]
    b=json.loads(guarded.read_bytes())["plans"]["global_guarded96"]
    c=json.loads(p04.read_bytes())["plans"]
    out={BASE:a,"guarded96":b["kept"],**{n:v["kept"] for n,v in c.items()}}
    expected={BASE:"23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7","guarded96":b["mask_sha256"],**{n:v["mask_sha256"] for n,v in c.items()}}
    need(set(out)==set(NAMES),"inventory")
    for n,m in out.items():
        need(canon(m)==expected[n] and set(m)=={str(l) for l in range(1,48)},"mask identity "+n)
        need(all(len(v)==96 and v==sorted(set(v)) and all(type(e)==int and 0<=e<128 for e in v) for v in m.values()),"mask shape")
    return out
def aggregate(plan,captures,masks):
    expected={n:canon(m) for n,m in masks.items()}
    layers=sorted(map(int,masks[BASE]));names=tuple(masks)
    allrows=plan["fit"]+plan["holdout"]
    need(len(allrows)==len({x["id"] for x in allrows})==len({x["line"] for x in allrows})==24,"disjoint aligned samples")
    need(set(captures)=={x["id"] for x in allrows},"capture inventory")
    result={}
    for subset in ("fit","holdout"):
        agg={l:{n:[[] for _ in GROUPS] for n in names} for l in layers}
        for d in DIRS:need(sum(x["direction"]==d for x in plan[subset])==2,"group coverage")
        for rec in plan[subset]:
            x=captures[rec["id"]]
            need(x["status"]=="LOCAL_OUTPUT_PROXY_ONLY" and x["id"]==rec["id"] and x["direction"]==rec["direction"] and x["line"]==rec["line"] and x["logit_equivalence"] is True and x["candidate_masks"]==expected,"capture identity")
            need(len(x["teacher_logits"])==9,"teacher steps")
            seen={}
            for o in x["observations"]:
                key=(o["layer"],o["phase"],o["step"]);need(key not in seen,"duplicate observation")
                need(o["row_count"]==(2 if o["phase"]=="prefill" else 1),"row count")
                need(all(math.isfinite(o[k]) and 0<=o[k]<=1e-4 for k in ("routed_drift","combined_drift")),"observer drift")
                need(set(o["losses"])==set(names),"candidate loss inventory")
                for v in o["losses"].values():need(set(v)=={"routed_error","combined_error"} and all(math.isfinite(z) and z>=0 for z in v.values()),"loss")
                seen[key]=o
            need(set(seen)=={(l,p,j) for l in layers for p,j in (("prefill",0),("decode",1),("decode",8))},"observation coverage")
            for l in layers:
                for n in names:
                    for p,steps in (("prefill",[0]),("decode",[1,8])):
                        g=GROUPS.index(rec["direction"]+"/"+p)
                        agg[l][n][g].append(sum(seen[l,p,j]["losses"][n]["combined_error"] for j in steps)/len(steps))
        need(all(len(v)==2 for row in agg.values() for g in row.values() for v in g),"all12 group coverage")
        result[subset]={l:{n:[sum(v)/len(v) for v in groups] for n,groups in row.items()} for l,row in agg.items()}
    return result
def means(data,choice):
    return [sum(data[l][choice[l]][j] for l in data)/len(data) for j in range(12)]
def obj(v):return (max(v[j] for j in PRIMARY),sum(v[j] for j in PRIMARY)/4,sum(v)/12)
def feasible(v,b):
    return all(v[j]<=b[j]+1e-12 for j in PRIMARY) and all(v[j]<=b[j]*1.02+1e-12 for j in SECONDARY) and sum(v)<=sum(b)+1e-12
def better(a,b):
    for x,y in zip(a,b):
        if x<y-1e-12:return True
        if x>y+1e-12:return False
    return False
def select(aggregate_data,masks):
    fit=aggregate_data["fit"];hold=aggregate_data["holdout"]
    choices={l:BASE for l in fit};base=means(fit,choices);current=base;changes=[]
    for _ in range(2*len(fit)):
        best=None
        for l in sorted(fit):
            for n in NAMES:
                if n==choices[l]:continue
                v=[a+(fit[l][n][j]-fit[l][choices[l]][j])/len(fit) for j,a in enumerate(current)]
                if feasible(v,base) and better(obj(v),obj(current)) and (best is None or better(obj(v),obj(best[0]))):best=(v,l,n)
        if best is None:break
        _,l,n=best;changes.append([l,choices[l],n]);choices[l]=n
        current=means(fit,choices);need(feasible(current,base),"independent fit guard")
    hbase=means(hold,{l:BASE for l in hold});h=means(hold,choices)
    improve=(obj(hbase)[1]-obj(h)[1])/max(obj(hbase)[1],1e-12)
    # Provisional pilot criterion, not a translation acceptance gate.
    accepted=bool(changes) and feasible(h,hbase) and improve>=.05
    candidate={str(l):masks[n][str(l)] for l,n in choices.items()}
    return {"status":"LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION" if accepted else "NO_ADMISSION_FROM_LOCAL_PROXY","selected_fit_only":{str(l):n for l,n in choices.items()},"fit_changes":changes,"fit_baseline":dict(zip(GROUPS,base)),"fit_selected":dict(zip(GROUPS,current)),"holdout_baseline":dict(zip(GROUPS,hbase)),"holdout_selected":dict(zip(GROUPS,h)),"holdout_primary_relative_improvement":improve,"candidate": {"kept":candidate,"mask_sha256":canon(candidate)} if accepted else None,"limits":"Fixed five-mask inventory; local output proxy only. No refitting on holdout; no validation translation fitting."}
def selftest():
    masks={n:{str(l):list(range(i,i+4)) for l in (1,2)} for i,n in enumerate(NAMES)}
    plan={s:[{"id":f"calibration:{d}:{s}:{j}","direction":d,"line":offset+2*i+j} for i,d in enumerate(DIRS) for j in range(2)] for s,offset in (("fit",0),("holdout",12))}
    captures={}
    for rec in plan["fit"]+plan["holdout"]:
        obs=[]
        for l in (1,2):
            for p,j in (("prefill",0),("decode",1),("decode",8)):
                loss={n:{"routed_error":1.0,"combined_error":.1 if n==("guarded96" if l==1 else "balanced") else .2 if n==BASE else .3} for n in NAMES}
                obs.append({"layer":l,"phase":p,"step":j,"row_count":2 if p=="prefill" else 1,"routed_drift":0.0,"combined_drift":0.0,"losses":loss})
        captures[rec["id"]]={"status":"LOCAL_OUTPUT_PROXY_ONLY",**rec,"logit_equivalence":True,"candidate_masks":{n:canon(m) for n,m in masks.items()},"teacher_logits":[{}]*9,"observations":obs}
    data=aggregate(plan,captures,masks);a=select(data,masks)
    need(a["status"]=="LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION" and a["selected_fit_only"]=={"1":"guarded96","2":"balanced"},"fit selection")
    altered={**data,"holdout":{l:{n:[.2 if n==BASE else .4]*12 for n in NAMES} for l in (1,2)}}
    b=select(altered,masks);need(b["candidate"] is None and b["selected_fit_only"]==a["selected_fit_only"],"holdout rejects, never refits")
    def reject(f):
        try:f()
        except ValueError:return
        raise AssertionError("expected rejection")
    reject(lambda:aggregate(plan,{k:v for k,v in captures.items() if k!=next(iter(captures))},masks))
    reject(lambda:aggregate({"fit":plan["fit"],"holdout":plan["fit"]},captures,masks))
    import copy
    q=copy.deepcopy(captures);q[next(iter(q))]["observations"][0]["combined_drift"]=1
    reject(lambda:aggregate(plan,q,masks))
    q=copy.deepcopy(captures);q[next(iter(q))]["observations"].pop()
    reject(lambda:aggregate(plan,q,masks))
    q=copy.deepcopy(captures);q[next(iter(q))]["candidate_masks"][BASE]="0"*64
    reject(lambda:aggregate(plan,q,masks))
    q=copy.deepcopy(captures);q[next(iter(q))]["observations"][0]["losses"][BASE]["combined_error"]=float("nan")
    reject(lambda:aggregate(plan,q,masks))
    print(json.dumps({"status":"PASS_SYNTHETIC_ONLY","groups":12,"fit_samples":12,"holdout_samples":12,"layer_choices":a["selected_fit_only"],"holdout_primary_relative_improvement":a["holdout_primary_relative_improvement"],"holdout_rejection_without_refit":True,"negative_checks":6}))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--verify-masks",nargs=3,type=Path);q=p.parse_args()
    if q.selftest:selftest()
    if q.verify_masks:
        m=masks_from(*q.verify_masks);print(json.dumps({"status":"EXACT_REAL_MASK_INVENTORY_VERIFIED","masks":{n:canon(v) for n,v in m.items()},"layers":47,"keep":96}))
    need(q.selftest or q.verify_masks,"real selection requires completed verified captures")
if __name__=="__main__":main()

