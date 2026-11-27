"""R23F conservative uniform96 mask construction, expanded fit only.
One frozen method: balanced per-example contribution shares, leave-one-aligned-line-out
stability, bounded swaps from original96. No local-loss fitting or fresh data.
"""
import argparse,hashlib,json,math,pathlib,collections,importlib.util
P=pathlib.Path
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"))
WEIGHTS=tuple(2 if i<4 else 1 for i in range(12))
PLAN="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
ORIGINAL="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
ORIGINAL_MASK="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
MASK_STATS="34a06701d47ce0e3ab888882a5b20c75745204d4"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
ADMISSION="b629b515a65d0342e275c77e1bf2c2f8c43913bab92c24ce76ec3b5b953c6f87"
ROOT=P.home()/".cache/mt-pruning"
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(x):return hashlib.sha1(b"blob "+str(len(x)).encode()+b"\0"+x).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file");return p.read_bytes()
def mean(v,n):
    need(bool(v) and all(len(x)==n for x in v),"nonempty group")
    return [math.fsum(x[i] for x in v)/len(v) for i in range(n)]
def balanced(rows,n):
    groups=[]
    for g in GROUPS:
        d,p=g.split("/")
        groups.append(mean([x["share"][p] for x in rows if x["direction"]==d],n))
    score=[math.fsum(w*v[i] for w,v in zip(WEIGHTS,groups))/sum(WEIGHTS) for i in range(n)]
    return groups,score
def share(q,n):
    # Equal mixture of observed routing mass and weighted output norm shares.
    # Normalize every example separately to avoid long prompts dominating.
    a=q["weight"];b=q["weighted_norm"]
    need(len(a)==len(b)==n and all(math.isfinite(z) and z>=0 for z in a+b),"share vector")
    x=math.fsum(a);y=math.fsum(b);need(x>0 and y>0,"positive mass")
    return [(u/x+v/y)/2 for u,v in zip(a,b)]
def choose(rows,base,n,limit=2):
    need(base==sorted(set(base)) and all(type(i) is int and 0<=i<n for i in base),"base mask")
    lines=sorted({r["line"] for r in rows});need(len(lines)>=3,"line support")
    full,score=balanced(rows,n)
    folds=[balanced([r for r in rows if r["line"]!=l],n)[1] for l in lines]
    support=[len({r["line"] for r in rows if i in r["observed"]}) for i in range(n)]
    anchor=set(base);kept=set(base);changes=[];coverage=[math.fsum(v[i] for i in base) for v in full];current=coverage[:]
    # A single exchange must be direction-balanced in the final cumulative mask.
    # At least 90% of leave-line-out folds must favor it, both experts observed
    # on >=2 aligned lines. Never treat missing observations as low importance.
    candidates=[]
    for drop in base:
        if support[drop]<2:continue
        for add in range(n):
            if add in anchor or support[add]<2:continue
            margin=score[add]-score[drop]
            if margin<=1e-12:continue
            votes=sum(v[add]-v[drop]>1e-12 for v in folds)
            if votes*10<9*len(folds):continue
            candidates.append((-margin,-votes,drop,add))
    candidates.sort()
    for neg,votes,drop,add in candidates:
        if len(changes)>=limit:break
        if drop not in kept or add in kept:continue
        after=[z+v[add]-v[drop] for z,v in zip(current,full)]
        if any(after[i]<coverage[i]*(1 if i<4 else .98)-1e-12 for i in range(12)):continue
        if math.fsum(after)<math.fsum(coverage)-1e-12:continue
        kept.remove(drop);kept.add(add);current=after
        changes.append(dict(drop=drop,add=add,full_score_gain=-neg,stable_folds=-votes,total_folds=len(folds),drop_line_support=support[drop],add_line_support=support[add]))
    need(len(kept)==len(base) and len(changes)<=limit,"same budget")
    return dict(kept=sorted(kept),swaps=changes,base_contribution_share=dict(zip(GROUPS,coverage)),selected_contribution_share=dict(zip(GROUPS,current)),aligned_lines=len(lines))
def selftest():
    rows=[]
    for j,d in enumerate(DIRS):
        for k in range(3):
            rows.append(dict(line=10*j+k,direction=d,share={"prefill":[.1,.2,.6,.1],"decode":[.1,.2,.6,.1]},observed={0,1,2,3}))
    q=choose(rows,[0,1],4)
    need(q["kept"]==[1,2] and len(q["swaps"])==1 and q["swaps"][0]["stable_folds"]==18,"stable swap oracle")
    for r in rows:r["observed"]={0,1,3}
    need(choose(rows,[0,1],4)["kept"]==[0,1],"unobserved add rejected")
    for r in rows:
        r["observed"]={0,1,2,3}
        if r["direction"]=="zh-my":r["share"]={"prefill":[.6,.3,.05,.05],"decode":[.6,.3,.05,.05]}
    need(choose(rows,[0,1],4)["kept"]==[0,1],"primary share protection")
    need(share({"weight":[1.,3.],"weighted_norm":[3.,1.]},2)==[.5,.5],"mixture")
    print(json.dumps(dict(status="PASS_SYNTHETIC_GROUPED_STABLE_SWAP",checks=4,real_mask="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--inputs",type=P);a.add_argument("--out",type=P);a.add_argument("--selftest",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.out,"inputs");d=q.inputs
    raw=read(d/"expanded.json");orig=read(d/"original.json");mraw=read(d/"masks.py")
    need(blob(raw)==PLAN and blob(orig)==ORIGINAL and blob(mraw)==MASK_STATS,"input pins")
    plan=json.loads(raw);fit=plan["fit"];hold=plan["holdout"]
    need(len(fit)==48 and len(hold)==24 and len({r["line"] for r in fit})==30 and len({r["line"] for r in hold})==14 and not ({r["line"] for r in fit}&{r["line"] for r in hold}),"group scope")
    need(all(sum(r["direction"]==di for r in fit)==8 for di in DIRS),"balance")
    base=json.loads(orig)["candidates"]["96"]["kept"];need(sha(canon(base))==ORIGINAL_MASK,"base mask")
    s=importlib.util.spec_from_file_location("masks",d/"masks.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    root=ROOT/"r23-results";old=root/"reap-full24";new=root/"expanded-fit-profile-v1"
    old_log=read(root/"reap-full24.log")+read(root/"reap-full24-resume-v2.log")
    # Both logs include terminal hash lines; only profile JSON lines are consumed.
    def logs(raw,st):
        out=[]
        for l in raw.splitlines():
            if l.startswith(b"{"):
                x=json.loads(l)
                if x.get("status")==st:out.append(x)
        return out
    prior=logs(old_log,"PROFILE_CAPTURED")[:12];need(len(prior)==12,"prior inventory")
    nlog=read(root/"expanded-fit-profile-v1-gate-v2.log")
    rec=logs(nlog,"EXPANDED_FIT_PROFILE_CAPTURED")
    done=logs(nlog,"EXPANDED_FIT_PROFILE_BATCH_COMPLETE_NOT_SELECTION")
    need(len(rec)==36 and len(done)==1 and done[0]["completed"]==48 and done[0]["holdout_capture"]=="NOT_RUN" and read(root/"expanded-fit-profile-v1-gate-v2.stderr")==b"","complete expanded job")
    need(sorted(p.name for p in new.glob("*.json"))==[f"{i:02d}.json" for i in range(12,48)],"expanded receipt inventory")
    profiles=[];hashes=[]
    for i,(r,h) in enumerate(zip(fit,prior+rec)):
        need(h["id"]==r["id"] and h["index"]==i,"ordered receipt")
        b=read((old if i<12 else new)/f"{i:02d}.json");digest=sha(b);need(digest==h["receipt_sha256"],"receipt bytes");hashes.append(digest)
        x=json.loads(b)
        need(x["id"]==r["id"] and x["index"]==i and x["subset"]=="fit" and x["teacher"]==TEACHER and x["engine_head"]==ENGINE and x["threads"]==8 and x["hook_calls"]==423 and x["num_experts"]==128 and x["top_k"]==8,"profile identity")
        z=x["teacher_logits"];need(len(z)==9 and [v["input_token"] for v in z[1:]]==r["teacher_prefix_tokens"],"fixed trajectory")
        if i>=12:need(x["plan_blob"]==PLAN and x["admission_sha256"]==ADMISSION and x["plain_logits"]==z==x["restored_logits"] and x["same_process_observer_equivalence"] is True and x["matched_r23a_logits"] is None,"expanded witness")
        need(set(x["profile"])=={f"{l}/{p}" for l in range(1,48) for p in ("prefill","decode")},"layer coverage")
        for key,y in x["profile"].items():
            m.check(y,128,8,x["scale"]);need(y["tokens"]==(r["prompt_tokens"] if key.endswith("/prefill") else 8),"token count")
        profiles.append(x)
    result={}
    for l in range(1,48):
        rows=[]
        for r,x in zip(fit,profiles):
            ps={p:x["profile"][f"{l}/{p}"] for p in ("prefill","decode")}
            rows.append(dict(line=r["line"],direction=r["direction"],share={p:share(v,128) for p,v in ps.items()},observed={i for i in range(128) if any(v["count"][i]>0 for v in ps.values())}))
        result[str(l)]=choose(rows,base[str(l)],128)
    kept={l:x["kept"] for l,x in result.items()};count=sum(len(x["swaps"]) for x in result.values())
    out=dict(status="FROZEN_STABLE96_NOT_ADMITTED" if count else "NO_STABLE_SWAP_FOUND",method="equal per-example route-mass/weighted-output-norm share mixture; balanced direction/phase with primary weight2; >=90% leave-aligned-line-out positive margin; >=2 observed lines each expert; max2 swaps per layer; cumulative group contribution share guards",fit_ids=[r["id"] for r in fit],fit_lines=30,profile_receipt_sha256=hashes,plan_blob=PLAN,base_mask_sha256=ORIGINAL_MASK,kept=kept,mask_sha256=sha(canon(kept)),swaps=count,layers=result,limits="contribution share is not local reconstruction error or translation quality; fresh holdout not evaluated")
    b=canon(out)+b"\n"
    import os
    with q.out.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=out["status"],fit_samples=48,fit_lines=30,swaps=count,layers_changed=sum(bool(x["swaps"]) for x in result.values()),mask_sha256=out["mask_sha256"],receipt_sha256=sha(b))),flush=True)
if __name__=="__main__":main()
