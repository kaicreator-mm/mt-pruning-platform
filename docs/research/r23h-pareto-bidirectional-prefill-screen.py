"""R23H: fit-only actual prefill KL-directed screen, 96 experts per layer.
Freeze bounded candidate inventory before actual evaluation. No fresh or validation."""
import argparse,hashlib,importlib.util,json,math,os,pathlib,sys,time,fcntl
P=pathlib.Path;R=P.home()/".cache/mt-pruning";RR=R/"r23-results"
USB=P("/mnt/mt-storage/mt-pruning/evidence/r23g-recovery-38ed4798")
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
STABLE="488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
LOGS=["aa9d1dd75249a084c4a8a4b142809493393bc0947658a0ea24a8e1490758e8bb","b72ce687ccef58628717de71d37ced00148dabb897feb0c91ac8f6c6659cf7fb"]
PINS={"proxy":"248737901db40b50efb0a26a9d5ef76d1928538b","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46","prop":"88468e8c2a5905628365da52e1575d5251516512"}
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en");GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"))
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pin(p,h,git=False):
    need(p.is_file() and not p.is_symlink(),"regular "+str(p));b=p.read_bytes();need((blob if git else sha)(b)==h,"pin "+str(p));return b
def load(d,n):
    p=d/(n+".py");pin(p,PINS[n],True);s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def mean(a):return math.fsum(a)/len(a)
def publish(p,x):
    b=canon(x)+b"\n"
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    return sha(b)
def event(status,**x):print(json.dumps(dict(status=status,**x),sort_keys=True),flush=True)
def local(rows,loss,revert):
    means={n:{g:mean([mean([loss[r["id"]][l,g.split("/")[1]]["original96" if n=="base" or l in revert else "stable96"] for l in range(1,48)]) for r in rows if r["direction"]==g.split("/")[0]]) for g in GROUPS} for n in ("base","candidate")}
    b,s=means["base"],means["candidate"];gain=(math.fsum(b[g]-s[g] for g in GROUPS[:4]))/math.fsum(b[g] for g in GROUPS[:4])
    v=[g for i,g in enumerate(GROUPS) if s[g]>b[g]*(1 if i<4 else 1.02)+1e-12];ok=gain>=.05 and not v and math.fsum(s.values())<=math.fsum(b.values())+1e-12
    return dict(primary_gain=gain,violations=v,pass_local=ok,means=means)
def prepare(d):
    proxy=load(d,"proxy");ep=json.loads(pin(USB/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True));rows=ep["fit"]
    need(len(rows)==48 and len({r["line"] for r in rows})==30 and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}),"fit only")
    base=json.loads(pin(d/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True))["candidates"]["96"]["kept"]
    stable=json.loads(pin(RR/"expanded-stable96-v1.json",STABLE))["kept"]
    g=json.loads(pin(USB/"candidate.json","e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"))["kept"]
    need(sha(canon(base))=="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7" and sha(canon(g))=="77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa","masks")
    need([l for l in g if g[l]!=stable[l]]==["38"],"anchor")
    rec=json.loads(pin(USB/"recovery-receipt.json",REC));need(rec["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and rec["primary_violations"]==["zh-my/prefill/kl"],"prior rejection")
    old={}
    for i,h in enumerate(rec["receipt_sha256"]):
        x=json.loads(pin(RR/"r23g-l38-propagation-v1"/f"{i:02d}.json",h));old[x["id"]]=x
    chosen=[r for r in rows if r["direction"] in ("zh-my","my-zh")];need(len(chosen)==16 and all(sum(r["direction"]==d for r in chosen)==8 for d in ("zh-my","my-zh")) and all(r["id"] in old for r in chosen if r["direction"]=="zh-my"),"balanced bidirectional primary rows")
    loss={};caps={};masks={"original96":base,"stable96":stable}
    for b,name in enumerate(("00-24","24-48")):
        ev=[json.loads(l) for l in pin(RR/f"stable96-recovery-fit-{name}.log",LOGS[b]).splitlines() if l.startswith(b"{")];ev=[e for e in ev if e.get("status")=="STABLE96_PROXY_CAPTURED"]
        need(len(ev)==24,"fit events")
        for i,e in enumerate(ev,b*24):
            r=rows[i];need(e["index"]==i and e["id"]==r["id"],"order")
            x=json.loads(pin(RR/"expanded-stable96-proxy-v1/fit"/f"{i:02d}.json",e["receipt_sha256"]));seen=proxy.checked_capture(x,r,masks,list(range(1,48)));caps[r["id"]]=x
            loss[r["id"]]={(l,p):{n:mean([seen[l,p,j]["losses"][n]["combined_error"] for j in js]) for n in masks} for l in range(1,48) for p,js in (("prefill",[0]),("decode",[1,8]))}
    f=json.loads(pin(RR/"expanded-stable96-proxy-v1/fit/decision.json","7821abc55b356520bdd02e4b60f32057a95d204b8cc723b894e2fdce3d9d100d"));need(f==proxy.decision(rows,caps,masks,list(range(1,48)),"fit"),"fit replay")
    # Diagnostic ranking: positive downstream MoE error delta on 8 exact samples.
    changed=[l for l in range(1,48) if g[str(l)]!=base[str(l)]]
    drift_rows=[r for r in chosen if r["direction"]=="zh-my" and r["id"] in old]\n    need(len(drift_rows)==8,"zh-my drift anchor")\n    drift={l:mean([next(q["moe_output_nmse"] for q in old[r["id"]]["candidate96"]["layer"] if q["layer"]==l and q["step"]==0)-next(q["moe_output_nmse"] for q in old[r["id"]]["original96"]["layer"] if q["layer"]==l and q["step"]==0) for r in drift_rows]) for l in changed}
    rank=sorted(changed,key=lambda l:(-drift[l],l))
    pareto_path=d/"pareto.json"
    raw=pin(pareto_path,"cbdc65097cce87cfc6ddb8934604c690e744b9f3d8f7f52b3708bf035c20fa2f")
    pr=json.loads(raw)
    need(pr["status"]=="R23H_PARETO_ROBUST_FIT_PROXY_CANDIDATES_NOT_ADMISSION" and pr["constraints"]==434 and pr["candidate_sha256"]=="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1","pareto provenance")
    candidates={};evaluated=[];seen=set()
    for c0 in pr["candidates"]:
        name=c0["name"];mask=c0["kept"]
        need(sha(canon(mask))==c0["mask_sha256"] and set(mask)==set(base),"pareto mask hash")
        revert=set()
        for ls in map(int,base):
            need(len(mask[str(ls)])==96 and mask[str(ls)]==sorted(set(mask[str(ls)])) and all(type(x) is int and 0<=x<128 for x in mask[str(ls)]),"expert mask")
            need(mask[str(ls)] in (base[str(ls)],stable[str(ls)]),"binary layer source")
            if stable[str(ls)]!=base[str(ls)] and mask[str(ls)]==base[str(ls)]:revert.add(ls)
        need(sum(len(set(mask[str(ls)])-set(base[str(ls)])) for ls in base)==c0["original96_swaps"],"swap count")
        a=local(rows,loss,revert)
        folds=[local([r for r in rows if r["line"]!=i],loss,revert)["pass_local"] for i in sorted({r["line"] for r in rows})]
        need(a["pass_local"] and all(folds) and sum(folds)==30,"independent local replay")
        need(abs(a["primary_gain"]-c0["full_primary_gain"])<1e-12 and abs(min(local([r for r in rows if r["line"]!=i],loss,revert)["primary_gain"] for i in sorted({r["line"] for r in rows}))-c0["minimum_fold_primary_gain"])<1e-12,"solver metric replay")
        item=dict(name=name,reverted_layers=sorted(revert),mask_sha256=c0["mask_sha256"],local=a,folds_passed=sum(folds),solver_folds_passed=c0["folds_passed"],original96_swaps=c0["original96_swaps"],kept=mask)
        evaluated.append({k:v for k,v in item.items() if k!="kept"})
        need(c0["mask_sha256"] not in seen,"duplicate candidate")
        seen.add(c0["mask_sha256"]);candidates[name]=item
    select=[x["name"] for x in pr["candidates"]]
    need(len(select)==6 and all(n in candidates for n in select),"six candidate inventory")
    return chosen,old,base,g,dict(status="R23H_PARETO_BIDIRECTIONAL_FIT_PREFILL_SCREEN_ONLY",teacher=TEACHER,prior_recovery_sha256=REC,pareto_report_sha256=pr["source_report_sha256"],pareto_input_sha256=sha(raw),ids=[r["id"] for r in chosen],evaluated_local_options=evaluated,screen=[candidates[n] for n in select],holdout="FORBIDDEN")
def metric(t,z):
    a=t.double();b=z.double();p=a.log_softmax(-1);q=b.log_softmax(-1)
    kl=float((p.exp()*(p-q)).sum(dim=-1).mean());nmse=float((a-b).square().sum()/a.square().sum())
    need(math.isfinite(kl) and kl>=-1e-12 and math.isfinite(nmse),"finite")
    return dict(kl=max(0.,kl),nmse=nmse,argmax_mismatch=int(a[-1].argmax()!=b[-1].argmax()))
def run(d):
    rows,old,base,g,plan=prepare(d);dst=RR/"r23h-pareto-bidirectional-prefill-v1";dst.mkdir(mode=0o700);ph=publish(dst/"plan.json",plan);event("R23H_PARETO_BIDIRECTIONAL_PLAN_FROZEN",sha256=ph,candidates=[dict(name=c["name"],reverted=c["reverted_layers"],gain=c["local"]["primary_gain"],folds=c["folds_passed"]) for c in plan["screen"]])
    drv=load(d,"driver");c=load(d,"collector");c.verify_code()
    corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
    ids={r["id"] for r in rows};ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};need(set(ctx)==ids,"corpus")
    for r in rows:need(sha(canon(ctx[r["id"]]["prompt_ids"]))==r["prompt_ids_sha256"],"prompt")
    with (R/"r23-admissions/batch.lock").open("a+") as lk:
        # Wait for already admitted cleanup, bounded; never overlap model work.
        deadline=time.monotonic()+7200
        while True:
            try:fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                need(time.monotonic()<deadline,"lock wait expired");time.sleep(10)
        event("R23H_LOCK_ACQUIRED");pre=drv.preflight();need(not pre["blockers"],str(pre))
        c.torch.set_num_threads(4);c.torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==TEACHER and ident["num_experts"]==128,"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=4);event("R23H_TEACHER_READY",teacher=TEACHER)
        def prefill(prompt,mask):
            try:
                e.set_mask(None if mask is None else {int(l):v for l,v in mask.items()})
                with c.torch.inference_mode():z=e.forward(prompt,e.new_cache(len(prompt)+9),"prefill")
                return c.logits_sha(z),z[[len(prompt)//2-1,len(prompt)-1]].clone()
            finally:e.set_mask(None)
        captures=[]
        for i,r in enumerate(rows):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            prompt=ctx[r["id"]]["prompt_ids"];prior=old.get(r["id"]);th,t=prefill(prompt,None)\n            if prior is not None:need(th==prior["original96"]["teacher_hashes"][0]["sha256"],"teacher logits")
            rec=dict(id=r["id"],direction=r["direction"],index=i,plan_sha256=ph,teacher_sha256=th,metrics={})
            for n,m,oldn in [("original96",base,"original96"),("r23g",g,"candidate96")]+[(x["name"],x["kept"],None) for x in plan["screen"]]:
                h,z=prefill(prompt,m);q=metric(t,z)
                if oldn:
                    o=prior[oldn]["logits"][0];need(h==prior[oldn]["masked_hashes"][0]["sha256"] and abs(q["kl"]-o["teacher_to_masked_kl"])<1e-12 and abs(q["nmse"]-o["nmse"])<1e-12 and q["argmax_mismatch"]==int(o["teacher_argmax"]!=o["masked_argmax"]),"prior actual prefill replay")
                rec["metrics"][n]=dict(q,logits_sha256=h,mask_sha256=sha(canon(m)))
            h,_=prefill(prompt,None);need(h==th,"teacher restoration");h=publish(dst/f"{i:02d}.json",rec);captures.append(rec);event("R23H_PARETO_BIDIRECTIONAL_PREFILL_CAPTURED",index=i,id=r["id"],sha256=h,kl={n:q["kl"] for n,q in rec["metrics"].items()})
        names=list(captures[0]["metrics"]);a={n:{k:(sum if k=="argmax_mismatch" else mean)([x["metrics"][n][k] for x in captures]) for k in ("kl","nmse","argmax_mismatch")} for n in names}
        good=[n for n in names[2:] if all(a[n][k]<=a["original96"][k]+1e-12 for k in a[n]) and a[n]["kl"]<a["r23g"]["kl"]]
        good.sort(key=lambda n:(a[n]["kl"],a[n]["nmse"],n))
        out=dict(status="R23H_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION",plan_sha256=ph,samples=8,metrics=a,eligible_for_full_fit_propagation_only=good,selected=good[0] if good else None,limits="Fixed fit prompt only, no decode or other direction real test, no quality or fresh admission.")
        h=publish(dst/"report.json",out);event(out["status"],report_sha256=h,**{k:v for k,v in out.items() if k!="status"})
if __name__=="__main__":run(P(sys.argv[1]))
