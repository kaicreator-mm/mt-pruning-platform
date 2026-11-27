"""R23H: conditional full fit propagation of the one predeclared screen winner.
No fresh, mask refit, or materialization. Require exact screen replay."""
import hashlib,json,math,os,pathlib,sys,fcntl,time,importlib.util
P=pathlib.Path;R=P.home()/".cache/mt-pruning";RR=R/"r23-results"
SCREEN=R/"r23h-pareto-screen-v1";SD=RR/"r23h-pareto-prefill-v1"
PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
USB=P("/mnt/mt-storage/mt-pruning/evidence/r23g-recovery-38ed4798")
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pin(p,h):
    need(p.is_file() and not p.is_symlink(),"regular "+str(p));b=p.read_bytes();need(sha(b)==h,"pin "+str(p));return b
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def pinblob(p,h):
    b=p.read_bytes();need(blob(b)==h,"blob "+str(p));return b
def publish(p,x):
    b=canon(x)+b"\n"
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    return sha(b)
def event(status,**x):print(json.dumps(dict(status=status,**x),sort_keys=True),flush=True)
def mean(v):return math.fsum(v)/len(v)
def load(p,h,n,git=True):
    (pinblob if git else pin)(p,h)
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def screen():
    need((SCREEN/"exit-code").read_text().strip()=="0" and (SCREEN/"run.stderr").read_bytes()==b"","terminal screen success")
    p=json.loads(pin(SD/"plan.json",PLAN));need(p["status"]=="R23H_PARETO_FIT_PREFILL_SCREEN_ONLY" and p["prior_recovery_sha256"]==REC and p["teacher"]==TEACHER and p["holdout"]=="FORBIDDEN","plan")
    names=["original96","r23g"]+[x["name"] for x in p["screen"]];need(len(names)==8 and len(set(names))==8 and len(p["ids"])==8 and len(set(p["ids"]))==8,"inventory")
    events=[json.loads(l) for l in (SCREEN/"run.log").read_bytes().splitlines() if l.startswith(b"{")]
    ev=[x for x in events if x["status"]=="R23H_PREFILL_CAPTURED"];end=[x for x in events if x["status"]=="R23H_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(ev)==8 and len(end)==1,"complete events")
    caps={}
    for i,x in enumerate(ev):
        z=json.loads(pin(SD/f"{i:02d}.json",x["sha256"]));need(x["index"]==z["index"]==i and x["id"]==z["id"]==p["ids"][i] and z["plan_sha256"]==PLAN and set(z["metrics"])==set(names),"capture")
        for n,q in z["metrics"].items():
            need(all(math.isfinite(q[k]) and q[k]>=0 for k in ("kl","nmse")) and q["argmax_mismatch"] in (0,1) and len(q["logits_sha256"])==64,"metric")
            if n not in names[:2]:need(q["mask_sha256"]==next(y["mask_sha256"] for y in p["screen"] if y["name"]==n),"mask")
        caps[z["id"]]=z
    a={n:{k:(sum if k=="argmax_mismatch" else mean)([x["metrics"][n][k] for x in caps.values()]) for k in ("kl","nmse","argmax_mismatch")} for n in names}
    good=sorted((n for n in names[2:] if all(a[n][k]<=a["original96"][k]+1e-12 for k in a[n]) and a[n]["kl"]<a["r23g"]["kl"]),key=lambda n:(a[n]["kl"],a[n]["nmse"],n))
    q=json.loads(pin(SD/"report.json",end[0]["report_sha256"]));need(q["plan_sha256"]==PLAN and q["samples"]==8 and q["metrics"]==a and q["eligible_for_full_fit_propagation_only"]==good and q["selected"]==(good[0] if good else None),"independent screen decision")
    need(end[0]["report_sha256"]=="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b" and good[0]=="all8","pinned selection")
    c=next(x for x in p["screen"] if x["name"]==good[0]);need(c["local"]["pass_local"] is True and c["folds_passed"]==30 and sha(canon(c["kept"]))==c["mask_sha256"] and set(c["kept"])==set(map(str,range(1,48))) and all(len(x)==96 and x==sorted(set(x)) and all(type(i) is int and 0<=i<128 for i in x) for x in c["kept"].values()),"frozen candidate")
    need(c["mask_sha256"]=="1f7402ae0fb540c86ace21914b45ea0c42a113e5f38072a2d3f86795ee8ad70b","all8 mask")
    return p,caps,q,end[0]["report_sha256"],c
def main():
    deadline=time.monotonic()+14400
    while not (SCREEN/"exit-code").exists():
        need(time.monotonic()<deadline,"screen wait expired");time.sleep(15)
    p,sc,q,sh,cand=screen()
    if cand is None:
        event("R23H_STOP_NO_FIT_PREFILL_CANDIDATE",screen_sha256=sh);return
    src=RR/"r23h-pareto-winner-propagation-v1";prev=R/"r23h-pareto-winner-run-v1"
    dst=RR/"r23h-pareto-winner-propagation-v2";dst.mkdir(mode=0o700)
    g=json.loads(pin(USB/"recovery-receipt.json",REC));need(g["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH","previous recovery")
    plan=json.loads(pinblob(USB/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915"));pp=json.loads(pinblob(USB/"prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde"))
    rows=pp["rows"]+[r for r in plan["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in pp["rows"]}]
    need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in plan["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in plan["holdout"]}),"fit scope")
    base=json.loads(pinblob(SCREEN/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"))["candidates"]["96"]["kept"];need(sha(canon(base))=="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7","base")
    old={}
    for i,r in enumerate(rows):
        x=json.loads(pin(RR/"r23g-l38-propagation-v1"/f"{i:02d}.json",g["receipt_sha256"][i]));need(x["id"]==r["id"],"previous row");old[r["id"]]=x
    ad=dict(status="R23H_WINNER_FIT_PROPAGATION_ONLY",screen_plan_sha256=PLAN,screen_report_sha256=sh,previous_recovery_sha256=REC,selected=cand["name"],mask_sha256=cand["mask_sha256"],ids=[r["id"] for r in rows],teacher=TEACHER,holdout="FORBIDDEN")
    ah=publish(dst/"admission.json",ad);need(ah=="f381a42350922ef14ca89257aa0349ce1d6f8c8887b43988dccb33321f1a70ce" and (src/"admission.json").read_bytes()==(dst/"admission.json").read_bytes(),"original admission");event("R23H_WINNER_ADMITTED",selected=cand["name"],mask_sha256=cand["mask_sha256"],admission_sha256=ah)
    d=SCREEN
    drv=load(d/"driver.py","0735238e15f13fd75e3f6939fc31aa834d8b5854","driver")
    c=load(d/"collector.py","02daf65bb7fe0526b2a700d17013054f7b0eea46","collector")
    prop=load(d/"prop.py","88468e8c2a5905628365da52e1575d5251516512","prop")
    checker=load(P(sys.argv[1])/"checked.py","87b12c30ebf801d2e4571b6f0f8348edbb61ba13","checked")
    c.verify_code()
    need(not (prev/"exit-code").exists() and (prev/"run.stderr").read_bytes()==b"","interrupted predecessor")
    log=(prev/"run.log").read_bytes()
    ev=[json.loads(l) for l in log.splitlines() if l.startswith(b"{")]
    prefix=[x for x in ev if x["status"]=="R23H_PROPAGATION_CAPTURED"]
    need(len(prefix)==10 and set(x.name for x in src.iterdir())=={"admission.json",*(f"{i:02d}.json" for i in range(10))},"exact prefix inventory")
    need(len([x for x in ev if x["status"]=="R23H_WINNER_ADMITTED" and x["admission_sha256"]==ah])==1 and len([x for x in ev if x["status"]=="R23H_PROPAGATION_TEACHER_READY" and x["teacher"]==TEACHER])==1,"prefix provenance")
    restored=[]
    for i,v in enumerate(prefix):
        r=rows[i];need(v["index"]==i and v["id"]==r["id"],"prefix order")
        x=json.loads(pin(src/f"{i:02d}.json",v["receipt_sha256"]))
        checker.checked(x,r,list(range(1,48)))
        need(x["index"]==i and x["teacher"]==TEACHER and x["mask_sha256"]==cand["mask_sha256"] and x["admission_sha256"]==ah and x["original96"]==old[r["id"]]["original96"],"prefix binding")
        if r["id"] in sc:
            z=sc[r["id"]]["metrics"][cand["name"]];v0=x["candidate96"]["logits"][0]
            need(x["candidate96"]["masked_hashes"][0]["sha256"]==z["logits_sha256"] and abs(v0["teacher_to_masked_kl"]-z["kl"])<1e-12 and abs(v0["nmse"]-z["nmse"])<1e-12 and int(v0["teacher_argmax"]!=v0["masked_argmax"])==z["argmax_mismatch"],"prefix screen replay")
        need(publish(dst/f"{i:02d}.json",x)==v["receipt_sha256"],"prefix byte identity");restored.append(x)
    rec=dict(status="R23H_RESTORED_INTERRUPTED_PREFIX",source_log_sha256=sha(log),source_stderr_sha256=sha((prev/"run.stderr").read_bytes()),source_admission_sha256=ah,receipt_sha256=[v["receipt_sha256"] for v in prefix],restored_count=len(restored),original_terminal="MISSING")
    event(rec["status"],recovery_sha256=sha(canon(rec)),**{k:v for k,v in rec.items() if k!="status"})
    for v in prefix:event(v["status"],**{k:z for k,z in v.items() if k!="status"})
    corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
    hist=pin(R/"run-d01b-r19/baseline/records.jsonl","37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8")
    ids={r["id"] for r in rows};ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};prev={x["id"]:x for x in map(json.loads,hist.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids};need(set(ctx)==set(prev)==ids,"input scope")
    for r in rows:
        x=ctx[r["id"]];z=prev[r["id"]];need(x["line"]==z["line"]==r["line"] and x["direction"]==z["direction"]==r["direction"] and len(x["prompt_ids"])==z["prompt_tokens"]==r["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and z["finish"]=="eos" and z["tokens"][:8]==r["teacher_prefix_tokens"],"input identity")
    with (R/"r23-admissions/batch.lock").open("a+") as lk:
        fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);need(not drv.preflight()["blockers"],"resources")
        c.torch.set_num_threads(8);c.torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==TEACHER and ident["num_experts"]==128,"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8);event("R23H_PROPAGATION_TEACHER_READY",teacher=TEACHER)
        ms={"original96":{int(l):v for l,v in base.items()},"candidate96":{int(l):v for l,v in cand["kept"].items()}};caps=list(restored)
        for i,r in enumerate(rows):
            if i<len(restored):continue
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            x=prop.run(c,e,ctx[r["id"]]["prompt_ids"],r["teacher_prefix_tokens"],ms);x.update(id=r["id"],line=r["line"],direction=r["direction"],consumed_prefix=r["teacher_prefix_tokens"],index=i,teacher=TEACHER,mask_sha256=cand["mask_sha256"],admission_sha256=ah)
            checker.checked(x,r,list(range(1,48)));need(x["original96"]==old[r["id"]]["original96"],"exact prior baseline")
            if r["id"] in sc:
                z=sc[r["id"]]["metrics"][cand["name"]];v=x["candidate96"]["logits"][0]
                need(x["candidate96"]["masked_hashes"][0]["sha256"]==z["logits_sha256"] and abs(v["teacher_to_masked_kl"]-z["kl"])<1e-12 and abs(v["nmse"]-z["nmse"])<1e-12 and int(v["teacher_argmax"]!=v["masked_argmax"])==z["argmax_mismatch"],"screen exact replay")
            h=publish(dst/f"{i:02d}.json",x);caps.append(x);event("R23H_PROPAGATION_CAPTURED",index=i,id=r["id"],receipt_sha256=h)
        groups={}
        for d in DIRS:
            for p in ("prefill","decode"):
                k=d+"/"+p;groups[k]={}
                for n in ("original96","candidate96"):
                    z=[v for x in caps if x["direction"]==d for v in x[n]["logits"] if (v["step"]==0)==(p=="prefill")]
                    need(len(z)==(8 if d=="zh-my" else 2)*(1 if p=="prefill" else 8),"coverage")
                    groups[k][n]=dict(kl=mean([v["teacher_to_masked_kl"] for v in z]),nmse=mean([v["nmse"] for v in z]),argmax_mismatch=sum(v["teacher_argmax"]!=v["masked_argmax"] for v in z),positions=len(z))
        viol=[g+"/"+k for g in groups if g.startswith(("zh-my/","my-zh/")) for k in ("kl","nmse","argmax_mismatch") if groups[g]["candidate96"][k]>groups[g]["original96"][k]+1e-12]
        out=dict(status="R23H_WINNER_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION",selected=cand["name"],mask_sha256=cand["mask_sha256"],admission_sha256=ah,samples=18,groups=groups,primary_violations=viol,primary_diagnostic_pass=not viol,limits="fit fixed trajectories only; no fresh, native quality or physical build")
        h=publish(dst/"report.json",out);event(out["status"],report_sha256=h,**{k:v for k,v in out.items() if k!="status"})
if __name__=="__main__":main()
