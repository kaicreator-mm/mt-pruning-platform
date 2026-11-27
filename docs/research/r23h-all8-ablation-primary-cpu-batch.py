"""R23H: conditional primary fit propagation for two predeclared all8 ablations.
Fixed fit trajectories only; no Fresh, mask refit, or materialization."""
import hashlib,json,math,os,pathlib,sys,fcntl,time,importlib.util
P=pathlib.Path;R=P.home()/".cache/mt-pruning";RR=R/"r23-results"
SCREEN=R/"r23h-pareto-screen-v1";SCREEN_RUN=R/"r23h-all8-ablation-screen-run-v2";D_ABLATION=SCREEN_RUN
SD=RR/"r23h-all8-ablation-prefill-v2";SD_OLD=RR/"r23h-pareto-prefill-v1"
PLAN="29e553c04e2cc61cbe999aac0604e9440f248480cadb6c8631f4758d87712acc"
OLD_PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
HYBRID_BLOBS={"all8_revert40":"255c95382a5888f9749d7626395182ba514c3c8d","all8_revert42":"f738ea900952ab19dbf6fd4cbca235dc40e56d05","all8_revert40_42":"97809cb5092ee6b903a4def8ee0139902d24f17d"}
HYBRID_MASKS={"all8_revert40":"02de6ad74290659e3bd9b301ef606c45c129e032abcbbaaf867fe61bb26af84e","all8_revert42":"054614572884a518ce2a6538a0f39c7884667d1551312216df767ee4720f1f0b","all8_revert40_42":"96d26e3699ee87f2560fbbf15a53b6016b7d96aadc744f5aade4b076a90dab2b"}
PRIMARY_CANDIDATES=("all8_revert42","all8_revert40_42")
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
USB=R/"r23g-recovery-ssd-38ed4798"
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
def checked(x,r,index,layers,admission_sha,mask_sha):
    need(x["status"]=="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY" and x["teacher_restored"] is True and x["index"]==index and x["id"]==r["id"] and x["direction"]==r["direction"] and x["line"]==r["line"] and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["teacher"]==TEACHER and x["mask_sha256"]==mask_sha and x["admission_sha256"]==admission_sha,"capture provenance")
    want={(l,"prefill" if j==0 else "decode",j) for l in layers for j in range(9)}
    for n in ("original96","candidate96"):
        a=x[n];ls=a["layer"]
        need(len(ls)==len(want) and {(q["layer"],q["phase"],q["step"]) for q in ls}==want,"layer coverage")
        for q in ls:
            need(all(math.isfinite(q[k]) and q[k]>=0 for k in ("hidden_input_nmse","moe_output_nmse","mean_overlap")) and q["mean_overlap"]<=8 and q["tokens"]==(2 if q["step"]==0 else 1) and type(q["exact_sets"]) is int and 0<=q["exact_sets"]<=q["tokens"],"layer values")
        for key in ("teacher_hashes","masked_hashes"):
            z=a[key];need(len(z)==9 and all(q["step"]==j and len(q["sha256"])==64 and type(q["next"]) is int and q["next"]>=0 for j,q in enumerate(z)),"token hash")
        need(len(a["logits"])==9 and all(q["step"]==j and math.isfinite(q["nmse"]) and math.isfinite(q["teacher_to_masked_kl"]) and q["teacher_argmax"]==a["teacher_hashes"][j]["next"] and q["masked_argmax"]==a["masked_hashes"][j]["next"] for j,q in enumerate(a["logits"])),"logits")
    need(x["original96"]["teacher_hashes"]==x["candidate96"]["teacher_hashes"],"teacher identity")
def screen():
    need((SCREEN_RUN/"exit-code").read_text().strip()=="0" and (SCREEN_RUN/"run.stderr").read_bytes()==b"","terminal ablation screen success")
    p=json.loads(pin(SD/"plan.json",PLAN));need(p["status"]=="R23H_ALL8_ABLATION_PREFILL_SCREEN_ONLY" and p["recovery_sha256"]==REC and p["teacher"]==TEACHER and p["holdout"]=="FORBIDDEN","ablation plan")
    names=["original96","r23g","all8","all10",*HYBRID_BLOBS]
    need(len(p["ids"])==8 and len(set(p["ids"]))==8 and set(p["masks"])==set(names),"ablation inventory")
    need((SCREEN/"exit-code").read_text().strip()=="0" and (SCREEN/"run.stderr").read_bytes()==b"","old control terminal")
    oldplan=json.loads(pin(SD_OLD/"plan.json",OLD_PLAN));oldreport=json.loads(pin(SD_OLD/"report.json",OLD_REPORT))
    need(oldplan["ids"]==p["ids"] and oldreport["selected"]=="all8" and oldreport["samples"]==8,"old screen ids/report")
    oldevents=[json.loads(l) for l in (SCREEN/"run.log").read_bytes().splitlines() if l.startswith(b"{")]
    oldcaps=[x for x in oldevents if x["status"]=="R23H_PREFILL_CAPTURED"];need(len(oldcaps)==8,"old control screen events")
    oldscreen={}
    for i,e in enumerate(oldcaps):
        z=json.loads(pin(SD_OLD/f"{i:02d}.json",e["sha256"]))
        need(z["index"]==i and z["id"]==p["ids"][i] and z["plan_sha256"]==OLD_PLAN,"old control receipt")
        oldscreen[z["id"]]=z
    all_cs=[]
    for n in HYBRID_BLOBS:
        raw=pinblob(D_ABLATION/(n+".json"),HYBRID_BLOBS[n]);x=json.loads(raw)
        need(x["status"]=="R23H_ALL8_SINGLE_LAYER_ABLATION_EXPLORATORY_NOT_ADMITTED" and x["method"]==n and x["mask_sha256"]==HYBRID_MASKS[n] and sha(canon(x["mask"]))==HYBRID_MASKS[n],"hybrid mask identity")
        need(p["hybrid_sources"][n]["blob_sha"]==HYBRID_BLOBS[n] and p["hybrid_sources"][n]["mask_sha256"]==HYBRID_MASKS[n] and p["masks"][n]==HYBRID_MASKS[n],"hybrid plan binding")
        need(set(x["mask"])==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for v in x["mask"].values()),"hybrid mask structure")
        all_cs.append(dict(name=n,mask_sha256=HYBRID_MASKS[n],kept=x["mask"]))
    cs=[x for x in all_cs if x["name"] in PRIMARY_CANDIDATES]
    events=[json.loads(l) for l in (SCREEN_RUN/"run.log").read_bytes().splitlines() if l.startswith(b"{")]
    ev=[x for x in events if x["status"]=="R23H_LAYER_HYBRID_PREFILL_CAPTURED"]
    done=[x for x in events if x["status"]=="R23H_ALL8_ABLATION_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(ev)==8 and len(done)==1,"ablation complete events")
    caps={}
    for i,e in enumerate(ev):
        z=json.loads(pin(SD/f"{i:02d}.json",e["receipt_sha256"]))
        need(e["index"]==z["index"]==i and e["id"]==z["id"]==p["ids"][i] and z["plan_sha256"]==PLAN and set(z["metrics"])==set(names),"ablation capture")
        for n,q in z["metrics"].items():
            need(all(math.isfinite(q[k]) and q[k]>=0 for k in ("kl","nmse")) and q["argmax_mismatch"] in (0,1) and len(q["logits_sha256"])==64 and q["mask_sha256"]==p["masks"][n],"ablation metric")
        for n in ("all8","all10"):
            need(z["metrics"][n]==oldscreen[z["id"]]["metrics"][n],"exact old-control replay")
        caps[z["id"]]=z
    a={n:{k:(sum if k=="argmax_mismatch" else mean)([x["metrics"][n][k] for x in caps.values()]) for k in ("kl","nmse","argmax_mismatch")} for n in names}
    good=[n for n in HYBRID_BLOBS if all(a[n][k]<=a["original96"][k]+1e-12 for k in a[n]) and a[n]["kl"]<a["r23g"]["kl"]]
    good.sort(key=lambda n:(a[n]["kl"],a[n]["nmse"],n))
    q=json.loads(pin(SD/"report.json",done[0]["report_sha256"]))
    need(q["status"]=="R23H_ALL8_ABLATION_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION" and q["plan_sha256"]==PLAN and q["samples"]==8 and q["metrics"]==a and q["eligible_for_primary_fit_only"]==good and q["selected"]==(good[0] if good else None),"ablation screen decision")
    need(done[0]["report_sha256"]=="f35caa64d979ff6c781239a6cab25061af04235df77a282d5c6585a1704de521" and good==list(PRIMARY_CANDIDATES),"pinned ablation result")
    return p,caps,q,done[0]["report_sha256"],cs
def prior_failed():
    prev=R/"r23h-pareto-all10-run-v2"
    d=RR/"r23h-pareto-all10-propagation-v2"
    deadline=time.monotonic()+14400
    while not (prev/"exit-code").is_file():
        need(time.monotonic()<deadline,"prior wait expired");time.sleep(15)
    need((prev/"exit-code").read_text().strip()=="0" and (prev/"run.stderr").read_bytes()==b"","prior terminal success required")
    ev=[json.loads(l) for l in (prev/"run.log").read_bytes().splitlines() if l.startswith(b"{")]
    end=[x for x in ev if x["status"]=="R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION"];caps=[x for x in ev if x["status"]=="R23H_PROPAGATION_CAPTURED"]
    need(len(end)==1 and len(caps)==18,"prior complete captures")
    ad=json.loads(pin(d/"admission.json","bfd5267613c493558d3e0deece69aa4472cce53ea6aea5ad859b5754117ded2f"))
    need(ad["selected"]=="all10" and ad["mask_sha256"]=="a9274a240cb1e990c77b110b9a6231e0ae1c905849a2b7e9295f795bd3a81fe2","prior candidate")
    q=json.loads(pin(d/"report.json",end[0]["report_sha256"]));need(q["admission_sha256"]==sha((d/"admission.json").read_bytes()) and q["samples"]==18 and q["selected"]=="all10" and q["mask_sha256"]==ad["mask_sha256"],"prior report")
    rs=[]
    for i,x in enumerate(caps):
        z=json.loads(pin(d/f"{i:02d}.json",x["receipt_sha256"]))
        need(z["index"]==x["index"]==i and z["id"]==x["id"]==ad["ids"][i] and z["mask_sha256"]==ad["mask_sha256"] and z["admission_sha256"]==q["admission_sha256"],"prior receipt binding")
        rs.append(z)
    groups={}
    for direction in DIRS:
        for phase in ("prefill","decode"):
            k=direction+"/"+phase;groups[k]={}
            for n in ("original96","candidate96"):
                z=[v for x in rs if x["direction"]==direction for v in x[n]["logits"] if (v["step"]==0)==(phase=="prefill")]
                need(len(z)==(8 if direction=="zh-my" else 2)*(1 if phase=="prefill" else 8),"prior coverage")
                groups[k][n]=dict(kl=mean([v["teacher_to_masked_kl"] for v in z]),nmse=mean([v["nmse"] for v in z]),argmax_mismatch=sum(v["teacher_argmax"]!=v["masked_argmax"] for v in z),positions=len(z))
    viol=[g+"/"+k for g in groups if g.startswith(("zh-my/","my-zh/")) for k in ("kl","nmse","argmax_mismatch") if groups[g]["candidate96"][k]>groups[g]["original96"][k]+1e-12]
    need(groups==q["groups"] and viol==q["primary_violations"] and q["primary_diagnostic_pass"]==(not viol),"prior replay")
    event("R23H_ALL8_ABLATION_CONDITIONAL_DECISION",prior_report_sha256=end[0]["report_sha256"],violations=viol,execute=bool(viol))
    return bool(viol),end[0]["report_sha256"]
def main():
    failed,prior_hash=prior_failed()
    if not failed:return
    lk=(R/"r23-admissions/batch.lock").open("a+")
    fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+14400
    while not (SCREEN_RUN/"exit-code").exists():
        need(time.monotonic()<deadline,"screen wait expired");time.sleep(15)
    p,sc,q,sh,cs=screen()
    d=SCREEN
    drv=load(d/"driver.py","0735238e15f13fd75e3f6939fc31aa834d8b5854","driver")
    c=load(d/"collector.py","02daf65bb7fe0526b2a700d17013054f7b0eea46","collector")
    prop=load(d/"prop.py","88468e8c2a5905628365da52e1575d5251516512","prop")
    c.verify_code()
    need(not drv.preflight()["blockers"],"resources")
    ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
    need(ident["teacher"]==TEACHER and ident["num_experts"]==128,"teacher")
    dst=RR/"r23h-all8-ablation-primary-v2";dst.mkdir(mode=0o700)
    g=json.loads(pin(USB/"recovery-receipt.json",REC));need(g["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH","previous recovery")
    plan=json.loads(pinblob(USB/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915"));pp=json.loads(pinblob(USB/"prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde"))
    rows=pp["rows"]+[r for r in plan["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in pp["rows"]}]
    need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in plan["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in plan["holdout"]}),"fit scope")
    base=json.loads(pinblob(SCREEN/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"))["candidates"]["96"]["kept"];need(sha(canon(base))=="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7","base")
    old={}
    for i,r in enumerate(rows):
        x=json.loads(pin(RR/"r23g-l38-propagation-v1"/f"{i:02d}.json",g["receipt_sha256"][i]));need(x["id"]==r["id"],"previous row");old[r["id"]]=x
    selected_rows=[(i,r) for i,r in enumerate(rows) if r["direction"] in ("zh-my","my-zh")]
    need(len(selected_rows)==10,"primary inventory")
    ad=dict(status="R23H_ALL8_ABLATION_PRIMARY_BATCH_FIT_ONLY",screen_plan_sha256=PLAN,screen_report_sha256=sh,previous_recovery_sha256=REC,prior_all10_report_sha256=prior_hash,candidates={c["name"]:c["mask_sha256"] for c in cs},ids=[r["id"] for i,r in selected_rows],indices=[i for i,r in selected_rows],teacher=TEACHER,holdout="FORBIDDEN")
    ah=publish(dst/"admission.json",ad);event(ad["status"],admission_sha256=ah)
    corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
    hist=pin(R/"run-d01b-r19/baseline/records.jsonl","37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8")
    ids={r["id"] for r in rows};ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};prev={x["id"]:x for x in map(json.loads,hist.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids};need(set(ctx)==set(prev)==ids,"input scope")
    for r in rows:
        x=ctx[r["id"]];z=prev[r["id"]];need(x["line"]==z["line"]==r["line"] and x["direction"]==z["direction"]==r["direction"] and len(x["prompt_ids"])==z["prompt_tokens"]==r["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and z["finish"]=="eos" and z["tokens"][:8]==r["teacher_prefix_tokens"],"input identity")
    c.torch.set_num_threads(8);c.torch.set_num_interop_threads(1)
    e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8);event("R23H_PROPAGATION_TEACHER_READY",teacher=TEACHER)
    masks={c0["name"]:{int(l):v for l,v in c0["kept"].items()} for c0 in cs};bm={int(l):v for l,v in base.items()};caps={c0["name"]:[] for c0 in cs}
    for i,r in selected_rows:
        need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
        prompt=ctx[r["id"]]["prompt_ids"];prefix=r["teacher_prefix_tokens"]
        t=prop.observe(c,e,prompt,prefix)
        a=prop.observe(c,e,prompt,prefix,bm);bc=prop.compare(t,a);del a
        need(bc==old[r["id"]]["original96"],"exact prior baseline")
        batch={}
        for c0 in cs:
            n=c0["name"];b=prop.observe(c,e,prompt,prefix,masks[n]);v=prop.compare(t,b);del b
            batch[n]=dict(status="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY",teacher_restored=True,original96=bc,candidate96=v,scope="common fixed input sequence; actual propagated states, not free-running translation; selected prefill rows plus every decode step",id=r["id"],line=r["line"],direction=r["direction"],consumed_prefix=prefix,index=i,teacher=TEACHER,mask_sha256=c0["mask_sha256"],admission_sha256=ah)
        z=prop.observe(c,e,prompt,prefix);need(prop.loghash(t)==prop.loghash(z),"teacher restoration");del t,z
        hashes={}
        for c0 in cs:
            n=c0["name"];x=batch[n];checked(x,r,i,list(range(1,48)),ah,c0["mask_sha256"])
            if r["id"] in sc:
                z=sc[r["id"]]["metrics"][n];v=x["candidate96"]["logits"][0]
                need(x["candidate96"]["masked_hashes"][0]["sha256"]==z["logits_sha256"] and abs(v["teacher_to_masked_kl"]-z["kl"])<1e-12 and abs(v["nmse"]-z["nmse"])<1e-12 and int(v["teacher_argmax"]!=v["masked_argmax"])==z["argmax_mismatch"],"screen exact replay")
            hashes[n]=publish(dst/f"{n}-{i:02d}.json",x);caps[n].append(x)
        event("R23H_ALL8_ABLATION_PRIMARY_CAPTURED",index=i,id=r["id"],receipt_sha256=hashes)
    groups={};viol={}
    for name,rs in caps.items():
        groups[name]={}
        for direction in ("zh-my","my-zh"):
            for phase in ("prefill","decode"):
                k=direction+"/"+phase;groups[name][k]={}
                for n in ("original96","candidate96"):
                    z=[v for x in rs if x["direction"]==direction for v in x[n]["logits"] if (v["step"]==0)==(phase=="prefill")]
                    need(len(z)==(8 if direction=="zh-my" else 2)*(1 if phase=="prefill" else 8),"coverage")
                    groups[name][k][n]=dict(kl=mean([v["teacher_to_masked_kl"] for v in z]),nmse=mean([v["nmse"] for v in z]),argmax_mismatch=sum(v["teacher_argmax"]!=v["masked_argmax"] for v in z),positions=len(z))
        viol[name]=[g+"/"+k for g in groups[name] for k in ("kl","nmse","argmax_mismatch") if groups[name][g]["candidate96"][k]>groups[name][g]["original96"][k]+1e-12]
    eligible=sorted((n for n in groups if not viol[n]),key=lambda n:(mean([groups[n][g]["candidate96"][k]/max(groups[n][g]["original96"][k],1e-12) for g in groups[n] for k in ("kl","nmse")]),n))
    out=dict(status="R23H_ALL8_ABLATION_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION",admission_sha256=ah,samples_per_candidate=10,candidates=ad["candidates"],groups=groups,violations=viol,eligible_for_secondary_fit_only=eligible,selected=eligible[0] if eligible else None,limits="Two candidates; 10 primary fit trajectories (8 zh-my, 2 my-zh), fixed teacher-forced prefixes and actual propagated decode states. Complete 8 secondary fit rows for any primary winner before full fit replay. No Fresh, holdout, free-running translation, or physical build.")
    h=publish(dst/"report.json",out);event(out["status"],report_sha256=h,**{k:v for k,v in out.items() if k!="status"})
if __name__=="__main__":main()
