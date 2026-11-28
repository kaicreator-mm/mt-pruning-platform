"""R23H: decode-targeted primary fit propagation for four frozen layer40 reversions.
Fixed fit trajectories only; no Fresh, mask refit, or materialization."""
import hashlib,json,math,os,pathlib,sys,fcntl,time,importlib.util
P=pathlib.Path;R=P.home()/".cache/mt-pruning";RR=R/"r23-results"
SCREEN=R/"r23h-pareto-screen-v1"
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
    sd=RR/"r23h-all8-one-swap-prefill-v1"
    p=json.loads(pin(sd/"plan.json","a80858f97ff201b3cf62f25c779ab0786bb6d0a60973204c6fc1f3dd1f8c700b"))
    q=json.loads(pin(RR/"r23h-all8-one-swap-primary-v1/report.json","0760d92157809e5e88a25af5188066060764f21470d5f581986bbf26676d2495"))
    replay=json.loads(pin(P(__file__).parent/"replay.json","f8f8a0dccc68188d76630a58b75661f1fa8dcac2be18f29e2f92d82adf0fd795"))
    plan=json.loads(pin(P(__file__).parent/"search-plan.json","09fb5b383f04097072fc151d0c3a50a025b2e12b033fde005f397b3ad02b26a5"))
    need(replay["status"]=="R23H_ONE_SWAP_PRIMARY_INDEPENDENT_REPLAY_PASS_REJECT_CANDIDATE" and replay["report_sha256"]==plan["parent_report_sha256"] and replay["violations"]==["zh-my/decode/nmse"] and q["eligible_for_secondary_fit_only"]==[],"prior rejection")
    base=next(x for x in p["candidates"] if x["name"]=="l42_remove63_add120")["kept"]
    need(sha(canon(base))==plan["base_mask_sha256"],"base")
    cs=[]
    for rem in (58,62):
        for add in (13,84):
            name=f"l42_63_120_l40_{rem}_{add}"
            mask={k:v[:] for k,v in base.items()}
            need(rem in mask["40"] and add not in mask["40"],"swap")
            mask["40"]=sorted((set(mask["40"])-{rem})|{add})
            h=sha(canon(mask))
            need(set(mask)==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for v in mask.values()),"mask")
            cs.append(dict(name=name,kept=mask,mask_sha256=h))
    need([{k:v for k,v in c.items() if k!="kept"} for c in cs]==plan["candidates"],"frozen inventory")
    return p,{},q,plan["parent_report_sha256"],cs
def prior_failed():
    return True,"0760d92157809e5e88a25af5188066060764f21470d5f581986bbf26676d2495"
def main():
    failed,prior_hash=prior_failed()
    if not failed:return
    lk=(R/"r23-admissions/batch.lock").open("a+")
    fcntl.flock(lk.fileno(),fcntl.LOCK_EX)
    p,sc,q,sh,cs=screen()
    d=SCREEN
    drv=load(d/"driver.py","0735238e15f13fd75e3f6939fc31aa834d8b5854","driver")
    c=load(d/"collector.py","02daf65bb7fe0526b2a700d17013054f7b0eea46","collector")
    prop=load(d/"prop.py","88468e8c2a5905628365da52e1575d5251516512","prop")
    c.verify_code()
    need(c.torch.__version__=="2.10.0+cpu","pinned CPU PyTorch runtime required")
    need(not drv.preflight()["blockers"],"resources")
    ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
    need(ident["teacher"]==TEACHER and ident["num_experts"]==128,"teacher")
    dst=RR/"r23h-l42-l40-decode-targeted-primary-v1";dst.mkdir(mode=0o700)
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
    ad=dict(status="R23H_L42_L40_DECODE_TARGETED_PRIMARY_BATCH_FIT_ONLY",screen_plan_sha256="a80858f97ff201b3cf62f25c779ab0786bb6d0a60973204c6fc1f3dd1f8c700b",parent_primary_report_sha256=sh,search_plan_sha256="09fb5b383f04097072fc151d0c3a50a025b2e12b033fde005f397b3ad02b26a5",previous_recovery_sha256=REC,prior_l42_report_sha256=prior_hash,independent_parent_replay_sha256="f8f8a0dccc68188d76630a58b75661f1fa8dcac2be18f29e2f92d82adf0fd795",candidates={c["name"]:c["mask_sha256"] for c in cs},ids=[r["id"] for i,r in selected_rows],indices=[i for i,r in selected_rows],teacher=TEACHER,holdout="FORBIDDEN")
    ah=publish(dst/"admission.json",ad);event(ad["status"],admission_sha256=ah)
    corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
    hist=pin(R/"run-d01b-r19/baseline/records.jsonl","37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8")
    ids={r["id"] for r in rows};ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};prev={x["id"]:x for x in map(json.loads,hist.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids};need(set(ctx)==set(prev)==ids,"input scope")
    for r in rows:
        x=ctx[r["id"]];z=prev[r["id"]];need(x["line"]==z["line"]==r["line"] and x["direction"]==z["direction"]==r["direction"] and len(x["prompt_ids"])==z["prompt_tokens"]==r["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and z["finish"]=="eos" and z["tokens"][:8]==r["teacher_prefix_tokens"],"input identity")
    c.torch.set_num_threads(8);c.torch.set_num_interop_threads(1)
    e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8);event("R23H_L42_L40_TEACHER_READY",teacher=TEACHER)
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
        event("R23H_L42_L40_DECODE_TARGETED_PRIMARY_CAPTURED",index=i,id=r["id"],receipt_sha256=hashes)
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
    out=dict(status="R23H_L42_L40_DECODE_TARGETED_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION",admission_sha256=ah,samples_per_candidate=10,candidates=ad["candidates"],groups=groups,violations=viol,primary_gate_pass=eligible,eligible_for_secondary_fit_only=[],selected=None,limits="Four frozen layer40 reversions on L42 parent; direct 10-row primary Fit propagation; separate 16-row bidirectional Prefill screen and independent replay before secondary Fit. No Fresh, Holdout, free-running translation or physical build.")
    h=publish(dst/"report.json",out);event(out["status"],report_sha256=h,**{k:v for k,v in out.items() if k!="status"})
if __name__=="__main__":main()
