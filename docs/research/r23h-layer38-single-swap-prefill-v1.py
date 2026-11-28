"""R23H: four one-swap layer-38 variants between Stable96 and R23G.
Fit-only actual Prefill KL/NMSE; all three baselines are remeasured in the same run."""
import hashlib,importlib.util,json,math,os,pathlib,sys,time,fcntl,traceback
P=pathlib.Path
R=P.home()/".cache/mt-pruning"
RR=R/"r23-results"
USB_PRIMARY=P("/mnt/mt-storage/mt-pruning/evidence/r23g-recovery-38ed4798")
USB_FALLBACK=R/"r23g-recovery-ssd-38ed4798"
USB=USB_PRIMARY if (USB_PRIMARY/"expanded.json").is_file() else USB_FALLBACK
PREVIOUS=RR/"r23h-factorial-prefill-v5"
PREVIOUS_RUN=R/"r23h-factorial-run-v5"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
PREVIOUS_PLAN_SHA="580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
PREVIOUS_REPORT_SHA="e906a1e16b38902ea3282afc44b9afcc190b1673b3ffb9cdfc0d01d456b77fa1"
EXPANDED_BLOB="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CANDIDATE_SHA="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
STABLE_SHA="488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef"
CORPUS_SHA="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
ORIGINAL_BLOB="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
ORIGINAL_MASK_SHA="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
R23G_MASK_SHA="77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
PINS={"driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46"}
SWAPS=[(41,89),(41,123),(43,89),(43,123)]
CANDIDATE_ORDER=["l38_swap_41_to_89","l38_swap_41_to_123","l38_swap_43_to_89","l38_swap_43_to_123"]
BASELINES=["original96","stable96","r23g"]
ALL_NAMES=BASELINES+CANDIDATE_ORDER

def need(x,m):
    if not x: raise ValueError(m)
def sha(b): return hashlib.sha256(b).hexdigest()
def blob(b): return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pin(p,h,git=False):
    need(p.is_file() and not p.is_symlink(),"regular "+str(p))
    b=p.read_bytes();need((blob if git else sha)(b)==h,"pin "+str(p));return b
def load(d,n):
    p=d/(n+".py");pin(p,PINS[n],True)
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def mean(a): return math.fsum(a)/len(a)
def publish(p,x):
    b=canon(x)+b"\n"
    with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
    return sha(b)
def event(status,**x):print(json.dumps(dict(status=status,**x),sort_keys=True),flush=True)
def check_mask(mask,label):
    need(set(mask)=={str(i) for i in range(1,48)},label+" layer inventory")
    for layer,experts in mask.items():
        need(type(experts) is list and len(experts)==96,label+" expert count "+layer)
        need(experts==sorted(set(experts)) and all(type(x) is int and 0<=x<128 for x in experts),label+" expert order/range "+layer)
def metric(t,z):
    a=t.double();b=z.double();p=a.log_softmax(-1);q=b.log_softmax(-1)
    kl=float((p.exp()*(p-q)).sum(dim=-1).mean())
    nmse=float((a-b).square().sum()/a.square().sum())
    need(math.isfinite(kl) and kl>=-1e-12 and math.isfinite(nmse),"finite metric")
    return dict(kl=max(0.,kl),nmse=nmse,argmax_mismatch=int(a[-1].argmax()!=b[-1].argmax()))
def aggregate(group,name):
    return {"kl":mean([x["metrics"][name]["kl"] for x in group]),
            "nmse":mean([x["metrics"][name]["nmse"] for x in group]),
            "argmax_mismatch":sum(x["metrics"][name]["argmax_mismatch"] for x in group)}
def prepare(d):
    expanded=json.loads(pin(USB/"expanded.json",EXPANDED_BLOB,True))
    rows=[r for r in expanded["fit"] if r["direction"] in ("zh-my","my-zh")]
    need(len(rows)==16 and all(sum(r["direction"]==x for r in rows)==8 for x in ("zh-my","my-zh")),"balanced Fit rows")
    original=json.loads(pin(d/"original.json",ORIGINAL_BLOB,True))["candidates"]["96"]["kept"]
    stable=json.loads(pin(RR/"expanded-stable96-v1.json",STABLE_SHA))["kept"]
    r23g=json.loads(pin(USB/"candidate.json",CANDIDATE_SHA))["kept"]
    need(sha(canon(original))==ORIGINAL_MASK_SHA and sha(canon(r23g))==R23G_MASK_SHA,"base masks")
    need(all(len(set(stable[str(l)])-set(original[str(l)]))==2 for l in (10,15,34)),"stable layer expert changes")
    need([int(l) for l in stable if stable[l]!=r23g[l]]==[38],"R23G differs from Stable96 only at layer38")
    need(sorted(set(stable["38"])-set(r23g["38"]))==[41,43],"layer38 removed experts")
    need(sorted(set(r23g["38"])-set(stable["38"]))==[89,123],"layer38 added experts")
    prev_plan=json.loads(pin(PREVIOUS/"plan.json",PREVIOUS_PLAN_SHA))
    prev_report=json.loads(pin(PREVIOUS/"report.json",PREVIOUS_REPORT_SHA))
    need(prev_plan["status"]=="R23H_FACTORIAL_FIT_PREFILL_SCREEN_ONLY" and prev_plan["teacher"]==TEACHER,"previous plan")
    need(prev_report["status"]=="R23H_FACTORIAL_BIDIRECTIONAL_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION","previous report")
    need(prev_plan["ids"]==[r["id"] for r in rows],"previous Fit IDs")
    prev_events=[json.loads(line) for line in (PREVIOUS_RUN/"run.log").read_bytes().splitlines() if line.startswith(b"{")]
    captures=[x for x in prev_events if x.get("status")=="R23H_FACTORIAL_PREFILL_CAPTURED"]
    need(len(captures)==16 and [x["index"] for x in captures]==list(range(16)),"previous receipt event inventory")
    previous=[]
    for i in range(16):
        raw=pin(PREVIOUS/f"{i:02d}.json",captures[i]["sha256"])
        previous.append(json.loads(raw))
    need([x["id"] for x in previous]==[r["id"] for r in rows],"previous receipt IDs")
    need(all(x["teacher_sha256"]==TEACHER for x in previous),"previous teacher hashes")
    corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl",CORPUS_SHA)
    ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in {r["id"] for r in rows}}
    need(len(ctx)==16,"calibration corpus coverage")
    for r in rows:need(sha(canon(ctx[r["id"]]["prompt_ids"]))==r["prompt_ids_sha256"],"prompt identity "+r["id"])
    candidates={}
    for (removed,added),name in zip(SWAPS,CANDIDATE_ORDER):
        layer38=set(stable["38"]);need(removed in layer38 and added not in layer38,"swap endpoints "+name)
        layer38.remove(removed);layer38.add(added)
        mask={str(l):(sorted(layer38) if l==38 else stable[str(l)]) for l in range(1,48)}
        check_mask(mask,name)
        need(mask["38"]==sorted(set(stable["38"])-{removed}|{added}),name+" exact layer38 swap")
        need(sum(len(set(mask[str(l)])-set(stable[str(l)])) for l in range(1,48))==1,name+" one-swap count")
        candidates[name]=dict(name=name,removed_expert=removed,added_expert=added,mask_sha256=sha(canon(mask)),kept=mask)
    plan=dict(status="R23H_LAYER38_SINGLE_SWAP_PREFILL_SCREEN_ONLY",teacher=TEACHER,
        source_previous_plan_sha256=PREVIOUS_PLAN_SHA,source_previous_report_sha256=PREVIOUS_REPORT_SHA,
        source_stable96_sha256=STABLE_SHA,source_r23g_mask_sha256=R23G_MASK_SHA,
        source_original96_mask_sha256=ORIGINAL_MASK_SHA,ids=[r["id"] for r in rows],
        directions={"zh-my":8,"my-zh":8},layer38_removed_experts=[41,43],layer38_added_experts=[89,123],
        baselines=[{"name":"original96","mask_sha256":sha(canon(original))},
                   {"name":"stable96","mask_sha256":sha(canon(stable))},
                   {"name":"r23g","mask_sha256":sha(canon(r23g))}],
        candidates=[candidates[n] for n in CANDIDATE_ORDER],holdout="FORBIDDEN",
        gate={"zh-my":"candidate KL strictly below original96; NMSE and argmax_mismatch no worse than original96",
              "my-zh":"candidate KL, NMSE, and argmax_mismatch no worse than R23G",
              "advance":"only if both direction gates pass; otherwise stop before Decode"})
    return rows,ctx,previous,original,stable,r23g,plan
def run(d):
    rows,ctx,previous,original,stable,r23g,plan=prepare(d)
    dst=RR/"r23h-layer38-single-swap-prefill-v1";dst.mkdir(mode=0o700)
    ph=publish(dst/"plan.json",plan)
    event("R23H_LAYER38_SINGLE_SWAP_PLAN_FROZEN",sha256=ph,candidates=[{"name":x["name"],"removed":x["removed_expert"],"added":x["added_expert"]} for x in plan["candidates"]])
    drv=load(d,"driver");c=load(d,"collector");c.verify_code()
    corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl",CORPUS_SHA)
    masks={"original96":original,"stable96":stable,"r23g":r23g}
    masks.update({x["name"]:x["kept"] for x in plan["candidates"]})
    with (R/"r23-admissions/batch.lock").open("a+") as lk:
        deadline=time.monotonic()+7200
        while True:
            try:fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:need(time.monotonic()<deadline,"lock wait expired");time.sleep(10)
        event("R23H_LAYER38_SINGLE_SWAP_LOCK_ACQUIRED")
        pre=drv.preflight();need(not pre["blockers"],str(pre))
        event("R23H_LAYER38_SINGLE_SWAP_RESOURCE_PREFLIGHT",preflight_status=pre["status"],blockers=pre["blockers"],gpu_pids=pre["gpu_pids"],model_processes=pre["model_processes"],memory=pre["memory"])
        c.torch.set_num_threads(8);c.torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
        need(ident["teacher"]==TEACHER and ident["num_experts"]==128,"exact teacher identity")
        event("R23H_LAYER38_SINGLE_SWAP_TEACHER_IDENTITY",teacher=ident["teacher"],num_experts=ident["num_experts"],identity=ident)
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8)
        event("R23H_LAYER38_SINGLE_SWAP_TEACHER_READY",teacher=TEACHER)
        def prefill(prompt,mask):
            try:
                e.set_mask(None if mask is None else {int(l):v for l,v in mask.items()})
                with c.torch.inference_mode():z=e.forward(prompt,e.new_cache(len(prompt)+9),"prefill")
                return c.logits_sha(z),z[[len(prompt)//2-1,len(prompt)-1]].clone()
            finally:e.set_mask(None)
        captures=[]
        for i,r in enumerate(rows):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            prompt=ctx[r["id"]]["prompt_ids"];old=previous[i]
            th,t=prefill(prompt,None)
            need(th==old["teacher_sha256"],"historical teacher hash "+r["id"])
            event("R23H_LAYER38_SINGLE_SWAP_TEACHER_HASH_CHECK",index=i,id=r["id"],actual=th,expected=old["teacher_sha256"],match=True)
            rec=dict(id=r["id"],direction=r["direction"],index=i,plan_sha256=ph,teacher_sha256=th,metrics={})
            for name,mask in masks.items():
                h,z=prefill(prompt,mask);q=metric(t,z)
                if name in ("original96","r23g"):
                    oldq=old["metrics"][name]
                    need(h==oldq["logits_sha256"] and abs(q["kl"]-oldq["kl"])<1e-12
                         and abs(q["nmse"]-oldq["nmse"])<1e-12
                         and q["argmax_mismatch"]==oldq["argmax_mismatch"],
                         "historical baseline exact replay "+name+"/"+r["id"])
                rec["metrics"][name]=dict(q,logits_sha256=h,mask_sha256=sha(canon(mask)))
            h,_=prefill(prompt,None);need(h==th,"teacher restoration")
            h=publish(dst/f"{i:02d}.json",rec);captures.append(rec)
            event("R23H_LAYER38_SINGLE_SWAP_PREFILL_CAPTURED",index=i,id=r["id"],sha256=h,kl={n:q["kl"] for n,q in rec["metrics"].items()})
        metrics={name:aggregate(captures,name) for name in ALL_NAMES}
        by_direction={d:{name:aggregate([x for x in captures if x["direction"]==d],name) for name in ALL_NAMES} for d in ("zh-my","my-zh")}
        eligible=[]
        for name in CANDIDATE_ORDER:
            zh,zh0=by_direction["zh-my"][name],by_direction["zh-my"]["original96"]
            my,myg=by_direction["my-zh"][name],by_direction["my-zh"]["r23g"]
            if zh["kl"]<zh0["kl"] and zh["nmse"]<=zh0["nmse"]+1e-12 and zh["argmax_mismatch"]<=zh0["argmax_mismatch"] and my["kl"]<=myg["kl"]+1e-12 and my["nmse"]<=myg["nmse"]+1e-12 and my["argmax_mismatch"]<=myg["argmax_mismatch"]:
                eligible.append(name)
        out=dict(status="R23H_LAYER38_SINGLE_SWAP_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION",
            plan_sha256=ph,samples=16,metrics=metrics,by_direction=by_direction,
            eligible_for_full_fit_propagation_only=eligible,selected=eligible[0] if eligible else None,
            limits="Fixed Fit Prefill only for zh-my/my-zh; all baselines remeasured in this run; no Decode, Fresh, Holdout, quality admission, or physical model build.")
        rh=publish(dst/"report.json",out)
        event(out["status"],report_sha256=rh,**{k:v for k,v in out.items() if k!="status"})
if __name__=="__main__":
    try:run(P(sys.argv[1]))
    except Exception as exc:
        event("R23H_LAYER38_SINGLE_SWAP_FATAL",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        raise
