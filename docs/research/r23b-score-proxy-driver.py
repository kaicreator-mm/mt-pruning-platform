"""Bounded R23B score mask and local proxy comparison.

Fit-only construction: read exactly 12 verified profiles, evaluate seven
same-budget masks on fit, freeze one layerwise candidate. Only then read
fresh holdout prompt/prefix records for rejection. No training, physical
materialization or translation acceptance. Default preflight, explicit admission.
"""
import argparse,hashlib,importlib.util,json,math,pathlib,sys,fcntl,os,time
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
PINS={"profile":"4316eda170d6c0ef1548c9d571409d96bcd43fd7","masks":"34a06701d47ce0e3ab888882a5b20c75745204d4","batch":"08abbbfbd930e532d41602dd29a5c70f2bc474e3","verifier":"e56b25b3fd92ccbd91455628606925a02af49450","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46","pilot":"19c8c851b1647cc0f495d6a7f1122219c42a0cdb","sparse":"1c38f6e928ff653fbbef584b298ff65e194900fe","selector":"f343ec873d6ab3801918cb20d795bc4d6a7c5155","fresh":"bc6d92ef87d23c954cd1c3f86527a39b52b5cc18"}
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"))
LAYERS=list(range(1,48));BASE="original96"
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def load(p,name):
    need(blob(p.read_bytes())==PINS[name],"source pin "+name)
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def check_scope(fit,hold):
    for rows in (fit,hold):
        need(len(rows)==12 and len({r["line"] for r in rows})==12 and all(r["id"].startswith("calibration:") for r in rows),"scope")
        need(all(sum(x["direction"]==d for x in rows)==2 for d in DIRS),"direction balance")
    need(not ({r["line"] for r in fit}&{r["line"] for r in hold}),"aligned isolation")
def fit_profiles(p,m,v,rows,old,receipt_hashes,admission_sha,root):
    need(len(receipt_hashes)==12 and receipt_hashes[:10]==p.LEGACY_RECEIPTS,"fit receipt inventory")
    result=[]
    for i,r in enumerate(rows[:12]):
        raw=v.read_regular(root/f"{i:02d}.json");need(sha(raw)==receipt_hashes[i],"profile bytes")
        x=v.strict(raw);need(x["index"]==i and x["id"]==r["id"] and x["direction"]==r["direction"] and x["subset"]=="fit" and x["teacher"]==v.TEACHER and x["engine_head"]==v.ENGINE and x["plan_blob"]==v.PLAN and x["threads"]==8 and x["hook_calls"]==423 and x["num_experts"]==128 and x["top_k"]==8,"profile identity")
        z=x["teacher_logits"];need(len(z)==9 and [t["input_token"] for t in z[1:]]==r["teacher_prefix_tokens"],"actual prefix")
        if i<10:
            need(x["admission_sha256"]==p.LEGACY_ADMISSION and x["matched_r23a_logits"] is True and p.normalized(z)==p.normalized(old[r["id"]]["teacher_logits"]),"legacy replay")
        else:
            need(x["admission_sha256"]==admission_sha and x["plain_logits"]==z==x["restored_logits"] and x["same_process_observer_equivalence"] is True,"new exact observer")
            dif=p.differences(old[r["id"]]["teacher_logits"],z);need(x["cross_process_r23a_differences"]==dif and x["matched_r23a_logits"]==(not dif),"cross-run ledger")
        for key,y in x["profile"].items():
            m.check(y,128,8,x["scale"]);need(y["tokens"]==(r["prompt_tokens"] if key.endswith("/prefill") else 8),"token coverage")
        result.append(x)
    return result
def aggregate(rows,captures,names,layers):
    need(set(captures)=={r["id"] for r in rows},"capture inventory")
    result={l:{n:[[] for _ in GROUPS] for n in names} for l in layers}
    for r in rows:
        x=captures[r["id"]];need(x["id"]==r["id"] and x["direction"]==r["direction"] and x["line"]==r["line"] and x["logit_equivalence"] is True,"capture identity")
        obs=x["observations"];seen={(o["layer"],o["phase"],o["step"]):o for o in obs}
        need(len(obs)==len(seen)==3*len(layers) and set(seen)=={(l,p,j) for l in layers for p,j in (("prefill",0),("decode",1),("decode",8))},"coverage")
        for o in obs:
            need(set(o["losses"])==set(names) and o["row_count"]==(2 if o["phase"]=="prefill" else 1) and all(math.isfinite(o[k]) and 0<=o[k]<=1e-4 for k in ("routed_drift","combined_drift")),"observation")
            need(all(set(z)=={"routed_error","combined_error"} and all(math.isfinite(v) and v>=0 for v in z.values()) for z in o["losses"].values()),"losses")
        for l in layers:
            for n in names:
                for phase,steps in (("prefill",[0]),("decode",[1,8])):
                    g=GROUPS.index(r["direction"]+"/"+phase)
                    result[l][n][g].append(math.fsum(seen[l,phase,j]["losses"][n]["combined_error"] for j in steps)/len(steps))
    need(all(len(v)==2 for cell in result.values() for a in cell.values() for v in a),"group coverage")
    return {l:{n:[math.fsum(v)/len(v) for v in a] for n,a in cell.items()} for l,cell in result.items()}
def means(data,choice):
    return [math.fsum(data[l][choice[l]][j] for l in sorted(data))/len(data) for j in range(12)]
def feasible(v,b):
    return all(v[j]<=b[j]+1e-12 for j in range(4)) and all(v[j]<=1.02*b[j]+1e-12 for j in range(4,12)) and math.fsum(v)<=math.fsum(b)+1e-12
def objective(v):return (max(v[:4]),math.fsum(v[:4])/4,math.fsum(v)/12)
def better(a,b):
    for x,y in zip(a,b):
        if x<y-1e-12:return True
        if x>y+1e-12:return False
    return False
def select_fit(data,masks):
    names=tuple(sorted(masks));choice={l:BASE for l in data};base=means(data,choice);cur=base;changes=[]
    for _ in range(2*len(data)):
        best=None
        for l in sorted(data):
            for n in names:
                if n==choice[l]:continue
                v=[a+(data[l][n][j]-data[l][choice[l]][j])/len(data) for j,a in enumerate(cur)]
                if feasible(v,base) and better(objective(v),objective(cur)) and (best is None or better(objective(v),objective(best[0]))):best=(v,l,n)
        if best is None:break
        _,l,n=best;changes.append([l,choice[l],n]);choice[l]=n;cur=means(data,choice);need(feasible(cur,base),"fit guard")
    kept={str(l):masks[choice[l]][str(l)] for l in sorted(data)}
    return dict(status="FIT_FROZEN_NOT_ADMITTED",selected_fit_only={str(l):n for l,n in choice.items()},changes=changes,fit_baseline=dict(zip(GROUPS,base)),fit_selected=dict(zip(GROUPS,cur)),fit_methods={n:dict(zip(GROUPS,means(data,{l:n for l in data}))) for n in names},kept=kept,mask_sha256=sha(canon(kept)))
def reject_holdout(data,fit):
    b=means(data,{l:BASE for l in data});a=means(data,{l:"selected96" for l in data})
    improvement=(objective(b)[1]-objective(a)[1])/max(objective(b)[1],1e-12)
    ok=bool(fit["changes"]) and feasible(a,b) and improvement>=.05
    return dict(status="LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION" if ok else "NO_ADMISSION_FROM_LOCAL_PROXY",candidate={"kept":fit["kept"],"mask_sha256":fit["mask_sha256"]} if ok else None,fit_mask_sha256=fit["mask_sha256"],holdout_baseline=dict(zip(GROUPS,b)),holdout_selected=dict(zip(GROUPS,a)),holdout_primary_relative_improvement=improvement,violations=[GROUPS[j] for j in range(12) if a[j]>b[j]*(1 if j<4 else 1.02)+1e-12],overall_guard=math.fsum(a)<=math.fsum(b)+1e-12)
def selftest():
    ms={BASE:{"1":[0,1],"2":[0,1]},"reap":{"1":[1,2],"2":[1,2]}}
    data={1:{BASE:[.2]*12,"reap":[.1]*12},2:{BASE:[.2]*12,"reap":[.3]*12}}
    f=select_fit(data,ms);need(f["selected_fit_only"]=={"1":"reap","2":BASE},"fit selection")
    good={1:{BASE:[.2]*12,"selected96":[.1]*12},2:{BASE:[.2]*12,"selected96":[.2]*12}}
    bad={1:{BASE:[.2]*12,"selected96":[.3]*12},2:{BASE:[.2]*12,"selected96":[.2]*12}}
    need(reject_holdout(good,f)["candidate"] is not None and reject_holdout(bad,f)["candidate"] is None,"holdout reject no refit")
    need(f["mask_sha256"]==sha(canon(f["kept"])),"mask pin")
    print(json.dumps(dict(status="PASS_SYNTHETIC_FIT_FREEZE_HOLDOUT_REJECT",real_capture="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--inputs",type=P);a.add_argument("--admission",type=P);a.add_argument("--selftest",action="store_true");a.add_argument("--execute",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs is not None and q.admission is not None,"explicit inputs and admission")
    d=q.inputs
    p,m,b,v,drv,c,pilot,s,sel=[load(d/(n+".py"),n) for n in ("profile","masks","batch","verifier","driver","collector","pilot","sparse","selector")]
    c.verify_code();w=c.load_witness(d/"witness.py")
    raw=q.admission.read_bytes();ad=v.strict(raw);need(raw==canon(ad) and ad["status"]=="R23B_SCORE_PROXY_ADMITTED" and ad["teacher"]==v.TEACHER and ad["engine_head"]==v.ENGINE and ad["plan_blob"]==v.PLAN and ad["fresh_blob"]==PINS["fresh"] and ad["threads"]==8 and ad["profile_job"]=="job_01M8E6CJ5G003989WQ8HGH7J9K","admission")
    corpus=ROOT/"run-d01b-r19/corpus/corpus.jsonl";baseline=ROOT/"run-d01b-r19/baseline/records.jsonl"
    c.pilot_inputs(d/"plan.json",corpus,baseline,d/"original.json",d/"guarded.json")
    plan,rows=v.frozen(d/"plan.json");fit=plan["fit"]
    freshraw=(d/"fresh.json").read_bytes();need(blob(freshraw)==PINS["fresh"],"fresh pin");fresh=v.strict(freshraw)["rows"];check_scope(fit,fresh)
    need(not ({r["line"] for r in rows}&{r["line"] for r in fresh}),"prior holdout isolation")
    need(ad["fit_ids"]==[r["id"] for r in fit] and ad["fresh_ids"]==[r["id"] for r in fresh],"admitted scope")
    first,_=v.first_receipt(ROOT/"r23-results/sparse-first.jsonl",ROOT/"r23-admissions/sparse-first.json",p.FIRST_SHA,rows[0]);old={rows[0]["id"]:first,**v.load_batch(ROOT/"r23-results/batch",rows)}
    profile_dir=ROOT/"r23-results/reap-full24"
    need(sorted(x.name for x in profile_dir.iterdir())==[f"{i:02d}.json" for i in range(24)],"completed profile inventory")
    need(sha((ROOT/"r23-admissions/reap-full24-resume-v2.json").read_bytes())==ad["resume_admission_sha256"],"resume admission pin")
    profiles=fit_profiles(p,m,v,rows,old,ad["fit_profile_sha256"],ad["resume_admission_sha256"],profile_dir)
    orig=sel.masks_from(d/"original.json",d/"guarded.json",d/"p04.json")[BASE]
    inventory=m.make(profiles,orig,LAYERS);masks={n:x["kept"] for n,x in inventory["methods"].items()}
    pre=drv.preflight();print(json.dumps(dict(pre,mask_hashes={n:sha(canon(z)) for n,z in masks.items()},fit_samples=12,fresh_samples=12)),flush=True)
    if not q.execute:return
    need(not pre["blockers"],"resources")
    dst=ROOT/"r23-results/score-proxy-v1"
    ctx={x["id"]:x for x in map(json.loads,corpus.read_bytes().splitlines()) if x["split"]=="calibration"}
    base={x["id"]:x for x in map(json.loads,baseline.read_bytes().splitlines()[1:]) if x["split"]=="calibration"}
    def run_capture(e,r,ms):
        row=ctx[r["id"]];hist=base[r["id"]];need(r["line"]==row["line"]==hist["line"] and r["direction"]==row["direction"]==hist["direction"] and hist["finish"]=="eos" and r["teacher_prefix_tokens"]==hist["tokens"][:8],"input binding")
        x=pilot.capture(c,w,s,e,dict(r,split="calibration"),row["prompt_ids"],{n:{int(l):ids for l,ids in z.items()} for n,z in ms.items()})
        need(x["candidate_masks"]=={n:sha(canon(z)) for n,z in ms.items()} and x["logit_equivalence"] is True and len(x["teacher_logits"])==9 and len(x["observations"])==141 and x["extra_oracle_evaluations"]==512,"capture contract")
        return b.annotate(x,r)
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);need(not drv.preflight()["blockers"] and not dst.exists(),"exclusive resource/path")
        dst.mkdir(mode=0o700);b.publish(dst/"admission.json",ad);b.publish(dst/"inventory.json",inventory)
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==v.TEACHER and ident["num_experts"]==128 and not drv.preflight()["blockers"],"teacher identity")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8)
        c.forward_sequence=lambda e,prompt,prefix,callback=None:b.forced_forward(c,e,prompt,prefix,callback)
        print(json.dumps(dict(status="TEACHER_READY",teacher=ident["teacher"],admission_sha256=sha(raw))),flush=True)
        results={}
        for i,r in enumerate(fit):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            x=run_capture(e,r,masks);results[r["id"]]=x;h=b.publish(dst/f"fit-{i:02d}.json",x)
            print(json.dumps(dict(status="FIT_CAPTURED",completed=i+1,id=r["id"],receipt_sha256=h,sparse=x["sparse_counterfactual_evaluations"])),flush=True)
        f=select_fit(aggregate(fit,results,tuple(masks),LAYERS),masks);fhash=b.publish(dst/"fit-frozen.json",f)
        print(json.dumps(dict(status=f["status"],mask_sha256=f["mask_sha256"],receipt_sha256=fhash,changes=len(f["changes"]))),flush=True)
        if not f["changes"]:
            b.publish(dst/"decision.json",dict(status="NO_ADMISSION_FROM_FIT",candidate=None));return
        ms={BASE:orig,"selected96":f["kept"]};results={}
        for i,r in enumerate(fresh):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            x=run_capture(e,r,ms);results[r["id"]]=x;h=b.publish(dst/f"fresh-{i:02d}.json",x)
            print(json.dumps(dict(status="FRESH_CAPTURED",completed=i+1,id=r["id"],receipt_sha256=h,sparse=x["sparse_counterfactual_evaluations"])),flush=True)
        out=reject_holdout(aggregate(fresh,results,tuple(ms),LAYERS),f);h=b.publish(dst/"decision.json",out)
        print(json.dumps(dict(out,receipt_sha256=h)),flush=True)
if __name__=="__main__":main()
