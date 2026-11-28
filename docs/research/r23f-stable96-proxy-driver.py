"""R23F stable uniform96 fixed-mask local proxy validation.
Fit48 first, fresh24 only after full fit acceptance. No refit, no holdout
candidate choice. Separate batches, immutable receipts and phase decisions.
"""
import argparse,fcntl,hashlib,importlib.util,json,math,os,pathlib,sys,time
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PLAN="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
ORIGINAL="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
BASE_SHA="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
CORPUS="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
BASELINE="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8"
PINS={"batch":"08abbbfbd930e532d41602dd29a5c70f2bc474e3","verifier":"e56b25b3fd92ccbd91455628606925a02af49450","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46","pilot":"19c8c851b1647cc0f495d6a7f1122219c42a0cdb","sparse":"1c38f6e928ff653fbbef584b298ff65e194900fe"}
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
GROUPS=tuple(d+"/"+p for d in DIRS for p in ("prefill","decode"));NAMES=("original96","stable96")
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(x):return hashlib.sha1(b"blob "+str(len(x)).encode()+b"\0"+x).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular input");return p.read_bytes()
def load(p,n):
    need(blob(read(p))==PINS[n],"source "+n)
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def checked_capture(x,r,masks,layers):
    expected={n:sha(canon(m)) for n,m in masks.items()}
    need(x["status"]=="LOCAL_OUTPUT_PROXY_ONLY" and x["candidate_masks"]==expected and x["logit_equivalence"] is True and x["id"]==r["id"] and x["line"]==r["line"] and x["direction"]==r["direction"],"sample identity")
    need(x["prefix_policy"]=="FIXED_HISTORICAL_PREFIX_LOCAL_PROXY" and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["threads"]==8,"fixed input")
    z=x["teacher_logits"];need(len(z)==9 and [q["input_token"] for q in z[1:]]==r["teacher_prefix_tokens"] and [(q["phase"],q["step"]) for q in z]==[("prefill",0)]+[("decode",j) for j in range(1,9)],"input trace")
    mis=[dict(step=j,expected=t,actual=q["next"]) for j,(t,q) in enumerate(zip(r["teacher_prefix_tokens"],z[:8]),1) if t!=q["next"]]
    need(x["historical_argmax_mismatches"]==mis and x["historical_greedy_match"]==(not mis),"argmax ledger")
    obs=x["observations"];seen={(o["layer"],o["phase"],o["step"]):o for o in obs}
    need(len(obs)==len(seen)==len(layers)*3 and set(seen)=={(l,p,j) for l in layers for p,j in (("prefill",0),("decode",1),("decode",8))},"layer coverage")
    for o in obs:
        need(set(o["losses"])==set(NAMES) and o["row_count"]==(2 if o["phase"]=="prefill" else 1) and all(math.isfinite(o[k]) and 0<=o[k]<=1e-4 for k in ("routed_drift","combined_drift")),"observer")
        need(all(set(a)=={"routed_error","combined_error"} and all(math.isfinite(v) and v>=0 for v in a.values()) for a in o["losses"].values()),"loss")
    return seen
def decision(rows,captures,masks,layers,stage):
    need(stage in ("fit","fresh") and len(rows)==(48 if stage=="fit" else 24) and len({r["id"] for r in rows})==len(rows) and set(captures)=={r["id"] for r in rows},"complete stage")
    counts={d:sum(r["direction"]==d for r in rows) for d in DIRS};need(set(counts.values())=={8 if stage=="fit" else 4},"direction balance")
    data={n:{g:[] for g in GROUPS} for n in NAMES}
    # Equal example weight within direction, equal layer weight, equal phases.
    # Also retain one line-level diagnostic: across direction examples sharing
    # an aligned line, report relative paired total loss change (not CI).
    byline={}
    for r in rows:
        seen=checked_capture(captures[r["id"]],r,masks,layers);scores={}
        for n in NAMES:
            scores[n]={p:math.fsum(math.fsum(seen[l,p,j]["losses"][n]["combined_error"] for j in steps)/len(steps) for l in layers)/len(layers) for p,steps in (("prefill",[0]),("decode",[1,8]))}
            for p,v in scores[n].items():data[n][r["direction"]+"/"+p].append(v)
        byline.setdefault(r["line"],[]).append([math.fsum(scores[n].values())/2 for n in NAMES])
    means={n:{g:math.fsum(v)/len(v) for g,v in groups.items()} for n,groups in data.items()}
    b=means["original96"];s=means["stable96"]
    gain=(math.fsum(b[g] for g in GROUPS[:4])-math.fsum(s[g] for g in GROUPS[:4]))/max(math.fsum(b[g] for g in GROUPS[:4]),1e-12)
    violations=[g for i,g in enumerate(GROUPS) if s[g]>b[g]*(1 if i<4 else 1.02)+1e-12];overall=math.fsum(s.values())<=math.fsum(b.values())+1e-12
    ok=gain>=.05 and not violations and overall
    line_changes=[math.fsum(v[1]-v[0] for v in vals)/len(vals) for vals in byline.values()]
    return dict(status=("FIT_LOCAL_PROXY_PASS_NOT_ADMISSION" if ok else "NO_ADMISSION_FROM_EXPANDED_FIT") if stage=="fit" else ("LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION" if ok else "NO_ADMISSION_FROM_EXPANDED_FRESH"),stage=stage,candidate_mask_sha256=sha(canon(masks["stable96"])),samples=len(rows),unique_aligned_lines=len(byline),means=means,primary_relative_improvement=gain,violations=violations,overall_guard=overall,line_improved=sum(x< -1e-12 for x in line_changes),line_regressed=sum(x>1e-12 for x in line_changes),line_tied=sum(abs(x)<=1e-12 for x in line_changes),limits="Fixed mask local teacher-state comparison; line diagnostics correlated within line; no product quality or deployment admission")
def selftest():
    masks={"original96":{"1":[0,1]},"stable96":{"1":[1,2]}}
    rows=[dict(id=f"calibration:{d}:{i}",direction=d,line=j*8+i,teacher_prefix_tokens=list(range(8))) for j,d in enumerate(DIRS) for i in range(8)]
    def records(loss):
        out={}
        for r in rows:
            z=[dict(phase="prefill" if j==0 else "decode",step=j,next=j,**({"input_token":j-1} if j else {})) for j in range(9)]
            obs=[dict(layer=1,phase=p,step=j,row_count=2 if j==0 else 1,routed_drift=0.,combined_drift=0.,losses={n:dict(routed_error=v,combined_error=v) for n,v in zip(NAMES,(.25,loss))}) for p,j in (("prefill",0),("decode",1),("decode",8))]
            out[r["id"]]=dict(status="LOCAL_OUTPUT_PROXY_ONLY",id=r["id"],direction=r["direction"],line=r["line"],candidate_masks={n:sha(canon(v)) for n,v in masks.items()},logit_equivalence=True,prefix_policy="FIXED_HISTORICAL_PREFIX_LOCAL_PROXY",consumed_prefix=r["teacher_prefix_tokens"],threads=8,teacher_logits=z,historical_argmax_mismatches=[],historical_greedy_match=True,observations=obs)
        return out
    a=decision(rows,records(.125),masks,[1],"fit")
    need(a["status"]=="FIT_LOCAL_PROXY_PASS_NOT_ADMISSION" and a["primary_relative_improvement"]==.5 and a["line_improved"]==48,"fit oracle")
    need(decision(rows,records(.3),masks,[1],"fit")["status"]=="NO_ADMISSION_FROM_EXPANDED_FIT","fit rejection")
    q=records(.125);q[rows[0]["id"]]["teacher_logits"][1]["input_token"]=99
    try:decision(rows,q,masks,[1],"fit")
    except ValueError:pass
    else:raise AssertionError("wrong token")
    print(json.dumps(dict(status="PASS_SYNTHETIC_EXPANDED_PROXY_DECISION",checks=3,real_capture="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--inputs",type=P);a.add_argument("--admission",type=P);a.add_argument("--execute",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.admission,"explicit inputs")
    d=q.inputs;b,v,drv,c,pilot,s=[load(d/(n+".py"),n) for n in ("batch","verifier","driver","collector","pilot","sparse")]
    c.verify_code();w=c.load_witness(d/"witness.py")
    pr=read(d/"expanded.json");orig=read(d/"original.json");need(blob(pr)==PLAN and blob(orig)==ORIGINAL,"input pins")
    plan=v.strict(pr);base=v.strict(orig)["candidates"]["96"]["kept"];need(sha(canon(base))==BASE_SHA,"base")
    fit=plan["fit"];fresh=plan["holdout"];need(len(fit)==48 and len(fresh)==24 and len({r["line"] for r in fit})==30 and len({r["line"] for r in fresh})==14 and not ({r["line"] for r in fit}&{r["line"] for r in fresh}),"scope")
    raw=read(q.admission);ad=v.strict(raw);need(raw==canon(ad) and ad["status"]=="R23F_STABLE96_PROXY_ADMITTED" and ad["plan_blob"]==PLAN and ad["teacher"]==v.TEACHER and ad["engine_head"]==v.ENGINE and ad["threads"]==8,"admission")
    stage=ad["stage"];need(stage in ("fit","fresh"),"stage");rows=fit if stage=="fit" else fresh
    first,stop=ad["first"],ad["stop"];need(type(first) is type(stop) is int and 0<=first<stop<=len(rows) and stop-first<=24 and ad["ids"]==[r["id"] for r in rows[first:stop]],"batch")
    candraw=read(ROOT/"r23-results/expanded-stable96-v1.json");need(sha(candraw)==ad["candidate_receipt_sha256"],"candidate receipt");cand=v.strict(candraw)
    need(cand["status"]=="FROZEN_STABLE96_NOT_ADMITTED" and cand["plan_blob"]==PLAN and cand["fit_ids"]==[r["id"] for r in fit] and cand["base_mask_sha256"]==BASE_SHA and cand["swaps"]>0,"fit-only candidate")
    kept=cand["kept"];need(sha(canon(kept))==ad["mask_sha256"]==cand["mask_sha256"] and set(kept)==set(base)=={str(i) for i in range(1,48)} and all(len(z)==96 and z==sorted(set(z)) and all(type(i) is int and 0<=i<128 for i in z) and len(set(z)-set(base[l]))<=2 for l,z in kept.items()),"stable mask")
    masks={"original96":base,"stable96":kept};layers=list(range(1,48))
    root=ROOT/"r23-results/expanded-stable96-proxy-v1";dst=root/stage
    if stage=="fresh":
        rr=read(root/"fit/decision.json");need(sha(rr)==ad["fit_decision_sha256"],"fit decision pin");dd=v.strict(rr)
        need(dd["status"]=="FIT_LOCAL_PROXY_PASS_NOT_ADMISSION" and dd["candidate_mask_sha256"]==cand["mask_sha256"] and dd["samples"]==48 and dd["unique_aligned_lines"]==30 and dd["primary_relative_improvement"]>=.05 and not dd["violations"] and dd["overall_guard"] is True,"fresh prerequisite")
    corpus=read(ROOT/"run-d01b-r19/corpus/corpus.jsonl");hist=read(ROOT/"run-d01b-r19/baseline/records.jsonl");need(sha(corpus)==CORPUS and sha(hist)==BASELINE,"corpus pins")
    ids={r["id"] for r in rows[first:stop]}
    ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids}
    old={x["id"]:x for x in map(json.loads,hist.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids}
    need(set(ctx)==set(old)==ids,"stage rows")
    for r in rows[first:stop]:
        x=ctx[r["id"]];y=old[r["id"]];need(x["line"]==y["line"]==r["line"] and x["direction"]==y["direction"]==r["direction"] and len(x["prompt_ids"])==y["prompt_tokens"]==r["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and y["finish"]=="eos" and y["tokens"][:8]==r["teacher_prefix_tokens"] and sha(canon(r["teacher_prefix_tokens"]))==r["teacher_prefix_sha256"],"input identity")
    def prefix():
        if first==0:need(not dst.exists(),"destination exists");return {}
        need(dst.is_dir() and not dst.is_symlink() and not (dst/"decision.json").exists(),"resume directory")
        hashes=ad["prior_receipts"];need(len(hashes)==first and sorted(z.name for z in dst.glob("*.json"))==[f"{i:02d}.json" for i in range(first)],"prior inventory")
        out={}
        for i,h in enumerate(hashes):
            rr=read(dst/f"{i:02d}.json");need(sha(rr)==h,"prior receipt");x=v.strict(rr);r=rows[i]
            need(x["index"]==i and x["stage"]==stage and x["teacher_digest"]==v.TEACHER and x["engine_head"]==v.ENGINE and x["plan_blob"]==PLAN and x["candidate_receipt_sha256"]==ad["candidate_receipt_sha256"] and x["extra_oracle_evaluations"]==512,"prior identity");checked_capture(x,r,masks,layers);out[r["id"]]=x
        return out
    prefix();pre=drv.preflight();print(json.dumps(dict(pre,stage=stage,first=first,stop=stop,mask_sha256=cand["mask_sha256"])),flush=True)
    if not q.execute:return
    need(not pre["blockers"],"resource preflight")
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);results=prefix();need(not drv.preflight()["blockers"],"exclusive resources")
        if first==0:dst.mkdir(mode=0o700,parents=True)
        ap=ROOT/"r23-admissions"/f"expanded-stable96-{stage}-{first:02d}-{stop:02d}.json"
        with ap.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==v.TEACHER and ident["num_experts"]==128 and not drv.preflight()["blockers"],"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8);c.forward_sequence=lambda e,prompt,prefix,callback=None:b.forced_forward(c,e,prompt,prefix,callback)
        print(json.dumps(dict(status="TEACHER_READY",teacher=ident["teacher"],admission_sha256=sha(raw),stage=stage)),flush=True)
        ms={n:{int(l):ids for l,ids in m.items()} for n,m in masks.items()}
        for i in range(first,stop):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            r=rows[i];t=time.monotonic();x=b.annotate(pilot.capture(c,w,s,e,dict(r,split="calibration"),ctx[r["id"]]["prompt_ids"],ms),r)
            x.update(index=i,stage=stage,teacher_digest=ident["teacher"],engine_head=v.ENGINE,plan_blob=PLAN,candidate_receipt_sha256=ad["candidate_receipt_sha256"],admission_sha256=sha(raw))
            checked_capture(x,r,masks,layers);need(x["extra_oracle_evaluations"]==512,"exhaustive oracle")
            results[r["id"]]=x;h=b.publish(dst/f"{i:02d}.json",x)
            print(json.dumps(dict(status="STABLE96_PROXY_CAPTURED",stage=stage,index=i,id=r["id"],completed=i+1,receipt_sha256=h,seconds=time.monotonic()-t)),flush=True)
        if stop==len(rows):
            out=decision(rows,results,masks,layers,stage);h=b.publish(dst/"decision.json",out)
            print(json.dumps(dict(out,receipt_sha256=h)),flush=True)
        else:print(json.dumps(dict(status="STABLE96_PROXY_PARTIAL_BATCH_COMPLETE",stage=stage,completed=stop,total=len(rows))),flush=True)
if __name__=="__main__":main()
