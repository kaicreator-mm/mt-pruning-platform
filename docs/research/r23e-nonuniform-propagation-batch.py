"""R23E fit-only nonuniform propagation diagnostic. No quality admission.
Runs only on restored host with exact resources, explicit admission and fixed masks.
CPU synthetic report tests do not load the teacher or contact the remote host.
"""
import argparse,fcntl,hashlib,importlib.util,json,math,os,pathlib,sys,time
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PLAN="cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde"
EXPANDED="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
ORIGINAL="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
BASE="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
MASK="2339adfdbb8a438ecfa046384cced8f652fd1a523e722a66bf4480002a0500c9"
RECEIPT="931e44f76ae47ba4c567b94335d7ffbb36f0cdf330bb20589ed55dd058983f17"
CORPUS="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
BASELINE="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8"
PINS={"prop":"88468e8c2a5905628365da52e1575d5251516512","batch":"08abbbfbd930e532d41602dd29a5c70f2bc474e3","verifier":"e56b25b3fd92ccbd91455628606925a02af49450","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46"}
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en");NAMES=("original96","candidate96")
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file");return p.read_bytes()
def load(p,n):
    need(blob(read(p))==PINS[n],"source "+n)
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def finite(v):return isinstance(v,(int,float)) and math.isfinite(v) and v>=0
def checked(x,r,layers,k=8):
    need(x["status"]=="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY" and x["teacher_restored"] is True and x["id"]==r["id"] and x["direction"]==r["direction"] and x["line"]==r["line"] and x["consumed_prefix"]==r["teacher_prefix_tokens"],"identity")
    want={(l,"prefill" if j==0 else "decode",j) for l in layers for j in range(9)}
    for n in NAMES:
        a=x[n];ls=a["layer"];seen={(q["layer"],q["phase"],q["step"]) for q in ls}
        need(len(ls)==len(seen)==len(want) and seen==want,"layer coverage")
        for q in ls:
            need(all(finite(q[z]) for z in ("hidden_input_nmse","moe_output_nmse","mean_overlap")) and q["mean_overlap"]<=k and q["tokens"]==(2 if q["step"]==0 else 1) and type(q["exact_sets"]) is int and 0<=q["exact_sets"]<=q["tokens"],"layer metrics")
        for arr in (a["teacher_hashes"],a["masked_hashes"]):
            need(len(arr)==9 and all(q["step"]==j and len(q["sha256"])==64 and all(c in "0123456789abcdef" for c in q["sha256"]) and type(q["next"]) is int for j,q in enumerate(arr)),"logit hashes")
        need(len(a["logits"])==9 and all(q["step"]==j and finite(q["nmse"]) and finite(q["teacher_to_masked_kl"]) and q["teacher_argmax"]==a["teacher_hashes"][j]["next"] and q["masked_argmax"]==a["masked_hashes"][j]["next"] for j,q in enumerate(a["logits"])),"logits")
    need(x["original96"]["teacher_hashes"]==x["candidate96"]["teacher_hashes"],"shared teacher")
def report(rows,captures,layers,k=8):
    need(len(rows)==12 and len({r["line"] for r in rows})==12 and all(sum(r["direction"]==d for r in rows)==2 for d in DIRS) and set(captures)=={r["id"] for r in rows},"fit scope")
    for r in rows:checked(captures[r["id"]],r,layers,k)
    def avg(v):return math.fsum(v)/len(v)
    out={}
    for n in NAMES:
        groups={}
        for d in DIRS:
            for phase in ("prefill","decode"):
                xs=[captures[r["id"]][n] for r in rows if r["direction"]==d]
                logits=[q for x in xs for q in x["logits"] if (q["step"]==0)==(phase=="prefill")]
                per={}
                for l in layers:
                    v=[q for x in xs for q in x["layer"] if q["layer"]==l and q["phase"]==phase]
                    per[str(l)]={key:avg([q[key] for q in v]) for key in ("hidden_input_nmse","moe_output_nmse","mean_overlap")}
                groups[d+"/"+phase]=dict(logit_nmse=avg([q["nmse"] for q in logits]),teacher_to_masked_kl=avg([q["teacher_to_masked_kl"] for q in logits]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in logits),logit_positions=len(logits),layers=per)
        out[n]=groups
    return dict(status="NONUNIFORM_PROPAGATION_DIAGNOSTIC_COMPLETE_NOT_ADMISSION",samples=12,aligned_lines=12,layer_phase_step_records_per_mask=12*len(layers)*9,groups=out,scope="Common historical input tokens, fixed masks, true propagated states. Diagnostic only; no free-running translation or deployment qualification, no fresh holdout.")
def selftest():
    rows=[dict(id=str(i),line=i,direction=DIRS[i//2],teacher_prefix_tokens=list(range(8))) for i in range(12)]
    caps={}
    for r in rows:
        x=dict(status="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY",teacher_restored=True,id=r["id"],line=r["line"],direction=r["direction"],consumed_prefix=r["teacher_prefix_tokens"])
        for n in NAMES:
            v=.1 if n=="original96" else .05
            h=[dict(step=j,sha256="a"*64,next=j) for j in range(9)]
            x[n]=dict(teacher_hashes=h,masked_hashes=h,logits=[dict(step=j,nmse=v,teacher_to_masked_kl=v,teacher_argmax=j,masked_argmax=j) for j in range(9)],layer=[dict(layer=1,phase="prefill" if j==0 else "decode",step=j,tokens=2 if j==0 else 1,hidden_input_nmse=v,moe_output_nmse=v,mean_overlap=8.,exact_sets=2 if j==0 else 1) for j in range(9)])
        caps[r["id"]]=x
    q=report(rows,caps,[1]);need(q["groups"]["candidate96"]["zh-my/decode"]["logit_nmse"]==.05 and q["layer_phase_step_records_per_mask"]==108,"report oracle")
    x=caps["0"];x["candidate96"]["layer"][0]["mean_overlap"]=9
    try:report(rows,caps,[1])
    except ValueError:pass
    else:raise AssertionError("invalid overlap accepted")
    x["candidate96"]["layer"][0]["mean_overlap"]=8;x["teacher_restored"]=False
    try:report(rows,caps,[1])
    except ValueError:pass
    else:raise AssertionError("unrestored teacher accepted")
    print(json.dumps(dict(status="PASS_SYNTHETIC_NONUNIFORM_PROPAGATION_REPORT",checks=3,real_capture="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--inputs",type=P);a.add_argument("--admission",type=P);a.add_argument("--execute",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.admission,"inputs");d=q.inputs
    p,b,v,drv,c=[load(d/(n+".py"),n) for n in ("prop","batch","verifier","driver","collector")]
    c.verify_code();pr=read(d/"prop-plan.json");ex=read(d/"expanded.json");ori=read(d/"original.json")
    need(blob(pr)==PLAN and blob(ex)==EXPANDED and blob(ori)==ORIGINAL,"plan pins")
    plan=v.strict(pr);expanded=v.strict(ex);rows=plan["rows"];need(plan["mask_sha256"]==MASK and plan["mask_receipt_sha256"]==RECEIPT and plan["plan_blob"]==EXPANDED,"plan identity")
    need(len(rows)==12 and len({r["line"] for r in rows})==12 and all(sum(r["direction"]==di for r in rows)==2 for di in DIRS) and all(r in expanded["fit"][12:] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in expanded["holdout"]}),"diagnostic fit scope")
    base=v.strict(ori)["candidates"]["96"]["kept"];mr=read(ROOT/"r23-results/nonuniform96avg-mask-v1.json");need(sha(mr)==RECEIPT,"mask receipt");mask=v.strict(mr)["kept"];need(sha(canon(base))==BASE and sha(canon(mask))==MASK and sum(map(len,mask.values()))==4512,"mask identity")
    layers=list(range(1,48));need(set(base)==set(mask)==set(map(str,layers)) and all(len(a)==96 for a in base.values()) and all(len(a) in (64,80,96,112,128) and a==sorted(set(a)) and all(type(i) is int and 0<=i<128 for i in a) for a in mask.values()),"mask shapes")
    raw=read(q.admission);ad=v.strict(raw);need(raw==canon(ad) and ad["status"]=="R23E_NONUNIFORM_PROPAGATION_DIAGNOSTIC_ADMITTED" and ad["plan_blob"]==PLAN and ad["teacher"]==v.TEACHER and ad["engine_head"]==v.ENGINE and ad["threads"]==8 and ad["mask_sha256"]==MASK and ad["ids"]==[r["id"] for r in rows],"admission")
    corpus=read(ROOT/"run-d01b-r19/corpus/corpus.jsonl");hist=read(ROOT/"run-d01b-r19/baseline/records.jsonl");need(sha(corpus)==CORPUS and sha(hist)==BASELINE,"corpus pins")
    ids={r["id"] for r in rows};ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};old={x["id"]:x for x in map(json.loads,hist.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids};need(set(ctx)==set(old)==ids,"rows")
    for r in rows:
        x=ctx[r["id"]];z=old[r["id"]];need(x["line"]==z["line"]==r["line"] and x["direction"]==z["direction"]==r["direction"] and len(x["prompt_ids"])==z["prompt_tokens"]==r["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and z["finish"]=="eos" and z["tokens"][:8]==r["teacher_prefix_tokens"] and sha(canon(r["teacher_prefix_tokens"]))==r["teacher_prefix_sha256"],"input binding")
    pre=drv.preflight();print(json.dumps(dict(pre,samples=12,holdout_capture="NOT_ALLOWED")),flush=True)
    if not q.execute:return
    dst=ROOT/"r23-results/nonuniform-propagation-v1"
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);need(not pre["blockers"] and not drv.preflight()["blockers"] and not dst.exists(),"exclusive resources")
        dst.mkdir(mode=0o700);b.publish(dst/"admission.json",ad)
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==v.TEACHER and ident["num_experts"]==128 and not drv.preflight()["blockers"],"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8)
        masks={"original96":{int(l):a for l,a in base.items()},"candidate96":{int(l):a for l,a in mask.items()}};caps={}
        print(json.dumps(dict(status="TEACHER_READY",teacher=ident["teacher"],admission_sha256=sha(raw))),flush=True)
        for i,r in enumerate(rows):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            t=time.monotonic();x=p.run(c,e,ctx[r["id"]]["prompt_ids"],r["teacher_prefix_tokens"],masks)
            x.update(id=r["id"],line=r["line"],direction=r["direction"],consumed_prefix=r["teacher_prefix_tokens"],index=i,teacher=ident["teacher"],engine_head=v.ENGINE,plan_blob=PLAN,mask_sha256=MASK,admission_sha256=sha(raw))
            checked(x,r,layers);caps[r["id"]]=x;h=b.publish(dst/f"{i:02d}.json",x)
            print(json.dumps(dict(status="NONUNIFORM_PROPAGATION_CAPTURED",index=i,id=r["id"],receipt_sha256=h,seconds=time.monotonic()-t)),flush=True)
        out=report(rows,caps,layers);h=b.publish(dst/"report.json",out)
        print(json.dumps(dict(status=out["status"],receipt_sha256=h,samples=12)),flush=True)
if __name__=="__main__":main()
