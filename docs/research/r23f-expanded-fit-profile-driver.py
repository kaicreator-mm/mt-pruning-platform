"""R23F expanded fit-only contribution profiles; no holdout capture or mask fitting.
Preserve the original twelve R23B fit receipts, then collect thirty-six new
fixed-trajectory samples under exact same-process plain/observed/restored checks.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,sys,time
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PLAN_BLOB="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
PINS={"profile":"4316eda170d6c0ef1548c9d571409d96bcd43fd7","batch":"08abbbfbd930e532d41602dd29a5c70f2bc474e3","verifier":"e56b25b3fd92ccbd91455628606925a02af49450","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46","reap":"b223d05813625d597c3a598f8373931324bc089f","masks":"34a06701d47ce0e3ab888882a5b20c75745204d4"}
CORPUS="9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
BASELINE="37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8"
AUDIT_SHA="a96037d8adc6e7fd06f8fd7b453eee09a445c6faa78034490783b1c51856c68e"
LEGACY_EXTRA=["2754d33afb59a281fd37f19b76c6e6574624e2b259a6c0831e0f9cc0a0d09c33","f2895b1cda2b1ba6db99e6dc3950bbe8d8aedccb35759acc5b05b623155ffc59"]
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def load(p,n,adapt=False):
    raw=p.read_bytes();need(blob(raw)==PINS[n],"source "+n)
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m
    if adapt:
        a='cross=differences(old["teacher_logits"],logits)';b='cross=None if old is None else differences(old["teacher_logits"],logits)'
        c='matched_r23a_logits=not cross';d='matched_r23a_logits=None if cross is None else not cross'
        src=raw.decode();need(src.count(a)==src.count(c)==1,"exact adaptation point")
        exec(compile(src.replace(a,b).replace(c,d),str(p),"exec"),m.__dict__)
    else:s.loader.exec_module(m)
    return m
def scope(plan):
    fit=plan["fit"];hold=plan["holdout"];old=set(plan["excluded_aligned_lines"])
    need(plan["status"]=="FROZEN_GROUPED_EXPANDED_R23F_CALIBRATION" and len(fit)==48 and len(hold)==24 and len({r["id"] for r in fit+hold})==72,"plan inventory")
    fl={r["line"] for r in fit};hl={r["line"] for r in hold}
    need(len(fl)==30 and len(hl)==14 and not fl&hl and not ({r["line"] for r in fit[12:]+hold}&old) and {r["line"] for r in fit[:12]}<=old,"aligned isolation")
    need(all(sum(r["direction"]==d for r in fit)==8 and sum(r["direction"]==d for r in hold)==4 for d in DIRS),"direction balance")
    return fit
def check_profile(x,r,m):
    need(x["id"]==r["id"] and x["direction"]==r["direction"] and x["subset"]=="fit" and x["hook_calls"]==423 and x["num_experts"]==128 and x["top_k"]==8 and x["threads"]==8,"profile identity")
    need(set(x["profile"])=={f"{l}/{p}" for l in range(1,48) for p in ("prefill","decode")},"profile coverage")
    for key,y in x["profile"].items():
        m.check(y,128,8,x["scale"]);need(y["tokens"]==(r["prompt_tokens"] if key.endswith("/prefill") else 8),"token count")
    z=x["teacher_logits"];need(len(z)==9 and [q["input_token"] for q in z[1:]]==r["teacher_prefix_tokens"],"actual prefix")
def capture(p,c,b,e,r,prompt):
    x=p.capture(c,b,e,r,prompt,None)
    need(x["matched_r23a_logits"] is None and x["cross_process_r23a_differences"] is None and x["plain_logits"]==x["teacher_logits"]==x["restored_logits"] and x["same_process_observer_equivalence"] is True,"same-process oracle/no historical R23A comparison")
    return b.annotate(x,r)
def selftest(p,c,b):
    import tempfile
    c.torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as d:
        q=P(d);model=c.tiny_checkpoint(q/"tiny",q/"none",vocab_size=256,seed=19);e=c.engine_module.Engine(c.engine_module.Checkpoint(model),threads=1)
        prompt=[4,8,15,16,23,42];prefix=c.greedy_prefix(e,prompt,8)
        r=dict(id="calibration:zh-my:synthetic",direction="zh-my",teacher_prefix_tokens=prefix)
        x=capture(p,c,b,e,r,prompt)
        need(x["hook_calls"]==18 and x["historical_greedy_match"] and x["same_process_observer_equivalence"] and x["matched_r23a_logits"] is None and x["cross_process_r23a_differences"] is None,"expanded profile")
    print(json.dumps(dict(status="PASS_SYNTHETIC_EXPANDED_PROFILE",hooks=18,logits=9,real_capture="NOT_RUN")),flush=True)
def main():
    a=argparse.ArgumentParser();a.add_argument("--inputs",type=P,required=True);a.add_argument("--admission",type=P);a.add_argument("--execute",action="store_true");a.add_argument("--selftest",action="store_true");q=a.parse_args();d=q.inputs
    p,b,v,drv,c,reap,m=[load(d/(n+".py"),n,n=="profile") for n in ("profile","batch","verifier","driver","collector","reap","masks")]
    c.verify_code()
    if q.selftest:selftest(p,c,b);return
    need(q.admission is not None,"explicit admission")
    pr=(d/"expanded.json").read_bytes();need(blob(pr)==PLAN_BLOB,"plan pin")
    plan=v.strict(pr);fit=scope(plan)
    raw=q.admission.read_bytes();ad=v.strict(raw);need(raw==canon(ad) and ad["status"]=="R23F_EXPANDED_FIT_PROFILE_ADMITTED" and ad["plan_blob"]==PLAN_BLOB and ad["teacher"]==v.TEACHER and ad["engine_head"]==v.ENGINE and ad["threads"]==8,"admission")
    first=ad["first"];stop=ad["stop"];need(type(first) is type(stop) is int and 12<=first<stop<=48 and stop-first<=36 and ad["ids"]==[r["id"] for r in fit[first:stop]],"bounded scope")
    audit=(ROOT/"r23-results/score-proxy-v1-replay-audit.json").read_bytes();need(sha(audit)==AUDIT_SHA and v.strict(audit)["decision"]=="NO_ADMISSION_FROM_LOCAL_PROXY","B rejection audit")
    corpus=(ROOT/"run-d01b-r19/corpus/corpus.jsonl").read_bytes();base=(ROOT/"run-d01b-r19/baseline/records.jsonl").read_bytes();need(sha(corpus)==CORPUS and sha(base)==BASELINE,"corpus pins")
    ids={r["id"] for r in fit}
    ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids}
    hist={x["id"]:x for x in map(json.loads,base.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids}
    need(set(ctx)==set(hist)==ids,"fit scope")
    for r in fit:
        x=ctx[r["id"]];z=hist[r["id"]]
        need(r["line"]==x["line"]==z["line"] and r["direction"]==x["direction"]==z["direction"] and len(x["prompt_ids"])==r["prompt_tokens"]==z["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and z["finish"]=="eos" and z["tokens"][:8]==r["teacher_prefix_tokens"] and sha(canon(r["teacher_prefix_tokens"]))==r["teacher_prefix_sha256"],"row binding")
    old=ROOT/"r23-results/reap-full24"
    for i,h in enumerate(p.LEGACY_RECEIPTS+LEGACY_EXTRA):
        data=v.read_regular(old/f"{i:02d}.json");need(sha(data)==h,"prior fit receipt");x=v.strict(data);check_profile(x,fit[i],m);need(x["teacher"]==v.TEACHER and x["engine_head"]==v.ENGINE and x["index"]==i,"prior fit identity")
    dst=ROOT/"r23-results/expanded-fit-profile-v1"
    def prior():
        if first==12:need(not dst.exists(),"first destination exists");return
        need(dst.is_dir() and not dst.is_symlink(),"resume directory")
        receipts=ad["prior_receipts"];need(len(receipts)==first-12 and sorted(z.name for z in dst.glob("*.json"))==[f"{i:02d}.json" for i in range(12,first)],"prior scope")
        for i,h in enumerate(receipts,12):
            raw=v.read_regular(dst/f"{i:02d}.json");need(sha(raw)==h,"prior hash");x=v.strict(raw);check_profile(x,fit[i],m);need(x["index"]==i and x["plan_blob"]==PLAN_BLOB and x["teacher"]==v.TEACHER and x["engine_head"]==v.ENGINE and x["plain_logits"]==x["teacher_logits"]==x["restored_logits"] and x["same_process_observer_equivalence"] and x["matched_r23a_logits"] is None,"prior new fit receipt")
    prior();pre=drv.preflight();print(json.dumps(dict(pre,first=first,stop=stop,fit_total=48,holdout_capture="NOT_ALLOWED")),flush=True)
    if not q.execute:return
    need(not pre["blockers"],"resource preflight")
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);prior();need(not drv.preflight()["blockers"],"exclusive resources")
        if first==12:dst.mkdir(mode=0o700)
        ap=ROOT/"r23-admissions"/f"expanded-fit-{first:02d}-{stop:02d}.json"
        with ap.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==v.TEACHER and ident["num_experts"]==128 and not drv.preflight()["blockers"],"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8)
        print(json.dumps(dict(status="TEACHER_READY",teacher=ident["teacher"],admission_sha256=sha(raw))),flush=True)
        for i in range(first,stop):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            r=fit[i];t=time.monotonic();x=capture(p,c,b,e,r,ctx[r["id"]]["prompt_ids"])
            x.update(index=i,subset="fit",teacher=ident["teacher"],engine_head=v.ENGINE,plan_blob=PLAN_BLOB,threads=8,admission_sha256=sha(raw),seconds=time.monotonic()-t)
            check_profile(x,r,m);h=b.publish(dst/f"{i:02d}.json",x)
            print(json.dumps(dict(status="EXPANDED_FIT_PROFILE_CAPTURED",id=r["id"],index=i,completed=i+1,receipt_sha256=h,hook_calls=x["hook_calls"],same_process_triple=True,seconds=x["seconds"])),flush=True)
        need(sorted(z.name for z in dst.glob("*.json"))==[f"{i:02d}.json" for i in range(12,stop)],"complete batch")
        print(json.dumps(dict(status="EXPANDED_FIT_PROFILE_BATCH_COMPLETE_NOT_SELECTION",completed=stop,total=48,holdout_capture="NOT_RUN")),flush=True)
if __name__=="__main__":main()
