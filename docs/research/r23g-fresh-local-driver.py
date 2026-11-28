"""R23G frozen candidate fresh-local validation. Separate explicit admission.
Only after independently replayed completed fit propagation passes. No new mask,
holdout fitting, propagation claim, model build or quality qualification."""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,re,sys,time
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
CAND="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
MASK="77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
BASE="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
PLAN="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
SENS="0e071705ea763d45e0b7034a20647cb16305b48ea737386a69d8014f903ad381"
PINS={"proxy":"248737901db40b50efb0a26a9d5ef76d1928538b","batch":"08abbbfbd930e532d41602dd29a5c70f2bc474e3","verifier":"e56b25b3fd92ccbd91455628606925a02af49450","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46","pilot":"19c8c851b1647cc0f495d6a7f1122219c42a0cdb","sparse":"1c38f6e928ff653fbbef584b298ff65e194900fe"}
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p.read_bytes()
def pin(p,h,git=False):
    b=read(p);need((blob if git else sha)(b)==h,"pin "+str(p));return b
def load(d,n):
    p=d/(n+".py");pin(p,PINS[n],True);s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def eligible(g,log,replay,report):
    return g["status"]=="R23G_FIT_PROPAGATION_REPLAY_PASS_FRESH_ELIGIBILITY_ONLY" and g["samples"]==18 and g["aligned_lines"]==16 and g["mask_sha256"]==MASK and g["candidate_sha256"]==CAND and g["sensitivity_sha256"]==SENS and g["primary_violations"]==[] and g["log_sha256"]==log and g["replay_sha256"]==replay and g["report_sha256"]==report
def fresh_ok(x):
    return x["status"]=="LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION" and x["stage"]=="fresh" and x["samples"]==24 and x["unique_aligned_lines"]==14 and x["candidate_mask_sha256"]==MASK and x["primary_relative_improvement"]>=.05 and not x["violations"] and x["overall_guard"] is True
def selftest():
    g=dict(status="R23G_FIT_PROPAGATION_REPLAY_PASS_FRESH_ELIGIBILITY_ONLY",samples=18,aligned_lines=16,mask_sha256=MASK,candidate_sha256=CAND,sensitivity_sha256=SENS,primary_violations=[],log_sha256="l",replay_sha256="r",report_sha256="p")
    need(eligible(g,"l","r","p"),"fit gate")
    need(not eligible(dict(g,primary_violations=["zh-my/prefill/kl"]),"l","r","p") and not eligible(g,"wrong","r","p") and not eligible(dict(g,samples=17),"l","r","p"),"reject bad prerequisite")
    f=dict(status="LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION",stage="fresh",samples=24,unique_aligned_lines=14,candidate_mask_sha256=MASK,primary_relative_improvement=.05,violations=[],overall_guard=True)
    need(fresh_ok(f) and not fresh_ok(dict(f,primary_relative_improvement=.049)) and not fresh_ok(dict(f,violations=["zh-my/prefill"])) and not fresh_ok(dict(f,unique_aligned_lines=13)),"unchanged local guard")
    print(json.dumps(dict(status="PASS_SYNTHETIC_R23G_FRESH_LOCAL_GATE",positive=2,negative=6,actual_fresh="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--inputs",type=P);a.add_argument("--gate-sha");a.add_argument("--fit-log-sha");a.add_argument("--fit-replay-sha");a.add_argument("--comment",type=int);a.add_argument("--head");a.add_argument("--execute",action="store_true");a.add_argument("--audit",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and all(re.fullmatch("[0-9a-f]{64}",x or "") for x in (q.gate_sha,q.fit_log_sha,q.fit_replay_sha)) and q.comment and q.comment>0 and re.fullmatch("[0-9a-f]{40}",q.head or "") and not(q.execute and q.audit),"explicit admission")
    d=q.inputs;v=load(d,"verifier");b=load(d,"batch");proxy=load(d,"proxy")
    root=ROOT/"r23-results";g=v.strict(pin(d/"fit-gate.json",q.gate_sha))
    fitlog=pin(root/"r23g-l38-propagation-v1.log",q.fit_log_sha);need(read(root/"r23g-l38-propagation-v1.stderr")==b"","fit stderr")
    fr=v.strict(pin(root/"r23g-l38-propagation-v1-replay.json",q.fit_replay_sha));need(fr["status"]=="R23G_EXACT_REPLAY_NOT_QUALITY" and fr["samples"]==18,"fit replay")
    actual=v.strict(pin(root/"r23g-l38-propagation-v1"/"report.json",fr["report_sha256"]));need(actual["primary_diagnostic_pass"] is True and not actual["primary_diagnostic_violations"] and actual["groups"]==g["groups"] and eligible(g,q.fit_log_sha,q.fit_replay_sha,fr["report_sha256"]),"fit rejected: fresh forbidden")
    plan=v.strict(pin(d/"expanded.json",PLAN,True));rows=plan["holdout"];need(len(rows)==24 and len({r["line"] for r in rows})==14 and all(sum(r["direction"]==di for r in rows)==4 for di in DIRS) and not ({r["line"] for r in rows}&{r["line"] for r in plan["fit"]}),"fresh scope")
    base=v.strict(pin(d/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True))["candidates"]["96"]["kept"];cand=v.strict(pin(d/"candidate.json",CAND));kept=cand["kept"];layers=list(range(1,48))
    need(sha(canon(base))==BASE and sha(canon(kept))==MASK and set(base)==set(kept)==set(map(str,layers)) and all(len(x)==96 and x==sorted(set(x)) and all(type(i) is int and 0<=i<128 for i in x) for m in (base,kept) for x in m.values()),"frozen masks")
    # Existing pinned proxy schema labels the second mask stable96; here its digest is R23G, not the rejected prior mask.
    masks={"original96":base,"stable96":kept}
    ad=dict(status="R23G_FRESH_LOCAL_ADMITTED",teacher=TEACHER,engine_head=ENGINE,plan_blob=PLAN,mask_sha256=MASK,candidate_sha256=CAND,fit_gate_sha256=q.gate_sha,fit_log_sha256=q.fit_log_sha,fit_replay_sha256=q.fit_replay_sha,ids=[r["id"] for r in rows],comment=q.comment,head=q.head,threads=8);ah=sha(canon(ad))
    dst=root/"r23g-fresh-local-v1";log=root/"r23g-fresh-local-v1.log";err=root/"r23g-fresh-local-v1.stderr"
    def checked(x,r,i):
        need(x["index"]==i and x["stage"]=="fresh" and x["teacher_digest"]==TEACHER and x["engine_head"]==ENGINE and x["plan_blob"]==PLAN and x["candidate_receipt_sha256"]==CAND and x["admission_sha256"]==ah and x["extra_oracle_evaluations"]==512,"capture provenance");proxy.checked_capture(x,r,masks,layers)
    if q.audit:
        need(read(dst/"admission.json")==canon(ad)+b"\n" and read(err)==b"" and sorted(p.name for p in dst.iterdir())==sorted(["admission.json","decision.json"]+[f"{i:02d}.json" for i in range(24)]),"exact inventory")
        z=[v.strict(l) for l in read(log).splitlines() if l.startswith(b"{")];ready=[x for x in z if x.get("status")=="TEACHER_READY"];events=[x for x in z if x.get("status")=="R23G_FRESH_LOCAL_CAPTURED"];done=[x for x in z if x.get("status")=="R23G_FRESH_LOCAL_COMPLETE"]
        need(len(ready)==len(done)==1 and ready[0]["teacher"]==TEACHER and ready[0]["admission_sha256"]==ah and len(events)==24,"complete log")
        caps={};hs=[]
        for i,(r,e) in enumerate(zip(rows,events)):
            need(e["index"]==i and e["id"]==r["id"],"log identity");raw=pin(dst/f"{i:02d}.json",e["receipt_sha256"]);x=v.strict(raw);checked(x,r,i);caps[r["id"]]=x;hs.append(sha(raw))
        raw=pin(dst/"decision.json",done[0]["receipt_sha256"]);f=v.strict(raw);need(f==proxy.decision(rows,caps,masks,layers,"fresh") and done[0]["decision"]==f["status"],"exact decision")
        print(json.dumps(dict(status="R23G_FRESH_LOCAL_REPLAY_PASS_NOT_QUALITY" if fresh_ok(f) else "R23G_FRESH_LOCAL_REPLAY_REJECT",admission_sha256=ah,fit_gate_sha256=q.gate_sha,mask_sha256=MASK,decision_sha256=sha(raw),receipt_sha256=hs,primary_relative_improvement=f["primary_relative_improvement"],violations=f["violations"],overall_guard=f["overall_guard"])),flush=True);return
    drv=load(d,"driver");c=load(d,"collector");pilot=load(d,"pilot");s=load(d,"sparse");c.verify_code();w=c.load_witness(d/"witness.py")
    corpus=pin(ROOT/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4");hist=pin(ROOT/"run-d01b-r19/baseline/records.jsonl","37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8")
    ids={r["id"] for r in rows};ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};old={x["id"]:x for x in map(json.loads,hist.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids};need(set(ctx)==set(old)==ids,"fresh inputs")
    for r in rows:
        x=ctx[r["id"]];z=old[r["id"]];need(x["line"]==z["line"]==r["line"] and x["direction"]==z["direction"]==r["direction"] and len(x["prompt_ids"])==z["prompt_tokens"]==r["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and z["finish"]=="eos" and z["tokens"][:8]==r["teacher_prefix_tokens"] and sha(canon(r["teacher_prefix_tokens"]))==r["teacher_prefix_sha256"],"input binding")
    pre=drv.preflight();print(json.dumps(dict(pre,admission_sha256=ah,samples=24,stage="fresh")),flush=True)
    if not q.execute:return
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);need(not pre["blockers"] and not drv.preflight()["blockers"] and not os.path.lexists(dst),"exclusive resources")
        dst.mkdir(mode=0o700);b.publish(dst/"admission.json",ad)
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==TEACHER and ident["num_experts"]==128 and not drv.preflight()["blockers"],"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8);c.forward_sequence=lambda e,prompt,prefix,callback=None:b.forced_forward(c,e,prompt,prefix,callback)
        ms={n:{int(l):ids for l,ids in m.items()} for n,m in masks.items()};caps={}
        print(json.dumps(dict(status="TEACHER_READY",teacher=TEACHER,admission_sha256=ah)),flush=True)
        for i,r in enumerate(rows):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            t=time.monotonic();x=b.annotate(pilot.capture(c,w,s,e,dict(r,split="calibration"),ctx[r["id"]]["prompt_ids"],ms),r)
            x.update(index=i,stage="fresh",teacher_digest=TEACHER,engine_head=ENGINE,plan_blob=PLAN,candidate_receipt_sha256=CAND,admission_sha256=ah);checked(x,r,i);caps[r["id"]]=x;h=b.publish(dst/f"{i:02d}.json",x)
            print(json.dumps(dict(status="R23G_FRESH_LOCAL_CAPTURED",index=i,id=r["id"],receipt_sha256=h,seconds=time.monotonic()-t)),flush=True)
        f=proxy.decision(rows,caps,masks,layers,"fresh");h=b.publish(dst/"decision.json",f);print(json.dumps(dict(status="R23G_FRESH_LOCAL_COMPLETE",decision=f["status"],receipt_sha256=h,primary_relative_improvement=f["primary_relative_improvement"],violations=f["violations"],overall_guard=f["overall_guard"])),flush=True)
if __name__=="__main__":main()
