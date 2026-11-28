"""R23F power recovery: verified immutable prefix, bounded resume, complete manifest.
Never contact a remote host, load a model, delete files or choose a mask.
Run on restored V100 host only after old jobs are terminal and resources idle.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,re
P=pathlib.Path
PLAN="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
WITNESS="5c96c7644d623f73bc9a3590542e76ffa1b9474c"
STATS="34a06701d47ce0e3ab888882a5b20c75745204d4"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
OLD_AD="b629b515a65d0342e275c77e1bf2c2f8c43913bab92c24ce76ec3b5b953c6f87"
CD="8e35997647591872f4c83f9dc7bfcae63052c604383bc1b2cb15bc8650eee63a"
ALLOC="04c2238ac5bb47609bc0ed452f80690b4037c531c9f3dc33b93394c89bf6c6bc"
ROOT=P.home()/".cache/mt-pruning"
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def strict(b):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,"duplicate key");d[k]=v
        return d
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p.read_bytes()
def pin(p,h,git=False):
    b=read(p);need((blob if git else sha)(b)==h,"pin "+str(p));return b
def publish(p,value):
    # Write new files only, with fsync of directory and contents.
    b=canon(value)+b"\n"
    import tempfile
    fd,tmp=tempfile.mkstemp(prefix=".recovery-unpublished-",dir=p.parent)
    try:
        with os.fdopen(fd,"wb") as f:f.write(b);f.flush();os.fsync(f.fileno())
        os.link(tmp,p)
        directory=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:os.unlink(tmp)
    return sha(b)
def check_index(rows,fit,first,stop):
    need(len(rows)==stop-first,"receipt count")
    for i,(x,r) in enumerate(zip(rows,fit[first:stop]),first):
        need(x["index"]==i and x["id"]==r["id"] and re.fullmatch("[0-9a-f]{64}",x["receipt_sha256"]) and re.fullmatch("[0-9a-f]{64}",x["admission_sha256"]),"ordered receipt identity")
def inventory(d,first,stop):
    need(d.is_dir() and not d.is_symlink(),"profile directory")
    got=sorted(p.name for p in d.iterdir())
    expected=[f"{i:02d}.json" for i in range(first,stop)]
    need(got==expected,"unexpected/missing receipt files; preserve and reconcile, never overwrite: "+str(got))
def validate_receipt(root,row,r,stats):
    i=row["index"];p=root/"r23-results"/("reap-full24" if i<12 else "expanded-fit-profile-v1")/f"{i:02d}.json"
    b=pin(p,row["receipt_sha256"]);x=strict(b)
    need(b==canon(x)+b"\n" and x["index"]==i and x["id"]==r["id"] and x["direction"]==r["direction"] and x["subset"]=="fit" and x["teacher"]==TEACHER and x["engine_head"]==ENGINE and x["threads"]==8 and x["admission_sha256"]==row["admission_sha256"] and x["hook_calls"]==423 and x["num_experts"]==128 and x["top_k"]==8,"profile identity")
    z=x["teacher_logits"]
    need(len(z)==9 and [q["input_token"] for q in z[1:]]==r["teacher_prefix_tokens"] and [(q["phase"],q["step"]) for q in z]==[("prefill",0)]+[("decode",j) for j in range(1,9)],"actual fixed input")
    need(all(re.fullmatch("[0-9a-f]{64}",q["sha256"]) and type(q["next"]) is int for q in z),"logit hashes")
    if i<10:
        need(x["matched_r23a_logits"] is True,"legacy oracle")
    else:
        need(x["same_process_observer_equivalence"] is True and x["plain_logits"]==z==x["restored_logits"],"same-process oracle")
    if i>=12:
        need(x["plan_blob"]==PLAN and x["matched_r23a_logits"] is None and x["cross_process_r23a_differences"] is None,"expanded provenance")
        mismatches=[dict(step=j,expected=t,actual=q["next"]) for j,(t,q) in enumerate(zip(r["teacher_prefix_tokens"],z[:8]),1) if t!=q["next"]]
        need(x["historical_argmax_mismatches"]==mismatches and x["historical_greedy_match"]==(not mismatches),"prefix ledger")
    need(set(x["profile"])=={f"{l}/{p}" for l in range(1,48) for p in ("prefill","decode")},"profile coverage")
    for k,v in x["profile"].items():
        stats.check(v,128,8,x["scale"]);need(v["tokens"]==(r["prompt_tokens"] if k.endswith("/prefill") else 8),"token coverage")
    return x
def resume_rows(log,fit,ad_sha):
    rows=[strict(l) for l in log.splitlines(keepends=True) if l.startswith(b"{") and l.endswith(b"\n")]
    caps=[x for x in rows if x.get("status")=="EXPANDED_FIT_PROFILE_CAPTURED"]
    done=[x for x in rows if x.get("status")=="EXPANDED_FIT_PROFILE_BATCH_COMPLETE_NOT_SELECTION"]
    ready=[x for x in rows if x.get("status")=="TEACHER_READY"]
    need(len(caps)==4 and len(done)==1 and done[0]["completed"]==done[0]["total"]==48 and done[0]["holdout_capture"]=="NOT_RUN","complete resumed batch")
    need(len(ready)==1 and ready[0]["teacher"]==TEACHER and ready[0]["admission_sha256"]==ad_sha,"resume teacher admission")
    out=[]
    for i,x in enumerate(caps,44):
        need(x["index"]==i and x["completed"]==i+1 and x["hook_calls"]==423 and x["same_process_triple"] is True,"resume oracle")
        out.append(dict(index=i,id=x["id"],receipt_sha256=x["receipt_sha256"],admission_sha256=ad_sha))
    check_index(out,fit,44,48);return out
def selftest():
    import tempfile
    fit=[dict(id=str(i)) for i in range(48)]
    rows=[dict(index=i,id=str(i),receipt_sha256="a"*64,admission_sha256="b"*64) for i in range(44)]
    check_index(rows,fit,0,44)
    def reject(f):
        try:f()
        except (ValueError,OSError):return
        raise AssertionError("bad evidence accepted")
    reject(lambda:check_index(rows[:-1],fit,0,44))
    reject(lambda:check_index(rows[:43]+[dict(rows[-1],id="wrong")],fit,0,44))
    reject(lambda:strict(b'{"a":1,"a":2}'))
    with tempfile.TemporaryDirectory() as d:
        p=P(d);publish(p/"12.json",{"value":1});inventory(p,12,13)
        reject(lambda:publish(p/"12.json",{"value":2}));need(strict(read(p/"12.json"))=={"value":1},"no overwrite")
        publish(p/"13.json",{"value":2});reject(lambda:inventory(p,12,13))
    ready=dict(status="TEACHER_READY",teacher=TEACHER,admission_sha256="b"*64)
    caps=[dict(status="EXPANDED_FIT_PROFILE_CAPTURED",index=i,completed=i+1,id=str(i),hook_calls=423,same_process_triple=True,receipt_sha256="c"*64) for i in range(44,48)]
    done=dict(status="EXPANDED_FIT_PROFILE_BATCH_COMPLETE_NOT_SELECTION",completed=48,total=48,holdout_capture="NOT_RUN")
    def log(x):return b"".join(canon(v)+b"\n" for v in x)
    need(len(resume_rows(log([ready]+caps+[done]),fit,"b"*64))==4,"resume inventory")
    reject(lambda:resume_rows(log([ready]+caps[:-1]+[done]),fit,"b"*64))
    reject(lambda:resume_rows(log([ready]+caps+[dict(done,holdout_capture="RUN")]),fit,"b"*64))
    print(json.dumps(dict(status="PASS_SYNTHETIC_POWER_RECOVERY",negative=7,real_disk="NOT_CHECKED")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--inputs",type=P);a.add_argument("--out",type=P);a.add_argument("--stage",choices=("preflight","resume-admission","finalize"),default="preflight");a.add_argument("--comment",type=int);a.add_argument("--head");a.add_argument("--resume-log",type=P);a.add_argument("--resume-stderr",type=P);a.add_argument("--resume-admission-sha");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs,"inputs");d=q.inputs
    plan=strict(pin(d/"expanded.json",PLAN,True));w=strict(pin(d/"witness.json",WITNESS,True));fit=plan["fit"];hold=plan["holdout"]
    need(len(fit)==48 and len(hold)==24 and len({r["line"] for r in fit})==30 and len({r["line"] for r in hold})==14 and not ({r["line"] for r in fit}&{r["line"] for r in hold}),"fit scope")
    need(w["plan_blob"]==PLAN and w["teacher"]==TEACHER and w["engine_head"]==ENGINE and w["expected_first"]==44 and w["expected_stop"]==48,"witness")
    rows=w["rows"];check_index(rows,fit,0,44)
    need(all(r["admission_sha256"]==OLD_AD for r in rows[12:]),"old admission")
    mp=d/"masks.py";pin(mp,STATS,True);sp=importlib.util.spec_from_file_location("masks",mp);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    root=ROOT;dest=root/"r23-results/expanded-fit-profile-v1"
    with (root/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        raw=pin(root/"r23-admissions/expanded-fit-12-48.json",OLD_AD);ad=strict(raw)
        need(raw==canon(ad) and ad["first"]==12 and ad["stop"]==48 and ad["ids"]==[r["id"] for r in fit[12:]] and ad["teacher"]==TEACHER and ad["engine_head"]==ENGINE and ad["plan_blob"]==PLAN,"old admission identity")
        for row,r in zip(rows,fit):validate_receipt(root,row,r,m)
        if q.stage=="finalize":
            need(q.out and q.resume_log and q.resume_stderr and q.resume_admission_sha,"resume evidence")
            need(read(q.resume_stderr)==b"","resume stderr")
            raw=pin(root/"r23-admissions/expanded-fit-44-48.json",q.resume_admission_sha);ad=strict(raw)
            need(raw==canon(ad) and ad["status"]=="R23F_EXPANDED_FIT_PROFILE_ADMITTED" and ad["first"]==44 and ad["stop"]==48 and ad["ids"]==[r["id"] for r in fit[44:]] and ad["prior_receipts"]==[r["receipt_sha256"] for r in rows[12:]] and ad["plan_blob"]==PLAN and ad["teacher"]==TEACHER and ad["engine_head"]==ENGINE and ad["threads"]==8,"resume admission scope")
            rows=rows+resume_rows(read(q.resume_log),fit,q.resume_admission_sha);inventory(dest,12,48)
            for row,r in zip(rows[44:],fit[44:]):validate_receipt(root,row,r,m)
            result=dict(status="EXPANDED_FIT48_RECOVERY_VERIFIED_NOT_SELECTION",plan_blob=PLAN,teacher=TEACHER,engine_head=ENGINE,witness_blob=WITNESS,rows=rows,fit_lines=30,holdout_capture="NOT_RUN",resume_admission_sha256=q.resume_admission_sha,resume_log_sha256=sha(read(q.resume_log)),resume_stderr_sha256=sha(read(q.resume_stderr)))
            h=publish(q.out,result);print(json.dumps(dict(status=result["status"],receipt_sha256=h,profiles=48)),flush=True);return
        inventory(dest,12,44)
        print(json.dumps(dict(status="DISK_PREFIX44_VERIFIED_NOT_RESUMED",profiles=44,remaining=4,holdout_capture="NOT_RUN")),flush=True)
        if q.stage=="resume-admission":
            need(q.out and type(q.comment) is int and q.comment>0 and q.head and re.fullmatch("[0-9a-f]{40}",q.head),"explicit coordinator admission")
            ad=dict(status="R23F_EXPANDED_FIT_PROFILE_ADMITTED",plan_blob=PLAN,teacher=TEACHER,engine_head=ENGINE,threads=8,first=44,stop=48,ids=[r["id"] for r in fit[44:]],prior_receipts=[r["receipt_sha256"] for r in rows[12:]],power_witness_blob=WITNESS,predecessor_admission_sha256=OLD_AD,cd_report_sha256=CD,allocation_sha256=ALLOC,admission_comment=q.comment,source_head=q.head)
            # Exact original driver expects canonical bytes WITHOUT final newline.
            with q.out.open("xb") as f:f.write(canon(ad));f.flush();os.fsync(f.fileno())
            print(json.dumps(dict(status="RESUME44_48_ADMISSION_READY_NOT_EXECUTED",admission_sha256=sha(canon(ad)))),flush=True)
if __name__=="__main__":main()
