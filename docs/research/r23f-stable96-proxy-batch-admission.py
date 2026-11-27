"""R23F fixed-mask proxy batch admission and exact independent decision replay.
No model load, no server connection, no generation, no selection using holdout.
Run only on restored host after all prior relevant jobs are terminal.
"""
import argparse,fcntl,hashlib,importlib.util,json,os,pathlib,re,sys,tempfile
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PLAN="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
ORIGINAL="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
BASE="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
PROXY="248737901db40b50efb0a26a9d5ef76d1928538b"
STABLE_SHA="1fd2e670bc55570668b32c18d179f497d2a8a50c1808b4c18e304668af9f291a"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
def need(x,m):
    if not x:raise ValueError(m)
def sha(x):return hashlib.sha256(x).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(x):return hashlib.sha1(b"blob "+str(len(x)).encode()+b"\0"+x).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p.read_bytes()
def strict(raw):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,"duplicate key");d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def pin(p,h,git=False):
    b=read(p);need((blob if git else sha)(b)==h,"pin "+str(p));return b
def load(p,name,h,git=False):
    pin(p,h,git);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def capture_rows(log,rows,stage,first,stop,ad_sha):
    lines=[strict(l) for l in log.splitlines(keepends=True) if l.startswith(b"{") and l.endswith(b"\n")]
    ready=[r for r in lines if r.get("status")=="TEACHER_READY"]
    need(len(ready)==1 and ready[0]["teacher"]==TEACHER and ready[0]["stage"]==stage and ready[0]["admission_sha256"]==ad_sha,"teacher admission")
    caps=[r for r in lines if r.get("status")=="STABLE96_PROXY_CAPTURED"]
    need(len(caps)==stop-first,"batch count")
    for i,(r,p) in enumerate(zip(caps,rows[first:stop]),first):
        need(r["stage"]==stage and r["index"]==i and r["completed"]==i+1 and r["id"]==p["id"] and re.fullmatch("[0-9a-f]{64}",r["receipt_sha256"]),"capture order")
    if stop<len(rows):
        end=[r for r in lines if r.get("status")=="STABLE96_PROXY_PARTIAL_BATCH_COMPLETE"]
        need(len(end)==1 and end[0]["stage"]==stage and end[0]["completed"]==stop and end[0]["total"]==len(rows),"partial marker")
        marker=None
    else:
        statuses=("FIT_LOCAL_PROXY_PASS_NOT_ADMISSION","NO_ADMISSION_FROM_EXPANDED_FIT") if stage=="fit" else ("LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION","NO_ADMISSION_FROM_EXPANDED_FRESH")
        end=[r for r in lines if r.get("status") in statuses]
        need(len(end)==1 and end[0]["stage"]==stage and end[0]["samples"]==len(rows) and re.fullmatch("[0-9a-f]{64}",end[0]["receipt_sha256"]),"decision marker");marker=end[0]
    return caps,marker
def admit_fresh(fit):
    return fit["status"]=="FIT_LOCAL_PROXY_PASS_NOT_ADMISSION" and fit["stage"]=="fit" and fit["samples"]==48 and fit["unique_aligned_lines"]==30 and fit["primary_relative_improvement"]>=.05 and not fit["violations"] and fit["overall_guard"] is True
def selftest():
    rows=[dict(id=str(i)) for i in range(48)]
    ad="a"*64
    ready=dict(status="TEACHER_READY",teacher=TEACHER,stage="fit",admission_sha256=ad)
    caps=[dict(status="STABLE96_PROXY_CAPTURED",stage="fit",index=i,completed=i+1,id=str(i),receipt_sha256="b"*64) for i in range(24)]
    end=dict(status="STABLE96_PROXY_PARTIAL_BATCH_COMPLETE",stage="fit",completed=24,total=48)
    def raw(x):return b"".join(canon(r)+b"\n" for r in x)
    need(len(capture_rows(raw([ready]+caps+[end]),rows,"fit",0,24,ad)[0])==24,"batch")
    def reject(f):
        try:f()
        except ValueError:return
        raise AssertionError("invalid evidence accepted")
    reject(lambda:capture_rows(raw([ready]+caps[:-1]+[end]),rows,"fit",0,24,ad))
    reject(lambda:capture_rows(raw([ready]+caps+[dict(end,completed=23)]),rows,"fit",0,24,ad))
    reject(lambda:capture_rows(raw([ready]+[dict(caps[0],id="wrong")]+caps[1:]+[end]),rows,"fit",0,24,ad))
    reject(lambda:capture_rows(raw([dict(ready,admission_sha256="x")]+caps+[end]),rows,"fit",0,24,ad))
    f=dict(status="FIT_LOCAL_PROXY_PASS_NOT_ADMISSION",stage="fit",samples=48,unique_aligned_lines=30,primary_relative_improvement=.1,violations=[],overall_guard=True)
    need(admit_fresh(f) and not admit_fresh(dict(f,violations=["zh-my/decode"])) and not admit_fresh(dict(f,primary_relative_improvement=.049)) and not admit_fresh(dict(f,unique_aligned_lines=29)),"fresh gate")
    print(json.dumps(dict(status="PASS_SYNTHETIC_PROXY_BATCH_ADMISSION",negative=7,real_capture="NOT_RUN")))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--inputs",type=P);a.add_argument("--stage",choices=("fit-first","fit-rest","fresh","audit"));a.add_argument("--candidate-sha");a.add_argument("--manifest",type=P);a.add_argument("--manifest-sha");a.add_argument("--out",type=P);a.add_argument("--comment",type=int);a.add_argument("--head");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.stage and q.candidate_sha and q.manifest and q.manifest_sha and q.out,"explicit evidence");d=q.inputs
    plan=strict(pin(d/"expanded.json",PLAN,True));fit=plan["fit"];fresh=plan["holdout"]
    need(len(fit)==48 and len(fresh)==24 and len({r["line"] for r in fit})==30 and len({r["line"] for r in fresh})==14 and not ({r["line"] for r in fit}&{r["line"] for r in fresh}),"grouped isolation")
    orig=strict(pin(d/"original.json",ORIGINAL,True))["candidates"]["96"]["kept"];need(sha(canon(orig))==BASE,"original mask")
    stable=load(d/"stable.py","stable",STABLE_SHA);proxy=load(d/"proxy.py","proxy",PROXY,True)
    root=ROOT/"r23-results";dst=root/"expanded-stable96-proxy-v1";candraw=pin(root/"expanded-stable96-v1.json",q.candidate_sha);cand=strict(candraw)
    need(cand["status"]=="FROZEN_STABLE96_NOT_ADMITTED" and cand["recovery_manifest_sha256"]==q.manifest_sha and cand["plan_blob"]==PLAN and cand["fit_ids"]==[r["id"] for r in fit] and cand["swaps"]>0 and cand["base_mask_sha256"]==BASE,"candidate")
    kept=cand["kept"];masksha=sha(canon(kept))
    need(masksha==cand["mask_sha256"] and set(kept)==set(orig)=={str(i) for i in range(1,48)} and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) and len(set(v)-set(orig[l]))<=2 for l,v in kept.items()),"mask")
    masks={"original96":orig,"stable96":kept};layers=list(range(1,48))
    # Replay the only frozen fit method, not a search over masks. No fresh observations used.
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        replay=P(tmp)/"candidate.json"
        subprocess.run([sys.executable,"-B",str(d/"stable.py"),"--inputs",str(d),"--out",str(replay),"--manifest",str(q.manifest),"--manifest-sha",q.manifest_sha],check=True)
        need(read(replay)==candraw,"exact candidate replay")
    def audit_batch(stage,rows,first,stop,prior,fit_sha=None):
        ap=ROOT/"r23-admissions"/f"expanded-stable96-{stage}-{first:02d}-{stop:02d}.json"
        raw=read(ap);ad=strict(raw);h=sha(raw)
        need(raw==canon(ad) and ad["status"]=="R23F_STABLE96_PROXY_ADMITTED" and ad["stage"]==stage and ad["first"]==first and ad["stop"]==stop and ad["ids"]==[r["id"] for r in rows[first:stop]] and ad["prior_receipts"]==prior and ad["plan_blob"]==PLAN and ad["teacher"]==TEACHER and ad["engine_head"]==ENGINE and ad["threads"]==8 and ad["candidate_receipt_sha256"]==q.candidate_sha and ad["mask_sha256"]==masksha and ad["recovery_manifest_sha256"]==q.manifest_sha,"batch admission")
        if stage=="fresh":need(ad["fit_decision_sha256"]==fit_sha,"fresh fit decision binding")
        prefix=root/f"stable96-recovery-{stage}-{first:02d}-{stop:02d}"
        need(read(prefix.with_suffix(".stderr"))==b"","batch stderr")
        caps,marker=capture_rows(read(prefix.with_suffix(".log")),rows,stage,first,stop,h);out={};hashes=[]
        for i,(r,c) in enumerate(zip(rows[first:stop],caps),first):
            rr=pin(dst/stage/f"{i:02d}.json",c["receipt_sha256"]);x=strict(rr)
            need(x["index"]==i and x["stage"]==stage and x["teacher_digest"]==TEACHER and x["engine_head"]==ENGINE and x["plan_blob"]==PLAN and x["candidate_receipt_sha256"]==q.candidate_sha and x["admission_sha256"]==h and x["extra_oracle_evaluations"]==512,"receipt provenance")
            proxy.checked_capture(x,r,masks,layers);out[r["id"]]=x;hashes.append(c["receipt_sha256"])
        return out,hashes,marker
    def check_decision(stage,rows,captures,marker):
        raw=pin(dst/stage/"decision.json",marker["receipt_sha256"]);result=strict(raw)
        need(canon(result)==canon(proxy.decision(rows,captures,masks,layers,stage)) and {k:v for k,v in marker.items() if k!="receipt_sha256"}==result,"exact decision replay")
        need(sorted(x.name for x in (dst/stage).iterdir())==sorted([f"{i:02d}.json" for i in range(len(rows))]+["decision.json"]),"complete stage directory")
        return result,sha(raw)
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        prev=[];fd=None
        if q.stage=="fit-first":need(not (dst/"fit").exists(),"fit already started");stage="fit";rows=fit;first=0;stop=24
        else:
            ac,prev,_=audit_batch("fit",fit,0,24,[])
            if q.stage=="fit-rest":
                need(sorted(x.name for x in (dst/"fit").iterdir())==[f"{i:02d}.json" for i in range(24)],"first fit exact inventory")
                stage="fit";rows=fit;first=24;stop=48
            else:
                bc,more,mark=audit_batch("fit",fit,24,48,prev);fd,fh=check_decision("fit",fit,{**ac,**bc},mark)
                if not admit_fresh(fd):
                    need(fd["status"]=="NO_ADMISSION_FROM_EXPANDED_FIT" and not (dst/"fresh").exists(),"fit rejection scope")
                    print(json.dumps(dict(status="STOP_FIT_REJECTED_NO_FRESH",fit_decision_sha256=fh)),flush=True);return
                if q.stage=="fresh":
                    need(not (dst/"fresh").exists(),"fresh already started")
                    stage="fresh";rows=fresh;first=0;stop=24;prev=[]
                else:
                    fc,_,mark=audit_batch("fresh",fresh,0,24,[],fh);result,rh=check_decision("fresh",fresh,fc,mark)
                    out=dict(status="FULL_STABLE96_PROXY_REPLAY_NOT_QUALITY",candidate_receipt_sha256=q.candidate_sha,mask_sha256=masksha,recovery_manifest_sha256=q.manifest_sha,fit_decision_sha256=fh,fresh_decision_sha256=rh,decision=result["status"],violations=result["violations"],primary_relative_improvement=result["primary_relative_improvement"],overall_guard=result["overall_guard"])
                    b=canon(out)+b"\n"
                    with q.out.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
                    print(json.dumps(dict(out,receipt_sha256=sha(b))),flush=True);return
        need(type(q.comment) is int and q.comment>0 and q.head and re.fullmatch("[0-9a-f]{40}",q.head),"explicit coordinator")
        out=dict(status="R23F_STABLE96_PROXY_ADMITTED",stage=stage,first=first,stop=stop,ids=[r["id"] for r in rows[first:stop]],prior_receipts=prev,plan_blob=PLAN,teacher=TEACHER,engine_head=ENGINE,threads=8,candidate_receipt_sha256=q.candidate_sha,mask_sha256=masksha,recovery_manifest_sha256=q.manifest_sha,admission_comment=q.comment,source_head=q.head)
        if stage=="fresh":out["fit_decision_sha256"]=fh
        b=canon(out)
        with q.out.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(status="PROXY_BATCH_ADMISSION_READY_NOT_EXECUTED",stage=stage,first=first,stop=stop,admission_sha256=sha(b))),flush=True)
if __name__=="__main__":main()
