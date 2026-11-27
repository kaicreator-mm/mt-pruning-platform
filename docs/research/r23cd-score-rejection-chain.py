"""R23CD conditional gate: run fit diagnostics only after score-proxy rejection.
An accepted mask takes priority; failed/incomplete predecessor is not rejection.
"""
import argparse,fcntl,hashlib,json,os,pathlib,time
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
PLAN_SHA="a9da78015d010484c16d6c8af9000a15d46946db00677c19f4fac4a0d562e41e"
ADMISSION_SHA="bc467020765d4d3728a28c5f5c6b161be10b51f337006a41fc535f37c33aedc5"
PLAN_BLOB="3c92bd0a944637ee7eb8d36d271417396b313026"
FRESH_BLOB="bc6d92ef87d23c954cd1c3f86527a39b52b5cc18"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file");return p.read_bytes()
def log_rows(raw):
    return [json.loads(l) for l in raw.splitlines(keepends=True) if l.endswith(b"\n")]
def receipts(rows,stage,ids):
    records=[x for x in rows if x.get("status")==stage]
    need(len(records)==len(ids) and all(x["id"]==i and x["completed"]==j+1 for j,(x,i) in enumerate(zip(records,ids))),"capture scope "+stage)
    return [x["receipt_sha256"] for x in records]
def decision_kind(decision,fit,rows,decision_hash):
    st=decision["status"]
    need(st in ("NO_ADMISSION_FROM_FIT","NO_ADMISSION_FROM_LOCAL_PROXY","LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION"),"unexpected decision")
    if st=="NO_ADMISSION_FROM_FIT":
        need(not fit["changes"] and decision["candidate"] is None,"fit rejection");return "REJECTED_FIT"
    r=[x for x in rows if x.get("status")==st]
    need(len(r)==1 and r[0]["receipt_sha256"]==decision_hash and {k:v for k,v in r[0].items() if k!="receipt_sha256"}==decision,"final log binding")
    need(decision["fit_mask_sha256"]==fit["mask_sha256"] and bool(fit["changes"]),"fit mask")
    if st=="NO_ADMISSION_FROM_LOCAL_PROXY":
        need(decision["candidate"] is None,"holdout rejection");return "REJECTED_FRESH"
    need(decision["candidate"]["kept"]==fit["kept"] and decision["candidate"]["mask_sha256"]==fit["mask_sha256"],"accepted mask binding")
    return "ACCEPTED_LOCAL_PROXY"
def selftest():
    fit={"changes":[],"mask_sha256":"m","kept":{}}
    need(decision_kind({"status":"NO_ADMISSION_FROM_FIT","candidate":None},fit,[],"h")=="REJECTED_FIT","fit")
    fit["changes"]=[1];d={"status":"NO_ADMISSION_FROM_LOCAL_PROXY","candidate":None,"fit_mask_sha256":"m"}
    need(decision_kind(d,fit,[dict(d,receipt_sha256="h")],"h")=="REJECTED_FRESH","fresh")
    d={"status":"LOCAL_PROXY_CANDIDATE_READY_FOR_REAL_VALIDATION","candidate":{"kept":{},"mask_sha256":"m"},"fit_mask_sha256":"m"}
    need(decision_kind(d,fit,[dict(d,receipt_sha256="h")],"h")=="ACCEPTED_LOCAL_PROXY","accepted")
    try:decision_kind(d,fit,[],"h")
    except ValueError:pass
    else:raise AssertionError("missing log accepted")
    print(json.dumps(dict(status="PASS_SYNTHETIC_REJECTION_CHAIN",checks=4)))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--plan",type=P);p.add_argument("--fresh",type=P);p.add_argument("--out",type=P);p.add_argument("--comment",type=int);p.add_argument("--head");p.add_argument("--timeout",type=int,default=3600);a=p.parse_args()
    if a.selftest:selftest();return
    need(a.plan and a.fresh and a.out and a.comment and a.head and 1<=a.timeout<=3600,"bounded gate")
    pr=read(a.plan);fr=read(a.fresh);need(blob(pr)==PLAN_BLOB and blob(fr)==FRESH_BLOB,"plan pins")
    fitids=[r["id"] for r in json.loads(pr)["plan"]["fit"]];freshids=[r["id"] for r in json.loads(fr)["rows"]]
    root=ROOT/"r23-results";dst=root/"score-proxy-v1";log=root/"score-proxy-v1.log";err=root/"score-proxy-v1.stderr";start=time.monotonic()
    while True:
        if err.exists():need(err.stat().st_size==0,"predecessor stderr not empty")
        if (dst/"decision.json").exists() and log.exists():
            raw=read(dst/"decision.json");dec=json.loads(raw)
            rows=log_rows(read(log));st=dec["status"]
            if st=="NO_ADMISSION_FROM_FIT" or any(x.get("status")==st and x.get("receipt_sha256")==sha(raw) for x in rows):break
        need(time.monotonic()-start<a.timeout,"predecessor incomplete");time.sleep(10)
    # The predecessor's exclusive lock must have been released before admission.
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        need(time.monotonic()-start<a.timeout and err.stat().st_size==0,"predecessor completion")
        ad=json.loads(read(dst/"admission.json"));need(sha(canon(ad))==ADMISSION_SHA,"predecessor admission")
        fitraw=read(dst/"fit-frozen.json");fit=json.loads(fitraw)
        markers=[x for x in rows if x.get("status")=="FIT_FROZEN_NOT_ADMITTED"];need(len(markers)==1 and markers[0]["receipt_sha256"]==sha(fitraw) and markers[0]["mask_sha256"]==fit["mask_sha256"] and sha(canon(fit["kept"]))==fit["mask_sha256"],"fit freeze")
        fh=receipts(rows,"FIT_CAPTURED",fitids)
        for i,h in enumerate(fh):need(sha(read(dst/f"fit-{i:02d}.json"))==h,"fit receipt")
        kind=decision_kind(dec,fit,rows,sha(raw))
        if kind!="REJECTED_FIT":
            for i,h in enumerate(receipts(rows,"FRESH_CAPTURED",freshids)):need(sha(read(dst/f"fresh-{i:02d}.json"))==h,"fresh receipt")
        if kind=="ACCEPTED_LOCAL_PROXY":
            print(json.dumps(dict(status="SKIP_CD_PRIORITIZE_ACCEPTED_CANDIDATE",decision_sha256=sha(raw))),flush=True);return
        need(sha(read(root/"fit-interaction-budget-plan-v1.json"))==PLAN_SHA,"CD plan pin")
        out=dict(status="R23CD_FIT_DIAGNOSTIC_ADMITTED",plan_sha256=PLAN_SHA,teacher="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94",engine_head="3212303e1e02994a5124db8c85195b91325c7a7c",fit_plan_blob=PLAN_BLOB,threads=8,fit_ids=fitids,predecessor_job="job_01M8E6JWSA000X1RJENQ3Z55AK",predecessor_decision_sha256=sha(raw),predecessor_outcome=kind,admission_comment=a.comment,source_head=a.head)
        with a.out.open("xb") as f:f.write(canon(out));f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(status="CD_CONDITIONAL_GATE_PASSED",outcome=kind,admission_sha256=sha(canon(out)))),flush=True)
if __name__=="__main__":main()
