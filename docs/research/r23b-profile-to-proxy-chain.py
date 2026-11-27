"""Bounded downstream admission after exact R23B profile completion.
Only consumes profile job's completed log and receipt digests, never rankings
or holdout losses. Fail closed on stderr, incomplete manifest or drift.
"""
import argparse,hashlib,json,pathlib,time,os
P=pathlib.Path
ROOT=P.home()/".cache/mt-pruning"
LEGACY="aa91a0c2ba3fd08fde8059483d57b52fc5cf6082e80e3590a6d112e82414ed0f"
RESUME="3af650d6d649149ac22f5f21578c3fa647510fd4afd271438dc60e2641bda043"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
PLAN="3c92bd0a944637ee7eb8d36d271417396b313026"
FRESH="bc6d92ef87d23c954cd1c3f86527a39b52b5cc18"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def records(raw):
    # Only parse a complete newline-terminated prefix, not partial tee output.
    return [json.loads(l) for l in raw.splitlines(keepends=True) if l.endswith(b"\n")]
def collect(old,new,ids):
    a=[x for x in old if x.get("status")=="PROFILE_CAPTURED"];b=[x for x in new if x.get("status")=="PROFILE_CAPTURED"]
    need(len(a)==10,"legacy receipt count")
    need(len(b)<=14,"resume receipt count")
    x=a+b
    for i,r in enumerate(x):
        need(r["id"]==ids[i] and r["index"]==i and r["completed"]==i+1 and r["hook_calls"]==423 and r["logit_checks"]==9,"profile receipt order")
        h=r["receipt_sha256"];need(len(h)==64 and all(c in "0123456789abcdef" for c in h),"receipt digest")
        if i>=10:need(r["restored_logit_checks"]==9 and 0<=r["cross_process_differences"]<=9,"resume oracle")
    marker=[r for r in new if r.get("status")=="R23B_FULL24_PROFILE_COMPLETE_NOT_QUALITY"]
    if not marker:return None
    need(len(marker)==1 and marker[0]["samples"]==24 and marker[0]["legacy_exact_cross_run_samples"]==10 and marker[0]["new_same_process_triple_samples"]==14 and marker[0]["hook_calls"]==10152 and len(x)==24,"final completion contract")
    ready=[r for r in new if r.get("status")=="TEACHER_READY"]
    need(len(ready)==1 and ready[0]["teacher"]==TEACHER and ready[0]["admission_sha256"]==RESUME,"resume teacher")
    return [r["receipt_sha256"] for r in x]
def selftest():
    ids=[str(i) for i in range(24)]
    rows=[dict(status="PROFILE_CAPTURED",id=i,index=j,completed=j+1,hook_calls=423,logit_checks=9,restored_logit_checks=9,cross_process_differences=0,receipt_sha256="1"*64) for j,i in enumerate(ids)]
    marker=dict(status="R23B_FULL24_PROFILE_COMPLETE_NOT_QUALITY",samples=24,legacy_exact_cross_run_samples=10,new_same_process_triple_samples=14,hook_calls=10152)
    ready=dict(status="TEACHER_READY",teacher=TEACHER,admission_sha256=RESUME)
    need(collect(rows[:10],[ready]+rows[10:11],ids) is None,"partial no admission")
    need(len(collect(rows[:10],[ready]+rows[10:]+[marker],ids))==24,"complete")
    def reject(f):
        try:f()
        except ValueError:return
        raise AssertionError("bad manifest accepted")
    reject(lambda:collect(rows[:10],[ready]+rows[10:23]+[marker],ids))
    reject(lambda:collect(rows[:10],[ready]+[dict(rows[10],id="wrong")]+rows[11:]+[marker],ids))
    reject(lambda:collect(rows[:10],[ready]+[dict(rows[10],restored_logit_checks=0)]+rows[11:]+[marker],ids))
    print(json.dumps(dict(status="PASS_SYNTHETIC_PROFILE_CHAIN_GATE",negative=3,real_admission="NOT_RUN")))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--plan",type=P);p.add_argument("--fresh",type=P);p.add_argument("--out",type=P);p.add_argument("--comment",type=int);p.add_argument("--source-head");p.add_argument("--timeout",type=int,default=1800);a=p.parse_args()
    if a.selftest:selftest();return
    need(a.plan and a.fresh and a.out and a.comment and a.comment>0 and a.source_head and 1<=a.timeout<=3600,"bounded chain admission")
    planraw=a.plan.read_bytes();freshraw=a.fresh.read_bytes();need(blob(planraw)==PLAN and blob(freshraw)==FRESH,"input pins")
    plan=json.loads(planraw)["plan"];fit=plan["fit"];fresh=json.loads(freshraw)["rows"];rows=fit+plan["holdout"]
    ids=[r["id"] for r in rows];need(len(ids)==24 and len(fresh)==12 and len({r["line"] for r in rows+fresh})==36,"scope")
    root=ROOT/"r23-results";ad=ROOT/"r23-admissions"
    need(sha((ad/"reap-full24.json").read_bytes())==LEGACY,"legacy admission")
    start=time.monotonic()
    while True:
        log=root/"reap-full24-resume-v2.log";err=root/"reap-full24-resume-v2.stderr"
        if err.exists():need(err.stat().st_size==0,"resume stderr not empty")
        if log.exists():
            hashes=collect(records((root/"reap-full24.log").read_bytes()),records(log.read_bytes()),ids)
            if hashes is not None:break
        need(time.monotonic()-start<a.timeout,"bounded wait expired")
        time.sleep(10)
    need(sha((ad/"reap-full24-resume-v2.json").read_bytes())==RESUME,"resume admission")
    dest=root/"reap-full24";need(sorted(x.name for x in dest.iterdir())==[f"{i:02d}.json" for i in range(24)],"profile inventory")
    for i,h in enumerate(hashes):
        q=dest/f"{i:02d}.json";need(q.is_file() and not q.is_symlink() and sha(q.read_bytes())==h,"immutable receipt "+str(i))
    out=dict(status="R23B_SCORE_PROXY_ADMITTED",teacher=TEACHER,engine_head="3212303e1e02994a5124db8c85195b91325c7a7c",plan_blob=PLAN,fresh_blob=FRESH,threads=8,profile_job="job_01M8E6CJ5G003989WQ8HGH7J9K",fit_ids=[r["id"] for r in fit],fresh_ids=[r["id"] for r in fresh],fit_profile_sha256=hashes[:12],resume_admission_sha256=RESUME,admission_comment=a.comment,source_head=a.source_head,profile_manifest_sha256=sha(canon(hashes)),scope="seven frozen score masks fit-only, one frozen combination, fresh holdout reject only")
    with a.out.open("xb") as f:f.write(canon(out));f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status="PROFILE_CHAIN_GATE_PASSED",profiles=24,fit_profiles=12,manifest_sha256=out["profile_manifest_sha256"],admission_sha256=sha(canon(out)))),flush=True)
if __name__=="__main__":main()
