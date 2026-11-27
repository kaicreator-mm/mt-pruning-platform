"""Bounded R23CD report replay, fit allocation and expanded-profile admission.
A complete diagnostic report and exact receipt inventory are prerequisites.
No holdout data or mask quality admission in this chain.
"""
import argparse,hashlib,importlib.util,json,math,os,pathlib,sys,subprocess,time,fcntl
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
CD_AD="fe652519d0ed823ea42cccb7041e1d183a7e11fedd82150111bdbe20ec79e5ac"
CD_PLAN="a9da78015d010484c16d6c8af9000a15d46946db00677c19f4fac4a0d562e41e"
B_DECISION="3fa7776480e9f62bc15ac0c3cb16f9fa05b047ceef0c2ef1e1d9df1219240468"
B_AUDIT="a96037d8adc6e7fd06f8fd7b453eee09a445c6faa78034490783b1c51856c68e"
PLAN="3c92bd0a944637ee7eb8d36d271417396b313026"
EXPANDED="a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CD_SHA="b193c3a3cf6ec23ea87539e924e43b4adf88465db648e234a8768f2908bf6527"
DP_SHA="e25890bc523cbb5bd8375b212df2a3b62b0e540aab439fbb7fa733095dffd74a"
INTERACTION="0bc99fcec770371e076c625a9ec9a540aaf65ba3"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file");return p.read_bytes()
def load(p,n,expected,git=False):
    need((blob if git else sha)(read(p))==expected,"source pin "+n)
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
def complete(logs,rows,report_sha):
    hits=[x for x in logs if x.get("status")=="CD_FIT_CAPTURED"]
    need(len(hits)==len(rows)==12 and all(x["completed"]==i+1 and x["id"]==r["id"] for i,(x,r) in enumerate(zip(hits,rows))),"CD receipt inventory")
    fin=[x for x in logs if x.get("status")=="FIT_ONLY_INTERACTION_BUDGET_DIAGNOSTIC_NOT_ADMISSION"]
    need(len(fin)==1 and fin[0]["receipt_sha256"]==report_sha,"CD report marker")
    ready=[x for x in logs if x.get("status")=="TEACHER_READY"]
    need(len(ready)==1 and ready[0]["teacher"]==TEACHER and ready[0]["admission_sha256"]==CD_AD,"CD teacher")
    return [x["receipt_sha256"] for x in hits],fin[0]
def report_equal(replay,report,marker):
    # JSON arrays deserialize as lists; the producer uses a tuple for groups.
    # Compare canonical serialized content, not Python container identity.
    need(canon(replay)==canon(report) and canon(marker["means"])==canon(report["means"]),"exact CD canonical report replay")
def selftest():
    rows=[dict(id=str(i)) for i in range(12)]
    logs=[dict(status="CD_FIT_CAPTURED",id=str(i),completed=i+1,receipt_sha256="h") for i in range(12)]
    logs+=[dict(status="FIT_ONLY_INTERACTION_BUDGET_DIAGNOSTIC_NOT_ADMISSION",receipt_sha256="r"),dict(status="TEACHER_READY",teacher=TEACHER,admission_sha256=CD_AD)]
    need(len(complete(logs,rows,"r")[0])==12,"complete")
    for bad in (logs[1:],logs[:-1],logs[:12]+[dict(logs[12],receipt_sha256="x"),logs[13]]):
        try:complete(bad,rows,"r")
        except ValueError:pass
        else:raise AssertionError("incomplete predecessor accepted")
    replay={"groups":("zh-my/prefill","zh-my/decode"),"means":{"original96":{"zh-my/prefill":0.25,"zh-my/decode":0.5}}}
    report=json.loads(canon(replay));marker={"means":report["means"]}
    need(replay!=report,"JSON roundtrip container witness")
    report_equal(replay,report,marker)
    for bad in (dict(report,groups=list(reversed(report["groups"]))),dict(report,means={"original96":{"zh-my/prefill":0.25000000000001,"zh-my/decode":0.5}})):
        try:report_equal(replay,bad,marker)
        except ValueError:pass
        else:raise AssertionError("changed report accepted")
    try:report_equal(replay,report,{"means":{}})
    except ValueError:pass
    else:raise AssertionError("changed marker accepted")
    print(json.dumps(dict(status="PASS_SYNTHETIC_CD_CHAIN_GATE_V2",negative=6,json_roundtrip_exact=True)))
def main():
    a=argparse.ArgumentParser();a.add_argument("--inputs",type=P);a.add_argument("--out",type=P);a.add_argument("--comment",type=int);a.add_argument("--head");a.add_argument("--selftest",action="store_true");q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.out and q.comment and q.head,"explicit gate args");d=q.inputs
    planraw=read(d/"plan.json");eraw=read(d/"expanded.json");need(blob(planraw)==PLAN and blob(eraw)==EXPANDED,"plans")
    fit=json.loads(planraw)["plan"]["fit"];expanded=json.loads(eraw);need(expanded["fit"][:12]==fit and len(expanded["fit"])==48 and len(expanded["holdout"])==24,"expanded scope")
    cd=load(d/"cd.py","cd",CD_SHA);interaction=load(d/"interaction.py","interaction",INTERACTION,True);dp=load(d/"dp.py","dp",DP_SHA)
    root=ROOT/"r23-results";dst=root/"interaction-budget-v1";log=root/"interaction-budget-v1.log";err=root/"interaction-budget-v1.stderr"
    auditraw=read(root/"score-proxy-v1-replay-audit.json");need(sha(auditraw)==B_AUDIT and json.loads(auditraw)["decision_sha256"]==B_DECISION,"B rejection audit")
    start=time.monotonic()
    while True:
        if err.exists():need(err.stat().st_size==0,"CD stderr")
        if (dst/"report.json").is_file() and log.is_file():
            raw=read(dst/"report.json");h=sha(raw);logs=[json.loads(l) for l in read(log).splitlines(keepends=True) if l.endswith(b"\n")]
            if any(x.get("status")=="FIT_ONLY_INTERACTION_BUDGET_DIAGNOSTIC_NOT_ADMISSION" and x.get("receipt_sha256")==h for x in logs):break
        need(time.monotonic()-start<900,"bounded CD dependency");time.sleep(10)
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        need(time.monotonic()-start<900 and read(err)==b"","CD completion")
        ad=json.loads(read(dst/"admission.json"));need(sha(canon(ad))==CD_AD and ad["predecessor_decision_sha256"]==B_DECISION,"CD admission")
        p=read(root/"fit-interaction-budget-plan-v1.json");need(sha(p)==CD_PLAN,"CD plan");masks={n:x["kept"] for n,x in json.loads(p)["masks"].items()}
        hashes,marker=complete(logs,fit,h);captures={}
        for i,(r,digest) in enumerate(zip(fit,hashes)):
            rr=read(dst/f"fit-{i:02d}.json");need(sha(rr)==digest,"CD receipt hash")
            x=json.loads(rr);need(x["index"]==i and x["teacher_digest"]==TEACHER and x["engine_head"]==ENGINE and x["admission_sha256"]==CD_AD and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["threads"]==8 and x["extra_oracle_evaluations"]==512,"CD identity")
            z=x["teacher_logits"];need(len(z)==9 and [q["input_token"] for q in z[1:]]==r["teacher_prefix_tokens"],"actual prefix")
            captures[r["id"]]=x
        report=json.loads(raw);replay=cd.report(fit,captures,interaction,masks,list(range(1,48)))
        report_equal(replay,report,marker)
        data=report["layer_group_combined_error"];curves={l:{k:v[n] for k,n in dp.KEYS.items()} for l,v in data.items()}
        allocation=dp.solve(curves,47*96);allocation["report_sha256"]=h
        ar=canon(allocation)+b"\n";out=root/"interaction-budget-v1-allocation.json"
        with out.open("xb") as f:f.write(ar);f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(status="CD_REPORT_REPLAY_PASS_NOT_QUALITY",report_sha256=h,receipts=12,allocation_status=allocation["status"],allocation_sha256=sha(ar),budget_used=allocation["budget_used"],fit_primary_improvement=allocation["fit_primary_improvement"],violations=allocation["violations"])),flush=True)
        a=dict(status="R23F_EXPANDED_FIT_PROFILE_ADMITTED",plan_blob=EXPANDED,teacher=TEACHER,engine_head=ENGINE,threads=8,first=12,stop=48,ids=[r["id"] for r in expanded["fit"][12:]],prior_receipts=[],predecessor_job="job_01M8E7139400NYNQNQTJT00TPG",cd_report_sha256=h,allocation_sha256=sha(ar),admission_comment=q.comment,source_head=q.head)
        with q.out.open("xb") as f:f.write(canon(a));f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(status="EXPANDED_FIT_CHAIN_GATE_PASSED",admission_sha256=sha(canon(a)),new_fit=36,holdout_capture="NOT_RUN")),flush=True)
if __name__=="__main__":main()
