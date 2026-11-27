"""R23G independent fit propagation gate. Read-only, stdlib, Windows/Linux.
Requires completed actual job evidence and exact previously frozen candidate.
Does not read fresh observations, construct a mask or run any model."""
import argparse,hashlib,json,math,pathlib
P=pathlib.Path
CAND="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
MASK="77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
AD="c92a54b33c8b9949fc8515831e78bf13b31dd1dce77ee7628e112bf519bb08b4"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
PRIOR="89a5715647088b7560411f379f90d368348dc4a1453bb32acce9e3dac7410c9c"
SENS="0e071705ea763d45e0b7034a20647cb16305b48ea737386a69d8014f903ad381"
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
NAMES=("original96","candidate96")
def need(v,m):
    if not v:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def read(p):
    need(p.is_file() and not p.is_symlink(),"regular file "+str(p));return p.read_bytes()
def strict(b):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,"duplicate key");d[k]=v
        return d
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def pin(p,h,git=False):
    b=read(p);need((blob if git else sha)(b)==h,"pin "+str(p));return strict(b)
def hex64(s):return type(s) is str and len(s)==64 and all(c in "0123456789abcdef" for c in s)
def finite(v):return type(v) in (int,float) and math.isfinite(v) and v>=0
def avg(xs):return math.fsum(xs)/len(xs)
def summarize(rows,caps):
    out={}
    for d in DIRS:
        for phase in ("prefill","decode"):
            g=d+"/"+phase;out[g]={}
            for n in NAMES:
                a=[q for r in rows if r["direction"]==d for q in caps[r["id"]][n]["logits"] if (q["step"]==0)==(phase=="prefill")]
                need(len(a)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"positions")
                out[g][n]=dict(nmse=avg([q["nmse"] for q in a]),kl=avg([q["teacher_to_masked_kl"] for q in a]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in a),positions=len(a))
    viol=[g+"/"+k for g in out if g.startswith(("zh-my/","my-zh/")) for k in ("nmse","kl","argmax_mismatch") if out[g]["candidate96"][k]>out[g]["original96"][k]+1e-12]
    return out,viol
def check_capture(x,r,i,ad=AD):
    need(x["status"]=="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY" and x["teacher_restored"] is True and x["index"]==i and x["id"]==r["id"] and x["direction"]==r["direction"] and x["line"]==r["line"] and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["teacher"]==TEACHER and x["mask_sha256"]==MASK and x["admission_sha256"]==ad,"capture provenance")
    want={(l,"prefill" if j==0 else "decode",j) for l in range(1,48) for j in range(9)}
    for n in NAMES:
        a=x[n];ls=a["layer"]
        need(len(ls)==len(want) and {(q["layer"],q["phase"],q["step"]) for q in ls}==want,"layer coverage")
        for q in ls:
            need(all(finite(q[k]) for k in ("hidden_input_nmse","moe_output_nmse","mean_overlap")) and q["mean_overlap"]<=8 and q["tokens"]==(2 if q["step"]==0 else 1) and type(q["exact_sets"]) is int and 0<=q["exact_sets"]<=q["tokens"],"layer values")
        for key in ("teacher_hashes","masked_hashes"):
            z=a[key];need(len(z)==9 and all(q["step"]==j and hex64(q["sha256"]) and type(q["next"]) is int and q["next"]>=0 for j,q in enumerate(z)),"token hash")
        need(len(a["logits"])==9 and all(q["step"]==j and finite(q["nmse"]) and finite(q["teacher_to_masked_kl"]) and q["teacher_argmax"]==a["teacher_hashes"][j]["next"] and q["masked_argmax"]==a["masked_hashes"][j]["next"] for j,q in enumerate(a["logits"])),"logits")
    need(x["original96"]["teacher_hashes"]==x["candidate96"]["teacher_hashes"],"teacher identity")
def events(log,rows):
    z=[strict(l) for l in log.splitlines() if l.startswith(b"{")]
    r=[x for x in z if x.get("status")=="TEACHER_READY"];need(len(r)==1 and r[0]["teacher"]==TEACHER and r[0]["admission_sha256"]==AD,"teacher event")
    c=[x for x in z if x.get("status")=="R23G_CAPTURED"];need(len(c)==18 and all(e["index"]==i and e["id"]==row["id"] and hex64(e["receipt_sha256"]) for i,(e,row) in enumerate(zip(c,rows))),"ordered receipts")
    d=[x for x in z if x.get("status")=="R23G_FIT_PROPAGATION_DIAGNOSTIC_NOT_QUALITY"];need(len(d)==1 and hex64(d[0]["receipt_sha256"]),"complete report marker")
    return c,d[0]
def selftest():
    rows=[dict(id=d+str(i),direction=d) for d in DIRS for i in range(8 if d=="zh-my" else 2)]
    caps={r["id"]:{n:dict(logits=[dict(step=j,nmse=.2 if n=="original96" else .1,teacher_to_masked_kl=.2 if n=="original96" else .1,teacher_argmax=0,masked_argmax=0) for j in range(9)]) for n in NAMES} for r in rows}
    need(not summarize(rows,caps)[1],"positive")
    caps[rows[0]["id"]]["candidate96"]["logits"][0]["teacher_to_masked_kl"]=2.
    need("zh-my/prefill/kl" in summarize(rows,caps)[1],"KL negative")
    try:events(b"",rows)
    except ValueError:pass
    else:raise AssertionError("empty log")
    print(json.dumps(dict(status="PASS_SYNTHETIC_R23G_INDEPENDENT_GATE",positive=1,negative=2)))
def main():
    a=argparse.ArgumentParser();a.add_argument("--selftest",action="store_true");a.add_argument("--inputs",type=P);a.add_argument("--log-sha");a.add_argument("--replay-sha");a.add_argument("--out",type=P);q=a.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.out and hex64(q.log_sha) and hex64(q.replay_sha),"explicit terminal evidence");d=q.inputs
    c=pin(d/"candidate.json",CAND);s=pin(d/"sensitivity.json",SENS);need(c["status"]=="FROZEN_L38_REVERSION_FOR_FIT_PROPAGATION_ONLY" and c["mask_sha256"]==sha(canon(c["kept"]))==MASK and c["predicted_local"]["eligible_for_fit_propagation_only"] is True and s["candidate_sha256"]==CAND and s["full"]["pass_local_algebra"] is True,"local prerequisite")
    ep=pin(d/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True);pp=pin(d/"prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True)
    rows=pp["rows"]+[r for r in ep["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in pp["rows"]}]
    need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in ep["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}),"fit only")
    ad=dict(status="R23G_FIT_PROPAGATION_ADMITTED",teacher=TEACHER,engine_head=ENGINE,mask_sha256=MASK,candidate_sha256=CAND,prior_replay_sha256=PRIOR,ids=[r["id"] for r in rows],comment=7218921977,head="37bca49d6202a1b33e2df5548ef2b85657913227",threads=8)
    need(sha(canon(ad))==AD,"admission identity")
    root=d/"fit";need(sorted(x.name for x in root.iterdir())==sorted(["admission.json","report.json"]+[f"{i:02d}.json" for i in range(18)]),"inventory")
    need(read(root/"admission.json")==canon(ad)+b"\n","admission bytes")
    log=read(d/"fit.log");need(sha(log)==q.log_sha and read(d/"fit.stderr")==b"","terminal log")
    ev,done=events(log,rows);rp=pin(d/"fit-replay.json",q.replay_sha)
    need(rp["status"]=="R23G_EXACT_REPLAY_NOT_QUALITY" and rp["samples"]==18 and rp["admission_sha256"]==AD and rp["receipt_sha256"]==[x["receipt_sha256"] for x in ev] and rp["report_sha256"]==done["receipt_sha256"],"replay identity")
    prior=pin(d/"prior-replay.json",PRIOR);need(prior["status"]=="NONUNIFORM_PROPAGATION_EXACT_REPLAY_NOT_QUALITY" and prior["samples"]==12,"prior")
    caps={}
    for i,r in enumerate(rows):
        x=pin(root/f"{i:02d}.json",ev[i]["receipt_sha256"]);check_capture(x,r,i)
        if i<12:
            old=pin(d/"prior"/f"{i:02d}.json",prior["receipt_sha256"][i]);need(old["id"]==r["id"] and old["original96"]==x["original96"],"prior exact baseline")
        caps[r["id"]]=x
    groups,viol=summarize(rows,caps);actual=pin(root/"report.json",done["receipt_sha256"])
    need(actual["status"]=="R23G_FIT_PROPAGATION_DIAGNOSTIC_NOT_QUALITY" and actual["samples"]==18 and actual["aligned_lines"]==16 and actual["groups"]==groups and actual["primary_diagnostic_violations"]==viol and actual["primary_diagnostic_pass"]==(not viol) and {k:v for k,v in done.items() if k!="receipt_sha256"}==actual,"independent replay")
    out=dict(status="R23G_FIT_PROPAGATION_REPLAY_PASS_FRESH_ELIGIBILITY_ONLY" if not viol else "R23G_FIT_PROPAGATION_REPLAY_REJECT_STOP_FRESH",mask_sha256=MASK,candidate_sha256=CAND,sensitivity_sha256=SENS,log_sha256=q.log_sha,replay_sha256=q.replay_sha,report_sha256=done["receipt_sha256"],samples=18,aligned_lines=16,primary_violations=viol,groups=groups,limits="Eligibility only; separate fresh admission required, no physical build, no translation quality. Secondary groups remain diagnostic at this fit stage.")
    raw=canon(out)+b"\n"
    with q.out.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=out["status"],sha256=sha(raw),primary_violations=viol)))
if __name__=="__main__":main()
