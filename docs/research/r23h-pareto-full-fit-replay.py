"""Cross-host full fit receipt gate for R23H Pareto candidates. No model or Fresh execution."""
import hashlib,json,math,pathlib,sys,tarfile
P=pathlib.Path
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
SCREEN="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
MASKS={"all8":"1f7402ae0fb540c86ace21914b45ea0c42a113e5f38072a2d3f86795ee8ad70b","all10":"a9274a240cb1e990c77b110b9a6231e0ae1c905849a2b7e9295f795bd3a81fe2"}
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
def check_capture(x,r,i,ad,mask):
    need(x["status"]=="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY" and x["teacher_restored"] is True and x["index"]==i and x["id"]==r["id"] and x["direction"]==r["direction"] and x["line"]==r["line"] and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["teacher"]==TEACHER and x["mask_sha256"]==mask and x["admission_sha256"]==ad,"capture provenance")
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

def main():
    d=P(sys.argv[1]);name=sys.argv[2];h=sys.argv[3]
    need(name in MASKS and hex64(h),"arguments")
    archive=d/(name+"-full-fit.tgz")
    raw=read(archive);need(sha(raw)==h,"transport hash")
    need(len(raw)<32*1024*1024,"archive bound")
    F="fit/";O="prior/";S="screen/"
    expected={F+"admission.json",F+"report.json",*(F+f"{i:02d}.json" for i in range(18)),*(O+f"{i:02d}.json" for i in range(18)),*(S+f"{i:02d}.json" for i in range(8)),S+"plan.json",S+"report.json","recovery.json","expanded.json","prop-plan.json","run.log","run.stderr","exit-code"}
    import io
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as t:
        members=t.getmembers();need(len(members)==len(expected) and {m.name for m in members}==expected and all(m.isfile() and 0<=m.size<=2*1024*1024 for m in members),"inventory")
        b={m.name:t.extractfile(m).read() for m in members}
    def exact(k,h,git=False):
        need((blob if git else sha)(b[k])==h,"exact "+k)
        return strict(b[k])
    ep=exact("expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True)
    pp=exact("prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True)
    rows=pp["rows"]+[r for r in ep["fit"] if r["direction"]=="zh-my" and r["id"] not in {x["id"] for x in pp["rows"]}]
    need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in ep["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}),"fit scope")
    p=exact(S+"plan.json",PLAN);q=exact(S+"report.json",SCREEN);prior=exact("recovery.json",REC)
    need(prior["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and len(prior["receipt_sha256"])==18,"prior")
    c=next(x for x in p["screen"] if x["name"]==name)
    need(c["mask_sha256"]==sha(canon(c["kept"]))==MASKS[name] and name in q["eligible_for_full_fit_propagation_only"] and c["folds_passed"]==30,"frozen mask")
    ad=strict(b[F+"admission.json"]);ah=sha(b[F+"admission.json"])
    need(ad["selected"]==name and ad["mask_sha256"]==MASKS[name] and ad["teacher"]==TEACHER and ad["ids"]==[r["id"] for r in rows] and ad["screen_plan_sha256"]==PLAN and ad["screen_report_sha256"]==SCREEN and ad["previous_recovery_sha256"]==REC and ad["holdout"]=="FORBIDDEN","admission")
    if name=="all8":need(ah=="f381a42350922ef14ca89257aa0349ce1d6f8c8887b43988dccb33321f1a70ce","all8 admission pin")
    need(b["exit-code"].strip()==b"0" and b["run.stderr"]==b"","terminal")
    ev=[strict(l) for l in b["run.log"].splitlines() if l.startswith(b"{")]
    captures=[x for x in ev if x["status"]=="R23H_PROPAGATION_CAPTURED"]
    terminal=[x for x in ev if x["status"]==("R23H_WINNER_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION" if name=="all8" else "R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION")]
    need(len(captures)==18 and len(terminal)==1,"terminal inventory")
    old=read(d/"pareto-actual-v1.tgz");need(sha(old)=="d6d83db734d1f4eda6eefa14c8861a551d739c91e5ae0511b1f2dc5de5b6be71","prior audited archive")
    with tarfile.open(fileobj=io.BytesIO(old),mode="r:gz") as t:
        for i in range(8):
            k=f"{i:02d}.json";m=t.getmember("r23-results/r23h-pareto-prefill-v1/"+k)
            need(m.isfile() and t.extractfile(m).read()==b[S+k],"audited prefill exact bytes")
    sc={p["ids"][i]:strict(b[S+f"{i:02d}.json"]) for i in range(8)}
    caps={}
    for i,r in enumerate(rows):
        e=captures[i];need(e["index"]==i and e["id"]==r["id"],"event order")
        x=exact(F+f"{i:02d}.json",e["receipt_sha256"]);check_capture(x,r,i,ah,MASKS[name])
        o=exact(O+f"{i:02d}.json",prior["receipt_sha256"][i]);need(o["id"]==r["id"] and o["original96"]==x["original96"],"exact baseline")
        if r["id"] in sc:
            z=sc[r["id"]];v=x["candidate96"]["logits"][0];m=z["metrics"][name]
            need(z["id"]==r["id"] and z["plan_sha256"]==PLAN and m["mask_sha256"]==MASKS[name] and m["logits_sha256"]==x["candidate96"]["masked_hashes"][0]["sha256"] and abs(m["kl"]-v["teacher_to_masked_kl"])<1e-12 and abs(m["nmse"]-v["nmse"])<1e-12 and m["argmax_mismatch"]==int(v["teacher_argmax"]!=v["masked_argmax"]),"prefill replay")
        caps[r["id"]]=x
    groups,viol=summarize(rows,caps)
    actual=exact(F+"report.json",terminal[0]["report_sha256"])
    need(actual["groups"]==groups and set(actual["primary_violations"])==set(viol) and actual["primary_diagnostic_pass"]==(not viol) and actual["samples"]==18 and actual["selected"]==name and actual["admission_sha256"]==ah and actual["mask_sha256"]==MASKS[name],"aggregate replay")
    need({k:v for k,v in terminal[0].items() if k!="report_sha256"}==actual,"terminal report binding")
    out=dict(status="R23H_PARETO_FULL_FIT_REPLAY_PASS_FRESH_ELIGIBILITY_ONLY" if not viol else "R23H_PARETO_FULL_FIT_REPLAY_REJECT_STOP_FRESH",candidate=name,archive_sha256=h,admission_sha256=ah,report_sha256=terminal[0]["report_sha256"],mask_sha256=MASKS[name],receipt_sha256=[x["receipt_sha256"] for x in captures],samples=18,groups=groups,primary_violations=viol,limits="Fixed fit trajectories only; separate fresh admission required. No product qualification.")
    rb=canon(out)+b"\n"
    with (d/(name+"-full-fit-replay.json")).open("xb") as f:f.write(rb)
    print(json.dumps(dict(status=out["status"],candidate=name,sha256=sha(rb),primary_violations=viol,groups=groups)))
if __name__=="__main__":main()
