"""R23G bounded fixed-trajectory propagation of single L38 reversion.
18 fit samples only, 12 matched R23E and all 8 zh-my. No holdout, no generation,
no physical build or product admission. Output is immutable and replayable.
"""
import argparse,fcntl,hashlib,importlib.util,json,math,os,pathlib,sys,time
P=pathlib.Path;ROOT=P.home()/".cache/mt-pruning"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE="3212303e1e02994a5124db8c85195b91325c7a7c"
MASK="77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
CAND="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
BASE="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
REPLAY="89a5715647088b7560411f379f90d368348dc4a1453bb32acce9e3dac7410c9c"
PINS={"prop":"88468e8c2a5905628365da52e1575d5251516512","batch":"08abbbfbd930e532d41602dd29a5c70f2bc474e3","verifier":"e56b25b3fd92ccbd91455628606925a02af49450","driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46","checked":"87b12c30ebf801d2e4571b6f0f8348edbb61ba13"}
DIRS=("zh-my","my-zh","zh-en","en-zh","en-my","my-en")
NAMES=("original96","candidate96")
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
def avg(xs):return math.fsum(xs)/len(xs)
def report(rows,caps):
    need(len(rows)==18 and len({r["id"] for r in rows})==18 and set(caps)=={r["id"] for r in rows},"exact rows")
    need([sum(r["direction"]==d for r in rows) for d in DIRS]==[8,2,2,2,2,2],"direction inventory")
    out={}
    for d in DIRS:
        for p in ("prefill","decode"):
            g=d+"/"+p;out[g]={}
            for n in NAMES:
                a=[q for r in rows if r["direction"]==d for q in caps[r["id"]][n]["logits"] if (q["step"]==0)==(p=="prefill")]
                need(len(a)==(8 if d=="zh-my" else 2)*(1 if p=="prefill" else 8),"logit inventory")
                for q in a:need(all(math.isfinite(q[k]) and q[k]>=0 for k in ("nmse","teacher_to_masked_kl")),"finite")
                out[g][n]=dict(nmse=avg([q["nmse"] for q in a]),kl=avg([q["teacher_to_masked_kl"] for q in a]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in a),positions=len(a))
    violations=[g+"/"+k for g in out if g.startswith(("zh-my/","my-zh/")) for k in ("nmse","kl","argmax_mismatch") if out[g]["candidate96"][k]>out[g]["original96"][k]+1e-12]
    return dict(status="R23G_FIT_PROPAGATION_DIAGNOSTIC_NOT_QUALITY",samples=18,aligned_lines=len({r["line"] for r in rows}),groups=out,primary_diagnostic_violations=violations,primary_diagnostic_pass=not violations,limits="Actual fixed historical trajectory; primary diagnostic requires no NMSE/KL/argmax mismatch regression; secondary reported, no free-running translation, holdout or admission.")
def selftest():
    rows=[dict(id=d+str(i),direction=d,line=i) for d in DIRS for i in range(8 if d=="zh-my" else 2)]
    caps={r["id"]:{n:dict(logits=[dict(step=j,nmse=.2 if n=="original96" else .1,teacher_to_masked_kl=.2 if n=="original96" else .1,teacher_argmax=0,masked_argmax=0) for j in range(9)]) for n in NAMES} for r in rows}
    need(report(rows,caps)["primary_diagnostic_pass"],"positive")
    caps[rows[0]["id"]]["candidate96"]["logits"][0]["nmse"]=2.
    need("zh-my/prefill/nmse" in report(rows,caps)["primary_diagnostic_violations"],"regression")
    try:report(rows[:-1],caps)
    except ValueError:pass
    else:raise AssertionError("incomplete")
    print(json.dumps(dict(status="PASS_SYNTHETIC_R23G_REPORT",positive=1,negative=2)))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--inputs",type=P);p.add_argument("--comment",type=int);p.add_argument("--head");p.add_argument("--execute",action="store_true");p.add_argument("--audit",action="store_true");q=p.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.comment and q.head and len(q.head)==40 and not(q.execute and q.audit),"explicit inputs")
    d=q.inputs
    v=load(d,"verifier");b=load(d,"batch");check=load(d,"checked")
    expanded=v.strict(pin(d/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True))
    plan=v.strict(pin(d/"prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True))
    rows=plan["rows"]+[r for r in expanded["fit"] if r["direction"]=="zh-my" and r["id"] not in {z["id"] for z in plan["rows"]}]
    need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in expanded["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in expanded["holdout"]}),"fit only")
    base=v.strict(pin(d/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True))["candidates"]["96"]["kept"]
    cand=v.strict(pin(d/"candidate.json",CAND));mask=cand["kept"];need(cand["status"]=="FROZEN_L38_REVERSION_FOR_FIT_PROPAGATION_ONLY" and cand["predicted_local"]["eligible_for_fit_propagation_only"] is True and sha(canon(base))==BASE and sha(canon(mask))==MASK,"frozen mask")
    stable=v.strict(pin(ROOT/"r23-results/expanded-stable96-v1.json","488a27afd5e75019cb6cc7355b9f847bbca515fc7570da081020a23b3e6213ef"))["kept"]
    layers=list(range(1,48));need(set(base)==set(mask)==set(stable)==set(map(str,layers)) and all(len(a)==96 and a==sorted(set(a)) and all(type(i) is int and 0<=i<128 for i in a) for m in (base,mask) for a in m.values()) and mask["38"]==base["38"] and [l for l in mask if mask[l]!=stable[l]]==["38"],"single layer reversion")
    root=ROOT/"r23-results";prior=v.strict(pin(root/"nonuniform-propagation-v1-exact-replay.json",REPLAY))
    need(prior["status"]=="NONUNIFORM_PROPAGATION_EXACT_REPLAY_NOT_QUALITY" and prior["samples"]==12,"prior exact replay")
    witnesses={}
    for i,r in enumerate(rows[:12]):
        x=v.strict(pin(root/"nonuniform-propagation-v1"/f"{i:02d}.json",prior["receipt_sha256"][i]));need(x["id"]==r["id"],"shared row");witnesses[r["id"]]=x["original96"]
    ad=dict(status="R23G_FIT_PROPAGATION_ADMITTED",teacher=TEACHER,engine_head=ENGINE,mask_sha256=MASK,candidate_sha256=CAND,prior_replay_sha256=REPLAY,ids=[r["id"] for r in rows],comment=q.comment,head=q.head,threads=8)
    ah=sha(canon(ad));dst=root/"r23g-l38-propagation-v1"
    def validate(x,r,i):
        check.checked(x,r,layers);need(x["index"]==i and x["admission_sha256"]==ah and x["teacher"]==TEACHER and x["mask_sha256"]==MASK,"provenance")
        if r["id"] in witnesses:need(x["original96"]==witnesses[r["id"]],"shared baseline exact")
    if q.audit:
        need(sorted(z.name for z in dst.iterdir())==sorted(["admission.json","report.json"]+[f"{i:02d}.json" for i in range(18)]),"complete inventory")
        need(v.strict(read(dst/"admission.json"))==ad,"admission replay")
        caps={};hashes=[]
        for i,r in enumerate(rows):
            raw=read(dst/f"{i:02d}.json");x=v.strict(raw);validate(x,r,i);caps[r["id"]]=x;hashes.append(sha(raw))
        raw=read(dst/"report.json");need(v.strict(raw)==report(rows,caps),"report replay")
        print(json.dumps(dict(status="R23G_EXACT_REPLAY_NOT_QUALITY",samples=18,receipt_sha256=hashes,report_sha256=sha(raw),admission_sha256=ah)));return
    drv=load(d,"driver");c=load(d,"collector");prop=load(d,"prop");c.verify_code()
    corpus=pin(ROOT/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
    hist=pin(ROOT/"run-d01b-r19/baseline/records.jsonl","37d13f9a7f9d9f56e9e810b5c869d216d2c6baa6a32cccb489818bd8ae7656e8")
    ids={r["id"] for r in rows};ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};old={x["id"]:x for x in map(json.loads,hist.splitlines()[1:]) if x["split"]=="calibration" and x["id"] in ids}
    need(set(ctx)==set(old)==ids,"input inventory")
    for r in rows:
        x=ctx[r["id"]];z=old[r["id"]];need(x["line"]==z["line"]==r["line"] and x["direction"]==z["direction"]==r["direction"] and len(x["prompt_ids"])==z["prompt_tokens"]==r["prompt_tokens"] and sha(canon(x["prompt_ids"]))==r["prompt_ids_sha256"] and z["finish"]=="eos" and z["tokens"][:8]==r["teacher_prefix_tokens"] and sha(canon(r["teacher_prefix_tokens"]))==r["teacher_prefix_sha256"],"input binding")
    pre=drv.preflight();print(json.dumps(dict(pre,admission_sha256=ah,samples=18,holdout_capture="NOT_ALLOWED")),flush=True)
    if not q.execute:return
    with (ROOT/"r23-admissions/batch.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);need(not pre["blockers"] and not drv.preflight()["blockers"] and not dst.exists(),"exclusive resources")
        dst.mkdir(mode=0o700);b.publish(dst/"admission.json",ad)
        torch=c.torch;torch.set_num_threads(8);torch.set_num_interop_threads(1)
        ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==TEACHER and ident["num_experts"]==128 and not drv.preflight()["blockers"],"teacher")
        e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8)
        masks={"original96":{int(l):a for l,a in base.items()},"candidate96":{int(l):a for l,a in mask.items()}};caps={}
        print(json.dumps(dict(status="TEACHER_READY",teacher=TEACHER,admission_sha256=ah)),flush=True)
        for i,r in enumerate(rows):
            need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
            t=time.monotonic();x=prop.run(c,e,ctx[r["id"]]["prompt_ids"],r["teacher_prefix_tokens"],masks)
            x.update(id=r["id"],line=r["line"],direction=r["direction"],consumed_prefix=r["teacher_prefix_tokens"],index=i,teacher=TEACHER,mask_sha256=MASK,admission_sha256=ah)
            validate(x,r,i);caps[r["id"]]=x;h=b.publish(dst/f"{i:02d}.json",x)
            print(json.dumps(dict(status="R23G_CAPTURED",index=i,id=r["id"],receipt_sha256=h,seconds=time.monotonic()-t)),flush=True)
        out=report(rows,caps);h=b.publish(dst/"report.json",out);print(json.dumps(dict(out,receipt_sha256=h)),flush=True)
if __name__=="__main__":main()
