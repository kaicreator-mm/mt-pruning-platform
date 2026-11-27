"""R23H independent cross-host audit for four Pareto primary-fit candidates.
No model execution, Fresh, holdout, or mask admission."""
import hashlib,io,json,math,pathlib,sys,tarfile
P=pathlib.Path
ARCHIVE_SHA="eef19f39837236b569aeaf6f789ff6cfe25674b2acd60dd38a1f507b128dce57"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
SCREEN="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
ALL10_REPORT="63a53755c176d7d6514f8a18d4908d9cadec622a9fb3492b2cc91cdbbea814fa"
ALL10_ADMISSION="bfd5267613c493558d3e0deece69aa4472cce53ea6aea5ad859b5754117ded2f"
BATCH_ADMISSION="e0b42e1f166915e14d2ec9a76ce31bde1e5c4f0965352dfc3c35b85ec027cdec"
BATCH_REPORT="3c0a1bb8759cd7942038562919e8ebea6e0d5a24e646d7d38a71873674885404"
NAMES=("all12","all14","primary4_10","primary4_14")
DIRS=("zh-my","my-zh")
def need(x,m):
    if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def strict(b):
    def pairs(xs):
        d={}
        for k,v in xs:need(k not in d,"duplicate key");d[k]=v
        return d
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def finite(x):return type(x) in (int,float) and math.isfinite(x) and x>=0
def avg(xs):return math.fsum(xs)/len(xs)
def read_tar(raw):
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as t:
        ms=t.getmembers();names={m.name for m in ms}
        need(all(m.isfile() and 0<=m.size<=2*1024*1024 for m in ms),"tar member types/bounds")
        need(len(names)==len(ms),"duplicate archive path")
        return {m.name:t.extractfile(m).read() for m in ms}
def parse(b):return strict(b)
def check_capture(x,r,i,ad,mask):
    need(x["status"]=="PROPAGATED_FIXED_TRAJECTORY_DIAGNOSTIC_NOT_QUALITY" and x["teacher_restored"] is True and x["index"]==i and x["id"]==r["id"] and x["direction"]==r["direction"] and x["line"]==r["line"] and x["consumed_prefix"]==r["teacher_prefix_tokens"] and x["teacher"]==TEACHER and x["mask_sha256"]==mask and x["admission_sha256"]==ad,"capture provenance")
    want={(l,"prefill" if j==0 else "decode",j) for l in range(1,48) for j in range(9)}
    for n in ("original96","candidate96"):
        a=x[n];ls=a["layer"]
        need(len(ls)==len(want) and {(q["layer"],q["phase"],q["step"]) for q in ls}==want,"layer coverage")
        for q in ls:
            need(all(finite(q[k]) for k in ("hidden_input_nmse","moe_output_nmse","mean_overlap")) and q["mean_overlap"]<=8 and q["tokens"]==(2 if q["step"]==0 else 1) and type(q["exact_sets"]) is int and 0<=q["exact_sets"]<=q["tokens"],"layer values")
        for key in ("teacher_hashes","masked_hashes"):
            z=a[key];need(len(z)==9 and all(q["step"]==j and len(q["sha256"])==64 and type(q["next"]) is int and q["next"]>=0 for j,q in enumerate(z)),"token hash")
        need(len(a["logits"])==9 and all(q["step"]==j and finite(q["nmse"]) and finite(q["teacher_to_masked_kl"]) and q["teacher_argmax"]==a["teacher_hashes"][j]["next"] and q["masked_argmax"]==a["masked_hashes"][j]["next"] for j,q in enumerate(a["logits"])),"logits")
    need(x["original96"]["teacher_hashes"]==x["candidate96"]["teacher_hashes"],"teacher identity")
def summarize(rows,caps):
    out={}
    for d in DIRS:
        for phase in ("prefill","decode"):
            g=d+"/"+phase;out[g]={}
            for n in ("original96","candidate96"):
                z=[q for r in rows if r["direction"]==d for q in caps[r["id"]][n]["logits"] if (q["step"]==0)==(phase=="prefill")]
                need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"position coverage")
                out[g][n]=dict(kl=avg([q["teacher_to_masked_kl"] for q in z]),nmse=avg([q["nmse"] for q in z]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in z),positions=len(z))
    viol={n:[g+"/"+k for g in out for k in ("kl","nmse","argmax_mismatch") if out[g]["candidate96"][k]>out[g]["original96"][k]+1e-12] for n in []}
    return out
def main():
    root=P(sys.argv[1]);raw=(root/"remaining4-primary.tgz").read_bytes();need(sha(raw)==ARCHIVE_SHA,"archive SHA")
    b=read_tar(raw)
    expected={"batch/admission.json","batch/report.json","batch/run.log","batch/run.stderr","batch/exit-code",
      *("batch/"+n+"-"+f"{i:02d}"+".json" for n in NAMES for i in (0,1,2,3,12,13,14,15,16,17)),
      "prior-all10/admission.json","prior-all10/report.json","prior-all10/run.log","prior-all10/run.stderr","prior-all10/exit-code",
      *("prior-all10/"+f"{i:02d}.json" for i in range(18)),
      "screen/plan.json","screen/report.json","screen/run.log","screen/run.stderr","screen/exit-code","screen/original.json",
      *("screen/"+f"{i:02d}.json" for i in range(8)),
      "recovery/recovery-receipt.json","recovery/expanded.json","recovery/prop-plan.json",
      *("prior-r23g/"+f"{i:02d}.json" for i in range(18))}
    need(set(b)==expected,"exact archive inventory")
    need(b["batch/exit-code"].strip()==b"0" and b["batch/run.stderr"]==b"" and b["prior-all10/exit-code"].strip()==b"0" and b["prior-all10/run.stderr"]==b"","terminal exits")
    def exact(k,h,git=False):
        x=b[k];need((blob if git else sha)(x)==h,"pin "+k);return parse(x)
    ep=exact("recovery/expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True)
    pp=exact("recovery/prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True)
    rec=exact("recovery/recovery-receipt.json",REC);need(rec["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and len(rec["receipt_sha256"])==18,"recovery")
    rows=pp["rows"]+[r for r in ep["fit"] if r["direction"]=="zh-my" and r["id"] not in {x["id"] for x in pp["rows"]}]
    need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in ep["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}),"fit-only rows")
    selected=[(i,r) for i,r in enumerate(rows) if r["direction"] in DIRS];need(len(selected)==10,"primary row inventory")
    plan=exact("screen/plan.json",PLAN);screen=exact("screen/report.json",SCREEN)
    need(plan["status"]=="R23H_PARETO_FIT_PREFILL_SCREEN_ONLY" and plan["holdout"]=="FORBIDDEN" and len(plan["screen"])==6 and screen["selected"]=="all8","frozen screen")
    base=exact("screen/original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True)["candidates"]["96"]["kept"]
    masksha={x["name"]:x["mask_sha256"] for x in plan["screen"]}
    for c in plan["screen"]:
        need(sha(canon(c["kept"]))==c["mask_sha256"] and len(c["kept"])==47 and all(len(v)==96 and v==sorted(set(v)) for v in c["kept"].values()),"screen mask")
    # Audit the completed all10 predecessor and exact baseline.
    pa=exact("prior-all10/admission.json",ALL10_ADMISSION)
    pr=exact("prior-all10/report.json",ALL10_REPORT)
    need(pa["selected"]=="all10" and pa["mask_sha256"]==masksha["all10"] and pr["status"]=="R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION" and pr["samples"]==18 and pr["selected"]=="all10" and pr["primary_diagnostic_pass"] is False,"all10 predecessor")
    pe=[parse(x) for x in b["prior-all10/run.log"].splitlines() if x.startswith(b"{")]
    pc=[x for x in pe if x["status"]=="R23H_PROPAGATION_CAPTURED"]
    pend=[x for x in pe if x["status"]=="R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(pc)==18 and len(pend)==1 and pend[0]["report_sha256"]==ALL10_REPORT,"all10 events")
    prior={}
    for i,(e,r) in enumerate(zip(pc,rows)):
        x=exact("prior-all10/"+f"{i:02d}.json",e["receipt_sha256"])
        need(e["index"]==i and e["id"]==r["id"] and x["id"]==r["id"] and x["mask_sha256"]==masksha["all10"] and x["admission_sha256"]==ALL10_ADMISSION,"all10 receipt")
        prior[r["id"]]=x
    # Confirm prior report aggregates from all10 receipts.
    ag={}
    for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
        for phase in ("prefill","decode"):
            g=d+"/"+phase;ag[g]={}
            for n in ("original96","candidate96"):
                z=[q for x in prior.values() if x["direction"]==d for q in x[n]["logits"] if (q["step"]==0)==(phase=="prefill")]
                need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"all10 coverage")
                ag[g][n]=dict(nmse=avg([q["nmse"] for q in z]),kl=avg([q["teacher_to_masked_kl"] for q in z]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in z),positions=len(z))
    need(ag==pr["groups"],"all10 aggregate replay")
    # Audit batch admission, events, every candidate receipt and report.
    ad=exact("batch/admission.json",BATCH_ADMISSION)
    need(ad["status"]=="R23H_REMAINING4_PRIMARY_BATCH_FIT_ONLY" and ad["prior_all10_report_sha256"]==ALL10_REPORT and ad["teacher"]==TEACHER and ad["holdout"]=="FORBIDDEN" and ad["indices"]==[i for i,r in selected] and ad["ids"]==[r["id"] for i,r in selected],"batch admission")
    need(ad["candidates"]=={n:masksha[n] for n in NAMES},"candidate masks")
    ev=[parse(x) for x in b["batch/run.log"].splitlines() if x.startswith(b"{")]
    ready=[x for x in ev if x["status"]=="R23H_PROPAGATION_TEACHER_READY"]
    captures=[x for x in ev if x["status"]=="R23H_REMAINING4_PRIMARY_CAPTURED"]
    done=[x for x in ev if x["status"]=="R23H_REMAINING4_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(ready)==1 and ready[0]["teacher"]==TEACHER and len(captures)==10 and len(done)==1 and done[0]["report_sha256"]==BATCH_REPORT,"batch events")
    screen_caps={plan["ids"][i]:exact("screen/"+f"{i:02d}.json",None) for i in range(8)}
    caps={n:{} for n in NAMES};layer_rows={n:[] for n in NAMES}
    for j,(idx,r) in enumerate(selected):
        e=captures[j];need(e["index"]==idx and e["id"]==r["id"],"batch order")
        for n in NAMES:
            h=e["receipt_sha256"][n];x=exact("batch/"+n+"-"+f"{idx:02d}"+".json",h)
            check_capture(x,r,idx,sha(b["batch/admission.json"]),masksha[n])
            need(x["original96"]==prior[r["id"]]["original96"],"exact original96 baseline")
            if r["id"] in screen_caps:
                z=screen_caps[r["id"]]["metrics"][n];v=x["candidate96"]["logits"][0]
                need(z["mask_sha256"]==masksha[n] and z["logits_sha256"]==x["candidate96"]["masked_hashes"][0]["sha256"] and abs(z["kl"]-v["teacher_to_masked_kl"])<1e-12 and abs(z["nmse"]-v["nmse"])<1e-12 and z["argmax_mismatch"]==int(v["teacher_argmax"]!=v["masked_argmax"]),"prefill screen replay")
            caps[n][r["id"]]=x
    reports={};all_viol={}
    for n in NAMES:
        groups={}
        for d in DIRS:
            for phase in ("prefill","decode"):
                g=d+"/"+phase;groups[g]={}
                for k in ("original96","candidate96"):
                    z=[q for r in rows if r["direction"]==d for q in caps[n][r["id"]][k]["logits"] if (q["step"]==0)==(phase=="prefill")]
                    need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"candidate coverage")
                    groups[g][k]=dict(kl=avg([q["teacher_to_masked_kl"] for q in z]),nmse=avg([q["nmse"] for q in z]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in z),positions=len(z))
        viol=[g+"/"+k for g in groups for k in ("kl","nmse","argmax_mismatch") if groups[g]["candidate96"][k]>groups[g]["original96"][k]+1e-12]
        reports[n]=groups;all_viol[n]=viol
    actual=exact("batch/report.json",BATCH_REPORT)
    need(actual["groups"]==reports and actual["violations"]==all_viol and actual["eligible_for_secondary_fit_only"]==[] and actual["selected"] is None and actual["samples_per_candidate"]==10,"batch report replay")
    need({k:v for k,v in done[0].items() if k not in ("status","report_sha256")}=={k:v for k,v in actual.items() if k!="status"},"terminal binding")
    # Layer-local attribution for exploratory mask recombination only.
    diagnostics={}
    for n in NAMES:
        changed=[]
        cm=next(x["kept"] for x in plan["screen"] if x["name"]==n)
        for l in range(1,48):
            old=set(base[str(l)]);new=set(cm[str(l)])
            if old!=new:changed.append(dict(layer=l,added=sorted(new-old),removed=sorted(old-new)))
        deltas={}
        for d in DIRS:
            vals={}
            for l in range(1,48):
                for phase in ("prefill","decode"):
                    dif=[]
                    for r in rows:
                        if r["direction"]!=d:continue
                        x=caps[n][r["id"]]
                        a={(q["layer"],q["phase"],q["step"]):q for q in x["original96"]["layer"]}
                        c={(q["layer"],q["phase"],q["step"]):q for q in x["candidate96"]["layer"]}
                        steps=[0] if phase=="prefill" else list(range(1,9))
                        dif.extend(c[(l,phase,j)]["moe_output_nmse"]-a[(l,phase,j)]["moe_output_nmse"] for j in steps)
                    vals[f"{l}/{phase}"]=avg(dif)
            deltas[d]=vals
        diagnostics[n]=dict(changed_layers=changed,layer_moe_output_nmse_delta=deltas)
    out=dict(status="R23H_REMAINING4_INDEPENDENT_RECEIPT_REPLAY_PASS_ALL4_REJECT_FRESH",archive_sha256=ARCHIVE_SHA,all10_report_sha256=ALL10_REPORT,batch_admission_sha256=BATCH_ADMISSION,batch_report_sha256=BATCH_REPORT,samples_per_candidate=10,violations=all_viol,groups=reports,layer_diagnostics=diagnostics,limits="Fixed fit trajectories only; 8 zh-my and 2 my-zh primary rows. No fresh, holdout, native translation, or physical build. Layer attribution is correlational and only supports exploratory mask recombination.")
    raw=canon(out)+b"\n";dst=root/"remaining4-primary-replay.json"
    with dst.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=out["status"],sha256=sha(raw),violations=all_viol,changed_layers={n:[x["layer"] for x in diagnostics[n]["changed_layers"]] for n in NAMES})))
if __name__=="__main__":main()
