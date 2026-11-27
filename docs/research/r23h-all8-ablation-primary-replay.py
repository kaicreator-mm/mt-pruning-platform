"""Independent receipt replay for R23H all8-ablation primary fit batch."""
import hashlib,io,json,math,pathlib,sys,tarfile
P=pathlib.Path
ARCHIVE_SHA="PENDING"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
PLAN="29e553c04e2cc61cbe999aac0604e9440f248480cadb6c8631f4758d87712acc"
SCREEN_REPORT="f35caa64d979ff6c781239a6cab25061af04235df77a282d5c6585a1704de521"
OLD_PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
ALL10_REPORT="63a53755c176d7d6514f8a18d4908d9cadec622a9fb3492b2cc91cdbbea814fa"
ALL10_ADMISSION="bfd5267613c493558d3e0deece69aa4472cce53ea6aea5ad859b5754117ded2f"
BATCH_ADMISSION="PENDING"
BATCH_REPORT="PENDING"
NAMES=("all8_revert42","all8_revert40_42")
MASKS={"all8_revert40":"02de6ad74290659e3bd9b301ef606c45c129e032abcbbaaf867fe61bb26af84e","all8_revert42":"054614572884a518ce2a6538a0f39c7884667d1551312216df767ee4720f1f0b","all8_revert40_42":"96d26e3699ee87f2560fbbf15a53b6016b7d96aadc744f5aade4b076a90dab2b"}
BLOBS={"all8_revert40":"255c95382a5888f9749d7626395182ba514c3c8d","all8_revert42":"f738ea900952ab19dbf6fd4cbca235dc40e56d05","all8_revert40_42":"97809cb5092ee6b903a4def8ee0139902d24f17d"}
SCREEN_MASKS=("all8_revert40","all8_revert42","all8_revert40_42")
SCREEN_NAMES=("original96","r23g","all8","all10",*SCREEN_MASKS)
INDICES=(0,1,2,3,12,13,14,15,16,17)
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
def main():
    root=P(sys.argv[1]);raw=(root/"all8-ablation-primary.tgz").read_bytes();need(sha(raw)==ARCHIVE_SHA,"archive SHA")
    b=read_tar(raw)
    expected={"batch/admission.json","batch/report.json","batch/run.log","batch/run.stderr","batch/exit-code",
      *("batch/"+n+"-"+f"{i:02d}"+".json" for n in NAMES for i in INDICES),
      "prior-all10/admission.json","prior-all10/report.json","prior-all10/run.log","prior-all10/run.stderr","prior-all10/exit-code",
      *("prior-all10/"+f"{i:02d}.json" for i in range(18)),
      "screen/plan.json","screen/report.json","screen/run.log","screen/run.stderr","screen/exit-code",
      *("screen/"+f"{i:02d}.json" for i in range(8)),*("screen/masks/"+n+".json" for n in SCREEN_MASKS),
      "old-screen/plan.json","old-screen/report.json","old-screen/run.log","old-screen/run.stderr","old-screen/exit-code","old-screen/original.json",
      *("old-screen/"+f"{i:02d}.json" for i in range(8)),
      "recovery/recovery-receipt.json","recovery/expanded.json","recovery/prop-plan.json",
      *("prior-r23g/"+f"{i:02d}.json" for i in range(18))}
    need(set(b)==expected,"exact archive inventory")
    need(b["batch/exit-code"].strip()==b"0" and b["batch/run.stderr"]==b"" and b["prior-all10/exit-code"].strip()==b"0" and b["prior-all10/run.stderr"]==b"" and b["screen/exit-code"].strip()==b"0" and b["screen/run.stderr"]==b"" and b["old-screen/exit-code"].strip()==b"0" and b["old-screen/run.stderr"]==b"","terminal exits")
    def exact(k,h,git=False):
        x=b[k];need((blob if git else sha)(x)==h,"pin "+k);return strict(x)
    ep=exact("recovery/expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True)
    pp=exact("recovery/prop-plan.json","cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde",True)
    rec=exact("recovery/recovery-receipt.json",REC);need(rec["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH" and len(rec["receipt_sha256"])==18,"recovery")
    rows=pp["rows"]+[r for r in ep["fit"] if r["direction"]=="zh-my" and r["id"] not in {x["id"] for x in pp["rows"]}]
    need(len(rows)==18 and len({r["line"] for r in rows})==16 and all(r in ep["fit"] for r in rows) and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}),"fit-only rows")
    selected=[(i,r) for i,r in enumerate(rows) if r["direction"] in DIRS]
    need([i for i,r in selected]==list(INDICES) and len(selected)==10,"primary row inventory")
    # Reconstruct and verify the failed all10 predecessor.
    pa=exact("prior-all10/admission.json",ALL10_ADMISSION);pr=exact("prior-all10/report.json",ALL10_REPORT)
    need(pa["selected"]=="all10" and pa["mask_sha256"]=="a9274a240cb1e990c77b110b9a6231e0ae1c905849a2b7e9295f795bd3a81fe2" and pr["status"]=="R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION" and pr["samples"]==18 and pr["selected"]=="all10" and pr["primary_diagnostic_pass"] is False,"all10 predecessor")
    pe=[strict(x) for x in b["prior-all10/run.log"].splitlines() if x.startswith(b"{")]
    pc=[x for x in pe if x["status"]=="R23H_PROPAGATION_CAPTURED"];pend=[x for x in pe if x["status"]=="R23H_ALL10_FIT_PROPAGATION_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(pc)==18 and len(pend)==1 and pend[0]["report_sha256"]==ALL10_REPORT,"all10 events")
    prior={}
    for i,(e,r) in enumerate(zip(pc,rows)):
        x=exact("prior-all10/"+f"{i:02d}.json",e["receipt_sha256"])
        need(e["index"]==i and e["id"]==r["id"] and x["id"]==r["id"] and x["mask_sha256"]==pa["mask_sha256"] and x["admission_sha256"]==sha(b["prior-all10/admission.json"]),"all10 receipt")
        prior[r["id"]]=x
    ag={}
    for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en"):
        for phase in ("prefill","decode"):
            g=d+"/"+phase;ag[g]={}
            for n in ("original96","candidate96"):
                z=[q for x in prior.values() if x["direction"]==d for q in x[n]["logits"] if (q["step"]==0)==(phase=="prefill")]
                need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"all10 coverage")
                ag[g][n]=dict(kl=avg([q["teacher_to_masked_kl"] for q in z]),nmse=avg([q["nmse"] for q in z]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in z),positions=len(z))
    need(ag==pr["groups"],"all10 aggregate replay")
    # Audit original Pareto screen and new all8-ablation prefill screen.
    oldplan=exact("old-screen/plan.json",OLD_PLAN);oldreport=exact("old-screen/report.json",OLD_REPORT)
    need(oldplan["status"]=="R23H_PARETO_FIT_PREFILL_SCREEN_ONLY" and oldplan["holdout"]=="FORBIDDEN" and oldreport["selected"]=="all8" and oldreport["samples"]==8,"old screen")
    oldev=[strict(x) for x in b["old-screen/run.log"].splitlines() if x.startswith(b"{")]
    oldcaps=[x for x in oldev if x["status"]=="R23H_PREFILL_CAPTURED"];need(len(oldcaps)==8,"old screen events")
    oldscreen={}
    for i,e in enumerate(oldcaps):
        x=exact("old-screen/"+f"{i:02d}.json",e["sha256"])
        need(e["index"]==i and x["index"]==i and x["id"]==oldplan["ids"][i] and e["id"]==x["id"] and x["plan_sha256"]==OLD_PLAN,"old screen receipt")
        oldscreen[x["id"]]=x
    base=exact("old-screen/original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True)["candidates"]["96"]["kept"]
    newplan=exact("screen/plan.json",PLAN);need(newplan["status"]=="R23H_ALL8_ABLATION_PREFILL_SCREEN_ONLY" and newplan["recovery_sha256"]==REC and newplan["teacher"]==TEACHER and newplan["holdout"]=="FORBIDDEN" and newplan["ids"]==oldplan["ids"],"new screen plan")
    masks={};masksha={}
    for n in SCREEN_MASKS:
        rawmask=b["screen/masks/"+n+".json"];need(blob(rawmask)==BLOBS[n],"mask blob "+n)
        z=strict(rawmask);m=z["mask"];need(z["status"]=="R23H_ALL8_SINGLE_LAYER_ABLATION_EXPLORATORY_NOT_ADMITTED" and z["method"]==n and z["mask_sha256"]==MASKS[n] and sha(canon(m))==MASKS[n],"mask identity "+n)
        need(newplan["hybrid_sources"][n]["blob_sha"]==BLOBS[n] and newplan["hybrid_sources"][n]["mask_sha256"]==MASKS[n] and newplan["masks"][n]==MASKS[n],"plan mask binding "+n)
        need(set(m)==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) for v in m.values()),"mask shape "+n)
        masks[n]=m;masksha[n]=MASKS[n]
    se=[strict(x) for x in b["screen/run.log"].splitlines() if x.startswith(b"{")]
    sce=[x for x in se if x["status"]=="R23H_LAYER_HYBRID_PREFILL_CAPTURED"]
    send=[x for x in se if x["status"]=="R23H_ALL8_ABLATION_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(sce)==8 and len(send)==1,"new screen events")
    scaps={}
    screen_names=list(SCREEN_NAMES)
    for i,e in enumerate(sce):
        x=exact("screen/"+f"{i:02d}.json",e["receipt_sha256"])
        need(e["index"]==i and x["index"]==i and e["id"]==x["id"]==newplan["ids"][i] and x["plan_sha256"]==PLAN and set(x["metrics"])==set(screen_names),"new screen capture")
        for n,q in x["metrics"].items():
            need(finite(q["kl"]) and finite(q["nmse"]) and q["argmax_mismatch"] in (0,1) and len(q["logits_sha256"])==64 and q["mask_sha256"]==newplan["masks"][n],"screen metric")
        for n in ("all8","all10"):
            need(x["metrics"][n]==oldscreen[x["id"]]["metrics"][n],"screen control replay")
        scaps[x["id"]]=x
    sm={n:{k:(sum if k=="argmax_mismatch" else avg)([x["metrics"][n][k] for x in scaps.values()]) for k in ("kl","nmse","argmax_mismatch")} for n in screen_names}
    eligible=[n for n in SCREEN_MASKS if all(sm[n][k]<=sm["original96"][k]+1e-12 for k in sm[n]) and sm[n]["kl"]<sm["r23g"]["kl"]]
    eligible.sort(key=lambda n:(sm[n]["kl"],sm[n]["nmse"],n))
    sr=exact("screen/report.json",SCREEN_REPORT)
    need(sr["status"]=="R23H_ALL8_ABLATION_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION" and sr["plan_sha256"]==PLAN and sr["samples"]==8 and sr["metrics"]==sm and sr["eligible_for_primary_fit_only"]==eligible and sr["selected"]==(eligible[0] if eligible else None) and eligible==["all8_revert42","all8_revert40_42"],"new screen aggregate")
    need(send[0]["report_sha256"]==SCREEN_REPORT,"screen terminal report")
    # Audit primary batch admission and receipts.
    ad=exact("batch/admission.json",BATCH_ADMISSION)
    need(ad["status"]=="R23H_ALL8_ABLATION_PRIMARY_BATCH_FIT_ONLY" and ad["screen_plan_sha256"]==PLAN and ad["screen_report_sha256"]==SCREEN_REPORT and ad["prior_all10_report_sha256"]==ALL10_REPORT and ad["teacher"]==TEACHER and ad["holdout"]=="FORBIDDEN" and ad["indices"]==list(INDICES) and ad["ids"]==[r["id"] for i,r in selected],"batch admission")
    need(ad["candidates"]=={n:masksha[n] for n in NAMES},"candidate masks")
    ev=[strict(x) for x in b["batch/run.log"].splitlines() if x.startswith(b"{")]
    ready=[x for x in ev if x["status"]=="R23H_PROPAGATION_TEACHER_READY"]
    captures=[x for x in ev if x["status"]=="R23H_ALL8_ABLATION_PRIMARY_CAPTURED"]
    done=[x for x in ev if x["status"]=="R23H_ALL8_ABLATION_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION"]
    need(len(ready)==1 and ready[0]["teacher"]==TEACHER and len(captures)==10 and len(done)==1 and done[0]["report_sha256"]==BATCH_REPORT,"batch events")
    caps={n:{} for n in NAMES}
    for j,(idx,r) in enumerate(selected):
        e=captures[j];need(e["index"]==idx and e["id"]==r["id"],"batch order")
        for n in NAMES:
            x=exact("batch/"+n+"-"+f"{idx:02d}"+".json",e["receipt_sha256"][n])
            check_capture(x,r,idx,sha(b["batch/admission.json"]),masksha[n])
            need(x["original96"]==prior[r["id"]]["original96"],"exact original96 baseline")
            if r["id"] in scaps:
                z=scaps[r["id"]]["metrics"][n];v=x["candidate96"]["logits"][0]
                need(z["mask_sha256"]==masksha[n] and z["logits_sha256"]==x["candidate96"]["masked_hashes"][0]["sha256"] and abs(z["kl"]-v["teacher_to_masked_kl"])<1e-12 and abs(z["nmse"]-v["nmse"])<1e-12 and z["argmax_mismatch"]==int(v["teacher_argmax"]!=v["masked_argmax"]),"prefill screen replay")
            caps[n][r["id"]]=x
    reports={};violations={}
    for n in NAMES:
        groups={}
        for d in DIRS:
            for phase in ("prefill","decode"):
                g=d+"/"+phase;groups[g]={}
                for k in ("original96","candidate96"):
                    z=[q for i,r in selected if r["direction"]==d for q in caps[n][r["id"]][k]["logits"] if (q["step"]==0)==(phase=="prefill")]
                    need(len(z)==(8 if d=="zh-my" else 2)*(1 if phase=="prefill" else 8),"candidate coverage")
                    groups[g][k]=dict(kl=avg([q["teacher_to_masked_kl"] for q in z]),nmse=avg([q["nmse"] for q in z]),argmax_mismatch=sum(q["teacher_argmax"]!=q["masked_argmax"] for q in z),positions=len(z))
        reports[n]=groups
        violations[n]=[g+"/"+k for g in groups for k in ("kl","nmse","argmax_mismatch") if groups[g]["candidate96"][k]>groups[g]["original96"][k]+1e-12]
    eligible_batch=sorted((n for n in NAMES if not violations[n]),key=lambda n:(avg([reports[n][g]["candidate96"][k]/max(reports[n][g]["original96"][k],1e-12) for g in reports[n] for k in ("kl","nmse")]),n))
    actual=exact("batch/report.json",BATCH_REPORT)
    need(actual["status"]=="R23H_ALL8_ABLATION_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION" and actual["groups"]==reports and actual["violations"]==violations and actual["eligible_for_secondary_fit_only"]==eligible_batch and actual["selected"]==(eligible_batch[0] if eligible_batch else None) and actual["samples_per_candidate"]==10,"batch report replay")
    need({k:v for k,v in done[0].items() if k not in ("status","report_sha256")}=={k:v for k,v in actual.items() if k!="status"},"terminal binding")
    diagnostics={}
    for n in NAMES:
        changed=[]
        for l in range(1,48):
            old=set(base[str(l)]);new=set(masks[n][str(l)])
            if old!=new:changed.append(dict(layer=l,added=sorted(new-old),removed=sorted(old-new)))
        deltas={}
        for d in DIRS:
            vals={}
            for l in range(1,48):
                for phase in ("prefill","decode"):
                    dif=[]
                    for i,r in selected:
                        if r["direction"]!=d:continue
                        x=caps[n][r["id"]]
                        a={(q["layer"],q["phase"],q["step"]):q for q in x["original96"]["layer"]}
                        c={(q["layer"],q["phase"],q["step"]):q for q in x["candidate96"]["layer"]}
                        steps=[0] if phase=="prefill" else list(range(1,9))
                        dif.extend(c[(l,phase,j)]["moe_output_nmse"]-a[(l,phase,j)]["moe_output_nmse"] for j in steps)
                    vals[f"{l}/{phase}"]=avg(dif)
            deltas[d]=vals
        diagnostics[n]=dict(changed_layers=changed,layer_moe_output_nmse_delta=deltas)
    out=dict(status="R23H_ALL8_ABLATION_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS",archive_sha256=ARCHIVE_SHA,screen_report_sha256=SCREEN_REPORT,batch_admission_sha256=BATCH_ADMISSION,batch_report_sha256=BATCH_REPORT,samples_per_candidate=10,violations=violations,groups=reports,eligible_for_secondary_fit_only=eligible_batch,selected=eligible_batch[0] if eligible_batch else None,layer_diagnostics=diagnostics,limits="Fixed fit trajectories only; 8 zh-my and 2 my-zh primary rows. No Fresh, holdout, free-running translation, or physical build. Layer attribution is correlational and only supports exploratory mask refinement.")
    raw=canon(out)+b"\n";dst=root/"all8-ablation-primary-replay.json"
    with dst.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=out["status"],sha256=sha(raw),violations=violations,eligible=eligible_batch,changed_layers={n:[x["layer"] for x in diagnostics[n]["changed_layers"]] for n in NAMES})))
if __name__=="__main__":main()
