"""R23G fixed hybrid local-loss replay and line-deletion sensitivity, offline only.
Not new inference, not a confidence interval, not admission or a new selection."""
import argparse,hashlib,importlib.util,json,math,pathlib
P=pathlib.Path
ATTR="71ee7ebd17c9efdc894be76ddcb2433b6a98c184873c41ea4f418320713da20f"
RECEIPT="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
def need(v,m):
    if not v:raise ValueError(m)
def mean(xs):need(len(xs)>0,"empty");return math.fsum(xs)/len(xs)
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p,h):
    need(p.is_file() and not p.is_symlink(),"regular file");b=p.read_bytes();need(sha(b)==h,"hash");return b
def sensitivity(rows,by,groups):
    def evaluate(rs):
        means={n:{g:mean([by[r["id"]][n][g.split("/")[1]] for r in rs if r["direction"]==g.split("/")[0]]) for g in groups} for n in ("base","hybrid")}
        b=means["base"];s=means["hybrid"];gain=(math.fsum(b[g] for g in groups[:4])-math.fsum(s[g] for g in groups[:4]))/max(math.fsum(b[g] for g in groups[:4]),1e-12)
        violations=[g for i,g in enumerate(groups) if s[g]>b[g]*(1 if i<4 else 1.02)+1e-12]
        overall=math.fsum(s.values())<=math.fsum(b.values())+1e-12
        return dict(means=means,primary_gain=gain,violations=violations,overall_guard=overall,pass_local_algebra=gain>=.05 and not violations and overall)
    full=evaluate(rows);folds=[]
    for line in sorted({r["line"] for r in rows}):
        z=evaluate([r for r in rows if r["line"]!=line]);folds.append(dict(omitted_line=line,primary_gain=z["primary_gain"],violations=z["violations"],overall_guard=z["overall_guard"],pass_local_algebra=z["pass_local_algebra"]))
    return dict(full=full,folds=folds,passed_folds=sum(z["pass_local_algebra"] for z in folds),total_folds=len(folds),min_primary_gain=min(z["primary_gain"] for z in folds),max_primary_gain=max(z["primary_gain"] for z in folds))
def selftest():
    groups=tuple(d+"/"+p for d in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en") for p in ("prefill","decode"))
    rows=[dict(id=g+str(i),direction=g,line=i) for g in ("zh-my","my-zh","zh-en","en-zh","en-my","my-en") for i in (0,1)]
    by={r["id"]:{n:{p:.2 if n=="base" else .1 for p in ("prefill","decode")} for n in ("base","hybrid")} for r in rows}
    z=sensitivity(rows,by,groups);need(z["passed_folds"]==2 and z["full"]["primary_gain"]==.5,"positive")
    by[rows[0]["id"]]["hybrid"]["prefill"]=.4;need(not sensitivity(rows,by,groups)["full"]["pass_local_algebra"],"regression")
    print(json.dumps(dict(status="PASS_SYNTHETIC_R23G_SENSITIVITY",checks=2)))
def main():
    p=argparse.ArgumentParser();p.add_argument("--selftest",action="store_true");p.add_argument("--inputs",type=P);p.add_argument("--out",type=P);q=p.parse_args()
    if q.selftest:selftest();return
    need(q.inputs and q.out,"inputs");d=q.inputs
    read(d/"analyze-v2.py",ATTR);sp=importlib.util.spec_from_file_location("attr",d/"analyze-v2.py");a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
    c=a.strict(read(d/"r23g-l38-reverted.json",RECEIPT));plan=a.strict(a.pinned(d/"expanded.json",a.PLAN,True));rows=plan["fit"]
    orig=a.strict(a.pinned(d/"original.json",a.ORIG,True))["candidates"]["96"]["kept"];stable=a.strict(a.pinned(d/"candidate.json",a.CAND))["kept"]
    need(len(rows)==48 and len({r["line"] for r in rows})==30 and not ({r["line"] for r in rows}&{r["line"] for r in plan["holdout"]}),"fit scope")
    h=c["kept"];need(h["38"]==orig["38"] and [k for k in h if h[k]!=stable[k]]==["38"] and a.sha(a.canon(h))==c["mask_sha256"],"one frozen reversion")
    proxy=a.pure_proxy(a.pinned(d/"proxy.py",a.PROXY,True));masks={"original96":orig,"stable96":stable};layers=list(range(1,48));caps={};by={};hashes=[]
    for b in range(2):
        events=[a.strict(x) for x in a.pinned(d/f"fit{b}.log",a.LOGS[b]).splitlines() if x.startswith(b"{")];need(a.read(d/f"fit{b}.stderr")==b"","stderr")
        ev=[x for x in events if x.get("status")=="STABLE96_PROXY_CAPTURED"];need(len(ev)==24,"batch")
        for i,e in enumerate(ev,b*24):
            r=rows[i];need(e["index"]==i and e["id"]==r["id"],"order")
            x=a.strict(a.pinned(d/"fit"/f"{i:02d}.json",e["receipt_sha256"]));need(x["admission_sha256"]==a.AD[b] and x["teacher_digest"]==a.TEACHER and x["engine_head"]==a.ENGINE and x["index"]==i and x["candidate_receipt_sha256"]==a.CAND,"provenance")
            seen=proxy.checked_capture(x,r,masks,layers);caps[r["id"]]=x;hashes.append(e["receipt_sha256"])
            by[r["id"]]={n:{phase:mean([mean([seen[l,phase,j]["losses"]["original96" if n=="base" or l==38 else "stable96"]["combined_error"] for j in steps]) for l in layers]) for phase,steps in (("prefill",[0]),("decode",[1,8]))} for n in ("base","hybrid")}
    f=a.strict(a.pinned(d/"fit"/"decision.json",a.DEC));need(f==proxy.decision(rows,caps,masks,layers,"fit"),"source decision replay")
    z=sensitivity(rows,by,a.GROUPS);need(all(abs(z["full"]["means"]["hybrid"][g]-c["predicted_local"]["means"][g])<1e-12 for g in a.GROUPS),"prediction reconciliation")
    out=dict(status="FIT_HYBRID_LOCAL_SENSITIVITY_NOT_INFERENCE",candidate_sha256=RECEIPT,samples=48,aligned_lines=30,receipt_sha256=hashes,**z,limits="Uses original and stable observed local layer errors under identical teacher trajectory. Hybrid layer-local arithmetic only; leave-one-aligned-line-out sensitivity on reused fit data, not independent holdout, CI or quality.")
    raw=json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"
    with q.out.open("xb") as f:f.write(raw)
    print(json.dumps(dict(status=out["status"],sha256=sha(raw),passed_folds=z["passed_folds"],total_folds=z["total_folds"],min_primary_gain=z["min_primary_gain"],max_primary_gain=z["max_primary_gain"],failed_folds=[v for v in z["folds"] if not v["pass_local_algebra"]])))
if __name__=="__main__":main()
