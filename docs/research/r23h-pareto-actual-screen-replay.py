"""Independent actual fit-prefill receipt audit; no model admission."""
import hashlib,json,math,pathlib,sys,tarfile
P=pathlib.Path
base=P(sys.argv[1])
def need(b,m):
    if not b:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pinned(b,h,m):
    need(sha(b)==h,m);return b
archive=base/"pareto-actual-v1.tgz"
pinned(archive.read_bytes(),"d6d83db734d1f4eda6eefa14c8861a551d739c91e5ae0511b1f2dc5de5b6be71","archive")
S="r23-results/r23h-pareto-prefill-v1/"
L="r23h-pareto-screen-v1/"
expected={S+"plan.json",S+"report.json",*(S+f"{i:02d}.json" for i in range(8)),*(L+n for n in ("run.log","run.stderr","exit-code"))}
with tarfile.open(archive,"r:gz") as t:
    members=t.getmembers();need(len(members)==len(expected) and {m.name for m in members}==expected and all(m.isfile() and m.size<=200000 for m in members),"inventory")
    data={m.name:t.extractfile(m).read() for m in members}
ph="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
rh="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
plan=json.loads(pinned(data[S+"plan.json"],ph,"plan"))
report=json.loads(pinned(data[S+"report.json"],rh,"report"))
src=json.loads(pinned((base/"input/r23h-pareto-screen-input.json").read_bytes(),"cbdc65097cce87cfc6ddb8934604c690e744b9f3d8f7f52b3708bf035c20fa2f","input"))
need(data[L+"exit-code"].strip()==b"0" and data[L+"run.stderr"]==b"","terminal")
events=[json.loads(x) for x in data[L+"run.log"].splitlines() if x.startswith(b"{")]
caps=[x for x in events if x["status"]=="R23H_PREFILL_CAPTURED"]
terminal=[x for x in events if x["status"]=="R23H_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"]
need(len(caps)==8 and len(terminal)==1 and terminal[0]["report_sha256"]==rh,"events")
need(plan["status"]=="R23H_PARETO_FIT_PREFILL_SCREEN_ONLY" and plan["holdout"]=="FORBIDDEN" and plan["pareto_input_sha256"]==sha((base/"input/r23h-pareto-screen-input.json").read_bytes()),"scope")
source={x["name"]:x for x in src["candidates"]}
masks={"original96":"23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7","r23g":"77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"}
for c in plan["screen"]:
    s=source[c["name"]];h=sha(canon(c["kept"]));need(h==s["mask_sha256"]==c["mask_sha256"] and c["kept"]==s["kept"] and c["folds_passed"]==30,"mask binding")
    masks[c["name"]]=h
need(len(masks)==8 and len(set(plan["ids"]))==8,"inventory")
rows=[]
for i,ev in enumerate(caps):
    z=json.loads(pinned(data[S+f"{i:02d}.json"],ev["sha256"],"capture"))
    need(z["index"]==ev["index"]==i and z["id"]==ev["id"]==plan["ids"][i] and z["id"].startswith("calibration:zh-my:") and z["plan_sha256"]==ph,"row")
    need(set(z["metrics"])==set(masks) and len(z["teacher_sha256"])==64,"metrics")
    for n,q in z["metrics"].items():
        need(q["mask_sha256"]==masks[n] and len(q["logits_sha256"])==64 and q["argmax_mismatch"] in (0,1) and all(math.isfinite(q[k]) and q[k]>=0 for k in ("kl","nmse")),"metric contract")
        need(q["kl"]==ev["kl"][n],"log binding")
    rows.append(z["metrics"])
def aggregate(rr):
    return {n:{k:(sum(rr0[n][k] for rr0 in rr) if k=="argmax_mismatch" else math.fsum(rr0[n][k] for rr0 in rr)/len(rr)) for k in ("kl","nmse","argmax_mismatch")} for n in masks}
def rank(a):
    return sorted((n for n in source if all(a[n][k]<=a["original96"][k]+1e-12 for k in a[n]) and a[n]["kl"]<a["r23g"]["kl"]),key=lambda n:(a[n]["kl"],a[n]["nmse"],n))
a=aggregate(rows);good=rank(a)
need(a==report["metrics"] and good==report["eligible_for_full_fit_propagation_only"] and report["selected"]==good[0]=="all8" and report["samples"]==8 and report["plan_sha256"]==ph,"decision")
folds=[rank(aggregate(rows[:i]+rows[i+1:])) for i in range(8)]
summary={n:dict(kl_improved_rows=sum(x[n]["kl"]<x["original96"]["kl"] for x in rows),nmse_improved_rows=sum(x[n]["nmse"]<x["original96"]["nmse"] for x in rows),leave_one_out_eligible=sum(n in f for f in folds),leave_one_out_first=sum(bool(f) and f[0]==n for f in folds)) for n in good}
out=dict(status="R23H_PARETO_ACTUAL_SCREEN_INDEPENDENT_RECEIPT_REPLAY_PASS_FIT_ONLY",archive_sha256=sha(archive.read_bytes()),plan_sha256=ph,report_sha256=rh,samples=8,candidates=6,eligible=good,selected=good[0],metrics=a,descriptive_leave_one_out=summary,limits="Recomputed from exact actual V100 receipts on separate machine. No decode, other directions, fresh, native translation or model admission. Leave-one-out is descriptive, not a replacement gate.")
raw=canon(out)+b"\n"
with (base/"pareto-actual-replay.json").open("xb") as f:f.write(raw)
print(json.dumps(out,sort_keys=True));print("SHA256="+sha(raw))
