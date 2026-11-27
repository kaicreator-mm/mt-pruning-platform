"""Independent replay of R23H layerwise-hybrid prefill screen receipts."""
import hashlib,io,json,math,pathlib,sys,tarfile
P=pathlib.Path
ARCH="007949e087e1c097473eb845cb13bff1ee9a2cef50eb4cab84413dd09fde766b"
OLD_ARCH="eef19f39837236b569aeaf6f789ff6cfe25674b2acd60dd38a1f507b128dce57"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
OLD_PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
NEW_PLAN="95b438dc7bec604c5d48feb5aa446b94aedc98c4d1a367a345c3157c7bf211f2"
MASKS={"minimax":"066ea30f6917a3f9425033f12efabe3becb919a9c0bc4b55b4a0be4878d6f220","balanced":"72d5bcbeda278221b54ed72457387cbb96ed80236366e0861109f4ac143c55b3"}
BLOBS={"minimax":"e8a9b24d182cba9f543eb5087ca96752fc211f7f","balanced":"4522d3a91f5383a0a655a7365c8807a2d8620639"}
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
def tar(raw):
 with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as t:
  ms=t.getmembers();need(len(ms)==len({m.name for m in ms}) and all(m.isfile() and m.size<=2*1024*1024 for m in ms),"tar")
  return {m.name:t.extractfile(m).read() for m in ms}
def finite(x):return type(x) in (int,float) and math.isfinite(x) and x>=0
def avg(x):return math.fsum(x)/len(x)
def main():
 root=P(sys.argv[1]);raw=(root/"layer-hybrid-prefill.tgz").read_bytes();need(sha(raw)==ARCH,"new archive SHA")
 b=tar(raw);oldraw=(root/"remaining4-primary.tgz").read_bytes();need(sha(oldraw)==OLD_ARCH,"old archive SHA");o=tar(oldraw)
 expected={"screen/plan.json","screen/report.json","screen/run.log","screen/run.stderr","screen/exit-code",*(f"screen/{i:02d}.json" for i in range(8)),"masks/minimax.json","masks/balanced.json"}
 need(set(b)==expected,"new inventory")
 plan=strict(b["screen/plan.json"]);need(sha(b["screen/plan.json"])==NEW_PLAN and plan["status"]=="R23H_LAYER_HYBRID_PREFILL_SCREEN_ONLY" and plan["teacher"]==TEACHER and plan["holdout"]=="FORBIDDEN","plan")
 for n in MASKS:
  x=strict(b[f"masks/{n}.json"]);need(blob(b[f"masks/{n}.json"])==BLOBS[n] and x["mask_sha256"]==MASKS[n] and sha(canon(x["mask"]))==MASKS[n] and x["method"]==n,"mask")
  need(plan["hybrid_sources"][n]["blob_sha"]==BLOBS[n] and plan["hybrid_sources"][n]["mask_sha256"]==MASKS[n] and plan["masks"][n]==MASKS[n],"plan mask binding")
 need(b["screen/exit-code"].strip()==b"0" and b["screen/run.stderr"]==b"","terminal")
 ev=[strict(x) for x in b["screen/run.log"].splitlines() if x.startswith(b"{")]
 ready=[x for x in ev if x["status"]=="R23H_LAYER_HYBRID_TEACHER_READY"]
 caps=[x for x in ev if x["status"]=="R23H_LAYER_HYBRID_PREFILL_CAPTURED"]
 done=[x for x in ev if x["status"]=="R23H_LAYER_HYBRID_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION"]
 need(len(ready)==1 and ready[0]["teacher"]==TEACHER and len(caps)==8 and len(done)==1,"events")
 ep=strict(o["recovery/expanded.json"]);pp=strict(o["recovery/prop-plan.json"])
 rows=[r for r in ep["fit"] if r["direction"]=="zh-my"]
 need(len(rows)==8 and len({r["line"] for r in rows})==8 and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}) and plan["ids"]==[r["id"] for r in rows],"fit rows")
 rec=strict(o["recovery/recovery-receipt.json"]);need(sha(o["recovery/recovery-receipt.json"])==REC and rec["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH","recovery")
 prior={};old_screen={}
 for i,h in enumerate(rec["receipt_sha256"]):
  x=strict(o[f"prior-r23g/{i:02d}.json"]);need(sha(o[f"prior-r23g/{i:02d}.json"])==h,"prior receipt");prior[x["id"]]=x
 oldplan=strict(o["screen/plan.json"]);oldreport=strict(o["screen/report.json"])
 need(sha(o["screen/plan.json"])==OLD_PLAN and sha(o["screen/report.json"])==OLD_REPORT and oldplan["ids"]==plan["ids"],"old screen")
 old_events=[strict(x) for x in o["screen/run.log"].splitlines() if x.startswith(b"{")]
 old_caps=[x for x in old_events if x["status"]=="R23H_PREFILL_CAPTURED"];need(len(old_caps)==8,"old screen events")
 for i,e in enumerate(old_caps):
  z=strict(o[f"screen/{i:02d}.json"]);need(sha(o[f"screen/{i:02d}.json"])==e["sha256"] and z["id"]==rows[i]["id"],"old screen receipt");old_screen[z["id"]]=z
 metrics={n:[] for n in ("original96","r23g","all8","all10","minimax","balanced")}
 for i,(e,r) in enumerate(zip(caps,rows)):
  x=strict(b[f"screen/{i:02d}.json"]);need(sha(b[f"screen/{i:02d}.json"])==e["receipt_sha256"] and e["index"]==i and e["id"]==r["id"]==x["id"] and x["index"]==i and x["plan_sha256"]==NEW_PLAN and x["teacher_sha256"]==TEACHER,"capture")
  p=prior[r["id"]];need(x["teacher_sha256"]==p["original96"]["teacher_hashes"][0]["sha256"] and x["metrics"]["original96"]["logits_sha256"]==p["original96"]["masked_hashes"][0]["sha256"] and x["metrics"]["r23g"]["logits_sha256"]==p["candidate96"]["masked_hashes"][0]["sha256"],"prior baseline replay")
  for n in metrics:
   q=x["metrics"][n];need(q["mask_sha256"]==plan["masks"][n] and len(q["logits_sha256"])==64 and all(finite(q[k]) for k in ("kl","nmse")) and type(q["argmax_mismatch"]) is int and q["argmax_mismatch"] in (0,1),"metric")
   metrics[n].append(q)
  for n in ("all8","all10"):
   q=x["metrics"][n];z=old_screen[r["id"]]["metrics"][n]
   need(q==z,"control replay "+n)
 report={n:{k:(sum if k=="argmax_mismatch" else avg)([x[k] for x in values]) for k in ("kl","nmse","argmax_mismatch")} for n,values in metrics.items()}
 eligible=[n for n in MASKS if all(report[n][k]<=report["original96"][k]+1e-12 for k in report[n]) and report[n]["kl"]<report["r23g"]["kl"]]
 eligible.sort(key=lambda n:(report[n]["kl"],report[n]["nmse"],n))
 actual=strict(b["screen/report.json"])
 need(actual["status"]=="R23H_LAYER_HYBRID_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION" and actual["plan_sha256"]==NEW_PLAN and actual["samples"]==8 and actual["metrics"]==report and actual["eligible_for_primary_fit_only"]==eligible and actual["selected"]==(eligible[0] if eligible else None),"report replay")
 need(done[0]["report_sha256"]==sha(b["screen/report.json"]) and {k:v for k,v in done[0].items() if k not in ("status","report_sha256")}=={k:v for k,v in actual.items() if k!="status"},"terminal report")
 out=dict(status="R23H_LAYER_HYBRID_PREFILL_INDEPENDENT_RECEIPT_REPLAY_PASS",archive_sha256=ARCH,plan_sha256=NEW_PLAN,samples=8,metrics=report,eligible_for_primary_fit_only=eligible,selected=eligible[0] if eligible else None,limits="Eight zh-my fit prompts, prefill only. Independent receipt/aggregation replay; no decode or other direction, no Fresh, no product admission.")
 rawout=canon(out)+b"\n";(root/"layer-hybrid-prefill-replay.json").write_bytes(rawout)
 print(json.dumps(dict(status=out["status"],sha256=sha(rawout),eligible=eligible,metrics=report),sort_keys=True))
if __name__=="__main__":main()
