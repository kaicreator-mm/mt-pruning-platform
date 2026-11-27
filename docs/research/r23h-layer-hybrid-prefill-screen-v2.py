"""R23H actual prefill screen for two layerwise hybrid masks.
Fit prompts only; no decode, holdout, Fresh, or model admission."""
import hashlib,importlib.util,json,math,os,pathlib,sys,time,fcntl
P=pathlib.Path;R=P.home()/".cache/mt-pruning";RR=R/"r23-results"
D=R/"r23h-pareto-screen-v1";USB=R/"r23g-recovery-ssd-38ed4798"
SD=RR/"r23h-pareto-prefill-v1";REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
OLD_PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
HYBRIDS={"minimax":"e8a9b24d182cba9f543eb5087ca96752fc211f7f","balanced":"4522d3a91f5383a0a655a7365c8807a2d8620639"}
PINS={"driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46"}
def need(x,m):
 if not x:raise ValueError(m)
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def pin(p,h,git=False):
 need(p.is_file() and not p.is_symlink(),"regular "+str(p));b=p.read_bytes();need((blob if git else sha)(b)==h,"pin "+str(p));return b
def publish(p,x):
 b=canon(x)+b"\n"
 with p.open("xb") as f:f.write(b);f.flush();os.fsync(f.fileno())
 return sha(b)
def event(status,**x):print(json.dumps(dict(status=status,**x),sort_keys=True),flush=True)
def load(n):
 p=D/(n+".py");pin(p,PINS[n],True);s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def mean(x):return math.fsum(x)/len(x)
def metric(t,z):
 a=t.double();b=z.double();p=a.log_softmax(-1);q=b.log_softmax(-1)
 kl=float((p.exp()*(p-q)).sum(dim=-1).mean());nmse=float((a-b).square().sum()/a.square().sum())
 need(math.isfinite(kl) and kl>=-1e-12 and math.isfinite(nmse),"finite")
 return dict(kl=max(0.,kl),nmse=nmse,argmax_mismatch=int(a[-1].argmax()!=b[-1].argmax()))
def main():
 d=P(sys.argv[1]);outdir=RR/"r23h-layer-hybrid-prefill-v2";outdir.mkdir(mode=0o700)
 rec=json.loads(pin(USB/"recovery-receipt.json",REC));need(rec["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH","recovery")
 ep=json.loads(pin(USB/"expanded.json","a1047d1a7c5624ed0e3a868bd5d76eef4c754915",True));rows=[r for r in ep["fit"] if r["direction"]=="zh-my"];need(len(rows)==8 and len({r["line"] for r in rows})==8 and not ({r["line"] for r in rows}&{r["line"] for r in ep["holdout"]}),"fit rows")
 base=json.loads(pin(D/"original.json","d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed",True))["candidates"]["96"]["kept"]
 oldplan=json.loads(pin(SD/"plan.json",OLD_PLAN));oldreport=json.loads(pin(SD/"report.json",OLD_REPORT))
 oldmask={x["name"]:x["kept"] for x in oldplan["screen"]}
 r23g=json.loads(pin(USB/"candidate.json","e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"))["kept"]
 masks={"original96":base,"r23g":r23g,"all8":oldmask["all8"],"all10":oldmask["all10"]}
 hyb={}
 for n,h in HYBRIDS.items():
  x=json.loads(pin(d/(n+".json"),h,True))
  need(x["method"]==n and x["status"]=="R23H_LAYERWISE_HYBRID_PROPOSALS_EXPLORATORY_NOT_ADMITTED" and x["mask_sha256"]==sha(canon(x["mask"])),"hybrid identity")
  need(set(x["mask"])==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for v in x["mask"].values()),"hybrid mask")
  hyb[n]=x;masks[n]=x["mask"]
 ids={r["id"] for r in rows};prior={}
 for i,h in enumerate(rec["receipt_sha256"]):
  x=json.loads(pin(RR/"r23g-l38-propagation-v1"/f"{i:02d}.json",h));prior[x["id"]]=x
 need(ids<=set(prior),"prior rows")
 old_events=[json.loads(l) for l in pin(D/"run.log",sha((D/"run.log").read_bytes())).splitlines() if l.startswith(b"{")]
 old_caps=[x for x in old_events if x["status"]=="R23H_PREFILL_CAPTURED"]
 need(len(old_caps)==8,"old screen events")
 old_screen={}
 for i,e in enumerate(old_caps):
  x=json.loads(pin(SD/f"{i:02d}.json",e["sha256"]));need(x["id"]==rows[i]["id"] and x["plan_sha256"]==OLD_PLAN,"old screen receipt");old_screen[x["id"]]=x
 plan=dict(status="R23H_LAYER_HYBRID_PREFILL_SCREEN_ONLY",teacher=TEACHER,recovery_sha256=REC,source_screen_plan_sha256=OLD_PLAN,source_screen_report_sha256=OLD_REPORT,hybrid_sources={n:{"blob_sha":HYBRIDS[n],"mask_sha256":hyb[n]["mask_sha256"],"method":n} for n in hyb},masks={n:sha(canon(m)) for n,m in masks.items()},ids=[r["id"] for r in rows],holdout="FORBIDDEN",limits="Eight zh-my fit prompts, prefill only. Not quality or admission.")
 ph=publish(outdir/"plan.json",plan);event("R23H_LAYER_HYBRID_PLAN_FROZEN",plan_sha256=ph,masks=plan["masks"])
 drv=load("driver");c=load("collector");c.verify_code()
 corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
 ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids};need(set(ctx)==ids,"corpus")
 for r in rows:need(sha(canon(ctx[r["id"]]["prompt_ids"]))==r["prompt_ids_sha256"],"prompt identity")
 with (R/"r23-admissions/batch.lock").open("a+") as lk:
  deadline=time.monotonic()+7200
  while True:
   try:fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:need(time.monotonic()<deadline,"lock wait");time.sleep(10)
  pre=drv.preflight();need(not pre["blockers"],str(pre))
  c.torch.set_num_threads(8);c.torch.set_num_interop_threads(1)
  ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True);need(ident["teacher"]==TEACHER and ident["num_experts"]==128,"teacher")
  e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=8);event("R23H_LAYER_HYBRID_TEACHER_READY",teacher=TEACHER)
  def prefill(prompt,mask):
   try:
    e.set_mask(None if mask is None else {int(l):v for l,v in mask.items()})
    with c.torch.inference_mode():z=e.forward(prompt,e.new_cache(len(prompt)+9),"prefill")
    return c.logits_sha(z),z[[len(prompt)//2-1,len(prompt)-1]].clone()
   finally:e.set_mask(None)
  caps=[]
  for i,r in enumerate(rows):
   need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
   prompt=ctx[r["id"]]["prompt_ids"];old=prior[r["id"]]
   th,t=prefill(prompt,None);need(th==old["original96"]["teacher_hashes"][0]["sha256"],"teacher logits")
   rec0=dict(id=r["id"],index=i,plan_sha256=ph,teacher_sha256=th,metrics={})
   for n,m in masks.items():
    h,z=prefill(prompt,m);q=metric(t,z)
    if n=="r23g":
     o=old["candidate96"]["logits"][0];need(h==old["candidate96"]["masked_hashes"][0]["sha256"] and abs(q["kl"]-o["teacher_to_masked_kl"])<1e-12 and abs(q["nmse"]-o["nmse"])<1e-12 and q["argmax_mismatch"]==int(o["teacher_argmax"]!=o["masked_argmax"]),"r23g replay")
    if n in ("all8","all10"):
     o=old_screen[r["id"]]["metrics"][n];need(h==o["logits_sha256"] and abs(q["kl"]-o["kl"])<1e-12 and abs(q["nmse"]-o["nmse"])<1e-12 and q["argmax_mismatch"]==o["argmax_mismatch"],"old screen replay")
    rec0["metrics"][n]=dict(q,logits_sha256=h,mask_sha256=sha(canon(m)))
   h,_=prefill(prompt,None);need(h==th,"teacher restoration")
   rh=publish(outdir/f"{i:02d}.json",rec0);caps.append(rec0);event("R23H_LAYER_HYBRID_PREFILL_CAPTURED",index=i,id=r["id"],receipt_sha256=rh,kl={n:x["kl"] for n,x in rec0["metrics"].items()})
  names=list(caps[0]["metrics"])
  metrics={n:{k:(sum if k=="argmax_mismatch" else mean)([x["metrics"][n][k] for x in caps]) for k in ("kl","nmse","argmax_mismatch")} for n in names}
  eligible=[n for n in hyb if all(metrics[n][k]<=metrics["original96"][k]+1e-12 for k in metrics[n]) and metrics[n]["kl"]<metrics["r23g"]["kl"]]
  eligible.sort(key=lambda n:(metrics[n]["kl"],metrics[n]["nmse"],n))
  report=dict(status="R23H_LAYER_HYBRID_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION",plan_sha256=ph,samples=8,metrics=metrics,eligible_for_primary_fit_only=eligible,selected=eligible[0] if eligible else None,limits="Fit prefill only; no decode, no other direction, no Fresh or product admission.")
  h=publish(outdir/"report.json",report);event(report["status"],report_sha256=h,**{k:v for k,v in report.items() if k!="status"})
if __name__=="__main__":main()
