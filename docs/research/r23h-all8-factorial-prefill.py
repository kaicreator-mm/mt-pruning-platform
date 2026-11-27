"""R23H all8 factorial prefill-only Fit screen; no decode or admission."""
import hashlib,importlib.util,json,math,os,pathlib,sys,time,fcntl
P=pathlib.Path
R=P.home()/".cache/mt-pruning";RR=R/"r23-results"
D=R/"r23h-pareto-screen-v1";USB=R/"r23g-recovery-ssd-38ed4798"
SD=RR/"r23h-pareto-prefill-v1";PREP=RR/"r23h-all8-factorial-prefill-v1"
RUN=R/"r23h-all8-factorial-prefill-run-v1"
PLAN="7b627cedeb7e3732af8e740aec213791e888324cb6c8db3dbe67cd747ca180ac"
PLAN_BLOB="415f2526c82356e216d74105301771b3060f3461"
OLD_PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
PRIMARY_REPORT="8ab5c2fc607ad49a589636df39ab46f801327d7c39cbbf70ce29b0fb2b23d2af"
PRIMARY_REPLAY="89ab4d14f73563ee6e1a05cdeeb2143cf67c277a43c78629c2468b310514c552"
TEACHER="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ORIGINAL_BLOB="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
EXPANDED_SHA="21b4a54f626c32727284f32f34f3b43b54ff6e68d56ac4499934ee652fb6d166"
CANDIDATE_SHA="e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
PINS={"driver":"0735238e15f13fd75e3f6939fc31aa834d8b5854","collector":"02daf65bb7fe0526b2a700d17013054f7b0eea46"}
MASKS={
 "all8_revert10":("7c7a154b17f56d896167c992e06f0f743e286688bf44714949058a2bacd9511d","faa4d6146a3df4192e913e2702a926afa3fb095d","b16b87ddd1058e0dd00286d0f1f7c7aa57e4925277346858f42bb36857479e7"),
 "all8_revert15":("cdaebcfe274af03118167638ad482f46dd8bae5474701b2bba34de2b8d747498","044a6f395d779489462df00a664cf7876732ab06","4c81baf0201fa0f085b8908625a2b46d9f4fb39bebc6cbef829138817a416c7f"),
 "all8_revert10_40_42":("776146272acf7673bac2f3b1393c85cc59be0316acac09bc466f561a69d9cbbd","adc46aceb97c4f35c8489b1b0cb02fbdf0295eee","2ffd628021e714e5ecf9145f9651a101788790e57f56bdf156b927dae65472ac"),
 "all8_revert15_40_42":("138cd339a7f7f9408bebf325fd8258d889fee156431a5f5b6be451d4cdbbf1fa","e453de1af1fef152b44f2446ecb5de6f9e18b4ae","31c5eeb44b3dc8acb6eeae97a2078faf1b96f9e129a24fc27726232b2b5c8df2"),
}
EXPECTED_CHANGED={"all8_revert10":[15,40,42],"all8_revert15":[10,40,42],"all8_revert10_40_42":[15],"all8_revert15_40_42":[10]}
def need(x,m):
 if not x:raise ValueError(m)
def canon(x):
 return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def pin(p,h,git=False):
 need(p.is_file() and not p.is_symlink(),"regular "+str(p))
 b=p.read_bytes();need((blob if git else sha)(b)==h,"pin "+str(p));return b
def publish(p,x):
 raw=canon(x)+b"\n"
 with p.open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return sha(raw)
def event(status,**x):
 print(json.dumps(dict(status=status,**x),sort_keys=True),flush=True)
def load(n):
 p=D/(n+".py");pin(p,PINS[n],True)
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def mean(xs):return math.fsum(xs)/len(xs)
def metric(t,z):
 a=t.double();b=z.double();p=a.log_softmax(-1);q=b.log_softmax(-1)
 kl=float((p.exp()*(p-q)).sum(dim=-1).mean());nmse=float((a-b).square().sum()/a.square().sum())
 need(math.isfinite(kl) and kl>=-1e-12 and math.isfinite(nmse),"finite metrics")
 return dict(kl=max(0.,kl),nmse=nmse,argmax_mismatch=int(a[-1].argmax()!=b[-1].argmax()))
def main():
 need(RUN.is_dir() and not (RUN/"run.log").exists() and not (RUN/"exit-code").exists(),"fresh run directory")
 need(all(not (PREP/f"{i:02d}.json").exists() for i in range(8)) and not (PREP/"report.json").exists(),"factorial output already exists")
 plan=json.loads(pin(PREP/"plan.json",PLAN))
 need(blob((PREP/"plan.json").read_bytes())==PLAN_BLOB and plan["status"]=="R23H_ALL8_FACTORIAL_PREFILL_SCREEN_ONLY" and plan["holdout"]=="FORBIDDEN","factorial plan")
 need(plan["source_primary_batch_report_sha256"]==PRIMARY_REPORT and plan["source_primary_independent_replay_sha256"]==PRIMARY_REPLAY,"prior primary replay binding")
 rec=json.loads(pin(USB/"recovery-receipt.json",REC));need(rec["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH","R23G recovery")
 prev=json.loads(pin(RR/"r23h-all8-ablation-primary-v3"/"report.json",PRIMARY_REPORT))
 need(prev["status"]=="R23H_ALL8_ABLATION_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION" and prev["eligible_for_secondary_fit_only"]==[],"previous primary rejection")
 replay=json.loads(pin(RUN/"primary-replay.json",PRIMARY_REPLAY))
 need(replay["status"]=="R23H_ALL8_ABLATION_PRIMARY_INDEPENDENT_RECEIPT_REPLAY_PASS" and replay["eligible_for_secondary_fit_only"]==[],"previous independent replay")
 oldplan=json.loads(pin(SD/"plan.json",OLD_PLAN));oldreport=json.loads(pin(SD/"report.json",OLD_REPORT))
 need(oldreport["selected"]=="all8" and oldreport["samples"]==8,"old screen")
 base=json.loads(pin(D/"original.json",ORIGINAL_BLOB,True))["candidates"]["96"]["kept"]
 oldmask={x["name"]:x["kept"] for x in oldplan["screen"]}
 r23g=json.loads(pin(USB/"candidate.json",CANDIDATE_SHA))["kept"]
 need(sha(canon(base))==plan["controls"]["original96"] and sha(canon(r23g))==plan["controls"]["r23g"] and sha(canon(oldmask["all8"]))==plan["controls"]["all8"] and sha(canon(oldmask["all10"]))==plan["controls"]["all10"],"control masks")
 candidates={}
 for n,(rh,bh,mh) in MASKS.items():
  raw=pin(PREP/(n+".json"),rh)
  need(blob(raw)==bh,"mask Git blob "+n)
  x=json.loads(raw);m=x["mask"]
  need(x["status"]=="R23H_ALL8_FACTORIAL_LAYER_REVERSION_EXPLORATORY_NOT_ADMITTED" and x["method"]==n and x["mask_sha256"]==mh and sha(canon(m))==mh,"mask receipt "+n)
  need(x["changed_layers"]==EXPECTED_CHANGED[n] and set(m)==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for v in m.values()),"mask shape "+n)
  need(plan["candidates"][n]["receipt_sha256"]==rh and plan["candidates"][n]["git_blob_sha"]==bh and plan["candidates"][n]["mask_sha256"]==mh,"plan mask binding "+n)
  candidates[n]=m
 expanded=json.loads(pin(USB/"expanded.json",EXPANDED_SHA))
 rows=[r for r in expanded["fit"] if r["direction"]=="zh-my"]
 need(len(rows)==8 and [r["id"] for r in rows]==plan["ids"] and not ({r["line"] for r in rows}&{r["line"] for r in expanded["holdout"]}),"fit-only rows")
 prior={}
 for i,h in enumerate(rec["receipt_sha256"]):
  x=json.loads(pin(RR/"r23g-l38-propagation-v1"/f"{i:02d}.json",h));prior[x["id"]]=x
 ids={r["id"] for r in rows};need(ids<=set(prior),"prior R23G rows")
 oldevents=[json.loads(l) for l in pin(D/"run.log",sha((D/"run.log").read_bytes())).splitlines() if l.startswith(b"{")]
 oldcaps=[x for x in oldevents if x["status"]=="R23H_PREFILL_CAPTURED"];need(len(oldcaps)==8,"old screen events")
 oldscreen={}
 for i,e in enumerate(oldcaps):
  x=json.loads(pin(SD/f"{i:02d}.json",e["sha256"]))
  need(x["id"]==rows[i]["id"] and x["plan_sha256"]==OLD_PLAN,"old screen receipt")
  oldscreen[x["id"]]=x
 corpus=pin(R/"run-d01b-r19/corpus/corpus.jsonl","9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4")
 ctx={x["id"]:x for x in map(json.loads,corpus.splitlines()) if x["split"]=="calibration" and x["id"] in ids}
 need(set(ctx)==ids,"corpus rows")
 for r in rows:need(sha(canon(ctx[r["id"]]["prompt_ids"]))==r["prompt_ids_sha256"],"prompt identity")
 masks={"original96":base,"r23g":r23g,"all8":oldmask["all8"],"all10":oldmask["all10"],**candidates}
 drv=load("driver");c=load("collector");c.verify_code()
 with (R/"r23-admissions/batch.lock").open("a+") as lk:
  deadline=time.monotonic()+7200
  while True:
   try:fcntl.flock(lk.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:need(time.monotonic()<deadline,"lock wait");time.sleep(10)
  pre=drv.preflight();need(not pre["blockers"],str(pre))
  need(c.torch.__version__=="2.10.0+cpu","pinned CPU engine required")
  ident=drv.checkpoint_identity(drv.SOURCE,cache_dir=None,fresh=True,exact_inventory=True)
  need(ident["teacher"]==TEACHER and ident["num_experts"]==128,"teacher identity")
  c.torch.set_num_threads(6);c.torch.set_num_interop_threads(1)
  e=c.engine_module.Engine(c.engine_module.Checkpoint(drv.SOURCE),threads=6)
  event("R23H_ALL8_FACTORIAL_TEACHER_READY",teacher=TEACHER,threads=6)
  def prefill(prompt,mask):
   try:
    e.set_mask(None if mask is None else {int(l):v for l,v in mask.items()})
    with c.torch.inference_mode():z=e.forward(prompt,e.new_cache(len(prompt)+9),"prefill")
    return c.logits_sha(z),z[[len(prompt)//2-1,len(prompt)-1]].clone()
   finally:e.set_mask(None)
  receipts=[]
  for i,r in enumerate(rows):
   need(not drv.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"]),"GPU busy")
   prompt=ctx[r["id"]]["prompt_ids"];old=prior[r["id"]]
   th,t=prefill(prompt,None)
   need(th==old["original96"]["teacher_hashes"][0]["sha256"],"teacher baseline")
   receipt=dict(status="R23H_ALL8_FACTORIAL_PREFILL_CAPTURED_NOT_ADMISSION",id=r["id"],index=i,plan_sha256=PLAN,teacher_sha256=th,metrics={})
   for n,m in masks.items():
    h,z=prefill(prompt,m);q=metric(t,z)
    if n=="original96":
     need(h==th,"original96 no-op")
    if n=="r23g":
     o=old["candidate96"]["logits"][0]
     need(h==old["candidate96"]["masked_hashes"][0]["sha256"] and abs(q["kl"]-o["teacher_to_masked_kl"])<1e-12 and abs(q["nmse"]-o["nmse"])<1e-12 and q["argmax_mismatch"]==int(o["teacher_argmax"]!=o["masked_argmax"]),"R23G replay")
    if n in ("all8","all10"):
     o=oldscreen[r["id"]]["metrics"][n]
     need(h==o["logits_sha256"] and abs(q["kl"]-o["kl"])<1e-12 and abs(q["nmse"]-o["nmse"])<1e-12 and q["argmax_mismatch"]==o["argmax_mismatch"],"old screen replay")
    receipt["metrics"][n]=dict(q,logits_sha256=h,mask_sha256=sha(canon(m)))
   h,_=prefill(prompt,None);need(h==th,"teacher restoration")
   rh=publish(PREP/f"{i:02d}.json",receipt);receipts.append(receipt)
   event("R23H_ALL8_FACTORIAL_PREFILL_CAPTURED",index=i,id=r["id"],receipt_sha256=rh)
  names=list(masks)
  metrics={n:{k:(sum if k=="argmax_mismatch" else mean)([x["metrics"][n][k] for x in receipts]) for k in ("kl","nmse","argmax_mismatch")} for n in names}
  eligible=[n for n in candidates if all(metrics[n][k]<=metrics["original96"][k]+1e-12 for k in metrics[n]) and metrics[n]["kl"]<metrics["r23g"]["kl"]]
  eligible.sort(key=lambda n:(metrics[n]["kl"],metrics[n]["nmse"],n))
  report=dict(status="R23H_ALL8_FACTORIAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION",plan_sha256=PLAN,samples=8,metrics=metrics,eligible_for_primary_fit_only=eligible,selected=eligible[0] if eligible else None,limits="Four layer-local reversion masks; eight frozen zh-my Fit prompts; prefill only. Passing masks may enter the 10-row primary Fit propagation. No decode, my-zh, Fresh, holdout, free-running translation, or product admission.")
  h=publish(PREP/"report.json",report);event(report["status"],report_sha256=h,**{k:v for k,v in report.items() if k!="status"})
if __name__=="__main__":main()
