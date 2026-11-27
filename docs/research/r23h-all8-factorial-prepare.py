"""Freeze four layer-local all8 factorial masks from pinned R23H Fit sources."""
import copy,hashlib,json,pathlib,os

P=pathlib.Path
R=P.home()/".cache/mt-pruning"
RR=R/"r23-results"
D=R/"r23h-pareto-screen-v1"
USB=R/"r23g-recovery-ssd-38ed4798"
SD=RR/"r23h-pareto-prefill-v1"
PREV=RR/"r23h-all8-ablation-primary-v3"
OUT=RR/"r23h-all8-factorial-prefill-v1"

OLD_PLAN="b6777f68cad735a5d59b2b4de07220fdc3fa007bbda7ee594a8d5bf65497577e"
OLD_REPORT="8f2663c26d8d0153fec68f9b22f37fc507ad86a7f064df34d951c7866cc2131b"
ORIGINAL_BLOB="d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
REC="ae81e98342add7ea9d366f0d1f1c7fa2d5e6f84c432331a6870ecc25ff72d5ad"
PRIMARY_REPORT="8ab5c2fc607ad49a589636df39ab46f801327d7c39cbbf70ce29b0fb2b23d2af"
PRIMARY_REPLAY="89ab4d14f73563ee6e1a05cdeeb2143cf67c277a43c78629c2468b310514c552"
ALL8="1f7402ae0fb540c86ace21914b45ea0c42a113e5f38072a2d3f86795ee8ad70b"
R23G="77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
ALL10="a9274a240cb1e990c77b110b9a6231e0ae1c905849a2b7e9295f795bd3a81fe2"

SPECS={
 "all8_revert10":(10,),
 "all8_revert15":(15,),
 "all8_revert10_40_42":(10,40,42),
 "all8_revert15_40_42":(15,40,42),
}
EXPECTED_CHANGED={
 "all8_revert10":[15,40,42],
 "all8_revert15":[10,40,42],
 "all8_revert10_40_42":[15],
 "all8_revert15_40_42":[10],
}

def need(x,m):
 if not x:raise ValueError(m)
def canon(x):
 return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def sha(b):
 return hashlib.sha256(b).hexdigest()
def blob(b):
 return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def pin(p,h,git=False):
 need(p.is_file() and not p.is_symlink(),"regular "+str(p))
 b=p.read_bytes()
 need((blob if git else sha)(b)==h,"pin "+str(p))
 return b
def publish(p,b):
 with p.open("xb") as f:
  f.write(b);f.flush();os.fsync(f.fileno())
 return sha(b)

def main():
 need(not OUT.exists(),"output already exists; preserve and use a new version")
 rec=json.loads(pin(USB/"recovery-receipt.json",REC))
 need(rec["status"]=="R23G_RECOVERED_DISK_REPLAY_REJECT_STOP_FRESH","R23G recovery")
 oldplan=json.loads(pin(SD/"plan.json",OLD_PLAN))
 oldreport=json.loads(pin(SD/"report.json",OLD_REPORT))
 need(oldreport["selected"]=="all8" and oldreport["samples"]==8,"old all8 screen")
 prev=json.loads(pin(PREV/"report.json",PRIMARY_REPORT))
 need(prev["status"]=="R23H_ALL8_ABLATION_PRIMARY_BATCH_COMPLETE_NOT_FRESH_ADMISSION" and prev["eligible_for_secondary_fit_only"]==[],"previous primary rejection")
 base=json.loads(pin(D/"original.json",ORIGINAL_BLOB,True))["candidates"]["96"]["kept"]
 all8={x["name"]:x["kept"] for x in oldplan["screen"]}["all8"]
 need(sha(canon(base))=="23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7","original96 mask")
 need(sha(canon(all8))==ALL8,"all8 mask")
 rows=[r for r in json.loads(pin(USB/"expanded.json","a1047d1a7c5624ed0e3e868bd5d76eef4c754915",True))["fit"] if r["direction"]=="zh-my"]
 need(len(rows)==8 and [r["id"] for r in rows]==oldplan["ids"],"same frozen zh-my Fit rows")
 OUT.mkdir(mode=0o700)
 candidates={}
 for name,restore in SPECS.items():
  mask=copy.deepcopy(all8)
  for layer in restore:mask[str(layer)]=copy.deepcopy(base[str(layer)])
  changed=[int(k) for k in mask if mask[k]!=base[k]]
  need(changed==EXPECTED_CHANGED[name],"changed-layer inventory "+name)
  need(set(mask)==set(map(str,range(1,48))) and all(len(v)==96 and v==sorted(set(v)) and all(type(i) is int and 0<=i<128 for i in v) for v in mask.values()),"mask shape "+name)
  mh=sha(canon(mask))
  receipt=dict(status="R23H_ALL8_FACTORIAL_LAYER_REVERSION_EXPLORATORY_NOT_ADMITTED",method=name,base_mask_sha256=ALL8,original96_mask_sha256=sha(canon(base)),restored_layers=list(restore),changed_layers=changed,mask_sha256=mh,mask=mask,limits="Exactly 96 experts in each of 47 MoE layers; derived from frozen all8 by restoring only listed layers to original96. Exploratory fit prefill screen only; not admission.")
  raw=canon(receipt)+b"\n"
  bh=publish(OUT/(name+".json"),raw)
  candidates[name]=dict(mask_sha256=mh,blob_sha=bh,restored_layers=list(restore),changed_layers=changed)
 controls=dict(original96=sha(canon(base)),r23g=R23G,all8=ALL8,all10=ALL10)
 plan=dict(status="R23H_ALL8_FACTORIAL_PREFILL_SCREEN_ONLY",teacher="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94",recovery_sha256=REC,source_screen_plan_sha256=OLD_PLAN,source_screen_report_sha256=OLD_REPORT,source_primary_batch_report_sha256=PRIMARY_REPORT,source_primary_independent_replay_sha256=PRIMARY_REPLAY,controls=controls,candidates=candidates,ids=[r["id"] for r in rows],holdout="FORBIDDEN",scope="Eight frozen zh-my calibration Fit prompts; prefill only. Same original96/R23G/all8/all10 controls. No decode, my-zh, Fresh, holdout, free-running translation, or product admission.",gate="Candidate must have KL, NMSE and argmax mismatch no worse than original96 on aggregate, and KL lower than R23G. Only passing candidates may enter the 10-row primary fit propagation.")
 raw=canon(plan)+b"\n";ph=publish(OUT/"plan.json",raw)
 print(json.dumps(dict(status=plan["status"],plan_sha256=ph,plan_blob_sha=blob(raw),controls=controls,candidates=candidates,ids=plan["ids"]),sort_keys=True))
if __name__=="__main__":main()
