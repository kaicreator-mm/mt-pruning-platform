"""R23H exploratory layer-wise hybrid mask proposal from audited fixed-fit receipts.
It never admits a model: local per-layer deltas are correlated diagnostics only."""
import hashlib,io,json,math,pathlib,tarfile
P=pathlib.Path
ARCH={"all8":"31ba7aeb8a9357b33fb1a147a44014c6754d2d7805ebe7183ef3750d8c201761","all10":"8650e78b971e6f5fcc307d5573b549f0b1ac46d4e15252c79ca34a7f41da6c56","remaining4":"eef19f39837236b569aeaf6f789ff6cfe25674b2acd60dd38a1f507b128dce57"}
NAMES=("all8","all10","all12","all14","primary4_10","primary4_14")
def sha(b):return hashlib.sha256(b).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def need(x,m):
 if not x:raise ValueError(m)
def tar(p,name):
 raw=p.read_bytes();need(sha(raw)==ARCH[name],name+" archive hash")
 with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as t:
  ms=t.getmembers();need(all(m.isfile() and m.size<=2*1024*1024 for m in ms),"archive bounds")
  return {m.name:t.extractfile(m).read() for m in ms}
def js(b):return json.loads(b)
def load_caps(d,kind,name,rows):
 out={}
 if kind in ("all8","all10"):
  for i,r in enumerate(rows):
   out[r["id"]]=js(d[f"fit/{i:02d}.json"])
 else:
  adm=js(d["batch/admission.json"])
  for idx,r in zip(adm["indices"],[x for x in rows if x["direction"] in ("zh-my","my-zh")]):
   out[r["id"]]=js(d[f"batch/{name}-{idx:02d}.json"])
 return out
def main():
 root=P(__import__("sys").argv[1])
 a8=tar(root/"all8-full-fit.tgz","all8");a10=tar(root/"all10-full-fit.tgz","all10");b=tar(root/"remaining4-primary.tgz","remaining4")
 ep=js(b["recovery/expanded.json"]);pp=js(b["recovery/prop-plan.json"])
 rows=pp["rows"]+[r for r in ep["fit"] if r["direction"]=="zh-my" and r["id"] not in {x["id"] for x in pp["rows"]}]
 need(len(rows)==18,"row inventory")
 plan=js(b["screen/plan.json"]);base=js(b["screen/original.json"])["candidates"]["96"]["kept"]
 masks={x["name"]:x["kept"] for x in plan["screen"]}
 need(set(NAMES)==set(masks),"mask inventory")
 allcaps={"all8":load_caps(a8,"all8","all8",rows),"all10":load_caps(a10,"all10","all10",rows)}
 for n in ("all12","all14","primary4_10","primary4_14"):allcaps[n]=load_caps(b,"batch",n,rows)
 # Only use the 8 zh-my + 2 my-zh primary fit rows for every candidate.
 selected=[r for r in rows if r["direction"] in ("zh-my","my-zh")]
 need(all(len(allcaps[n])==18 for n in ("all8","all10")) and all(len(allcaps[n])==10 for n in NAMES[2:]),"candidate receipt counts")
 baseline={}
 for d in ("zh-my","my-zh"):
  for l in range(1,48):
   z=[]
   for n in ("all8","all10"):
    for r in selected:
     if r["direction"]!=d:continue
     x=allcaps[n][r["id"]]["original96"]
     z.extend(q["moe_output_nmse"] for q in x["layer"] if q["layer"]==l and q["phase"]=="decode" and q["step"]>0)
   need(z,"baseline layer values");baseline[d,l]=math.fsum(z)/len(z)
 # Reaggregate each candidate on same primary rows, using its observed full-mask execution.
 deltas={}
 for n in NAMES:
  deltas[n]={}
  for d in ("zh-my","my-zh"):
   for l in range(1,48):
    z=[];h=[]
    for r in selected:
     if r["direction"]!=d:continue
     x=allcaps[n][r["id"]]
     orig={(q["layer"],q["phase"],q["step"]):q for q in x["original96"]["layer"]}
     cand={(q["layer"],q["phase"],q["step"]):q for q in x["candidate96"]["layer"]}
     for j in range(1,9):
      z.append(cand[(l,"decode",j)]["moe_output_nmse"]-orig[(l,"decode",j)]["moe_output_nmse"])
      h.append(cand[(l,"decode",j)]["hidden_input_nmse"]-orig[(l,"decode",j)]["hidden_input_nmse"])
    deltas[n][d,l]=dict(moe=math.fsum(z)/len(z),hidden=math.fsum(h)/len(h),relative=math.fsum(z)/len(z)/max(baseline[d,l],1e-12),observations=len(z))
 # Add baseline as a valid per-layer choice (zero local delta).
 opts=("original96",)+NAMES
 for n in NAMES:
  for l in range(1,48):
   v=set(masks[n][str(l)]);need(len(v)==96 and len(v)==len(masks[n][str(l)]),"candidate mask")
 need(all(len(set(base[str(l)]))==96 for l in range(1,48)),"base mask")
 candidates={}
 methods={
  "minimax":lambda x,y: max(x,y)+0.20*abs(x-y),
  "balanced":lambda x,y: 0.50*(x+y)+0.50*max(x,y),
  "myzh_guard":lambda x,y: 0.30*x+0.70*y+0.20*max(0.0,y-x),
 }
 for method,scorefn in methods.items():
  kept={};choices={};tot={"zh-my":0.0,"my-zh":0.0};counts={}
  for l in range(1,48):
   choices_l=[]
   for n in opts:
    x=y=0.0
    if n!="original96":
     x=deltas[n]["zh-my",l]["relative"];y=deltas[n]["my-zh",l]["relative"]
    choices_l.append((scorefn(x,y),max(x,y),x+y,n))
   chosen=min(choices_l,key=lambda z:(z[0],z[1],z[2],z[3]))[3]
   vals=base[str(l)] if chosen=="original96" else masks[chosen][str(l)]
   kept[str(l)]=sorted(set(vals));choices[str(l)]=chosen;counts[chosen]=counts.get(chosen,0)+1
   if chosen!="original96":
    tot["zh-my"]+=deltas[chosen]["zh-my",l]["moe"];tot["my-zh"]+=deltas[chosen]["my-zh",l]["moe"]
  need(all(len(v)==96 and len(v)==len(set(v)) for v in kept.values()),"hybrid invariant")
  candidates[method]=dict(mask=kept,mask_sha256=sha(canon(kept)),layer_sources=choices,source_counts=counts,estimated_sum_layer_moe_nmse_delta=tot)
 # Include all source masks and local diagnostics, no new quality claim.
 out=dict(status="R23H_LAYERWISE_HYBRID_PROPOSALS_EXPLORATORY_NOT_ADMITTED",teacher="0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94",source_archives=ARCH,source_masks={n:sha(canon(masks[n])) for n in NAMES},base_mask_sha256=sha(canon(base)),rows=[r["id"] for r in selected],method="per-layer choice among frozen original96/all8/all10/all12/all14/primary4_10/primary4_14 masks, minimizing local decode MoE-output NMSE proxy across both zh-my and my-zh",candidates=candidates,layer_deltas={n:{d:{str(l):deltas[n][d,l] for l in range(1,48)} for d in ("zh-my","my-zh")} for n in NAMES},limits="This is an exploratory recombination heuristic using fixed-fit layer-local metrics. Per-layer deltas are conditional on each source model's full mask and are not causal under recombination. The hybrids are untested; run fresh prefill screen and primary full-model propagation before any admission.")
 raw=canon(out)+b"\n";dst=root/"r23h-layer-hybrid-plan.json";dst.write_bytes(raw)
 print(json.dumps(dict(status=out["status"],sha256=sha(raw),candidates={k:dict(mask_sha256=v["mask_sha256"],source_counts=v["source_counts"],estimated_sum_layer_moe_nmse_delta=v["estimated_sum_layer_moe_nmse_delta"]) for k,v in candidates.items()}),sort_keys=True))
if __name__=="__main__":main()
